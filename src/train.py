"""DAY 2 학습·평가 파이프라인 (회귀: y = log10(cycle_life), 지표 = 역변환 후 MAPE).

실행 (프로젝트 루트에서):
    python -m src.train

순서 (가이드 docs/DAY2_따라치기_가이드.md 의 STEP 번호와 같다)
    STEP 1-2  피처 표 + 제외 규칙                 (features.py)
    STEP 3    Batch1 -> 프로토콜 단위 hold-out 분리 (seed 고정, Batch2를 보기 전에 정함)
    STEP 4-6  기준선 / CV 모델 / ablation         -> Batch1 비-hold-out 셀만 사용 (선택은 전부 여기서)
    STEP 7    선택 확정 후 hold-out, Batch2, Batch3를 한 번씩 평가하고 표 저장
    STEP 8    오류 분석 (셀별 예측, 가장 크게 틀린 5셀)

누수 방지 규칙
    - 스케일러는 Pipeline 안에 있어서, fit할 때마다 그 fit에 들어온 학습 셀에만 맞춰진다.
    - 하이퍼파라미터(GridSearchCV), 모델 선택, 피처 ablation은 Batch1 비-hold-out 셀의 CV만 본다.
    - Batch2/3는 run_selection()이 끝나 선택이 확정된 뒤에만 main()에서 처음 꺼낸다.
"""
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import plots
from src.features import AUX, CAND, CORE, build_feature_table, model_cells

SEED = 42             # hold-out 분할과 RandomForest에 쓰는 유일한 seed. 바꿔 가며 다시 돌리지 않는다.
HOLDOUT_FRAC = 0.2    # Batch1 프로토콜 20개 중 20% = 4개 프로토콜을 hold-out으로
N_OUTER = 4           # 바깥 CV: 남은 16개 프로토콜을 4묶음으로 (Train MAPE 계산용)
N_INNER = 3           # 안쪽 CV: GridSearchCV가 하이퍼파라미터를 고를 때 쓰는 fold 수
TARGET_MAPE = 9.1     # 원논문(과제 문서) 목표 MAPE (%)
CANDIDATES = ["ElasticNet", "Ridge", "RandomForest"]   # 최종 모델 후보 (Dummy는 기준선이라 후보 아님)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
# Gap 부호 규칙 (성능 표 머리말과 모든 Gap 비고에 그대로 쓴다)
GAP_RULE = "Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가"


def mape(life_true, log_pred):
    """예측 log10 수명을 10**로 되돌린 뒤 실제 cycle_life와 비교한 MAPE(%)."""
    life_true = np.asarray(life_true, dtype=float)
    life_pred = 10 ** np.asarray(log_pred, dtype=float)
    return float(np.mean(np.abs(life_pred - life_true) / life_true) * 100)


def _neg_mape_from_log(y_log_true, y_log_pred):
    # GridSearchCV는 '클수록 좋은' 점수를 고르므로 MAPE에 -를 붙인다
    return -mape(10 ** np.asarray(y_log_true), y_log_pred)


SCORER = make_scorer(_neg_mape_from_log)


# ---------------------------------------------------------------- STEP 3. 분할
def split_holdout(b1):
    """Batch1을 프로토콜(policy) 단위로 train / hold-out으로 나눈다. 같은 프로토콜은 한쪽에만 있다."""
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRAC, random_state=SEED)
    tr_idx, ho_idx = next(gss.split(b1, groups=b1.policy))
    train, hold = b1.iloc[tr_idx].copy(), b1.iloc[ho_idx].copy()
    assert not set(train.policy) & set(hold.policy), "같은 프로토콜이 train과 hold-out에 동시에 있음"
    print(f"[STEP 3] seed={SEED} | train {len(train)}셀 / {train.policy.nunique()}프로토콜"
          f" | hold-out {len(hold)}셀 / {hold.policy.nunique()}프로토콜")
    print("  hold-out 프로토콜별 셀 수:", hold.groupby("policy").size().to_dict())
    print(f"  hold-out 수명 범위: {hold.cycle_life.min():.0f} ~ {hold.cycle_life.max():.0f}")
    return train, hold


# ---------------------------------------------------------------- STEP 4-5. 모델
def make_model(name):
    """모델 이름 -> (Pipeline[스케일러 + 모델], 하이퍼파라미터 그리드). 그리드가 비면 튜닝 없음."""
    models = {
        "Dummy": (DummyRegressor(strategy="mean"), {}),          # 학습 셀 log 수명의 평균만 예측
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0]}),
        "ElasticNet": (ElasticNet(max_iter=50000),
                       {"model__alpha": [0.0001, 0.001, 0.01, 0.1], "model__l1_ratio": [0.1, 0.5, 0.9]}),
        # RF는 비교용이라 튜닝하지 않고 셀 36개에 맞게 얕은 나무로 고정
        "RandomForest": (RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3,
                                               random_state=SEED), {}),
    }
    est, grid = models[name]
    return Pipeline([("scaler", StandardScaler()), ("model", est)]), grid


def fit_model(name, X, y, groups):
    """그리드가 있으면 GroupKFold(프로토콜) GridSearchCV로 고른 뒤 X 전체로 다시 fit한다."""
    pipe, grid = make_model(name)
    if not grid:
        return pipe.fit(X, y)
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=N_INNER), scoring=SCORER)
    return gs.fit(X, y, groups=groups)


def cv_mape(name, data, feats):
    """Train(Batch1 CV) MAPE: 바깥 GroupKFold의 fold별 MAPE 평균.

    fold마다 fit_model을 새로 부르므로 스케일러와 하이퍼파라미터가 그 fold의 학습 부분만 본다
    (중첩 CV: 바깥 4-fold로 점수, 안쪽 3-fold로 튜닝).
    """
    X, y, g = data[feats], data.logL, data.policy
    scores = []
    for tr, va in GroupKFold(n_splits=N_OUTER).split(X, y, g):
        m = fit_model(name, X.iloc[tr], y.iloc[tr], g.iloc[tr])
        scores.append(mape(data.cycle_life.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


# ---------------------------------------------------------------- STEP 6. ablation
def ablation(name, train):
    """dQ_log_var 단독 -> +avgC -> +Tavg. Batch1 CV MAPE가 낮아질 때만 그 피처를 남긴다."""
    feats = [CORE]
    best = cv_mape(name, train, feats)
    if name == "Dummy":
        # Dummy는 피처를 무시하고 평균만 예측한다. 계산은 같은 코드(입력 = CORE 한 열)로 하되
        # 표에는 피처 '(none)', 채택 여부 '-'로 적어 '피처를 골랐다'는 오해를 막는다.
        return [{"model": name, "step": "1", "features": "(none)", "cv_mape": best, "kept": "-"}], feats, best
    rows = [{"model": name, "step": "1", "features": CORE, "cv_mape": best, "kept": True}]
    for i, f in enumerate([AUX, CAND], start=2):
        trial = feats + [f]
        score = cv_mape(name, train, trial)
        kept = score < best              # 멈춤 규칙: CV가 좋아지지 않으면 이 피처는 버린다
        rows.append({"model": name, "step": str(i), "features": "+".join(trial),
                     "cv_mape": score, "kept": kept})
        if kept:
            feats, best = trial, score
    return rows, feats, best


def run_selection(train):
    """모든 모델의 ablation을 돌리고, 후보 중 최종 CV MAPE가 가장 낮은 (모델, 피처)를 고른다."""
    rows, finals = [], {}
    for name in ["Dummy"] + CANDIDATES:
        r, feats, best = ablation(name, train)
        rows += r
        finals[name] = (feats, best)
        for x in r:
            label = {True: "채택", False: "CV 기준 버림"}.get(x["kept"], "-")   # Dummy는 '-'
            print(f"  {name:12s} step {x['step']} {x['features']:45s} CV MAPE {x['cv_mape']:6.2f}%"
                  f"  {label}")
    best_name = min(CANDIDATES, key=lambda n: finals[n][1])
    feats, cv = finals[best_name]
    print(f"[STEP 6] 최종 선택(Batch1 CV만으로): {best_name} + {feats} | CV MAPE {cv:.2f}%")
    return pd.DataFrame(rows), (best_name, feats, cv)


# ---------------------------------------------------------------- STEP 7. 최종 평가
def predict_table(model, feats, data):
    """셀별 예측 표: cell_id, protocol, y_true(cycle_life), y_pred(역변환), 절대 % 오차."""
    out = data[["cell_id", "batch", "policy", "newstructure", "protrusion", "cycle_life"]].copy()
    out = out.rename(columns={"policy": "protocol", "cycle_life": "y_true"})
    out["y_pred"] = 10 ** model.predict(data[feats])
    out["abs_pct_err"] = np.abs(out.y_pred - out.y_true) / out.y_true * 100
    return out.sort_values("abs_pct_err", ascending=False).reset_index(drop=True)


def record_comparison(comp, train, hold, b2):
    """선택이 끝난 뒤, 각 (모델, 단계)를 train에 fit해 hold-out과 Batch2 MAPE를 '기록만' 한다."""
    hold_m, b2_m = [], []
    for _, r in comp.iterrows():
        # Dummy 행은 피처 '(none)'로 적혀 있지만, 계산은 ablation과 같은 입력(CORE 한 열)으로 한다
        feats = [CORE] if r.model == "Dummy" else r.features.split("+")
        m = fit_model(r.model, train[feats], train.logL, train.policy)
        hold_m.append(mape(hold.cycle_life, m.predict(hold[feats])))
        b2_m.append(mape(b2.cycle_life, m.predict(b2[feats])))
    comp = comp.assign(holdout_mape=hold_m, b2_mape_recorded_only=b2_m)
    return comp.round(2)


def performance_table(cv, valid, test2, test3):
    """과제 표준 리포트 표. 단위는 %p.

    부호 규칙 (GAP_RULE): Gap = 뒤 항목 MAPE - 앞 항목 MAPE. 행 이름의 '앞-뒤' 순서와 반대로 뺀다.
    MAPE는 낮을수록 좋으므로 (+) = 뒤 항목의 오차가 더 큼 = 오차 증가.
        Gap(Train-Valid)        = Valid - Train
        Gap(Valid-Test)         = Test(Batch2) - Valid
        Gap(Target-Test)        = Test(Batch2) - 9.1
        Gap(Batch2-Batch3)      = Test(Batch3) - Test(Batch2)
        Gap(Target-Test Batch3) = Test(Batch3) - 9.1
    """
    rule = f" [{GAP_RULE}]"
    rows = [
        ("Train (Batch1 CV)", cv, "프로토콜 GroupKFold 4-fold 평균 (중첩 CV)"),
        ("Valid (Batch1 Hold-out)", valid, "프로토콜 단위 hold-out"),
        ("Test (Batch2)", test2, "1차 테스트셋 최종 성능 (39셀, 돌출 셀 포함)"),
        ("Gap (Train-Valid)", valid - cv, "Valid − Train. (+) : 과적합 의심" + rule),
        ("Gap (Valid-Test)", test2 - valid, "Test − Valid. (+) : 배치 간 일반화 저하 의심" + rule),
        ("Gap (Target-Test)", test2 - TARGET_MAPE,
         f"Test − Target. (+): 목표 미달; Target : 원논문 {TARGET_MAPE}%" + rule),
        ("Test (Batch3)", test3, "선택: 2차 테스트셋 (44셀, 돌출 셀 포함)"),
        ("Gap (Batch2-Batch3)", test3 - test2,
         "선택: Batch3 − Batch2. (+) : Batch3 오차가 Batch2보다 큼, (−) : Batch3 오차가 더 작음" + rule),
        ("Gap (Target-Test Batch3)", test3 - TARGET_MAPE,
         f"선택: Test(Batch3) − Target. (+): 목표 미달; Target : 원논문 {TARGET_MAPE}%" + rule),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def sensitivity_table(pred2, pred3):
    """민감도 확인 전용: ΔQ 돌출 셀을 뺐을 때 MAPE. 어떤 선택에도 쓰지 않는다."""
    rows = []
    for name, p in [("Batch2", pred2), ("Batch3", pred3)]:
        keep = p[~p.protrusion]
        rows.append({"batch": name, "n_all": len(p), "mape_all": p.abs_pct_err.mean(),
                     "n_protrusion": int(p.protrusion.sum()), "n_without": len(keep),
                     "mape_without_protrusion": keep.abs_pct_err.mean()})
    return pd.DataFrame(rows).round(2)


def to_markdown(df):
    """tabulate 없이 DataFrame을 마크다운 표 문자열로 바꾼다."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- STEP 8. 오류 분석
def error_summary(pred, train_range):
    """가장 크게 틀린 5셀과, 셀 무리별 평균 오차. (원인은 가설일 뿐이다)"""
    p = pred.assign(life_vs_train=np.select(
        [pred.y_true < train_range[0], pred.y_true > train_range[1]],
        ["below", "above"], "inside"))
    print("  가장 크게 틀린 5셀:")
    cols = ["cell_id", "protocol", "newstructure", "protrusion", "life_vs_train",
            "y_true", "y_pred", "abs_pct_err"]
    print(p[cols].head(5).round(1).to_string(index=False))
    print("  newstructure 여부별 평균 오차(%):",
          p.groupby("newstructure").abs_pct_err.agg(["size", "mean"]).round(1).to_dict("index"))
    print(f"  학습 수명 범위({train_range[0]:.0f}~{train_range[1]:.0f}) 대비 위치별 평균 오차(%):",
          p.groupby("life_vs_train").abs_pct_err.agg(["size", "mean"]).round(1).to_dict("index"))
    print("  예측 > 실제(과대예측) 비율: {:.0f}%".format(100 * (p.y_pred > p.y_true).mean()))
    return p


# ---------------------------------------------------------------- 전체 실행
def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def main():
    clean = model_cells(build_feature_table())                         # STEP 1-2
    b1 = clean[clean.batch == "b1"]
    train, hold = split_holdout(b1)                                     # STEP 3
    comp, (name, feats, cv) = run_selection(train)                      # STEP 4-6 (train만 사용)
    # ===== 여기서 선택이 확정된다. 아래부터 hold-out, Batch2, Batch3를 처음이자 한 번 평가 =====
    b2, b3 = clean[clean.batch == "b2"], clean[clean.batch == "b3"]
    # refit 정책: 최종 모델은 hold-out을 뺀 Batch1(train)에만 fit하고, 그 '같은 모델'로
    # Valid(hold-out)와 Test(Batch2/3)를 평가한다. hold-out을 다시 학습에 넣으면 Valid와 Test가
    # 서로 다른 모델의 점수가 되므로, 셀 7개를 덜 쓰는 대신 두 점수가 같은 모델을 재도록 했다.
    final = fit_model(name, train[feats], train.logL, train.policy)
    pred_ho, pred2, pred3 = (predict_table(final, feats, d) for d in (hold, b2, b3))
    perf = performance_table(cv, pred_ho.abs_pct_err.mean(), pred2.abs_pct_err.mean(),
                             pred3.abs_pct_err.mean())
    comp = record_comparison(comp, train, hold, b2)
    sens = sensitivity_table(pred2, pred3)
    for df, fn in [(perf, "model_performance.csv"), (comp, "model_comparison.csv"),
                   (pred2, "predictions_batch2.csv"), (pred3, "predictions_batch3.csv"),
                   (pred_ho, "predictions_holdout.csv"), (sens, "sensitivity_protrusion.csv")]:
        save(df, fn)
    with open(os.path.join(RESULTS_DIR, "model_performance.md"), "w") as fh:
        fh.write(f"최종 모델: {name} + {feats} (seed={SEED})\n\n"
                 f"Gap 부호 규칙: {GAP_RULE}. 단위는 %p.\n\n" + to_markdown(perf) + "\n")
    report(name, feats, final, perf, comp, sens, pred2, pred3, train, b1)
    plots.make_all(pred2, pred3, comp, (train.cycle_life.min(), train.cycle_life.max()))


def report(name, feats, final, perf, comp, sens, pred2, pred3, train, b1):
    """STEP 7-8 결과를 화면에 출력한다."""
    best = final.best_params_ if hasattr(final, "best_params_") else "고정 파라미터"
    print(f"[STEP 7] 최종 모델 {name} {feats} | 하이퍼파라미터 {best}")
    print("Gap 부호 규칙:", GAP_RULE)
    print(perf.to_string(index=False))
    print("  목표 9.1% 달성(Batch2):", bool(perf.iloc[2, 1] <= TARGET_MAPE))
    print("모델 비교 (b2_mape_recorded_only 는 기록만, 선택에 사용 안 함):")
    print(comp.to_string(index=False))
    print("민감도(돌출 셀 제외, 선택에 사용 안 함):")
    print(sens.to_string(index=False))
    rng = (train.cycle_life.min(), train.cycle_life.max())
    # 오류 분석의 below/inside/above 기준은 최종 모델이 실제로 본 학습 셀(hold-out 제외)의 범위다.
    # Batch1 전체(36셀)의 최대 1074는 hold-out 셀이라 학습 범위에 들어가지 않는다.
    print(f"[STEP 8] 학습 셀({len(train)}셀) 수명 범위 {rng[0]:.0f}~{rng[1]:.0f}"
          f" | 참고: Batch1 전체({len(b1)}셀) 범위 {b1.cycle_life.min():.0f}~{b1.cycle_life.max():.0f}"
          f" (최대 {b1.cycle_life.max():.0f} 셀은 hold-out에 있음)")
    for tag, p in [("Batch2", pred2), ("Batch3", pred3)]:
        print(f"--- {tag} 오류 분석")
        error_summary(p, rng)


if __name__ == "__main__":
    main()
