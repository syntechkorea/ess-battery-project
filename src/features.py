"""DAY 2 피처 테이블 만들기 (셀 1개 = 1행).

실행 (프로젝트 루트에서):
    python -m src.features

하는 일
    1) 캐시(data/cache/b*.pkl)를 읽는다                      -> src/load_data.load_all
    2) 셀 정보 + 전처리 플래그 표를 만든다                    -> src/eda_utils.cell_table
    3) cycle 100 이하 데이터로만 피처를 계산한다              -> src/eda_utils.early_features
    4) cycle 10 실측 충전 전류(avgC_0to80_meas)를 붙인다      -> src/eda_utils.charge_current_table
    5) 제외 규칙(타깃을 믿을 수 없는 셀)을 표시한다           -> src/eda_utils.exclusion_rule
새로 만드는 계산은 'ΔQ 돌출 셀 표시' 하나뿐이다 (01_EDA.ipynb Q3와 같은 정의).

누수 방지: 여기서 만드는 피처는 cycle_life, n_cycles, knee를 전혀 읽지 않는다.
"""
import numpy as np
import pandas as pd

from src import eda_utils as U          # DAY 1에서 만든 함수를 그대로 쓴다 (다시 짜지 않는다)
from src.load_data import load_all

# DAY 1 보고서 4-2의 피처 구성: 핵심 1 + 보조 1 + 후보 1
CORE = "dQ_log_var"                     # ΔQ(V)=Q100-Q10 분산의 log10
AUX = "avgC_0to80_meas"                 # cycle 10 실측 평균 충전 속도 (SOC 0->80%)
CAND = "Tavg_mean_2_100"                # cycle 2~100 평균 온도 (후보: Batch1 CV로만 채택 여부 결정)
FEATURES = [CORE, AUX, CAND]

PROTRUSION_AH = 0.002                   # ΔQ 최대값이 +2mAh 초과 = '돌출 셀' (EDA Q3와 같은 기준)


def build_feature_table(all_cells=None):
    """139셀 전체의 표를 만든다. 제외 셀도 지우지 않고 excl 열로만 표시한다."""
    if all_cells is None:
        all_cells = load_all()
    df = U.cell_table(all_cells)                        # cell_id, batch, policy, cycle_life, 플래그 ...
    rows = []
    for cells in all_cells.values():
        for c in cells:
            f = U.early_features(c)                     # cycle <= 100 만 사용
            rows.append({"cell_id": c["cell_id"], CORE: f[CORE], CAND: f[CAND],
                         "protrusion": bool(np.max(U.delta_q(c)) > PROTRUSION_AH)})
    df = df.merge(pd.DataFrame(rows), on="cell_id")
    cc = U.charge_current_table(all_cells)              # 캐시 CSV(data/cache/charge_current_c10.csv)
    df = df.merge(cc[["cell_id", AUX]], on="cell_id")
    df["excl"] = U.exclusion_rule(df)                   # NaN 수명 또는 80% 미도달
    df["logL"] = np.log10(df.cycle_life)                # 타깃 y = log10(cycle_life)
    return df


def model_cells(df):
    """모델링에 쓰는 셀만 남긴다 (제외 규칙 적용). 배치별 셀 수와 '왜 빠졌나'를 출력한다."""
    clean = df[~df.excl].copy()
    reason = df[df.excl].assign(
        nan_life=lambda d: d.cycle_life_missing,
        not_80pct_only=lambda d: ~d.cycle_life_missing & d.no_80pct_in_data)
    print("모델링 셀 수:", clean.groupby("batch").size().to_dict())
    print("제외 이유 (nan_life = 수명 NaN, not_80pct_only = 수명은 있으나 80% 미도달):")
    print(reason.groupby("batch")[["nan_life", "not_80pct_only"]].sum().to_string())
    return clean


def check_features(clean):
    """피처에 NaN이 없는지, 배치별로 몇 개인지 확인한다 (Batch2/3는 개수만 본다)."""
    n_nan = int(clean[FEATURES].isna().sum().sum())
    print("피처 NaN 개수 (0이어야 함):", n_nan)
    assert n_nan == 0, "피처에 NaN이 있음"
    b1 = clean[clean.batch == "b1"]
    print("Batch1 피처 요약:")
    print(b1[FEATURES + ["cycle_life"]].describe().round(4).to_string())
    print("Batch1 피처 vs log10(수명) Pearson r:",
          b1[FEATURES].corrwith(b1.logL).round(3).to_dict())


if __name__ == "__main__":
    table = build_feature_table()
    clean = model_cells(table)
    check_features(clean)
