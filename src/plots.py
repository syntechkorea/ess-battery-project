"""DAY 2 그림 3종 (01_EDA.ipynb와 같은 스타일: AppleGothic, 배치 고정 색, 왼쪽 정렬 제목).

저장 위치: results/figures/  (python -m src.train 이 make_all()을 부른다)
    M1_pred_vs_actual.png   예측 vs 실제 (Batch2 / Batch3, 대각선 = 완벽한 예측)
    M2_error_by_cell.png    셀별 절대 % 오차 (큰 순서)
    M3_ablation.png         피처 단계별 Batch1 CV MAPE (Batch2는 기록만, 회색)
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
BC = {"b1": "#2a78d6", "b2": "#eb6834", "b3": "#1baf7a"}    # 01_EDA와 같은 배치 색
BC_NEW = {"b2": "#7a2f12", "b3": "#0b5e40"}                 # newstructure 셀 = 같은 배치 색의 진한 버전
GREY = "#9a9a96"


def set_style():
    mpl.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "figure.dpi": 90,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "lines.linewidth": 1.2, "figure.constrained_layout.use": True})


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight")


def batch_name(b):
    return "Batch2" if b == "b2" else "Batch3"


def plot_pred_vs_actual(pred2, pred3, train_range):
    """log 축 산점도. 원 = 기존 구조, 세모 = newstructure, 빈 마커 = ΔQ 돌출 셀, 회색 띠 = 학습 수명 범위."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (b, p) in zip(axes, [("b2", pred2), ("b3", pred3)]):
        ax.axvspan(*train_range, color="#f1f0ec", zorder=0)
        # 범례는 직접 만든다: 그 배치에 실제로 있는 셀 무리만 넣는다 (Batch3에는 기존 구조 셀이 없음)
        handles = [Patch(color="#f1f0ec", label=f"학습 셀 수명 범위 ({train_range[0]:.0f}~{train_range[1]:.0f})")]
        for ns, mk, lab in [(False, "o", "기존 구조"), (True, "^", "newstructure")]:
            g = p[p.newstructure == ns]
            if g.empty:
                continue
            face = np.where(g.protrusion, "white", BC[b])
            ax.scatter(g.y_true, g.y_pred, s=30, marker=mk, facecolors=face, edgecolors=BC[b])
            handles.append(Line2D([], [], ls="", marker=mk, color=BC[b], label=f"{lab} ({len(g)}셀)"))
        if p.protrusion.any():
            handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=BC[b],
                                  label=f"ΔQ 돌출 셀 (빈 마커, {int(p.protrusion.sum())}셀)"))
        ax.plot([300, 2100], [300, 2100], color="#52514e", ls="--", lw=1)
        handles.append(Line2D([], [], color="#52514e", ls="--", lw=1, label="y = x"))
        ax.set(xscale="log", yscale="log", xlim=(300, 2100), ylim=(300, 2100),
               xlabel="실제 cycle_life", ylabel="예측 cycle_life")
        ax.set_title(f"{batch_name(b)}  MAPE {p.abs_pct_err.mean():.1f}%  (n={len(p)})")
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("M1. 예측 vs 실제 (빈 마커 = ΔQ 돌출 셀, 점선 = 완벽한 예측)", x=0.01, ha="left")
    savefig(fig, "M1_pred_vs_actual.png")
    return fig


def plot_error_by_cell(pred2, pred3):
    """셀별 절대 % 오차 막대 (큰 순서). 밝은 막대 = 기존 구조, 진한 막대 = newstructure, 빗금 = 돌출 셀."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4))
    for ax, (b, p) in zip(axes, [("b2", pred2), ("b3", pred3)]):
        bars = ax.bar(p.cell_id, p.abs_pct_err, color=np.where(p.newstructure, BC_NEW[b], BC[b]))
        for bar, pr in zip(bars, p.protrusion):
            bar.set_hatch("///" if pr else "")
        ax.axhline(p.abs_pct_err.mean(), color="#52514e", ls="--", lw=1)
        # 패널마다 실제로 있는 무리만 범례에 쓴다 (Batch3 = 44셀 모두 newstructure)
        n_old, n_new = int((~p.newstructure).sum()), int(p.newstructure.sum())
        handles = []
        if n_old:
            handles.append(Patch(color=BC[b], label=f"기존 구조 ({n_old}셀)"))
        if n_new:
            handles.append(Patch(color=BC_NEW[b], label=f"newstructure ({n_new}셀)"))
        handles.append(Patch(facecolor="white", edgecolor="#52514e", hatch="///", label="ΔQ 돌출 셀"))
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
        ax.set_ylabel("절대 % 오차")
        ax.set_title(f"{batch_name(b)} (점선 = 평균 {p.abs_pct_err.mean():.1f}%)", fontsize=10, loc="left")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.suptitle("M2. 셀별 오차 (밝은 막대 = 기존 구조, 진한 막대 = newstructure, 빗금 = ΔQ 돌출 셀)",
                 x=0.01, ha="left")
    savefig(fig, "M2_error_by_cell.png")
    return fig


def plot_ablation(comp):
    """모델 x 피처 단계별 MAPE. 선택 기준은 Batch1 CV(진한 막대)뿐이다."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    labels = comp.model + "\n" + comp.features.str.replace("+", "\n+", regex=False)
    x = np.arange(len(comp))
    w = 0.27
    ax.bar(x - w, comp.cv_mape, w, color=BC["b1"], label="Batch1 CV (선택 기준)")
    ax.bar(x, comp.holdout_mape, w, color="#9cc2ef", label="Batch1 hold-out (Valid)")
    ax.bar(x + w, comp.b2_mape_recorded_only, w, color=GREY, label="Batch2 (기록만, 선택에 안 씀)")
    for xi, kept, cv in zip(x, comp.kept, comp.cv_mape):
        if str(kept) == "False":            # Dummy('-')와 채택(True)에는 표시하지 않는다
            ax.text(xi - w, cv + 0.8, "CV 기준\n버림", ha="center", va="bottom", fontsize=7, color=BC["b1"])
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, comp.b2_mape_recorded_only.max() * 1.25)    # 위쪽 범례와 막대가 겹치지 않게
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("M3. 피처 단계별 비교 (ablation): 단계 = dQ_log_var → +avgC → +Tavg (Dummy는 피처 없음)",
                 x=0.01, ha="left")
    savefig(fig, "M3_ablation.png")
    return fig


def make_all(pred2, pred3, comp, train_range):
    set_style()
    for fig in (plot_pred_vs_actual(pred2, pred3, train_range), plot_error_by_cell(pred2, pred3),
                plot_ablation(comp)):
        plt.close(fig)
    print("그림 저장:", os.path.relpath(FIG_DIR), sorted(f for f in os.listdir(FIG_DIR) if f.startswith("M")))
