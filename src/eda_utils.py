"""Reusable helpers for notebooks/01_EDA.ipynb.

Cycle indexing convention (used everywhere in this module):
    "cycle n" = the dataset's own summary['cycle'] label n = cache row n-1.
    So cycle 10 -> Qdlin row 9, cycle 100 -> Qdlin row 99, cycle 2 -> row 1.
    In Batch1 row 0 (= cycle 1) is an empty MATLAB entry (Qdlin NaN, QD=0); cycle 2 onward is valid in all batches.

Nothing in early_features() reads cycle_life, so its outputs are safe as model inputs.
"""
import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from src.load_data import BATCH_FILES, CACHE_DIR, DATA_DIR

NOMINAL_AH = 1.1
EOL_AH = 0.8 * NOMINAL_AH
QD_VALID = (0.5, 1.3)          # summary QD outside this range is treated as a spike / empty cycle
QDLIN_MAX = 1.3                 # Qdlin rows whose max exceeds this are corrupt rows
ROW = {"c2": 1, "c10": 9, "c100": 99}
FIG_DIR = os.path.join(os.path.dirname(CACHE_DIR), "..", "results", "figures")
FIG_DIR = os.path.normpath(FIG_DIR)
POLICY_RE = re.compile(r"^([\d.]+)C\((\d+)%\)-([\d.]+)C")


def parse_policy(policy):
    """'5.4C(50%)-3.6C' -> C1=5.4, Q1=50, C2=3.6 and the nominal average C-rate from 0 to 80% SOC.

    avg_C_0to80 = 0.8 / (Q1/100/C1 + (80-Q1)/100/C2); VarCharge policies have no fixed C1/C2 (NaN).
    """
    m = POLICY_RE.match(policy)
    out = {
        "varcharge": policy.startswith("VarCharge"),
        "slowcycle": "SLOWCYCLE" in policy,
        "newstructure": "newstructure" in policy,
        "C1": np.nan, "Q1": np.nan, "C2": np.nan, "avg_C_0to80": np.nan,
    }
    if m:
        c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
        hours = q1 / 100 / c1 + max(80 - q1, 0) / 100 / c2
        out.update(C1=c1, Q1=q1, C2=c2, avg_C_0to80=0.8 / hours)
    return out


def cell_table(all_cells, flags_csv=os.path.join(CACHE_DIR, "cell_flags.csv")):
    """One row per cell: id, batch, policy, cycle_life, n_cycles, parsed policy, preprocess flags."""
    rows = []
    for batch, cells in all_cells.items():
        for c in cells:
            rows.append({"cell_id": c["cell_id"], "batch": batch, "policy": c["policy"],
                         "cycle_life": c["cycle_life"], "n_cycles": c["n_cycles"], **parse_policy(c["policy"])})
    df = pd.DataFrame(rows)
    flags = pd.read_csv(flags_csv)
    keep = ["cell_id", "qd_min", "qd_last", "cycle_life_missing", "no_80pct_in_data",
            "short_summary", "summary_anomaly", "flag_any", "reason"]
    df = df.merge(flags[keep], on="cell_id", how="left")
    df["censored_b1"] = (df.batch == "b1") & df.no_80pct_in_data & ~df.cycle_life_missing
    return df


def exclusion_rule(df):
    """Model-time exclusion (EDA keeps every cell): no target, or target not backed by a measured 80% crossing."""
    return df.cycle_life_missing | df.no_80pct_in_data


def clean_qd(qd):
    """Summary QD with empty cycles and spikes (outside QD_VALID) set to NaN."""
    q = qd.astype(float).copy()
    q[(q < QD_VALID[0]) | (q > QD_VALID[1])] = np.nan
    return q


def delta_q(cell, a="c100", b="c10"):
    """Q_a(V) - Q_b(V) on the Vdlin grid."""
    q = cell["Qdlin"].astype(float)
    return q[ROW[a]] - q[ROW[b]]


def early_features(cell):
    """Early-cycle (<= cycle 100) features for one cell. Does not use cycle_life."""
    s = cell["summary"]
    dq = delta_q(cell)
    qd = clean_qd(s["QD"])[:100]
    cyc = np.arange(1, 101)
    ok = np.isfinite(qd) & (cyc >= 2)
    slope, icpt = np.polyfit(cyc[ok], qd[ok], 1)
    ir = s["IR"][:100].astype(float)
    ir[~(ir > 0)] = np.nan                       # IR == 0 means "not measured" (all of b2c40-c46)
    tavg = s["Tavg"][1:100].astype(float)
    tavg[(tavg <= 0) | (tavg > 80)] = np.nan     # b2 has logging spikes (e.g. -8.7 / 73.6 degC)
    tmax = s["Tmax"][1:100].astype(float)
    tmax[(tmax <= 0) | (tmax > 80)] = np.nan
    nanmax = lambda a: np.nanmax(a) if np.isfinite(a).any() else np.nan
    return {
        "dQ_log_var": np.log10(np.var(dq)),
        "dQ_log_abs_min": np.log10(abs(np.min(dq))),
        "dQ_log_abs_mean": np.log10(abs(np.mean(dq))),
        "dQ_skew": stats.skew(dq),
        "dQ_kurt": stats.kurtosis(dq),
        "dQ_at_2V": dq[-1],
        "QD_c2": qd[1],
        "QD_max_minus_c2": np.nanmax(qd[1:100]) - qd[1],
        "QD_c100_minus_c2": qd[99] - qd[1],
        "QD_slope_2_100": slope,
        "QD_icpt_2_100": icpt,
        "IR_c2": ir[1],
        "IR_min_2_100": np.nanmin(ir[1:100]) if np.isfinite(ir[1:100]).any() else np.nan,
        "IR_c100_minus_c2": ir[99] - ir[1],
        "Tavg_mean_2_100": np.nanmean(tavg) if np.isfinite(tavg).any() else np.nan,
        "Tmax_max_2_100": nanmax(tmax),
        "chargetime_mean_2_6": np.nanmean(s["chargetime"][1:6]),
    }


def knee_point(qd, end=None, start=2, min_seg=30, smooth=5):
    """Two-segment continuous piecewise-linear fit (Bacon-Watts limit) on a median-smoothed QD curve.

    Uses cycles start..end (end = last cycle to consider). Returns knee cycle and pre/post fade rates (Ah/cycle).
    EDA only: uses the whole life curve, so it must never become a model feature.
    """
    q = clean_qd(qd)
    if end is not None:
        q = q[: int(end)]
    x = np.arange(1, len(q) + 1).astype(float)
    m = np.isfinite(q) & (x >= start)
    x, y = x[m], pd.Series(q[m]).rolling(smooth, center=True, min_periods=1).median().to_numpy()
    if len(x) < 2 * min_seg + 1:
        return {"knee": np.nan, "rate_pre": np.nan, "rate_post": np.nan}
    best = (np.inf, None)
    for k in range(min_seg, len(x) - min_seg, 2):
        xb = x[k]
        A = np.column_stack([np.ones_like(x), x, np.clip(x - xb, 0, None)])
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = res[0] if res.size else np.sum((A @ coef - y) ** 2)
        if sse < best[0]:
            best = (sse, (xb, coef))
    xb, coef = best[1]
    return {"knee": xb, "rate_pre": -coef[1], "rate_post": -(coef[1] + coef[2])}


def vif(X):
    """Variance inflation factor per column of a DataFrame (OLS of each column on the others)."""
    Z = (X - X.mean()) / X.std(ddof=0)
    out = {}
    for col in Z.columns:
        y = Z[col].to_numpy()
        A = np.column_stack([np.ones(len(Z)), Z.drop(columns=col).to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
        out[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out, name="VIF")


def read_charge_current(batch, idx, cycle=10):
    """Lazy h5py read of raw I/t/Qc for one cycle of one cell (I is stored in C-rate units).

    Returns dict with the raw arrays and summary numbers: peak charge C-rate, minutes to 80% SOC
    (Qc >= 0.88 Ah) and the measured average C-rate over 0-80% SOC.
    """
    row = ROW.get(f"c{cycle}", cycle - 1)
    with h5py.File(os.path.join(DATA_DIR, BATCH_FILES[batch]), "r") as f:
        cyc = f[f["batch"]["cycles"][idx, 0]]
        arr = {k: f[cyc[k][row, 0]][()].ravel() for k in ("I", "t", "Qc")}
    k80 = np.argmax(arr["Qc"] >= EOL_AH) if (arr["Qc"] >= EOL_AH).any() else None
    t80 = arr["t"][k80] if k80 else np.nan
    pre = arr["I"][: k80 + 1] if k80 else arr["I"]
    return {**arr, "I_peak_C": float(np.nanmax(pre)), "t_to80_min": float(t80),
            "avgC_0to80_meas": 0.8 / (t80 / 60) if k80 else np.nan}


def charge_current_table(all_cells, cycle=10, cache=os.path.join(CACHE_DIR, "charge_current_c10.csv")):
    """Measured charge-current descriptors for every cell at one early cycle (cached as CSV after first read)."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for batch, cells in all_cells.items():
        for c in cells:
            r = read_charge_current(batch, c["index"], cycle)
            rows.append({"cell_id": c["cell_id"], "I_peak_C": r["I_peak_C"],
                         "t_to80_min": r["t_to80_min"], "avgC_0to80_meas": r["avgC_0to80_meas"]})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight")


def read_qdlin_row(batch, idx, raw_row):
    """Lazy h5py read of one raw Qdlin row (raw_row is the 0-based index in the MATLAB cycles struct)."""
    with h5py.File(os.path.join(DATA_DIR, BATCH_FILES[batch]), "r") as f:
        cyc = f[f["batch"]["cycles"][idx, 0]]
        return f[cyc["Qdlin"][raw_row, 0]][()].ravel().astype(float)
