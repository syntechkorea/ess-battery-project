"""Flag (never delete) problematic cells. Output: data/cache/cell_flags.csv.

Flags (one boolean column each, plus a joined 'reason' column):
    cycle_life_missing    cycle_life is NaN
    no_80pct_in_data      summary QD never drops to 80% of nominal capacity (1.1Ah), so the
                          reported cycle_life is not backed by a measured 80% crossing
    short_summary         fewer than MIN_CYCLES cycles, or summary length != raw cycle count
    summary_anomaly       NaN values in QD/IR within the first 100 cycles, or QD above 1.3Ah
                          (nominal is 1.1Ah) in any cycle
    b1_continuation_candidate  (Batch2 only) same policy string as a Batch1 cell AND the Batch2
                          first-cycle QD/IR continue where that Batch1 cell ended.
                          The data does not state continuation explicitly: barcode/channel_id
                          are unreadable opaque objects, so any candidate is 'unverified'.
"""
import os

import numpy as np
import pandas as pd

from src.load_data import CACHE_DIR, load_all

NOMINAL_AH = 1.1
EOL_AH = 0.8 * NOMINAL_AH
EOL_TOL = 0.01          # QD this close above 0.88Ah counts as reaching 80%
MIN_CYCLES = 150
QD_MAX_AH = 1.3
CONT_QD_TOL = 0.02      # |QD_b2[first] - QD_b1[last]| in Ah
CONT_IR_TOL = 0.002     # |IR_b2[first] - IR_b1[last]| in Ohm


def _first_valid(a):
    a = a[(a > 0) & np.isfinite(a)]
    return a[0] if a.size else np.nan


def _last_valid(a):
    a = a[(a > 0) & np.isfinite(a)]
    return a[-1] if a.size else np.nan


def continuation_candidates(b1_cells, b2_cell):
    """Batch1 cell ids with the same policy whose end state matches this Batch2 cell's start."""
    out = []
    for c1 in b1_cells:
        if c1["policy"] != b2_cell["policy"]:
            continue
        dq = abs(_first_valid(b2_cell["summary"]["QD"]) - _last_valid(c1["summary"]["QD"]))
        dir_ = abs(_first_valid(b2_cell["summary"]["IR"]) - _last_valid(c1["summary"]["IR"]))
        if dq <= CONT_QD_TOL and dir_ <= CONT_IR_TOL:
            out.append(c1["cell_id"])
    return out


def flag_cells(all_cells):
    rows = []
    for batch, cells in all_cells.items():
        for c in cells:
            s = c["summary"]
            qd, ir = s["QD"], s["IR"]
            qd_min = np.nanmin(qd[qd > 0]) if (qd > 0).any() else np.nan
            f = {}
            f["cycle_life_missing"] = bool(np.isnan(c["cycle_life"]))
            f["no_80pct_in_data"] = bool(not qd_min <= EOL_AH + EOL_TOL)
            f["short_summary"] = bool(c["n_cycles"] < MIN_CYCLES or len(qd) != c["n_cycles"])
            nan_early = np.isnan(qd[1:100]).any() or np.isnan(ir[1:100]).any()
            f["summary_anomaly"] = bool(nan_early or (qd > QD_MAX_AH).any())
            cont = continuation_candidates(all_cells["b1"], c) if batch == "b2" else []
            f["b1_continuation_candidate"] = bool(cont)
            reasons = [k for k, v in f.items() if v]
            if cont:
                reasons[reasons.index("b1_continuation_candidate")] = (
                    "b1_continuation_candidate(unverified:" + "|".join(cont) + ")")
            rows.append({
                "cell_id": c["cell_id"], "batch": batch, "policy": c["policy"],
                "cycle_life": c["cycle_life"], "n_cycles": c["n_cycles"],
                "qd_min": qd_min, "qd_last": qd[-1], **f,
                "flag_any": bool(reasons), "reason": ";".join(reasons),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = flag_cells(load_all())
    df.to_csv(os.path.join(CACHE_DIR, "cell_flags.csv"), index=False)
    flag_cols = [c for c in df.columns if c in (
        "cycle_life_missing", "no_80pct_in_data", "short_summary",
        "summary_anomaly", "b1_continuation_candidate")]
    print(df.groupby("batch")[flag_cols + ["flag_any"]].sum())
    print(df.loc[df.flag_any, ["cell_id", "cycle_life", "n_cycles", "qd_min", "reason"]].to_string(index=False))
