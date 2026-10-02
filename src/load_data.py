"""Load Severson et al. (2019) batch .mat files (MATLAB v7.3 / HDF5) into per-batch pickle caches.

Each cell is read lazily through h5py (object references are dereferenced one cell at a time),
so the ~8GB of raw data is never held in memory.

Kept per cell (dict):
    cell_id, batch, index        e.g. 'b1c0', 'b1', 0
    cycle_life                   float (NaN if missing)
    policy                       readable charge policy string, e.g. '3.6C(80%)-3.6C'
    summary                      dict of 1-D arrays over all cycles:
                                 QD, QC, IR, Tmax, Tavg, Tmin, chargetime, cycle
    n_cycles                     number of entries in the raw cycles struct
    Vdlin                        (1000,) voltage grid for Qdlin / dQdV
    Qdlin, Tdlin, dQdV           (100, 1000) float32, first 100 cycles (rows of NaN = empty cycle)

Not kept: raw per-cycle V, I, T, Qd, Qc, t (~1000 points x 6 arrays x 100 cycles x 139 cells
is a few hundred MB and the linearly interpolated Qdlin/Tdlin cover the early-cycle features).
The 'barcode' and 'channel_id' fields are stored as MATLAB opaque string objects and read back as
uint32 metadata (no readable text), so they are not kept.
"""
import os
import pickle

import h5py
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
N_EARLY = 100
N_GRID = 1000
SUMMARY_FIELDS = {
    "QD": "QDischarge", "QC": "QCharge", "IR": "IR", "Tmax": "Tmax",
    "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime", "cycle": "cycle",
}


def _text(f, ref):
    return "".join(chr(c) for c in f[ref][()].ravel())


def _early_matrix(f, refs):
    """Stack the first N_EARLY per-cycle arrays into (N_EARLY, N_GRID); empty cycles stay NaN."""
    out = np.full((N_EARLY, N_GRID), np.nan, dtype=np.float32)
    for i in range(min(N_EARLY, refs.shape[0])):
        arr = f[refs[i, 0]][()]
        if arr.size == N_GRID:
            out[i] = arr.ravel()
    return out


def read_cell(f, batch, i):
    b = f["batch"]
    cycles = f[b["cycles"][i, 0]]
    summ = f[b["summary"][i, 0]]
    cl = f[b["cycle_life"][i, 0]][()].ravel()
    return {
        "cell_id": f"{batch}c{i}",
        "batch": batch,
        "index": i,
        "cycle_life": float(cl[0]) if cl.size else np.nan,
        "policy": _text(f, b["policy_readable"][i, 0]),
        "summary": {k: summ[v][()].ravel() for k, v in SUMMARY_FIELDS.items()},
        "n_cycles": cycles["V"].shape[0],
        "Vdlin": f[b["Vdlin"][i, 0]][()].ravel(),
        "Qdlin": _early_matrix(f, cycles["Qdlin"]),
        "Tdlin": _early_matrix(f, cycles["Tdlin"]),
        "dQdV": _early_matrix(f, cycles["discharge_dQdV"]),
    }


def load_batch(batch, use_cache=True):
    """Return list of cell dicts for 'b1'/'b2'/'b3'. Reads the cache if present, else builds it."""
    cache = os.path.join(CACHE_DIR, f"{batch}.pkl")
    if use_cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with h5py.File(os.path.join(DATA_DIR, BATCH_FILES[batch]), "r") as f:
        n = f["batch"]["summary"].shape[0]
        cells = [read_cell(f, batch, i) for i in range(n)]
    with open(cache, "wb") as fh:
        pickle.dump(cells, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cells


def load_all(use_cache=True):
    return {b: load_batch(b, use_cache) for b in BATCH_FILES}


if __name__ == "__main__":
    for batch, cells in load_all().items():
        cl = np.array([c["cycle_life"] for c in cells])
        print(f"{batch}: {len(cells)} cells, cycle_life NaN={int(np.isnan(cl).sum())}, "
              f"min={np.nanmin(cl):.0f}, max={np.nanmax(cl):.0f}")
