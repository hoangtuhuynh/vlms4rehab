"""
Convert the raw 100 Hz sensor CSVs into compact per-trial ``.npz`` arrays.

Each output file ``<out_dir>/<trial_id>.npz`` holds:
    t        (T,)    seconds
    X        (T, C)  float32 sensor channels, names in ``CHANNELS``
    prim     (T,)    int8 primitive code (index into PRIMITIVES, -1 if unmapped)
    markers  (T,)    raw MarkerNames strings

Column names differ across files (``Time_s`` vs ``Times``, ``ElbowFlexionRT_deg`` vs
``ElbowFlexionRTdeg``), so they are normalized by lowercasing and stripping
underscores. Left-paretic recordings (files prefixed ``RT_``) were mirrored by the
dataset curators so the performing arm is in the RT columns; RT/LT channels are
therefore renamed to ``p_`` (paretic arm) / ``np_`` (non-paretic arm).

Usage:
    python -m data.sensor.preprocess --root "<.../Stroke Rehab data file/files>" --out sensor_work/trials_25hz
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data.sensor.build_index import INDEX_PATH


RAW_FS = 100

PRIMITIVES = ["reach", "reposition", "transport", "stabilize", "idle"]

_JOINTS = [
    "elbowflexion", "shouldertotalflexion", "shoulderflexion", "shoulderabduction",
    "shoulderrotationout", "wristextension", "wristradial", "wristsupination",
]
_ACCEL_SEGMENTS = ["hand", "forearm", "upperarm"]
_ROT_SEGMENTS = ["hand", "forearm", "upperarm"]

# (normalized raw column, output channel name)
_CHANNEL_MAP = (
    [(f"{j}{side}deg", f"{arm}_{j}_deg") for arm, side in (("p", "rt"), ("np", "lt")) for j in _JOINTS]
    + [(f"{t}deg", f"trunk_{t}_deg") for t in (
        "lumbarflexion", "lumbarlateralrt", "lumbaraxialrt",
        "thoracicflexion", "thoraciclateralrt", "thoracicaxialrt")]
    + [(f"{s}accelsensor{a}{side}mg", f"{arm}_{s}_acc_{a}_mg")
       for arm, side in (("p", "rt"), ("np", "lt")) for s in _ACCEL_SEGMENTS for a in "xyz"]
    + [(f"{s}accelsensor{a}mg", f"{s}_acc_{a}_mg") for s in ("upperspine", "lowerspine", "pelvis") for a in "xyz"]
    + [(f"{side}{s}rot{a}", f"{arm}_{s}_rot_{a}")
       for arm, side in (("p", "rt"), ("np", "lt")) for s in _ROT_SEGMENTS for a in "xyz"]
    + [(f"{s}rot{a}", f"{s}_rot_{a}") for s in ("upperspine", "lowerspine", "pelvis") for a in "xyz"]
)
CHANNELS = [name for _, name in _CHANNEL_MAP]


def marker_to_primitive(marker: str) -> str | None:
    """Same keyword rules as PrimitiveLabelUtils.convert_labels_to_prims_times."""
    m = marker.lower()
    if "reach" in m:
        return "reach"
    if "reposition" in m or "retract" in m:
        return "reposition"
    if "transport" in m:
        return "transport"
    if "stabilize" in m:
        return "stabilize"
    if "idle" in m or "rest" in m:
        return "idle"
    return None


def _normalize(col) -> str:
    return str(col).replace("_", "").lower()


def load_trial_csv(path: str, fs_out: int) -> dict:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [_normalize(c) for c in df.columns]
    t = pd.to_numeric(df["times"], errors="coerce").to_numpy()  # "Time_s" and "Times" both normalize to this

    X = df[[raw for raw, _ in _CHANNEL_MAP]].apply(pd.to_numeric, errors="coerce")
    X = X.interpolate(limit_direction="both").fillna(0.0)

    step = max(1, RAW_FS // fs_out)
    if step > 1:
        # Box-filter before decimating to avoid aliasing.
        X = X.rolling(step, center=True, min_periods=1).mean()
    X = X.to_numpy(dtype=np.float32)[::step]
    t = t[::step].astype(np.float32)

    markers = df["markernames"].fillna("").astype(str).to_numpy()[::step]
    prim_lookup = {p: i for i, p in enumerate(PRIMITIVES)}
    prim = np.array([prim_lookup.get(marker_to_primitive(m), -1) for m in markers], dtype=np.int8)
    return {"t": t, "X": X, "prim": prim, "markers": markers.astype("U64")}


def _process_one(args):
    root, rel_path, out_path, fs_out = args
    try:
        d = load_trial_csv(os.path.join(root, rel_path), fs_out)
    except Exception as e:  # keep going; report at the end
        return {"rel_path": rel_path, "error": repr(e)}
    np.savez_compressed(out_path, **d)
    markers = pd.Series(d["markers"])
    return {
        "rel_path": rel_path,
        "error": "",
        "n_samples": len(d["t"]),
        "duration_s": float(d["t"][-1] - d["t"][0]) if len(d["t"]) else 0.0,
        "label_side": "left" if markers.str.startswith("l_").mean() > 0.5 else "right",
        "unmapped_label_frac": float((d["prim"] < 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--out", default=os.path.join("sensor_work", "trials_25hz"))
    parser.add_argument("--fs", type=int, default=25, help="Output sampling rate in Hz (raw is 100).")
    parser.add_argument("--all", action="store_true", help="Also process non-standard trials (partial/merged clips).")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    if not args.all:
        index = index[index.is_standard_rep]
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.root, r.rel_path, os.path.join(args.out, f"{r.trial_id}.npz"), args.fs) for r in index.itertuples()]
    with ProcessPoolExecutor(args.jobs) as ex:
        results = pd.DataFrame(ex.map(_process_one, jobs, chunksize=4))

    out = index.merge(results, on="rel_path", how="left")
    # s40 has an empty metadata.xml; fall back to the side the labels were recorded on.
    out["paretic_side"] = out["paretic_side"].fillna(out["label_side"])
    ok = out[out.error.fillna("") == ""]
    ok.to_csv(os.path.join(args.out, "trials.csv"), index=False)

    failed = out[out.error.fillna("") != ""]
    print(f"Processed {len(ok)} trials into {args.out} ({len(failed)} failed)")
    if len(failed):
        print(failed[["rel_path", "error"]].to_string(index=False))
    print(f"Median duration: {ok.duration_s.median():.1f}s; unmapped label frac (mean): {ok.unmapped_label_frac.mean():.4f}")


if __name__ == "__main__":
    main()
