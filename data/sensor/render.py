"""
Turn a preprocessed sensor trial into VLM input: a multi-panel plot image and/or a
compact text table.

The plot never shows the trial id or file name, since those contain the activity.

Usage (pre-render all plots next to the .npz files):
    python -m data.sensor.render --data sensor_work/trials_25hz --jobs 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data.sensor.preprocess import CHANNELS

PLOT_DIR = "plots"

_ARM_JOINTS = [
    ("elbowflexion", "elbow flexion"),
    ("shoulderflexion", "shoulder flexion"),
    ("shoulderabduction", "shoulder abduction"),
    ("wristextension", "wrist extension"),
    ("wristsupination", "wrist supination"),
]
_TRUNK = [("trunk_thoracicflexion_deg", "thoracic flexion"), ("trunk_lumbarflexion_deg", "lumbar flexion")]

# (column header, source) for the text table; source is a channel name or "<arm>_hand_acc".
_TEXT_COLUMNS = (
    [(f"p_{short}", f"p_{j}_deg") for j, short in (
        ("elbowflexion", "elb"), ("shoulderflexion", "shflex"), ("shoulderabduction", "shabd"),
        ("wristextension", "wrext"), ("wristsupination", "wrsup"))]
    + [(f"np_{short}", f"np_{j}_deg") for j, short in (
        ("elbowflexion", "elb"), ("shoulderflexion", "shflex"), ("shoulderabduction", "shabd"))]
    + [("trunk_flex", "trunk_thoracicflexion_deg"), ("p_hand_acc", "p_hand_acc"), ("np_hand_acc", "np_hand_acc")]
)

TEXT_LEGEND = (
    "Columns: t = time (s); p_ = paretic (affected) arm, np_ = non-paretic arm; "
    "elb = elbow flexion, shflex = shoulder flexion, shabd = shoulder abduction, "
    "wrext = wrist extension, wrsup = wrist supination, trunk_flex = thoracic trunk flexion "
    "(all joint angles in degrees); hand_acc = hand acceleration magnitude in g with gravity removed (near 0 when the hand is still)."
)


def load_trial(data_dir: str, trial_id: str) -> dict:
    with np.load(os.path.join(data_dir, f"{trial_id}.npz")) as z:
        return {k: z[k] for k in ("t", "X", "prim")}


def _channel(trial: dict, name: str) -> np.ndarray:
    X = trial["X"]
    if name.endswith("_hand_acc"):
        arm = name.split("_")[0]
        return np.linalg.norm(X[:, [CHANNELS.index(f"{arm}_hand_acc_{a}_mg") for a in "xyz"]], axis=1) / 1000.0
    return X[:, CHANNELS.index(name)]


def plot_path(data_dir: str, trial_id: str) -> str:
    return os.path.join(data_dir, PLOT_DIR, f"{trial_id}.png")


def render_plot(trial: dict, out_path: str, width_px: int = 1120, height_px: int = 840, dpi: int = 100) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = trial["t"] - trial["t"][0]
    fig, axes = plt.subplots(4, 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, sharex=True)
    for ax, arm, title in ((axes[0], "p", "Paretic (affected) arm joint angles"),
                           (axes[1], "np", "Non-paretic arm joint angles")):
        for j, label in _ARM_JOINTS:
            ax.plot(t, _channel(trial, f"{arm}_{j}_deg"), lw=1, label=label)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("deg")
    axes[1].sharey(axes[0])
    for name, label in _TRUNK:
        axes[2].plot(t, _channel(trial, name), lw=1, label=label)
    axes[2].set_title("Trunk", fontsize=10, loc="left")
    axes[2].set_ylabel("deg")
    axes[3].plot(t, _channel(trial, "p_hand_acc"), lw=1, label="paretic hand")
    axes[3].plot(t, _channel(trial, "np_hand_acc"), lw=1, label="non-paretic hand")
    axes[3].set_title("Hand acceleration magnitude (gravity removed)", fontsize=10, loc="left")
    axes[3].set_ylabel("g")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_text(trial: dict, hz: float = 2.0, max_rows: int = 120) -> str:
    t = trial["t"] - trial["t"][0]
    duration = float(t[-1]) if len(t) else 0.0
    hz = min(hz, max_rows / max(duration, 1e-6))
    grid = np.arange(0.0, duration + 1e-6, 1.0 / hz)
    cols = {"t": grid}
    for header, src in _TEXT_COLUMNS:
        cols[header] = np.interp(grid, t, _channel(trial, src))
    df = pd.DataFrame(cols)
    fmt = {c: ("{:.1f}" if c == "t" else "{:.2f}" if c.endswith("_acc") else "{:.0f}") for c in df.columns}
    lines = [",".join(df.columns)]
    lines += [",".join(fmt[c].format(v) for c, v in zip(df.columns, row)) for row in df.itertuples(index=False)]
    return (f"Sensor recording, duration {duration:.1f} s, sampled at {hz:.2g} Hz.\n"
            f"{TEXT_LEGEND}\n" + "\n".join(lines))


def _render_one(args):
    data_dir, trial_id, overwrite = args
    out = plot_path(data_dir, trial_id)
    if overwrite or not os.path.exists(out):
        render_plot(load_trial(data_dir, trial_id), out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("sensor_work", "trials_25hz"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta = pd.read_csv(os.path.join(args.data, "trials.csv"))
    jobs = [(args.data, tid, args.overwrite) for tid in meta.trial_id]
    with ProcessPoolExecutor(args.jobs) as ex:
        n = sum(1 for _ in ex.map(_render_one, jobs, chunksize=8))
    print(f"Rendered {n} plots into {os.path.join(args.data, PLOT_DIR)}")


if __name__ == "__main__":
    main()
