"""
Stick-figure view of a sensor trial: a strip of body poses at evenly spaced times, used
as a second image next to the signal plot.

The idea of showing sensor-derived stick figures comes from Jasper (Xizhe) Liu's Study 30
skeleton renderer (github.com/XmanJL/vlms4rehab_ext, data/study30/skeleton.py). Here the
arms are posed with a simple 3-D chain (shoulder flexion, shoulder abduction, elbow
flexion, wrist extension, thoracic flexion) and drawn from a slightly oblique front view,
so hand height and forward reach are both visible. Segment lengths are generic and
shoulder rotation is ignored, so it is a schematic, not a metric reconstruction.

Usage (pre-render all strips next to the .npz files, or a few with --trials;
add --frames for one image per pose instead of one strip, or --adaptive for a per-trial
number of pose images placed by arm motion):
    python -m data.sensor.skeleton --data sensor_work/trials_25hz --jobs 8
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data.sensor.preprocess import CHANNELS
from data.sensor.render import load_trial

SKELETON_DIR = "skeletons"
FRAMES_DIR = "skeleton_frames"
FRAME_COUNT = 64
FRAME_SIZE_PX = 280  # a multiple of 28, so Qwen2.5-VL keeps it as 10 x 10 = 100 image tokens
ADAPTIVE_FRAMES_DIR = "skeleton_frames_adaptive"
MAX_FRAMES = 96  # 9,600 image tokens; about 17k tokens ran out of memory on a 40 GB A100
FRAMES_PER_MOVEMENT = 4
ARM_ANGLES = ["p_shoulderflexion_deg", "p_shoulderabduction_deg", "p_elbowflexion_deg", "p_wristextension_deg"]
AFFECTED_COLOR = "#e41a1c"
OTHER_COLOR = "#377eb8"


VIEW_AZIMUTH_DEG = 30.0
TRUNK_LEN, NECK_LEN, HEAD_OFFSET, HALF_SHOULDER = 1.65, 0.38, 0.42, 0.42
UPPER_ARM_LEN, FOREARM_LEN, HAND_LEN = 0.92, 0.82, 0.3


def _angle(row: np.ndarray, name: str, low: float, high: float) -> float:
    value = row[CHANNELS.index(name)]
    return float(np.clip(value, low, high)) if np.isfinite(value) else 0.0


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r), 0, -math.sin(r)], [0, 1, 0], [math.sin(r), 0, math.cos(r)]])


def _rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    r = math.radians(deg)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(r) * k + (1 - math.cos(r)) * k @ k


def _arm(row: np.ndarray, arm: str) -> tuple:
    """Elbow, wrist and hand offsets from the shoulder, in the arm's own frame
    (x = outward, y = forward, z = up), before trunk flexion."""
    shflex = _angle(row, f"{arm}_shoulderflexion_deg", -60, 180)
    shabd = _angle(row, f"{arm}_shoulderabduction_deg", -60, 180)
    elb = _angle(row, f"{arm}_elbowflexion_deg", 0, 160)
    wrext = _angle(row, f"{arm}_wristextension_deg", -90, 90)
    shoulder_rot = _rot_y(shabd) @ _rot_x(shflex)
    upper = shoulder_rot @ np.array([0.0, 0.0, -1.0])
    hinge = shoulder_rot @ np.array([1.0, 0.0, 0.0])
    fore = _rot_axis(hinge, elb) @ upper
    hand = _rot_axis(hinge, elb - wrext) @ upper
    elbow = UPPER_ARM_LEN * upper
    wrist = elbow + FOREARM_LEN * fore
    return elbow, wrist, wrist + HAND_LEN * hand


def pose(row: np.ndarray) -> dict:
    """2-D joint positions for one time step. The affected arm (p_) is on the image right."""
    lean = _rot_x(-_angle(row, "trunk_thoracicflexion_deg", -60, 90))
    to_world = lambda v: lean @ v
    neck = to_world(np.array([0.0, 0.0, TRUNK_LEN]))
    shoulder_center = to_world(np.array([0.0, 0.0, TRUNK_LEN + NECK_LEN]))
    points = {"pelvis": np.zeros(3), "neck": neck, "shoulder_center": shoulder_center,
              "head": to_world(np.array([0.0, 0.0, TRUNK_LEN + NECK_LEN + HEAD_OFFSET]))}
    for arm, side in (("np", -1.0), ("p", 1.0)):
        flip = np.diag([side, 1.0, 1.0])
        shoulder = shoulder_center + to_world(np.array([side * HALF_SHOULDER, 0.0, 0.0]))
        elbow, wrist, hand = _arm(row, arm)
        points.update({f"shoulder_{arm}": shoulder,
                       f"elbow_{arm}": shoulder + to_world(flip @ elbow),
                       f"wrist_{arm}": shoulder + to_world(flip @ wrist),
                       f"hand_{arm}": shoulder + to_world(flip @ hand)})
    az = math.radians(VIEW_AZIMUTH_DEG)
    return {name: np.array([p[0] * math.cos(az) + p[1] * math.sin(az), p[2]]) for name, p in points.items()}


def draw_pose(ax, joints: dict, title: str, lw: float = 2) -> None:
    from matplotlib.patches import Circle

    links = [("pelvis", "neck", "#222222")]
    for arm, color in (("np", OTHER_COLOR), ("p", AFFECTED_COLOR)):
        links += [("shoulder_center", f"shoulder_{arm}", color), (f"shoulder_{arm}", f"elbow_{arm}", color),
                  (f"elbow_{arm}", f"wrist_{arm}", color), (f"wrist_{arm}", f"hand_{arm}", color)]
    for a, b, color in links:
        pts = np.stack([joints[a], joints[b]])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round")
    ax.add_patch(Circle(joints["head"], 0.22, fill=False, color="#222222", lw=0.75 * lw))
    ax.set_xlim(-2.1, 2.7)
    ax.set_ylim(-0.2, 4.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")


def render_skeleton_strip(trial: dict, out_path: str, k: int = 16, cols: int = 8,
                          width_px: int = 1120, height_px: int = 336, dpi: int = 100) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = trial["t"] - trial["t"][0]
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    for ax, i in zip(axes.reshape(-1), np.linspace(0, len(t) - 1, k).round().astype(int)):
        draw_pose(ax, pose(trial["X"][i]), f"t={t[i]:.1f}s")
    fig.suptitle(f"Body pose at {k} evenly spaced times (schematic, front view turned 30 degrees; "
                 "forward reach points right). Red = paretic (affected) arm, blue = non-paretic arm.",
                 fontsize=9, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.05, hspace=0.25)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def arm_speed(trial: dict) -> np.ndarray:
    """Smoothed angular speed (deg/s) of the paretic arm's four joint angles, one value per sample."""
    from scipy.ndimage import uniform_filter1d

    angles = np.nan_to_num(trial["X"][:, [CHANNELS.index(c) for c in ARM_ANGLES]])
    speed = np.linalg.norm(np.diff(angles, axis=0), axis=1) / np.diff(trial["t"])
    return uniform_filter1d(np.r_[0.0, speed], 7)


def movement_frame_indices(trial: dict, min_k: int = FRAME_COUNT, max_k: int = MAX_FRAMES) -> np.ndarray:
    """Sample indices for the adaptive pose images: FRAMES_PER_MOVEMENT per estimated movement,
    clipped to [min_k, max_k], spaced by accumulated arm motion so moving stretches get more poses.

    Each arm-speed peak (prominence 20 deg/s) counts as one movement; on this dataset that count
    matches the labeled reach/transport/reposition primitives with a median ratio of 1.0."""
    from scipy.signal import find_peaks

    speed = arm_speed(trial)
    movements = len(find_peaks(speed, prominence=20, distance=6)[0])
    k = int(np.clip(FRAMES_PER_MOVEMENT * movements, min_k, max_k))
    weight = speed + 0.25 * speed.mean() + 1e-6  # the constant part keeps still stretches from getting no poses
    cum = np.r_[0.0, np.cumsum(weight)]
    return np.searchsorted(cum / cum[-1], np.linspace(0, 1, k)).clip(0, len(speed) - 1)


def render_skeleton_frames(trial: dict, out_paths: list, indices=None, size_px: int = FRAME_SIZE_PX,
                           dpi: int = 100) -> list:
    """One image per pose at the given sample indices (default: len(out_paths) evenly spaced poses)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(out_paths)
    t = trial["t"] - trial["t"][0]
    if indices is None:
        indices = np.linspace(0, len(t) - 1, k).round().astype(int)
    for n, (path, i) in enumerate(zip(out_paths, indices), start=1):
        fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        draw_pose(ax, pose(trial["X"][i]), f"Pose {n}/{k}, t={t[i]:.1f}s", lw=4)
        ax.title.set_fontsize(12)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return out_paths


def skeleton_path(data_dir: str, trial_id: str) -> str:
    return os.path.join(data_dir, SKELETON_DIR, f"{trial_id}.png")


def skeleton_frame_paths(data_dir: str, trial_id: str, k: int = FRAME_COUNT, frames_dir: str = FRAMES_DIR) -> list:
    return [os.path.join(data_dir, frames_dir, trial_id, f"pose_{n:02d}.png") for n in range(1, k + 1)]


def adaptive_frames(data_dir: str, trial_id: str, render: bool = False, overwrite: bool = False) -> list:
    """Paths of a trial's adaptive pose images (their number depends on the trial); renders missing ones if asked."""
    trial = load_trial(data_dir, trial_id)
    indices = movement_frame_indices(trial)
    paths = skeleton_frame_paths(data_dir, trial_id, len(indices), ADAPTIVE_FRAMES_DIR)
    if render and (overwrite or not all(os.path.exists(p) for p in paths)):
        render_skeleton_frames(trial, paths, indices)
    return paths


def _render_one(args):
    data_dir, trial_id, overwrite, frames = args
    if frames == "adaptive":
        return adaptive_frames(data_dir, trial_id, render=True, overwrite=overwrite)[0]
    if frames:
        outs = skeleton_frame_paths(data_dir, trial_id)
        if overwrite or not all(os.path.exists(p) for p in outs):
            render_skeleton_frames(load_trial(data_dir, trial_id), outs)
        return outs[0]
    out = skeleton_path(data_dir, trial_id)
    if overwrite or not os.path.exists(out):
        render_skeleton_strip(load_trial(data_dir, trial_id), out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("sensor_work", "trials_25hz"))
    parser.add_argument("--trials", nargs="+", help="only these trial ids (default: all in trials.csv)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frames", action="store_true",
                        help=f"save {FRAME_COUNT} separate pose images per trial instead of one strip")
    parser.add_argument("--adaptive", action="store_true",
                        help=f"save {FRAME_COUNT}-{MAX_FRAMES} pose images per trial, placed by arm motion")
    args = parser.parse_args()

    frames = "adaptive" if args.adaptive else args.frames
    trial_ids = args.trials or list(pd.read_csv(os.path.join(args.data, "trials.csv")).trial_id)
    jobs = [(args.data, tid, args.overwrite, frames) for tid in trial_ids]
    with ProcessPoolExecutor(args.jobs) as ex:
        n = sum(1 for _ in ex.map(_render_one, jobs, chunksize=8))
    what, out_dir = {"adaptive": ("sets of adaptive pose images", ADAPTIVE_FRAMES_DIR),
                     True: (f"sets of {FRAME_COUNT} pose images", FRAMES_DIR)}.get(frames, ("stick-figure strips", SKELETON_DIR))
    print(f"Rendered {n} {what} into {os.path.join(args.data, out_dir)}")


if __name__ == "__main__":
    main()
