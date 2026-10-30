"""
Export supervised fine-tuning data for sensor-based activity identification.

Writes one JSONL per split and input mode (``<out>/<split>_<mode>.jsonl``) in the
ShareGPT/LLaMA-Factory style also read by ``data/sensor/train_lora.py``:

    {"id": ..., "images": ["<plot.png>"],
     "messages": [{"role": "user", "content": "<image>..."},
                  {"role": "assistant", "content": "FINAL_ANSWER: Brushing"}]}

Text-mode records have ``"images": []`` and the sensor table inside the prompt;
image_text records have both the plot and the table; image_skeleton_text records have
the plot and the stick-figure strip (in that order) plus the table; poses records have only
the separate stick-figure images, in time order.

Usage:
    python -m data.sensor.export_sft --data sensor_work/trials_25hz --out sensor_work/sft
"""

import argparse
import json
import os

import pandas as pd

from data.sensor.prompts import ACTIVITY_NAMES, PLOT_MODES, POSE_MODES, SKELETON_MODES, TEXT_MODES, build_prompt
from data.sensor.render import load_trial, plot_path, render_text
from data.sensor.skeleton import skeleton_frame_paths, skeleton_path

IMAGE_TOKEN = "<image>"


def make_record(data_dir: str, row, mode: str, style: str) -> dict:
    images = [plot_path(data_dir, row.trial_id)] if mode in PLOT_MODES else []
    if mode in SKELETON_MODES:
        images.append(skeleton_path(data_dir, row.trial_id))
    if mode in POSE_MODES:
        images.extend(skeleton_frame_paths(data_dir, row.trial_id))
    sensor_text = render_text(load_trial(data_dir, row.trial_id)) if mode in TEXT_MODES else None
    user = IMAGE_TOKEN * len(images) + build_prompt(mode, style, sensor_text)
    return {
        "id": row.trial_id,
        "images": images,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"FINAL_ANSWER: {ACTIVITY_NAMES[row.activity]}"},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("sensor_work", "trials_25hz"))
    parser.add_argument("--out", default=os.path.join("sensor_work", "sft"))
    parser.add_argument("--modes", default="image,text", help="comma-separated: image, text, image_text, image_skeleton_text, poses")
    parser.add_argument("--prompt", default="kinematic", choices=["kinematic", "plain"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pd.read_csv(os.path.join(args.data, "trials.csv"))
    for mode in args.modes.split(","):
        for split, df in meta.groupby("split"):
            path = os.path.join(args.out, f"{split}_{mode}.jsonl")
            with open(path, "w") as f:
                for row in df.itertuples():
                    f.write(json.dumps(make_record(args.data, row, mode, args.prompt)) + "\n")
            print(f"{path}: {len(df)} records")


if __name__ == "__main__":
    main()
