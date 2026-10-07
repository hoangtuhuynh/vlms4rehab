"""
Export supervised fine-tuning data for sensor-based activity identification.

Writes one JSONL per split and input mode (``<out>/<split>_<mode>.jsonl``) in the
ShareGPT/LLaMA-Factory style also read by ``data/sensor/train_lora.py``:

    {"id": ..., "images": ["<plot.png>"],
     "messages": [{"role": "user", "content": "<image>..."},
                  {"role": "assistant", "content": "FINAL_ANSWER: Brushing"}]}

Text-mode records have ``"images": []`` and the sensor table inside the prompt.

Usage:
    python -m data.sensor.export_sft --data sensor_work/trials_25hz --out sensor_work/sft
"""

import argparse
import json
import os

import pandas as pd

from data.sensor.prompts import ACTIVITY_NAMES, build_prompt
from data.sensor.render import load_trial, plot_path, render_text

IMAGE_TOKEN = "<image>"


def make_record(data_dir: str, row, mode: str, style: str) -> dict:
    if mode == "image":
        images = [plot_path(data_dir, row.trial_id)]
        user = IMAGE_TOKEN + build_prompt("image", style)
    else:
        images = []
        user = build_prompt("text", style, render_text(load_trial(data_dir, row.trial_id)))
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
    parser.add_argument("--modes", default="image,text")
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
