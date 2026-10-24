"""
Break down a sensor activity-identification run (lmms_eval ``--log_samples`` output)
by activity, impairment level, and patient, and list the most common mistakes.

Usage:
    python -m data.sensor.analyze_results logs/strokerehab_sensor_identification_image/qwen2_5_vl_7b_instruct_image
    python -m data.sensor.analyze_results <run_dir_or_samples.jsonl> [<another> ...]
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*([^\n<]+)", re.IGNORECASE)


def find_samples(path: str) -> str:
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "**", "*samples*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *samples*.jsonl under {path}")
    return files[-1]


def load_samples(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            resp = d["resps"]
            while isinstance(resp, list):
                resp = resp[0]
            m = FINAL_ANSWER_RE.search(resp)
            rows.append({
                "subject": d["doc"]["subject"],
                "impairment": d["doc"]["impairment"],
                "activity": d["target"],
                "prediction": m.group(1).strip(" *") if m else "(no answer)",
                "correct": float(d["accuracy"]),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = df.groupby(by).correct.agg(right="sum", trials="count", accuracy="mean")
    out["right"] = out.right.astype(int)
    out["accuracy"] = (100 * out.accuracy).round(1)
    return out.sort_values("accuracy", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folder(s) or samples .jsonl file(s)")
    parser.add_argument("--top", type=int, default=8, help="number of most common mistakes to show")
    args = parser.parse_args()

    for run in args.runs:
        path = find_samples(run)
        df = load_samples(path)
        print(f"\n=== {path}")
        print(f"Overall: {int(df.correct.sum())}/{len(df)} = {100 * df.correct.mean():.1f}%")
        for by in ("activity", "impairment", "subject"):
            print(f"\nBy {by}:\n{summarize(df, by).to_string()}")
        wrong = df[df.correct == 0]
        mistakes = wrong.groupby(["activity", "prediction"]).size().sort_values(ascending=False).head(args.top)
        print(f"\nMost common mistakes (true activity -> answer), {len(wrong)} mistakes in total:")
        print(mistakes.to_string())


if __name__ == "__main__":
    main()
