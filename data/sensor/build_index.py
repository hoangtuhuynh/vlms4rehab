"""
Build a trial-level index of the wearable-sensor (IMU) StrokeRehab dataset.

The dataset is organized as ``<root>/sNN/<activity>/<file>.csv`` with one
``metadata.xml`` per subject. Each CSV is one trial sampled at 100 Hz with joint
angles, accelerometer and segment-rotation channels plus per-sample primitive
labels (``MarkerNames``).

Usage:
    python -m data.sensor.build_index --root "<.../Stroke Rehab data file/files>"
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd


INDEX_PATH = os.path.join("data", "sensor", "sensor_trials.csv")

# Folder name -> label used by the identification task (matches sr_id_doc_to_target).
ACTIVITY_LABELS = {
    "brushing": "brushing",
    "combing": "combing",
    "deodorant": "deodorant",
    "drinking": "drinking",
    "face wash": "face wash",
    "facewash": "face wash",
    "feeding": "feeding",
    "glasses": "glasses",
    "RTT": "rtt exercise",
    "shelf": "shelf exercise",
}

# e.g. "... brushing 3.csv", "... brushing 2-1.csv", "... shelf left side  4.csv"
REP_RE = re.compile(r"(\d+)(?:-\d+)?\s*\.csv$", re.IGNORECASE)


def subject_to_patient(subject: str) -> str:
    """'s02' -> 'S0002', 's17' -> 'S00017' (the ID convention used in data/utils_strokerehab.py)."""
    return f"S000{int(subject[1:])}"


def read_subject_metadata(path: str) -> dict:
    fields = {
        "paretic_side": "paretic_side",
        "gender": "gender",
        "age": "age",
        "UE_FMA_score": "fma",
        "impairment_level": "impairment",
        "time_since_stroke": "time_since_stroke",
        "training_test_set": "split",
    }
    root = ET.parse(path).getroot()
    out = {}
    for tag, key in fields.items():
        el = root.find(tag)
        out[key] = el.get("value") if el is not None else None
    return out


def build_index(root: str) -> pd.DataFrame:
    rows = []
    for subject in sorted(d for d in os.listdir(root) if re.fullmatch(r"s\d+", d)):
        subj_dir = os.path.join(root, subject)
        meta_path = os.path.join(subj_dir, "metadata.xml")
        meta = read_subject_metadata(meta_path) if os.path.exists(meta_path) else {}
        for folder in sorted(os.listdir(subj_dir)):
            act_dir = os.path.join(subj_dir, folder)
            if not os.path.isdir(act_dir) or folder not in ACTIVITY_LABELS:
                continue
            for fn in sorted(os.listdir(act_dir)):
                if not fn.lower().endswith(".csv"):
                    continue
                m = REP_RE.search(fn)
                low = fn.lower()
                side = "left" if "left" in low else ("right" if "right" in low else "")
                rel_path = os.path.join(subject, folder, fn)
                rows.append({
                    "trial_id": os.path.splitext(rel_path)[0].replace(os.sep, "__").replace(" ", "_"),
                    "subject": subject,
                    "patient": subject_to_patient(subject),
                    "activity_folder": folder,
                    "activity": ACTIVITY_LABELS[folder],
                    "rep": int(m.group(1)) if m else None,
                    "side_performed": side,
                    # Standard trials are numbered repetitions 1-5; everything else is a
                    # partial clip ("reaches", "reaching to cup"), a merged file ("all",
                    # "combined") or a failed attempt.
                    "is_standard_rep": bool(m) and "failed" not in low,
                    "rel_path": rel_path,
                    "size_bytes": os.path.getsize(os.path.join(act_dir, fn)),
                    **meta,
                })
    df = pd.DataFrame(rows)
    for col in ("age", "fma", "time_since_stroke"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Path to the dataset 'files' directory (contains s01, s02, ...).")
    parser.add_argument("--out", default=INDEX_PATH)
    args = parser.parse_args()

    df = build_index(args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    std = df[df.is_standard_rep]
    print(f"Wrote {len(df)} trials ({len(std)} standard reps) from {df.subject.nunique()} subjects to {args.out}")
    print(std.groupby("split").agg(subjects=("subject", "nunique"), trials=("trial_id", "count")))
    print(std.activity.value_counts().to_string())


if __name__ == "__main__":
    main()
