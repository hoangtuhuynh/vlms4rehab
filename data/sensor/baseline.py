"""
Classical activity-identification baseline on the preprocessed sensor trials.

Each trial is summarized by per-channel statistics and classified with a random
forest trained on the ``training`` split. Results are reported on the ``test``
split (same 8 subjects as ``strokerehab_test_set.txt``) and on the ``none`` split
(10 extra subjects, including the severe cohort), broken down by the repo's
impairment cohorts.

Usage:
    python -m data.sensor.baseline --data sensor_work/trials_25hz
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from data.sensor.preprocess import CHANNELS
from data.utils_strokerehab import MILD_SUBJECTS, MODERATE_SUBJECTS, SEVERE_SUBJECTS

COHORTS = {
    **{s: "mild" for s in MILD_SUBJECTS.split(",")},
    **{s: "moderate" for s in MODERATE_SUBJECTS.split(",")},
    **{s: "severe" for s in SEVERE_SUBJECTS.split(",")},
}

FEATURE_SETS = {
    "all": CHANNELS,
    "joint_angles": [c for c in CHANNELS if c.endswith("_deg")],
    "paretic_arm": [c for c in CHANNELS if c.startswith("p_")],
}


def trial_features(X: np.ndarray, duration_s: float) -> np.ndarray:
    q10, q50, q90 = np.percentile(X, [10, 50, 90], axis=0)
    dX = np.diff(X, axis=0) if len(X) > 1 else np.zeros_like(X)
    return np.concatenate([
        X.mean(0), X.std(0), X.min(0), X.max(0), q10, q50, q90,
        np.abs(dX).mean(0), [duration_s],
    ])


def load_features(data_dir: str, channels: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(os.path.join(data_dir, "trials.csv"))
    cols = [CHANNELS.index(c) for c in channels]
    feats = []
    for r in meta.itertuples():
        X = np.load(os.path.join(data_dir, f"{r.trial_id}.npz"))["X"][:, cols]
        feats.append(trial_features(X, r.duration_s))
    meta["cohort"] = meta.patient.map(COHORTS).fillna("unknown")
    return meta, np.nan_to_num(np.vstack(feats))


def report(name: str, meta: pd.DataFrame, y_true, y_pred) -> dict:
    row = {"eval": name, "n": len(y_true), "accuracy": accuracy_score(y_true, y_pred)}
    correct = pd.Series(np.asarray(y_true) == np.asarray(y_pred), index=meta.index)
    for cohort, acc in correct.groupby(meta.cohort).mean().items():
        row[f"acc_{cohort}"] = acc
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("sensor_work", "trials_25hz"))
    parser.add_argument("--out", default=os.path.join("sensor_work", "baseline"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for fs_name, channels in FEATURE_SETS.items():
        meta, F = load_features(args.data, channels)
        y = meta.activity.to_numpy()
        tr = (meta.split == "training").to_numpy()

        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=args.seed, class_weight="balanced")
        cv_pred = cross_val_predict(clf, F[tr], y[tr], groups=meta.subject[tr], cv=GroupKFold(5))
        rows.append({"features": fs_name, **report("train_subject_cv", meta[tr], y[tr], cv_pred)})

        clf.fit(F[tr], y[tr])
        for split in ("test", "none"):
            m = (meta.split == split).to_numpy()
            pred = clf.predict(F[m])
            rows.append({"features": fs_name, **report(split, meta[m], y[m], pred)})
            if fs_name == "all":
                labels = sorted(set(y))
                cm = pd.DataFrame(confusion_matrix(y[m], pred, labels=labels), index=labels, columns=labels)
                cm.to_csv(os.path.join(args.out, f"confusion_{split}.csv"))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "results.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
        print("\nConfusion matrix (test, all features):")
        print(pd.read_csv(os.path.join(args.out, "confusion_test.csv"), index_col=0).to_string())


if __name__ == "__main__":
    main()
