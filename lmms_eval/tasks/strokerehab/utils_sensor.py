"""
Activity identification from wearable-sensor data instead of video.

Trials come from ``data/sensor/preprocess.py``; set ``STROKEREHAB_SENSOR_DIR`` to the
preprocessed directory (default ``sensor_work/trials_25hz`` relative to the repo root).
``lmms_eval_specific_kwargs.mode`` selects the input: "image" (rendered plot) or
"text" (serialized table, no visual input).
"""

import os
from functools import partial

import datasets
import pandas as pd

from data.sensor.prompts import ACTIVITY_NAMES, build_prompt
from data.sensor.render import load_trial, plot_path, render_plot, render_text
from lmms_eval.tasks.strokerehab.utils_identification import OutputToResultsFilter, _parse_final_answer  # noqa: F401

SENSOR_DIR = os.environ.get("STROKEREHAB_SENSOR_DIR", os.path.join("sensor_work", "trials_25hz"))


def load_sensor_dataset(split: str = "test", reps: str = "all"):
    """split: "training", "test", "none" (10 held-out subjects incl. severe), or "all".
    reps: "all" or "first" (one trial per subject and activity, like the video task)."""
    df = pd.read_csv(os.path.join(SENSOR_DIR, "trials.csv"))
    if split != "all":
        df = df[df.split == split]
    if reps == "first":
        df = df.sort_values("rep").groupby(["subject", "activity"]).head(1)
    df = df.sort_values("trial_id").reset_index(drop=True)
    return datasets.DatasetDict({"test": datasets.Dataset.from_pandas(df)})


load_test = partial(load_sensor_dataset, split="test")
load_test_first = partial(load_sensor_dataset, split="test", reps="first")
load_heldout = partial(load_sensor_dataset, split="none")


def _kwargs(lmms_eval_specific_kwargs):
    kw = lmms_eval_specific_kwargs or {}
    return kw.get("mode", "image"), kw.get("prompt", "kinematic")


def sr_sensor_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    mode, _ = _kwargs(lmms_eval_specific_kwargs)
    if mode == "text":
        return []
    path = plot_path(SENSOR_DIR, doc["trial_id"])
    if not os.path.exists(path):
        render_plot(load_trial(SENSOR_DIR, doc["trial_id"]), path)
    return [path]


def sr_sensor_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    mode, style = _kwargs(lmms_eval_specific_kwargs)
    sensor_text = render_text(load_trial(SENSOR_DIR, doc["trial_id"])) if mode == "text" else None
    return build_prompt(mode, style, sensor_text)


def sr_sensor_doc_to_target(doc):
    return ACTIVITY_NAMES[doc["activity"]]


def sr_sensor_process_results(doc, results):
    pred = _parse_final_answer(results[0]).strip(" *").lower()
    gt = sr_sensor_doc_to_target(doc).lower().strip()
    return {"accuracy": float(pred == gt)}
