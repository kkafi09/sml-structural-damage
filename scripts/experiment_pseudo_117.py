"""Evaluate memory-bounded LightGBM self-training with OOF pseudo-labels."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def select_training_sample(features: pd.DataFrame, labels: pd.Series,
                           sample_size: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Take a reproducible stratified subset to keep student fitting memory bounded."""
    sampled_x, _, sampled_y, _ = train_test_split(
        features, labels, train_size=sample_size, random_state=seed, stratify=labels
    )
    return sampled_x, sampled_y


def score_predictions(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Return Macro F1 and per-grade F1 on an untouched evaluation set."""
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    scores = f1_score(labels, predictions, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "per_grade_f1": {grade: float(value) for grade, value in zip(LABELS, scores)},
    }


def main() -> None:
    """Train one student on a subset plus pseudo-labeled calibration rows."""
    cache = ROOT / "reports" / "catboost_encoded_data_101.pkl"
    with cache.open("rb") as cache_file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(cache_file)
    print(f"Loaded prepared split from {cache.name}", flush=True)

    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    teacher_oof = stack["calibration_oof_probabilities"]
    if not np.array_equal(np.asarray(cal_y), stack["calibration_labels"]):
        raise ValueError("Cached calibration rows do not match OOF stack rows")
    selected = teacher_oof.max(axis=1) >= 0.75
    pseudo_y = np.asarray(LABELS)[teacher_oof.argmax(axis=1)][selected]
    pseudo_x = cal_x.loc[selected]
    print(f"OOF pseudo-labels selected: {int(selected.sum()):,}/{len(selected):,}", flush=True)

    sampled_x, sampled_y = select_training_sample(train_x, train_y, 180_000, 2026)
    del train_x
    gc.collect()
    train_x = pd.concat([sampled_x, pseudo_x], ignore_index=True)
    train_y = np.concatenate([sampled_y.to_numpy(), pseudo_y])
    row_weights = np.concatenate([
        np.ones(len(sampled_y), dtype=float),
        np.full(len(pseudo_y), 0.35, dtype=float),
    ])
    del sampled_x, sampled_y, pseudo_x, cal_x, cal_y
    gc.collect()

    for column in CATEGORICAL:
        if column in train_x and isinstance(train_x[column].dtype, pd.CategoricalDtype):
            categories = train_x[column].cat.categories
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    model = lgb.LGBMClassifier(
        objective="multiclass", metric="None", n_estimators=74,
        learning_rate=0.06, num_leaves=63, min_child_samples=100,
        colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
        reg_lambda=3.0,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=3, random_state=3026, verbosity=-1,
    )
    model.fit(train_x, train_y, sample_weight=row_weights, categorical_feature=CATEGORICAL)
    student = model.predict_proba(eval_x)
    del model, train_x, eval_x
    gc.collect()

    evaluation_labels = np.asarray(stack["evaluation_labels"])
    baseline = stack["evaluation_probabilities"]
    bias_report = json.loads(
        (ROOT / "reports" / "validation_exp_087_interaction_bias.json").read_text()
    )
    bias = np.array([bias_report["bias_by_grade"][grade] for grade in LABELS])
    baseline_logits = np.log(np.clip(baseline, 1e-8, 1.0)) + bias
    student_logits = np.log(np.clip(student, 1e-8, 1.0))
    results = {"student_raw": score_predictions(student, evaluation_labels)}
    for weight in (0.15, 0.30, 0.50):
        results[f"stack_blend_{weight:.2f}"] = score_predictions(
            np.exp((1.0 - weight) * baseline_logits + weight * student_logits),
            evaluation_labels,
        )
    report = {
        "method": "OOF stack pseudo-labels at confidence >=0.75; 180k stratified training sample; evaluation untouched",
        "pseudo_rows": int(selected.sum()),
        "pseudo_weight": 0.35,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
        "results": results,
    }
    path = ROOT / "reports" / "validation_exp_117_pseudo_label.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), student_evaluation_probabilities=student)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
