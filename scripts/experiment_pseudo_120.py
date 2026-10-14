"""Test full-size LightGBM self-training from leakage-safe OOF pseudo-labels."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def score(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Calculate Macro F1 and class-level F1 on the untouched evaluation set."""
    prediction = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    per_grade = f1_score(labels, prediction, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "per_grade_f1": {label: float(value) for label, value in zip(LABELS, per_grade)},
    }


def main() -> None:
    """Fit a single student on all fit rows plus low-weight OOF pseudo-labels."""
    cache = ROOT / "reports" / "catboost_encoded_data_101.pkl"
    with cache.open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    print("Prepared feature cache loaded", flush=True)

    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    teacher_oof = stack["calibration_oof_probabilities"]
    if not np.array_equal(np.asarray(cal_y), stack["calibration_labels"]):
        raise ValueError("Calibration rows are not aligned with OOF predictions")
    chosen = teacher_oof.max(axis=1) >= 0.75
    pseudo_x = cal_x.loc[chosen]
    pseudo_y = np.asarray(LABELS)[teacher_oof.argmax(axis=1)][chosen]
    print(f"Pseudo-label rows: {len(pseudo_y):,}", flush=True)

    for column in CATEGORICAL:
        if column in train_x and isinstance(train_x[column].dtype, pd.CategoricalDtype):
            categories = train_x[column].cat.categories
            pseudo_x[column] = pseudo_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    augmented_x = pd.concat([train_x, pseudo_x], ignore_index=True)
    augmented_y = np.concatenate([train_y.to_numpy(), pseudo_y])
    weights = np.concatenate([
        np.ones(len(train_y), dtype=float), np.full(len(pseudo_y), 0.15, dtype=float)
    ])
    del train_x, train_y, cal_x, cal_y, pseudo_x, pseudo_y
    gc.collect()
    print(f"Student matrix: {augmented_x.shape}; fitting", flush=True)

    student = lgb.LGBMClassifier(
        objective="multiclass", metric="None", n_estimators=74,
        learning_rate=0.06, num_leaves=63, min_child_samples=100,
        colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
        reg_lambda=3.0, max_bin=127,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=2, random_state=3026, verbosity=-1,
    )
    student.fit(augmented_x, augmented_y, sample_weight=weights,
                categorical_feature=CATEGORICAL)
    student_eval = student.predict_proba(eval_x)
    del student, augmented_x, augmented_y, eval_x
    gc.collect()

    baseline = stack["evaluation_probabilities"]
    evaluation_labels = np.asarray(stack["evaluation_labels"])
    bias_report = json.loads(
        (ROOT / "reports" / "validation_exp_087_interaction_bias.json").read_text()
    )
    bias = np.array([bias_report["bias_by_grade"][grade] for grade in LABELS])
    base_logits = np.log(np.clip(baseline, 1e-8, 1.0)) + bias
    student_logits = np.log(np.clip(student_eval, 1e-8, 1.0))
    results = {"student_raw": score(student_eval, evaluation_labels)}
    for blend_weight in (0.10, 0.20, 0.30):
        blended = np.exp((1.0 - blend_weight) * base_logits + blend_weight * student_logits)
        results[f"stack_blend_{blend_weight:.2f}"] = score(blended, evaluation_labels)
    report = {
        "method": "Full 80% fit split plus low-weight OOF pseudo-labels from calibration; evaluation untouched",
        "seed": 2026,
        "confidence_threshold": 0.75,
        "pseudo_weight": 0.15,
        "pseudo_rows": int(chosen.sum()),
        "baseline_evaluation_macro_f1": 0.5985512292352484,
        "results": results,
    }
    path = ROOT / "reports" / "validation_exp_120_full_pseudo.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), student_evaluation_probabilities=student_eval)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
