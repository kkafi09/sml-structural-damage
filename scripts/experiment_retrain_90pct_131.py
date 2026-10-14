"""Compare an additional 10% of supervised fit data using cached features."""

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


def score(logits: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Macro F1 from class logits and labels."""
    prediction = np.asarray(LABELS)[logits.argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro"))


def main() -> None:
    """Fit one fixed model on fit plus calibration rows; keep eval untouched."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    if not np.array_equal(np.asarray(eval_y), stack["evaluation_labels"]):
        raise ValueError("Cached evaluation labels are not aligned with stack predictions")
    for column in CATEGORICAL:
        if column in train_x and isinstance(train_x[column].dtype, pd.CategoricalDtype):
            categories = train_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    train90 = pd.concat([train_x, cal_x], ignore_index=True)
    labels90 = np.concatenate([train_y.to_numpy(), cal_y.to_numpy()])
    del train_x, train_y, cal_x, cal_y
    gc.collect()
    print(f"90% fit matrix {train90.shape}; evaluation untouched", flush=True)
    model = lgb.LGBMClassifier(
        objective="multiclass", metric="None", n_estimators=120,
        learning_rate=0.06, num_leaves=63, min_child_samples=100,
        colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
        reg_lambda=3.0, max_bin=127,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=2, random_state=7131, verbosity=-1,
    )
    model.fit(train90, labels90, categorical_feature=CATEGORICAL)
    evaluation_probability = model.predict_proba(eval_x)
    del model, train90, labels90, eval_x
    gc.collect()
    baseline = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    student = np.log(np.clip(evaluation_probability, 1e-8, 1.0))
    bias_report = json.loads(
        (ROOT / "reports" / "validation_exp_087_interaction_bias.json").read_text()
    )
    bias = np.array([bias_report["bias_by_grade"][grade] for grade in LABELS])
    eval_labels = np.asarray(stack["evaluation_labels"])
    result = {
        "student_raw": score(student, eval_labels),
        "baseline_fixed_bias": score(baseline + bias, eval_labels),
        "fixed_5pct_blend": score(0.95 * baseline + 0.05 * student + bias, eval_labels),
        "fixed_10pct_blend": score(0.90 * baseline + 0.10 * student + bias, eval_labels),
    }
    report = {
        "method": "LightGBM multiclass fit on 90% cached split; no eval labels used",
        "fit_rows": 480_119,
        "evaluation_macro_f1_baseline": 0.5985512292352484,
        "scores": result,
        "note": "Training/cal/eval encodings all come from the same cached 80%-fit feature construction; the additional calibration labels enter only as supervised rows.",
    }
    path = ROOT / "reports" / "validation_exp_131_fit90pct.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        evaluation_probabilities=evaluation_probability)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
