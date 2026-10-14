"""Evaluate an XGBoost model with native categorical splits."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, optimize_class_bias
from scripts.train_lightgbm_ensemble import prepare_data


def macro_f1_metric(labels, probabilities):
    """Return Macro F1 for XGBoost's probability callback."""
    predictions = np.asarray(probabilities).reshape(len(labels), len(LABELS)).argmax(axis=1)
    return float(f1_score(labels, predictions, average="macro"))


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    seed = 2026
    train_x, train_y, calibration_x, calibration_y, evaluation_x, evaluation_y = (
        prepare_data(seed)
    )
    high_cardinality_ids = [
        "id_prefix6", "id_prefix8", "id_prefix9", "id_prefix10"
    ]
    train_x = train_x.drop(columns=high_cardinality_ids, errors="ignore")
    calibration_x = calibration_x.drop(columns=high_cardinality_ids, errors="ignore")
    evaluation_x = evaluation_x.drop(columns=high_cardinality_ids, errors="ignore")
    train_codes = np.array([LABELS.index(label) for label in train_y])
    calibration_codes = np.array([LABELS.index(label) for label in calibration_y])
    evaluation_codes = np.array([LABELS.index(label) for label in evaluation_y])
    class_weights = np.ones(len(LABELS), dtype=float)
    categorical_columns = list(train_x.select_dtypes(include="category").columns)
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(LABELS), eval_metric="mlogloss",
        n_estimators=650, learning_rate=0.06, max_depth=7, min_child_weight=30,
        subsample=0.9, colsample_bytree=0.85, reg_lambda=8.0,
        tree_method="hist", enable_categorical=True, max_cat_to_onehot=4,
        max_cat_threshold=16, n_jobs=4, random_state=2718,
        early_stopping_rounds=75,
    )
    model.fit(
        train_x, train_codes, sample_weight=class_weights[train_codes],
        eval_set=[(calibration_x, calibration_codes)],
        verbose=50,
    )
    calibration_probabilities = model.predict_proba(calibration_x)
    evaluation_probabilities = model.predict_proba(evaluation_x)
    calibration_labels = [LABELS[index] for index in calibration_codes]
    evaluation_labels = [LABELS[index] for index in evaluation_codes]
    calibrated = optimize_class_bias(
        calibration_probabilities, calibration_labels,
        evaluation_probabilities, evaluation_labels,
    )
    summary = {
        "seed": seed,
        "model_random_state": 2718,
        "method": "XGBoost on location/structure target interactions and long ID neighbors",
        "parameters": {
            "n_estimators": 650, "learning_rate": 0.06, "max_depth": 7,
            "min_child_weight": 30, "subsample": 0.9,
            "colsample_bytree": 0.85, "reg_lambda": 8.0,
            "max_cat_to_onehot": 4, "max_cat_threshold": 16,
            "class_weights": class_weights.tolist(),
        },
        "categorical_columns": categorical_columns,
        "best_iteration": int(model.best_iteration),
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            evaluation_codes, evaluation_probabilities.argmax(axis=1), average="macro"
        )),
        "calibration": calibrated,
    }
    report_path = ROOT / "reports" / "validation_exp_094_xgboost_interactions.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "reports" / "xgboost_holdout_094.npz",
        calibration_probabilities=calibration_probabilities,
        evaluation_probabilities=evaluation_probabilities,
        calibration_labels=np.asarray(calibration_labels),
        evaluation_labels=np.asarray(evaluation_labels),
    )
    print(
        f"XGBoost best_iteration={model.best_iteration}; "
        f"base={summary['evaluation_macro_f1_uncalibrated']:.6f}; "
        f"calibrated={calibrated['evaluation_macro_f1']:.6f}", flush=True
    )
    print(f"Saved {report_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
