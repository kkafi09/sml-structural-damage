"""Evaluate calibrated one-vs-rest LightGBM models on a fresh holdout."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import (
    CATEGORICAL,
    LABELS,
    add_target_statistics,
    cross_fit_target_statistics,
    load_data,
    optimize_class_bias,
)


def main() -> None:
    """Fit one binary classifier per grade and evaluate bias calibration."""
    seed = 2026
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    calibration_x, evaluation_x, calibration_y, evaluation_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=seed + 1, stratify=holdout_y
    )
    raw_train_x = train_x
    train_x, calibration_x = cross_fit_target_statistics(
        raw_train_x, train_y, calibration_x, seed=seed
    )
    evaluation_x = add_target_statistics(raw_train_x, train_y, evaluation_x)

    calibration_probabilities = np.zeros((len(calibration_y), len(LABELS)))
    evaluation_probabilities = np.zeros((len(evaluation_y), len(LABELS)))
    best_iterations = {}
    for index, grade in enumerate(LABELS):
        train_binary = train_y.eq(grade).astype("int8")
        calibration_binary = calibration_y.eq(grade).astype("int8")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1600, learning_rate=0.04,
            num_leaves=31, max_depth=-1, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, n_jobs=4, random_state=seed + index, verbosity=-1,
        )
        model.fit(
            train_x, train_binary,
            eval_set=[(calibration_x, calibration_binary)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                       lgb.log_evaluation(200)],
            categorical_feature=CATEGORICAL,
        )
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        calibration_probabilities[:, index] = model.predict_proba(calibration_x)[:, positive_index]
        evaluation_probabilities[:, index] = model.predict_proba(evaluation_x)[:, positive_index]
        best_iterations[grade] = int(model.best_iteration_)
        print(f"{grade}: best_iteration={model.best_iteration_}", flush=True)

    calibration_base = [LABELS[index] for index in calibration_probabilities.argmax(axis=1)]
    evaluation_base = [LABELS[index] for index in evaluation_probabilities.argmax(axis=1)]
    calibrated = optimize_class_bias(
        calibration_probabilities, calibration_y,
        evaluation_probabilities, evaluation_y,
    )
    summary = {
        "seed": seed,
        "method": "One-vs-rest LightGBM; 80/10/10 stratified train/calibration/evaluation",
        "best_iterations": best_iterations,
        "calibration_macro_f1_uncalibrated": float(f1_score(
            calibration_y, calibration_base, average="macro"
        )),
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            evaluation_y, evaluation_base, average="macro"
        )),
        "evaluation_classification_report_uncalibrated": classification_report(
            evaluation_y, evaluation_base, labels=LABELS, output_dict=True,
            zero_division=0,
        ),
        "calibration": calibrated,
    }
    path = ROOT / "reports" / "validation_exp_007_ovr.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Base held-out Macro F1: {summary['evaluation_macro_f1_uncalibrated']:.6f}; "
          f"calibrated: {calibrated['evaluation_macro_f1']:.6f}", flush=True)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
