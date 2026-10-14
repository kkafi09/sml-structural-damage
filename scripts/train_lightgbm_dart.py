"""Evaluate DART boosting with calibration-based iteration selection."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import (
    CATEGORICAL,
    LABELS,
    PROFILE_COLUMNS,
    add_target_statistics,
    cross_fit_target_statistics,
    load_data,
    optimize_class_bias,
)


def main() -> None:
    """Fit DART and pick its iteration from calibration Macro F1."""
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
    train_x = train_x.drop(columns=PROFILE_COLUMNS)
    calibration_x = calibration_x.drop(columns=PROFILE_COLUMNS)
    evaluation_x = evaluation_x.drop(columns=PROFILE_COLUMNS)
    model = lgb.LGBMClassifier(
        objective="multiclass", boosting_type="dart", n_estimators=600,
        learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=100, colsample_bytree=0.9,
        subsample=0.85, subsample_freq=1, reg_lambda=3.0,
        drop_rate=0.1, skip_drop=0.5, max_drop=50,
        n_jobs=4, random_state=seed, verbosity=-1,
    )
    model.fit(
        train_x, train_y,
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.log_evaluation(100)],
    )
    iterations = [100, 150, 200, 250, 300, 400, 500, 600]
    results = []
    best = None
    best_cal_prob = None
    best_eval_prob = None
    for iteration in iterations:
        cal_prob = model.predict_proba(calibration_x, num_iteration=iteration)
        eval_prob = model.predict_proba(evaluation_x, num_iteration=iteration)
        calibrated = optimize_class_bias(cal_prob, calibration_y, eval_prob, evaluation_y)
        result = {
            "iteration": iteration,
            "calibration_macro_f1_after_bias": calibrated["calibration_macro_f1"],
            "evaluation_macro_f1": calibrated["uncalibrated_evaluation_macro_f1"],
            "evaluation_macro_f1_after_bias": calibrated["evaluation_macro_f1"],
            "bias_by_grade": calibrated["bias_by_grade"],
        }
        results.append(result)
        print(f"Iteration {iteration}: base={result['evaluation_macro_f1']:.6f}; "
              f"calibrated={result['evaluation_macro_f1_after_bias']:.6f}", flush=True)
        if best is None or result["calibration_macro_f1_after_bias"] > \
                best["calibration_macro_f1_after_bias"]:
            best = result
            best_cal_prob = cal_prob
            best_eval_prob = eval_prob
    path = ROOT / "reports" / "validation_exp_067_dart.json"
    summary = {
        "seed": seed,
        "method": "LightGBM DART; 80/10/10 stratified train/calibration/evaluation",
        "selected_iteration_by_calibration": best["iteration"],
        "selected_evaluation_macro_f1_after_bias":
            best["evaluation_macro_f1_after_bias"],
        "iterations": results,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import numpy as np
    np.savez_compressed(
        ROOT / "reports" / "lightgbm_dart_holdout_067.npz",
        calibration_probabilities=best_cal_prob,
        evaluation_probabilities=best_eval_prob,
        calibration_labels=calibration_y.to_numpy(),
        evaluation_labels=evaluation_y.to_numpy(),
    )
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
