"""Search LightGBM tree sizes with calibration separated from final evaluation."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
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
    macro_f1_metric,
    optimize_class_bias,
)


def main() -> None:
    """Fit candidate tree shapes, select on calibration, and report evaluation scores."""
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
    candidates = [
        {"num_leaves": 63, "min_child_samples": 100, "learning_rate": 0.06,
         "reg_lambda": 3.0, "max_bin": 255},
        {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.05,
         "reg_lambda": 5.0, "max_bin": 255},
        {"num_leaves": 127, "min_child_samples": 150, "learning_rate": 0.04,
         "reg_lambda": 8.0, "max_bin": 255},
        {"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.04,
         "reg_lambda": 1.0, "max_bin": 511},
    ]
    results = []
    selected = None
    for number, parameters in enumerate(candidates, start=1):
        model = lgb.LGBMClassifier(
            objective="multiclass", n_estimators=1800, max_depth=-1,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            n_jobs=4, random_state=seed + number, verbosity=-1, **parameters,
        )
        model.fit(
            train_x, train_y, eval_set=[(calibration_x, calibration_y)],
            eval_metric=macro_f1_metric,
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                       lgb.log_evaluation(100)],
            categorical_feature=CATEGORICAL,
        )
        cal_prob = model.predict_proba(calibration_x)
        eval_prob = model.predict_proba(evaluation_x)
        calibrated = optimize_class_bias(cal_prob, calibration_y, eval_prob, evaluation_y)
        base_pred = [LABELS[index] for index in eval_prob.argmax(axis=1)]
        result = {
            "candidate": number,
            "parameters": parameters,
            "best_iteration": int(model.best_iteration_),
            "calibration_macro_f1_after_bias": calibrated["calibration_macro_f1"],
            "evaluation_macro_f1": float(f1_score(
                evaluation_y, base_pred, average="macro"
            )),
            "evaluation_macro_f1_after_bias": calibrated["evaluation_macro_f1"],
            "bias_by_grade": calibrated["bias_by_grade"],
            "evaluation_classification_report_after_bias":
                calibrated["evaluation_classification_report"],
        }
        results.append(result)
        print(f"Candidate {number}: base={result['evaluation_macro_f1']:.6f}; "
              f"calibrated={result['evaluation_macro_f1_after_bias']:.6f}; "
              f"best_iteration={model.best_iteration_}", flush=True)
        if selected is None or result["calibration_macro_f1_after_bias"] > \
                selected["calibration_macro_f1_after_bias"]:
            selected = result
    summary = {
        "seed": seed,
        "method": "LightGBM tree search; 80/10/10 stratified train/calibration/evaluation",
        "selected_by": "calibration Macro F1 after class-bias tuning",
        "selected_candidate": selected["candidate"],
        "selected_evaluation_macro_f1_after_bias":
            selected["evaluation_macro_f1_after_bias"],
        "candidates": results,
    }
    path = ROOT / "reports" / "validation_exp_011_tree_search.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
