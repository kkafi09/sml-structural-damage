"""Evaluate LightGBM's multiclass one-vs-all objective."""

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
    PROFILE_COLUMNS,
    add_neighbor_statistics,
    add_target_statistics,
    cross_fit_neighbor_statistics,
    cross_fit_target_statistics,
    load_data,
    macro_f1_metric,
    optimize_class_bias,
)


def main() -> None:
    """Fit multiclass OVA and evaluate after independent class-bias calibration."""
    seed = 2026
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    calibration_raw, evaluation_raw, calibration_y, evaluation_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=seed + 1,
        stratify=holdout_y,
    )
    train_te, calibration_x = cross_fit_target_statistics(
        train_x, train_y, calibration_raw, seed=seed
    )
    evaluation_x = add_target_statistics(train_x, train_y, evaluation_raw)
    train_neighbor, calibration_neighbor = cross_fit_neighbor_statistics(
        train_x, train_y, calibration_raw, seed=seed + 1
    )
    evaluation_neighbor = add_neighbor_statistics(train_x, train_y, evaluation_raw)
    neighbor_columns = [
        column for column in train_neighbor if column.startswith("id_neighbor_")
    ]
    train_x = train_te.drop(columns=PROFILE_COLUMNS)
    train_x = train_x.join(train_neighbor[neighbor_columns])
    calibration_x = calibration_x.drop(columns=PROFILE_COLUMNS).join(
        calibration_neighbor[neighbor_columns]
    )
    evaluation_x = evaluation_x.drop(columns=PROFILE_COLUMNS).join(evaluation_neighbor)
    train_x = train_x.drop(columns=["building_id_order"])
    calibration_x = calibration_x.drop(columns=["building_id_order"])
    evaluation_x = evaluation_x.drop(columns=["building_id_order"])

    model = lgb.LGBMClassifier(
        objective="multiclassova", metric="None", n_estimators=1000,
        learning_rate=0.06, num_leaves=63, max_depth=-1,
        min_child_samples=100, colsample_bytree=0.9,
        subsample=0.85, subsample_freq=1, reg_lambda=3.0,
        n_jobs=4, random_state=seed, verbosity=-1,
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
    base_predictions = [LABELS[index] for index in eval_prob.argmax(axis=1)]
    summary = {
        "seed": seed,
        "method": "LightGBM multiclassova; OOF target/neighbor features; 80/10/10 split",
        "best_iteration": int(model.best_iteration_),
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            evaluation_y, base_predictions, average="macro"
        )),
        "evaluation_classification_report_uncalibrated": classification_report(
            evaluation_y, base_predictions, labels=LABELS, output_dict=True,
            zero_division=0,
        ),
        "calibration": calibrated,
    }
    path = ROOT / "reports" / "validation_exp_024_ova.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Base evaluation: {summary['evaluation_macro_f1_uncalibrated']:.6f}; "
          f"calibrated: {calibrated['evaluation_macro_f1']:.6f}", flush=True)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
