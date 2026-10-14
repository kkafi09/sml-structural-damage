"""Train LightGBM with cross-fitted ward spatial target distributions."""

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import (
    CATEGORICAL, LABELS, add_spatial_target_statistics,
    cross_fit_spatial_target_statistics, macro_f1_metric, optimize_class_bias,
)


def main() -> None:
    """Add spatial features to cached leakage-safe matrices and fit two objectives."""
    cache_path = ROOT / "reports" / "catboost_encoded_data_101.pkl"
    with cache_path.open("rb") as cache_file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(cache_file)

    train_spatial, cal_spatial = cross_fit_spatial_target_statistics(
        train_x, train_y, cal_x, seed=2028
    )
    eval_spatial = add_spatial_target_statistics(train_x, train_y, eval_x)
    spatial_columns = [c for c in train_spatial if c.startswith("spatial_")]
    train_x = train_x.join(train_spatial[spatial_columns])
    cal_x = cal_x.join(cal_spatial[spatial_columns])
    eval_x = eval_x.join(eval_spatial[spatial_columns])

    weights = {
        "Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
        "Grade 4": 0.95, "Grade 5": 0.80,
    }
    models = [
        lgb.LGBMClassifier(
            objective="multiclass", metric="None", class_weight=weights,
            n_estimators=1200, learning_rate=0.06, num_leaves=63,
            min_child_samples=100, colsample_bytree=0.9,
            subsample=0.85, subsample_freq=1, reg_lambda=3.0,
            n_jobs=4, random_state=2026, verbosity=-1,
        ),
        lgb.LGBMClassifier(
            objective="multiclassova", metric="None", n_estimators=1200,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, n_jobs=4, random_state=2027, verbosity=-1,
        ),
    ]
    cal_probs, eval_probs, iterations = [], [], []
    for model in models:
        model.fit(
            train_x, train_y, eval_set=[(cal_x, cal_y)],
            eval_metric=macro_f1_metric,
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                       lgb.log_evaluation(100)],
            categorical_feature=CATEGORICAL,
        )
        cal_probs.append(model.predict_proba(cal_x))
        eval_probs.append(model.predict_proba(eval_x))
        iterations.append(int(model.best_iteration_))
        print(f"{model.objective_}: best_iteration={model.best_iteration_}", flush=True)

    cal_prob = 0.6 * cal_probs[0] + 0.4 * cal_probs[1]
    eval_prob = 0.6 * eval_probs[0] + 0.4 * eval_probs[1]
    calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
    report = {
        "method": "LightGBM with cross-fitted ward-neighbor spatial target features",
        "best_iterations": iterations,
        "spatial_features": spatial_columns,
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_103_spatial_encoded.json").write_text(
        __import__("json").dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "lightgbm_spatial_holdout_103.npz",
        calibration_multiclass=cal_probs[0], evaluation_multiclass=eval_probs[0],
        calibration_ova=cal_probs[1], evaluation_ova=eval_probs[1],
        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy(),
    )
    print(f"Spatial model holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
