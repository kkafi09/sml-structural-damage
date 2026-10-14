"""Blend calibrated multiclass and one-vs-all LightGBM predictions."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import (
    CATEGORICAL,
    LABELS,
    PROFILE_COLUMNS,
    add_neighbor_statistics,
    add_region_neighbor_statistics,
    add_target_statistics,
    cross_fit_neighbor_statistics,
    cross_fit_region_neighbor_statistics,
    cross_fit_target_statistics,
    load_data,
    macro_f1_metric,
    optimize_class_bias,
)


def prepare_data(seed: int):
    """Prepare leakage-safe target and neighbor encodings for all three splits."""
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    cal_raw, eval_raw, cal_y, eval_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=seed + 1, stratify=holdout_y
    )
    train_te, cal_x = cross_fit_target_statistics(train_x, train_y, cal_raw, seed)
    eval_x = add_target_statistics(train_x, train_y, eval_raw)
    train_neighbors, cal_neighbors = cross_fit_neighbor_statistics(
        train_x, train_y, cal_raw, seed + 1
    )
    eval_neighbors = add_neighbor_statistics(train_x, train_y, eval_raw)
    train_region_neighbors, cal_region_neighbors = cross_fit_region_neighbor_statistics(
        train_x, train_y, cal_raw, seed + 2
    )
    eval_region_neighbors = add_region_neighbor_statistics(train_x, train_y, eval_raw)
    neighbor_columns = [
        column for column in train_neighbors if column.startswith("id_neighbor_")
    ]
    region_neighbor_columns = [
        column for column in train_region_neighbors
        if column.startswith("region_neighbor_")
    ]
    train_x = train_te.drop(columns=PROFILE_COLUMNS).join(
        train_neighbors[neighbor_columns]
    ).join(train_region_neighbors[region_neighbor_columns])
    cal_x = cal_x.drop(columns=PROFILE_COLUMNS).join(
        cal_neighbors[neighbor_columns]
    ).join(cal_region_neighbors[region_neighbor_columns])
    eval_x = eval_x.drop(columns=PROFILE_COLUMNS).join(eval_neighbors).join(
        eval_region_neighbors
    )
    for frame in (train_x, cal_x, eval_x):
        frame.drop(columns=["building_id_order"], inplace=True)
    return train_x, train_y, cal_x, cal_y, eval_x, eval_y


def main() -> None:
    """Fit both objectives and select their probability blend on calibration data."""
    seed = 2026
    train_x, train_y, cal_x, cal_y, eval_x, eval_y = prepare_data(seed)
    weight = {
        "Grade 1": 0.90, "Grade 2": 1.45, "Grade 3": 1.40,
        "Grade 4": 1.00, "Grade 5": 0.75,
    }
    models = [
        lgb.LGBMClassifier(
            objective="multiclass", metric="None", class_weight=weight,
            n_estimators=1000, learning_rate=0.06, num_leaves=63,
            min_child_samples=100, colsample_bytree=0.9,
            subsample=0.85, subsample_freq=1, reg_lambda=3.0,
            n_jobs=4, random_state=seed + 17, verbosity=-1,
        ),
        lgb.LGBMClassifier(
            objective="multiclassova", metric="None", n_estimators=1000,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, n_jobs=4, random_state=seed + 18, verbosity=-1,
        ),
    ]
    cal_predictions = []
    eval_predictions = []
    iterations = []
    for model in models:
        model.fit(
            train_x, train_y, eval_set=[(cal_x, cal_y)],
            eval_metric=macro_f1_metric,
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                       lgb.log_evaluation(100)],
            categorical_feature=CATEGORICAL,
        )
        cal_predictions.append(model.predict_proba(cal_x))
        eval_predictions.append(model.predict_proba(eval_x))
        iterations.append(int(model.best_iteration_))
        print(f"{model.objective_}: best_iteration={model.best_iteration_}", flush=True)

    best = None
    results = []
    for multiclass_weight in np.linspace(0.0, 1.0, 21):
        cal_prob = (multiclass_weight * cal_predictions[0]
                    + (1.0 - multiclass_weight) * cal_predictions[1])
        eval_prob = (multiclass_weight * eval_predictions[0]
                     + (1.0 - multiclass_weight) * eval_predictions[1])
        calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
        base_score = float(f1_score(
            eval_y, [LABELS[index] for index in eval_prob.argmax(axis=1)],
            average="macro",
        ))
        result = {
            "multiclass_weight": float(multiclass_weight),
            "calibration_macro_f1_after_bias": calibrated["calibration_macro_f1"],
            "evaluation_macro_f1": base_score,
            "evaluation_macro_f1_after_bias": calibrated["evaluation_macro_f1"],
            "bias_by_grade": calibrated["bias_by_grade"],
        }
        results.append(result)
        if best is None or result["calibration_macro_f1_after_bias"] > \
                best["calibration_macro_f1_after_bias"]:
            best = result
    path = ROOT / "reports" / "validation_exp_088_weighted_interactions.json"
    summary = {
        "seed": seed,
        "method": "Multiclass + multiclassova LightGBM; 80/10/10 split",
        "best_iterations": iterations,
        "selected_weight_by_calibration": best["multiclass_weight"],
        "selected_evaluation_macro_f1_after_bias":
            best["evaluation_macro_f1_after_bias"],
        "blends": results,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "reports" / "lightgbm_holdout_088.npz",
        calibration_multiclass=cal_predictions[0],
        evaluation_multiclass=eval_predictions[0],
        calibration_ova=cal_predictions[1],
        evaluation_ova=eval_predictions[1],
        calibration_labels=cal_y.to_numpy(),
        evaluation_labels=eval_y.to_numpy(),
    )
    print(f"Selected blend evaluation Macro F1: "
          f"{best['evaluation_macro_f1_after_bias']:.6f}", flush=True)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
