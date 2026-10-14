"""Evaluate CatBoost with survey-ID prefixes as native categorical features."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, load_data as load_engineered_data, optimize_class_bias


CATEGORICAL = [
    "district_id", "vdcmun_id", "ward_id", "land_surface_condition",
    "foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
    "position", "plan_configuration", "superstructure", "id_prefix6", "id_prefix8",
    "id_prefix9", "id_prefix10",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load data and add two survey-ID prefix categories."""
    features, labels = load_engineered_data()
    features = features.drop(columns=[
        "building_id_order", "structure_profile", "id_profile", "ward_profile",
        "age_band", "floor_band", "area_band", "height_band", "building_profile",
    ])
    for column in CATEGORICAL:
        features[column] = features[column].fillna("__MISSING__").astype(str)
    return features, labels


def main() -> None:
    """Train CatBoost with early stopping and calibrate grade decisions."""
    seed = 2026
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    calibration_x, evaluation_x, calibration_y, evaluation_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=seed + 1,
        stratify=holdout_y,
    )
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro;use_weights=false",
        iterations=500, depth=8, learning_rate=0.08, l2_leaf_reg=12,
        auto_class_weights="SqrtBalanced", max_ctr_complexity=1,
        cat_features=CATEGORICAL, random_seed=seed, thread_count=4,
        allow_writing_files=False, verbose=20,
    )
    model.fit(
        train_x, train_y, eval_set=(calibration_x, calibration_y),
        early_stopping_rounds=60, use_best_model=True, verbose=25,
    )
    cal_prob = model.predict_proba(calibration_x)
    eval_prob = model.predict_proba(evaluation_x)
    calibrated = optimize_class_bias(cal_prob, calibration_y, eval_prob, evaluation_y)
    base_predictions = [LABELS[index] for index in eval_prob.argmax(axis=1)]
    summary = {
        "seed": seed,
        "method": "CatBoost with native prefix6/prefix8 categories; 80/10/10 split",
        "parameters": {
            "iterations": 500, "depth": 8, "learning_rate": 0.08,
            "l2_leaf_reg": 12, "auto_class_weights": "SqrtBalanced",
            "max_ctr_complexity": 1,
        },
        "best_iteration": int(model.get_best_iteration()),
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            evaluation_y, base_predictions, average="macro"
        )),
        "evaluation_classification_report_uncalibrated": classification_report(
            evaluation_y, base_predictions, labels=LABELS, output_dict=True,
            zero_division=0,
        ),
        "calibration": calibrated,
    }
    path = ROOT / "reports" / "validation_exp_047_catboost_profiles.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "reports" / "catboost_holdout_047.npz",
        calibration_probabilities=cal_prob,
        evaluation_probabilities=eval_prob,
        calibration_labels=np.asarray(calibration_y),
        evaluation_labels=np.asarray(evaluation_y),
    )
    print(f"Base evaluation Macro F1: {summary['evaluation_macro_f1_uncalibrated']:.6f}; "
          f"calibrated: {calibrated['evaluation_macro_f1']:.6f}", flush=True)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
