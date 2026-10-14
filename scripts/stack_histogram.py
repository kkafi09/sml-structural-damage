"""Test a cross-validated nonlinear meta-model on base-model probabilities."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, optimize_class_bias, load_data
from scripts.stack_validation_models import stack_features


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    xgb7 = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    xgb6 = np.load(ROOT / "reports" / "xgboost_holdout_039.npz")
    xgb_profile = np.load(ROOT / "reports" / "xgboost_holdout_045.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    lgbm_profile = np.load(ROOT / "reports" / "lightgbm_holdout_041.npz", allow_pickle=True)
    lgbm_region = np.load(ROOT / "reports" / "lightgbm_holdout_044.npz", allow_pickle=True)
    catboost = np.load(ROOT / "reports" / "catboost_holdout_047.npz", allow_pickle=True)
    cal_labels = xgb7["calibration_labels"]
    eval_labels = xgb7["evaluation_labels"]
    for data in (xgb6, xgb_profile, lgbm, lgbm_profile, lgbm_region, catboost):
        if not np.array_equal(data["calibration_labels"], cal_labels):
            raise ValueError("Calibration rows differ")
        if not np.array_equal(data["evaluation_labels"], eval_labels):
            raise ValueError("Evaluation rows differ")
    lgb_calibrations = [
        0.6 * item["calibration_multiclass"] + 0.4 * item["calibration_ova"]
        for item in (lgbm, lgbm_profile, lgbm_region)
    ]
    lgb_evaluations = [
        0.6 * item["evaluation_multiclass"] + 0.4 * item["evaluation_ova"]
        for item in (lgbm, lgbm_profile, lgbm_region)
    ]
    cal_x = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        xgb_profile["calibration_probabilities"], *lgb_calibrations,
        catboost["calibration_probabilities"],
    )
    eval_x = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        xgb_profile["evaluation_probabilities"], *lgb_evaluations,
        catboost["evaluation_probabilities"],
    )
    raw_features, raw_labels = load_data()
    _, raw_holdout, _, raw_holdout_labels = train_test_split(
        raw_features, raw_labels, test_size=0.2, random_state=2026,
        stratify=raw_labels,
    )
    raw_calibration, raw_evaluation, raw_calibration_labels, raw_evaluation_labels = train_test_split(
        raw_holdout, raw_holdout_labels, test_size=0.5, random_state=2027,
        stratify=raw_holdout_labels,
    )
    if not np.array_equal(raw_calibration_labels.to_numpy(), cal_labels):
        raise ValueError("Raw calibration rows do not align with model probabilities")
    if not np.array_equal(raw_evaluation_labels.to_numpy(), eval_labels):
        raise ValueError("Raw evaluation rows do not align with model probabilities")
    raw_columns = [
        column for column in raw_features.select_dtypes(include=np.number).columns
        if column != "building_id_order"
    ]
    calibration_raw_values = raw_calibration[raw_columns].to_numpy(dtype=float)
    evaluation_raw_values = raw_evaluation[raw_columns].to_numpy(dtype=float)
    categorical_raw_columns = [
        "district_id", "vdcmun_id", "ward_id", "land_surface_condition",
        "foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
        "position", "plan_configuration", "superstructure",
    ]
    calibration_codes = np.column_stack([
        pd.Categorical(
            raw_calibration[column], categories=raw_features[column].astype("category").cat.categories
        ).codes
        for column in categorical_raw_columns
    ])
    evaluation_codes = np.column_stack([
        pd.Categorical(
            raw_evaluation[column], categories=raw_features[column].astype("category").cat.categories
        ).codes
        for column in categorical_raw_columns
    ])
    cal_x = np.column_stack([cal_x, calibration_raw_values, calibration_codes])
    eval_x = np.column_stack([eval_x, evaluation_raw_values, evaluation_codes])
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    y_eval = np.array([LABELS.index(label) for label in eval_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for leaves in (7, 15, 31):
        for minimum_leaf in (100, 300):
            oof = np.zeros((len(y_cal), len(LABELS)), dtype=float)
            for fit_i, held_i in cv.split(cal_x, y_cal):
                model = HistGradientBoostingClassifier(
                    learning_rate=0.08, max_iter=180, max_leaf_nodes=leaves,
                    min_samples_leaf=minimum_leaf, l2_regularization=5.0,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=20, random_state=2026,
                )
                model.fit(cal_x[fit_i], y_cal[fit_i])
                oof[held_i] = model.predict_proba(cal_x[held_i])
            score = f1_score(y_cal, oof.argmax(axis=1), average="macro")
            print(
                f"leaves={leaves}, min_leaf={minimum_leaf}: OOF F1={score:.6f}",
                flush=True,
            )
            if best is None or score > best["score"]:
                best = {
                    "leaves": leaves, "minimum_leaf": minimum_leaf,
                    "score": float(score), "oof": oof,
                }
    final = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=180, max_leaf_nodes=best["leaves"],
        min_samples_leaf=best["minimum_leaf"], l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=2026,
    )
    final.fit(cal_x, y_cal)
    eval_probabilities = final.predict_proba(eval_x)
    calibrated = optimize_class_bias(
        best["oof"], cal_labels, eval_probabilities, eval_labels
    )
    report = {
        "method": "OOF HistGradientBoosting stack of three XGBoost, three LightGBM and CatBoost",
        "parameters": {"max_leaf_nodes": best["leaves"],
                       "min_samples_leaf": best["minimum_leaf"]},
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            y_eval, eval_probabilities.argmax(axis=1), average="macro"
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_046_hist_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Stacked HGB evaluation Macro F1={calibrated['evaluation_macro_f1']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
