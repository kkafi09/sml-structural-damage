"""Stack out-of-sample model probabilities using calibration-only cross-validation."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, optimize_class_bias


def stack_features(*probability_arrays):
    """Convert base-model probabilities into concatenated log-probability features."""
    return np.concatenate([
        np.log(np.clip(probabilities, 1e-6, 1.0))
        for probabilities in probability_arrays
    ], axis=1)


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    xgb7 = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    xgb6 = np.load(ROOT / "reports" / "xgboost_holdout_039.npz")
    xgb_profile = np.load(ROOT / "reports" / "xgboost_holdout_045.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    lgbm_profile = np.load(
        ROOT / "reports" / "lightgbm_holdout_041.npz", allow_pickle=True
    )
    lgbm_region = np.load(
        ROOT / "reports" / "lightgbm_holdout_044.npz", allow_pickle=True
    )
    catboost = np.load(ROOT / "reports" / "catboost_holdout_047.npz", allow_pickle=True)
    cal_labels = xgb7["calibration_labels"]
    eval_labels = xgb7["evaluation_labels"]
    for data in (xgb6, xgb_profile, lgbm, lgbm_profile, lgbm_region, catboost):
        if not np.array_equal(data["calibration_labels"], cal_labels):
            raise ValueError("Calibration rows differ between base models")
        if not np.array_equal(data["evaluation_labels"], eval_labels):
            raise ValueError("Evaluation rows differ between base models")
    lgb_cal = 0.6 * lgbm["calibration_multiclass"] + 0.4 * lgbm["calibration_ova"]
    lgb_eval = 0.6 * lgbm["evaluation_multiclass"] + 0.4 * lgbm["evaluation_ova"]
    lgb_profile_cal = 0.6 * lgbm_profile["calibration_multiclass"] + 0.4 * lgbm_profile["calibration_ova"]
    lgb_profile_eval = 0.6 * lgbm_profile["evaluation_multiclass"] + 0.4 * lgbm_profile["evaluation_ova"]
    lgb_region_cal = 0.6 * lgbm_region["calibration_multiclass"] + 0.4 * lgbm_region["calibration_ova"]
    lgb_region_eval = 0.6 * lgbm_region["evaluation_multiclass"] + 0.4 * lgbm_region["evaluation_ova"]
    calibration_x = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        xgb_profile["calibration_probabilities"],
        lgb_cal, lgb_profile_cal, lgb_region_cal,
        catboost["calibration_probabilities"],
    )
    evaluation_x = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        xgb_profile["evaluation_probabilities"],
        lgb_eval, lgb_profile_eval, lgb_region_eval,
        catboost["evaluation_probabilities"],
    )
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for regularization in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof_probabilities = np.zeros((len(y_cal), len(LABELS)), dtype=float)
        for fit_indices, held_indices in cv.split(calibration_x, y_cal):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=regularization, max_iter=500),
            )
            model.fit(calibration_x[fit_indices], y_cal[fit_indices])
            oof_probabilities[held_indices] = model.predict_proba(
                calibration_x[held_indices]
            )
        score = f1_score(y_cal, oof_probabilities.argmax(axis=1), average="macro")
        print(f"C={regularization}: calibration OOF Macro F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {
                "regularization": regularization,
                "score": float(score),
                "oof_probabilities": oof_probabilities,
            }
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best["regularization"], max_iter=500),
    )
    final_model.fit(calibration_x, y_cal)
    evaluation_probabilities = final_model.predict_proba(evaluation_x)
    calibrated = optimize_class_bias(
        best["oof_probabilities"], cal_labels,
        evaluation_probabilities, eval_labels,
    )
    evaluation_uncalibrated = f1_score(
        eval_labels,
        [LABELS[index] for index in evaluation_probabilities.argmax(axis=1)],
        average="macro",
    )
    report = {
        "method": "OOF logistic stacking of three XGBoost, three LightGBM and CatBoost",
        "selected_regularization": best["regularization"],
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(evaluation_uncalibrated),
        "calibration": calibrated,
    }
    path = ROOT / "reports" / "validation_exp_040_stacking.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_040.npz",
        calibration_oof_probabilities=best["oof_probabilities"],
        evaluation_probabilities=evaluation_probabilities,
        calibration_labels=cal_labels,
        evaluation_labels=eval_labels,
        selected_regularization=best["regularization"],
    )
    print(
        f"Stacked evaluation Macro F1={evaluation_uncalibrated:.6f}; "
        f"calibrated={calibrated['evaluation_macro_f1']:.6f}", flush=True
    )


if __name__ == "__main__":
    main()
