"""Measure whether the location-interaction LightGBM model improves the best stack."""

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

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Cross-validate stack regularization and score a fixed holdout."""
    files = [
        np.load(ROOT / "reports" / name, allow_pickle=True)
        for name in (
            "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
            "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
            "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz",
            "lightgbm_holdout_088.npz",
        )
    ]
    cal_y_labels, eval_y_labels = files[0]["calibration_labels"], files[0]["evaluation_labels"]
    for data in files[1:]:
        if not np.array_equal(data["calibration_labels"], cal_y_labels):
            raise ValueError("Calibration rows do not align")
        if not np.array_equal(data["evaluation_labels"], eval_y_labels):
            raise ValueError("Evaluation rows do not align")
    xgb = [data for data in files[:3]]
    lgb = [0.6 * data["calibration_multiclass"] + 0.4 * data["calibration_ova"]
           for data in files[3:]]
    lgb_eval = [0.6 * data["evaluation_multiclass"] + 0.4 * data["evaluation_ova"]
                for data in files[3:]]
    x_cal = stack_features(*(d["calibration_probabilities"] for d in xgb), *lgb)
    x_eval = stack_features(*(d["evaluation_probabilities"] for d in xgb), *lgb_eval)
    y_cal = np.array([LABELS.index(label) for label in cal_y_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y_cal), len(LABELS)))
        for fit_i, held_i in cv.split(x_cal, y_cal):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=500))
            model.fit(x_cal[fit_i], y_cal[fit_i])
            oof[held_i] = model.predict_proba(x_cal[held_i])
        score = float(f1_score(y_cal, oof.argmax(axis=1), average="macro"))
        print(f"C={c_value}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=500))
    final.fit(x_cal, y_cal)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_y_labels, eval_prob, eval_y_labels)
    report = {
        "method": "OOF stack: three XGBoost, baseline/profile LightGBM, location-interaction and weighted LightGBM",
        "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(label) for label in eval_y_labels], eval_prob.argmax(axis=1), average="macro"
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_089_weighted_interaction_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_089.npz",
        calibration_oof_probabilities=best["oof"],
        evaluation_probabilities=eval_prob,
        calibration_labels=cal_y_labels,
        evaluation_labels=eval_y_labels,
        selected_C=best["C"],
    )
    print(f"Best holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
