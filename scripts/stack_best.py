"""Reproduce the strongest leakage-safe validation stack found so far."""

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
    """Run the experiment and save its report and prediction artifacts."""
    xgb7 = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    xgb6 = np.load(ROOT / "reports" / "xgboost_holdout_039.npz")
    xgb_seed42 = np.load(ROOT / "reports" / "xgboost_holdout_053.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    lgbm_profile = np.load(ROOT / "reports" / "lightgbm_holdout_041.npz", allow_pickle=True)
    cal_labels, eval_labels = xgb7["calibration_labels"], xgb7["evaluation_labels"]
    for data in (xgb6, xgb_seed42, lgbm, lgbm_profile):
        if not np.array_equal(data["calibration_labels"], cal_labels):
            raise ValueError("Calibration rows do not align")
        if not np.array_equal(data["evaluation_labels"], eval_labels):
            raise ValueError("Evaluation rows do not align")
    lgb_cal = [
        0.6 * data["calibration_multiclass"] + 0.4 * data["calibration_ova"]
        for data in (lgbm, lgbm_profile)
    ]
    lgb_eval = [
        0.6 * data["evaluation_multiclass"] + 0.4 * data["evaluation_ova"]
        for data in (lgbm, lgbm_profile)
    ]
    x_cal = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        xgb_seed42["calibration_probabilities"],
        *lgb_cal
    )
    x_eval = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        xgb_seed42["evaluation_probabilities"],
        *lgb_eval
    )
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y_cal), len(LABELS)), dtype=float)
        for fit_i, held_i in cv.split(x_cal, y_cal):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c_value, max_iter=500)
            )
            model.fit(x_cal[fit_i], y_cal[fit_i])
            oof[held_i] = model.predict_proba(x_cal[held_i])
        score = float(f1_score(y_cal, oof.argmax(axis=1), average="macro"))
        print(f"C={c_value}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    final = make_pipeline(
        StandardScaler(), LogisticRegression(
            C=best["C"], max_iter=500
        )
    )
    final.fit(x_cal, y_cal)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_labels, eval_prob, eval_labels)
    report = {
        "method": "OOF logistic stack: three XGBoost configurations/seeds plus two LightGBM blends",
        "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(label) for label in eval_labels], eval_prob.argmax(axis=1),
            average="macro",
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_054_seed_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_054.npz",
        calibration_oof_probabilities=best["oof"],
        evaluation_probabilities=eval_prob,
        calibration_labels=cal_labels,
        evaluation_labels=eval_labels,
        selected_C=best["C"],
    )
    print(f"Best holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
