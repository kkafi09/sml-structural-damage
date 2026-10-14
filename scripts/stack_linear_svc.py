"""Cross-validate a linear margin stack over OOF probability features."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


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
            raise ValueError("Calibration rows do not align")
        if not np.array_equal(data["evaluation_labels"], eval_labels):
            raise ValueError("Evaluation rows do not align")
    lgb_cal = [
        0.6 * item["calibration_multiclass"] + 0.4 * item["calibration_ova"]
        for item in (lgbm, lgbm_profile, lgbm_region)
    ]
    lgb_eval = [
        0.6 * item["evaluation_multiclass"] + 0.4 * item["evaluation_ova"]
        for item in (lgbm, lgbm_profile, lgbm_region)
    ]
    x_cal = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        xgb_profile["calibration_probabilities"], *lgb_cal,
        catboost["calibration_probabilities"],
    )
    x_eval = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        xgb_profile["evaluation_probabilities"], *lgb_eval,
        catboost["evaluation_probabilities"],
    )
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for class_weight in (None, "balanced"):
        for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
            oof_scores = np.zeros((len(y_cal), len(LABELS)), dtype=float)
            for fit_i, held_i in cv.split(x_cal, y_cal):
                model = make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=c_value, class_weight=class_weight,
                              max_iter=5000, dual="auto", random_state=2026),
                )
                model.fit(x_cal[fit_i], y_cal[fit_i])
                oof_scores[held_i] = model.decision_function(x_cal[held_i])
            oof_probabilities = softmax(oof_scores, axis=1)
            score = float(f1_score(y_cal, oof_scores.argmax(axis=1), average="macro"))
            print(f"C={c_value}, weights={class_weight}: OOF F1={score:.6f}", flush=True)
            if best is None or score > best["score"]:
                best = {"C": c_value, "class_weight": class_weight,
                        "score": score, "oof_probabilities": oof_probabilities}
    final_model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=best["C"], class_weight=best["class_weight"],
                  max_iter=5000, dual="auto", random_state=2026),
    )
    final_model.fit(x_cal, y_cal)
    eval_scores = final_model.decision_function(x_eval)
    eval_probabilities = softmax(eval_scores, axis=1)
    calibrated = optimize_class_bias(
        best["oof_probabilities"], cal_labels, eval_probabilities, eval_labels
    )
    report = {
        "method": "OOF LinearSVC stack over three XGBoost, three LightGBM and CatBoost",
        "parameters": {"C": best["C"], "class_weight": best["class_weight"]},
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(label) for label in eval_labels], eval_scores.argmax(axis=1),
            average="macro",
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_049_svc_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"LinearSVC stack evaluation F1={calibrated['evaluation_macro_f1']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
