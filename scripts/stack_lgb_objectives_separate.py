"""Let OOF logistic stacking learn separate weights for each LightGBM objective."""

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
sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main():
    """Run the experiment and save its report and prediction artifacts."""
    xgb = [np.load(ROOT / "reports" / name) for name in (
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz")]
    lgb = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in (
        "lightgbm_holdout_030.npz", "lightgbm_holdout_041.npz",
        "lightgbm_holdout_085.npz")]
    arrays = xgb + lgb
    cal_labels, eval_labels = arrays[0]["calibration_labels"], arrays[0]["evaluation_labels"]
    for item in arrays[1:]:
        if not np.array_equal(item["calibration_labels"], cal_labels) or not np.array_equal(
            item["evaluation_labels"], eval_labels):
            raise ValueError("Prediction rows are not aligned")
    cal_inputs = [item["calibration_probabilities"] for item in xgb]
    eval_inputs = [item["evaluation_probabilities"] for item in xgb]
    for item in lgb:
        cal_inputs.extend([item["calibration_multiclass"], item["calibration_ova"]])
        eval_inputs.extend([item["evaluation_multiclass"], item["evaluation_ova"]])
    x, x_eval = stack_features(*cal_inputs), stack_features(*eval_inputs)
    y = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_idx, held_idx in cv.split(x, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=700))
            model.fit(x[fit_idx], y[fit_idx])
            oof[held_idx] = model.predict_proba(x[held_idx])
        score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        print(f"C={c_value}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=700))
    final.fit(x, y)
    evaluation = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_labels, evaluation, eval_labels)
    report = {"method": "Separate multiclass and OVA probability features in OOF logistic stack",
              "selected_C": best["C"], "calibration_oof_macro_f1": best["score"],
              "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_115_lgb_objectives_separate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "stacking_holdout_115.npz",
                        calibration_oof_probabilities=best["oof"],
                        evaluation_probabilities=evaluation,
                        calibration_labels=cal_labels, evaluation_labels=eval_labels,
                        selected_C=best["C"])
    print(f"Separate objectives holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
