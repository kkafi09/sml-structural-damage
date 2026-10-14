"""Cross-validate a broad stack of diverse leakage-safe model predictions."""

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


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "xgboost_holdout_094.npz",
        "lightgbm_holdout_030.npz", "lightgbm_holdout_041.npz",
        "lightgbm_holdout_085.npz", "lightgbm_holdout_096.npz",
        "catboost_encoded_holdout_101.npz", "lightgbm_spatial_holdout_103.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    cal_labels, eval_labels = arrays[0]["calibration_labels"], arrays[0]["evaluation_labels"]
    for data in arrays[1:]:
        if not np.array_equal(data["calibration_labels"], cal_labels) or not np.array_equal(
            data["evaluation_labels"], eval_labels
        ):
            raise ValueError("Prediction rows are not aligned")
    cal_probs, eval_probs = [], []
    for data in arrays:
        if "calibration_probabilities" in data:
            cal_probs.append(data["calibration_probabilities"])
            eval_probs.append(data["evaluation_probabilities"])
        else:
            cal_probs.append(0.6 * data["calibration_multiclass"]
                             + 0.4 * data["calibration_ova"])
            eval_probs.append(0.6 * data["evaluation_multiclass"]
                              + 0.4 * data["evaluation_ova"])
    x_cal, x_eval = stack_features(*cal_probs), stack_features(*eval_probs)
    y_cal = np.array([LABELS.index(value) for value in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros((len(y_cal), len(LABELS)))
        for fit_idx, held_idx in cv.split(x_cal, y_cal):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c_value, max_iter=700)
            )
            model.fit(x_cal[fit_idx], y_cal[fit_idx])
            oof[held_idx] = model.predict_proba(x_cal[held_idx])
        score = float(f1_score(y_cal, oof.argmax(axis=1), average="macro"))
        print(f"C={c_value}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    final = make_pipeline(
        StandardScaler(), LogisticRegression(C=best["C"], max_iter=700)
    )
    final.fit(x_cal, y_cal)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_labels, eval_prob, eval_labels)
    report = {
        "method": "OOF logistic stack across ten diverse models",
        "base_models": names,
        "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"],
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_106_all_diverse_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_106.npz",
        calibration_oof_probabilities=best["oof"],
        evaluation_probabilities=eval_prob,
        calibration_labels=cal_labels,
        evaluation_labels=eval_labels,
        selected_C=best["C"],
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
