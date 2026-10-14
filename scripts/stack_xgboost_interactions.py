"""Test the interaction-feature XGBoost model in the strongest LightGBM stack."""

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


def probabilities(data):
    """Return class probabilities, blending multiclass/OVA LightGBM outputs."""
    if "calibration_probabilities" in data:
        return data["calibration_probabilities"], data["evaluation_probabilities"]
    calibration = 0.6 * data["calibration_multiclass"] + 0.4 * data["calibration_ova"]
    evaluation = 0.6 * data["evaluation_multiclass"] + 0.4 * data["evaluation_ova"]
    return calibration, evaluation


def main() -> None:
    """CV-select logistic stack regularization and score the held-out split."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
        "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz",
        "xgboost_holdout_094.npz",
    ]
    data = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    cal_labels, eval_labels = data[0]["calibration_labels"], data[0]["evaluation_labels"]
    for item in data[1:]:
        if not np.array_equal(item["calibration_labels"], cal_labels) or not np.array_equal(
            item["evaluation_labels"], eval_labels
        ):
            raise ValueError("Prediction rows are not aligned")
    arrays = [probabilities(item) for item in data]
    x = stack_features(*(item[0] for item in arrays))
    x_eval = stack_features(*(item[1] for item in arrays))
    y = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, val_i in cv.split(x, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=700))
            model.fit(x[fit_i], y[fit_i])
            oof[val_i] = model.predict_proba(x[val_i])
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"C={c}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=700))
    final.fit(x, y)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_labels, eval_prob, eval_labels)
    report = {"method": "Best location-interaction stack plus interaction-feature XGBoost",
              "base_models": names, "selected_C": best["C"],
              "calibration_oof_macro_f1": best["score"], "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_095_xgboost_interaction_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_095.npz",
        calibration_oof_probabilities=best["oof"], evaluation_probabilities=eval_prob,
        calibration_labels=cal_labels, evaluation_labels=eval_labels,
        selected_C=best["C"],
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
