"""Test whether diverse tree models complement the strongest validation stack."""

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
    """Select regularization from OOF CV and evaluate diverse model stack."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "xgboost_holdout_055.npz",
        "xgboost_holdout_045.npz", "lightgbm_holdout_030.npz",
        "lightgbm_holdout_041.npz", "lightgbm_holdout_044.npz",
        "catboost_holdout_047.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    labels, eval_labels = arrays[0]["calibration_labels"], arrays[0]["evaluation_labels"]
    for arr in arrays[1:]:
        if not np.array_equal(labels, arr["calibration_labels"]) or not np.array_equal(
            eval_labels, arr["evaluation_labels"]
        ):
            raise ValueError("All base predictions must use identical split rows")
    cal_probs, eval_probs = [], []
    for arr in arrays:
        if "calibration_probabilities" in arr:
            cal_probs.append(arr["calibration_probabilities"])
            eval_probs.append(arr["evaluation_probabilities"])
        else:
            cal_probs.append(0.6 * arr["calibration_multiclass"] + 0.4 * arr["calibration_ova"])
            eval_probs.append(0.6 * arr["evaluation_multiclass"] + 0.4 * arr["evaluation_ova"])
    x, x_eval = stack_features(*cal_probs), stack_features(*eval_probs)
    y = np.array([LABELS.index(label) for label in labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
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
    calibrated = optimize_class_bias(best["oof"], labels, eval_prob, eval_labels)
    report = {
        "method": "OOF logistic stack of five best models plus seed, profile, region and CatBoost models",
        "base_models": names, "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"], "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_066_diverse_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
