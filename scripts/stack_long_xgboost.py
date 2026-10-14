"""Test the long-neighbor XGBoost probabilities in the best stack."""

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
    """Compare regularized logistic stack variants on calibration OOF rows."""
    names = ["xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
             "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
             "lightgbm_holdout_041.npz", "xgboost_holdout_069.npz"]
    data = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    labels, eval_labels = data[0]["calibration_labels"], data[0]["evaluation_labels"]
    for row in data[1:]:
        if not np.array_equal(labels, row["calibration_labels"]) or not np.array_equal(
            eval_labels, row["evaluation_labels"]
        ):
            raise ValueError("Prediction rows do not align")
    cal_probs, eval_probs = [], []
    for row in data:
        if "calibration_probabilities" in row:
            cal_probs.append(row["calibration_probabilities"])
            eval_probs.append(row["evaluation_probabilities"])
        else:
            cal_probs.append(0.6 * row["calibration_multiclass"] + 0.4 * row["calibration_ova"])
            eval_probs.append(0.6 * row["evaluation_multiclass"] + 0.4 * row["evaluation_ova"])
    x, x_eval = stack_features(*cal_probs), stack_features(*eval_probs)
    y = np.array([LABELS.index(label) for label in labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, val_i in cv.split(x, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500))
            model.fit(x[fit_i], y[fit_i])
            oof[val_i] = model.predict_proba(x[val_i])
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"C={c}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=500))
    final.fit(x, y)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], labels, eval_prob, eval_labels)
    report = {"method": "Best five-model stack plus long-radius XGBoost",
              "selected_C": best["C"], "calibration_oof_macro_f1": best["score"],
              "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_070_long_xgboost_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
