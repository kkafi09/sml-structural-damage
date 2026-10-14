"""Fit a cross-validated second-stage calibrator on stacked OOF probabilities."""

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
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Select a calibration model on OOF rows and report held-out score."""
    data = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    p = np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0)
    p_eval = np.clip(data["evaluation_probabilities"], 1e-8, 1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, val_i in cv.split(p, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
            model.fit(np.log(p[fit_i]), y[fit_i])
            oof[val_i] = model.predict_proba(np.log(p[val_i]))
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"C={c}: calibration CV Macro F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=1000))
    final.fit(np.log(p), y)
    result = final.predict_proba(np.log(p_eval))
    score = float(f1_score(y_eval, result.argmax(1), average="macro"))
    calibrated = optimize_class_bias(
        best["oof"], data["calibration_labels"], result, data["evaluation_labels"]
    )
    report = {
        "method": "CV-selected logistic recalibration of best-stack OOF probabilities",
        "selected_C": best["C"], "calibration_cv_macro_f1": best["score"],
        "evaluation_macro_f1": score, "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_065_meta_calibrator.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
