"""Evaluate per-class Platt calibration on best-stack out-of-fold probabilities."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def platt_fit_predict(train_p, train_y, query_p, c_value):
    """Fit independent binary logistic calibrators to each class score."""
    outputs = np.zeros((len(query_p), len(LABELS)))
    for grade in range(len(LABELS)):
        model = LogisticRegression(C=c_value, max_iter=300)
        model.fit(train_p[:, grade:grade + 1], (train_y == grade).astype(int))
        outputs[:, grade] = model.predict_proba(query_p[:, grade:grade + 1])[:, 1]
    return outputs


def main() -> None:
    """Cross-validate calibration strength and score the untouched evaluation rows."""
    data = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    p = np.clip(data["calibration_oof_probabilities"], 1e-6, 1 - 1e-6)
    p_eval = np.clip(data["evaluation_probabilities"], 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    eval_logits = np.log(p_eval / (1 - p_eval))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        oof = np.zeros_like(p)
        for fit_i, val_i in cv.split(logits, y):
            oof[val_i] = platt_fit_predict(logits[fit_i], y[fit_i], logits[val_i], c)
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"C={c}: calibration CV F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c, "score": score, "oof": oof}
    eval_cal = platt_fit_predict(logits, y, eval_logits, best["C"])
    score = float(f1_score(y_eval, eval_cal.argmax(1), average="macro"))
    report = {
        "method": "One-vs-rest Platt scaling of each stacked class probability",
        "selected_C": best["C"], "calibration_cv_macro_f1": best["score"],
        "evaluation_macro_f1": score,
    }
    (ROOT / "reports" / "validation_exp_068_platt.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
