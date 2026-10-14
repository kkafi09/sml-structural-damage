"""Fit a nonlinear meta-classifier over OOF probabilities from the best stack."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def probability_features(probabilities):
    """Build compact probability, logit, margin, and entropy meta-features."""
    p = np.clip(probabilities, 1e-8, 1.0)
    sorted_p = np.sort(p, axis=1)
    return np.column_stack([
        p, np.log(p), np.log(p / np.maximum(1.0 - p, 1e-8)),
        sorted_p[:, -1] - sorted_p[:, -2],
        -np.sum(p * np.log(p), axis=1),
    ])


def main() -> None:
    """Select HGB complexity using calibration CV and score evaluation data."""
    data = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    p, p_eval = data["calibration_oof_probabilities"], data["evaluation_probabilities"]
    x, x_eval = probability_features(p), probability_features(p_eval)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for leaves, iterations, l2 in ((5, 100, 2.0), (7, 150, 2.0),
                                   (10, 150, 5.0), (15, 200, 5.0),
                                   (20, 250, 10.0)):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, val_i in cv.split(x, y):
            model = HistGradientBoostingClassifier(
                max_iter=iterations, max_leaf_nodes=leaves,
                learning_rate=0.08, l2_regularization=l2,
                min_samples_leaf=100, random_state=2026,
            )
            model.fit(x[fit_i], y[fit_i])
            oof[val_i] = model.predict_proba(x[val_i])
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"leaves={leaves}, iterations={iterations}, l2={l2}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"params": (leaves, iterations, l2), "score": score, "oof": oof}
    leaves, iterations, l2 = best["params"]
    model = HistGradientBoostingClassifier(
        max_iter=iterations, max_leaf_nodes=leaves, learning_rate=0.08,
        l2_regularization=l2, min_samples_leaf=100, random_state=2026,
    )
    model.fit(x, y)
    eval_prob = model.predict_proba(x_eval)
    calibrated = optimize_class_bias(
        best["oof"], data["calibration_labels"], eval_prob, data["evaluation_labels"]
    )
    report = {"method": "HGB meta-classifier over best-stack probability features",
              "parameters": {"max_leaf_nodes": leaves, "max_iter": iterations, "l2": l2},
              "calibration_oof_macro_f1": best["score"], "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_071_hgb_probability_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
