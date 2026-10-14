"""Cross-validate beta calibration on the best stack's OOF probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import optimize_class_bias


def beta_features(probability: np.ndarray, grade: int) -> np.ndarray:
    """Return the two log-odds features used by classwise beta calibration."""
    p = np.clip(probability[:, grade], 1e-7, 1.0 - 1e-7)
    return np.column_stack([np.log(p), np.log1p(-p)])


def fit_beta(train_probability: np.ndarray, train_y: np.ndarray,
             query_probability: np.ndarray, regularization: float) -> np.ndarray:
    """Fit binary beta maps per grade, then normalize their predicted scores."""
    calibrated = np.zeros_like(query_probability, dtype=float)
    for grade in range(len(LABELS)):
        model = LogisticRegression(C=regularization, solver="lbfgs", max_iter=200)
        model.fit(beta_features(train_probability, grade), (train_y == grade).astype(int))
        calibrated[:, grade] = model.predict_proba(beta_features(query_probability, grade))[:, 1]
    return calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)


def main() -> None:
    """Select beta calibration strength on OOF folds and evaluate holdout."""
    data = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    probability = data["calibration_oof_probabilities"]
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    eval_labels = data["evaluation_labels"]
    eval_y = np.array([LABELS.index(label) for label in eval_labels])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    configurations = []
    best = None
    for c_value in (0.1, 1.0, 10.0):
        oof = np.zeros_like(probability)
        for fold, (fit_idx, held_idx) in enumerate(folds.split(probability, y), 1):
            oof[held_idx] = fit_beta(probability[fit_idx], y[fit_idx],
                                     probability[held_idx], c_value)
            print(f"C={c_value} fold={fold} complete", flush=True)
        score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        configurations.append({"C": c_value, "calibration_oof_macro_f1": score})
        print(f"C={c_value}: OOF Macro F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    evaluation_probability = fit_beta(
        probability, y, data["evaluation_probabilities"], best["C"]
    )
    calibrated = optimize_class_bias(
        best["oof"], data["calibration_labels"],
        evaluation_probability, eval_labels,
    )
    report = {
        "method": "Classwise beta calibration on OOF stack probabilities",
        "configurations": configurations, "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            eval_y, evaluation_probability.argmax(axis=1), average="macro"
        )),
        "calibration": calibrated,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_130_beta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "stacking_holdout_130.npz",
                        calibration_oof_probabilities=best["oof"],
                        evaluation_probabilities=evaluation_probability,
                        calibration_labels=data["calibration_labels"],
                        evaluation_labels=eval_labels)
    print(f"Evaluation Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
