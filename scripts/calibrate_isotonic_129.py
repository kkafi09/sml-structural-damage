"""Cross-validate per-grade isotonic probability calibration on OOF stack outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import optimize_class_bias


def fit_transform(train_probability: np.ndarray, train_labels: np.ndarray,
                  query_probability: np.ndarray) -> np.ndarray:
    """Fit one isotonic mapping per class and normalize transformed probabilities."""
    transformed = np.zeros_like(query_probability, dtype=float)
    for grade in range(len(LABELS)):
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(train_probability[:, grade], (train_labels == grade).astype(float))
        transformed[:, grade] = model.predict(query_probability[:, grade])
    totals = transformed.sum(axis=1, keepdims=True)
    return transformed / np.maximum(totals, 1e-12)


def main() -> None:
    """Select isotonic calibration on folds and evaluate the untouched holdout."""
    data = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    probability = data["calibration_oof_probabilities"]
    labels_text = data["calibration_labels"]
    y = np.array([LABELS.index(label) for label in labels_text])
    evaluation_labels = data["evaluation_labels"]
    evaluation_y = np.array([LABELS.index(label) for label in evaluation_labels])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros_like(probability)
    for fold, (fit_idx, held_idx) in enumerate(folds.split(probability, y), 1):
        oof[held_idx] = fit_transform(probability[fit_idx], y[fit_idx], probability[held_idx])
        print(f"fold={fold} complete", flush=True)
    raw_cv = float(f1_score(y, oof.argmax(axis=1), average="macro"))
    final_probability = fit_transform(probability, y, data["evaluation_probabilities"])
    calibrated = optimize_class_bias(oof, labels_text, final_probability, evaluation_labels)
    report = {
        "method": "Per-class isotonic calibration cross-fitted on stack OOF probabilities",
        "calibration_oof_macro_f1_uncalibrated": raw_cv,
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            evaluation_y, final_probability.argmax(axis=1), average="macro"
        )),
        "calibration": calibrated,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_129_isotonic.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "stacking_holdout_129.npz",
                        calibration_oof_probabilities=oof,
                        evaluation_probabilities=final_probability,
                        calibration_labels=labels_text, evaluation_labels=evaluation_labels)
    print(f"Evaluation Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
