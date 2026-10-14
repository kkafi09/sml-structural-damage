"""Tune a calibration-only convex blend of base and encoded CatBoost stacks."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    base = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    encoded = np.load(ROOT / "reports" / "stacking_holdout_102.npz", allow_pickle=True)
    if not np.array_equal(base["calibration_labels"], encoded["calibration_labels"]):
        raise ValueError("Calibration labels do not align")
    if not np.array_equal(base["evaluation_labels"], encoded["evaluation_labels"]):
        raise ValueError("Evaluation labels do not align")
    y = np.array([LABELS.index(label) for label in base["calibration_labels"]])
    results = []
    for weight in np.linspace(0.0, 1.0, 21):
        cal = (weight * base["calibration_oof_probabilities"]
               + (1 - weight) * encoded["calibration_oof_probabilities"])
        score = float(f1_score(y, cal.argmax(axis=1), average="macro"))
        results.append((score, float(weight), cal))
        print(f"base_weight={weight:.2f}: OOF F1={score:.6f}", flush=True)
    score, weight, cal = max(results, key=lambda item: item[0])
    eval_prob = (weight * base["evaluation_probabilities"]
                 + (1 - weight) * encoded["evaluation_probabilities"])
    calibrated = optimize_class_bias(
        cal, base["calibration_labels"], eval_prob, base["evaluation_labels"]
    )
    report = {
        "method": "OOF-selected convex blend of six-model base and CatBoost stacks",
        "selected_base_weight": weight,
        "calibration_oof_macro_f1": score,
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_105_encoded_stack_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_105.npz",
        calibration_oof_probabilities=cal,
        evaluation_probabilities=eval_prob,
        calibration_labels=base["calibration_labels"],
        evaluation_labels=base["evaluation_labels"], selected_base_weight=weight,
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
