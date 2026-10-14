"""Tune per-class logit scales and offsets on best-stack OOF predictions."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def main() -> None:
    """Coordinate-search class-specific confidence scaling on calibration OOF rows."""
    data = np.load(ROOT / "reports" / "stacking_holdout_086.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    z = np.log(np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0))
    z_eval = np.log(np.clip(data["evaluation_probabilities"], 1e-8, 1.0))
    scale = np.ones(len(LABELS))
    bias = np.array([-0.176, 0.174, 0.149, 0.049, -0.196])
    for cycle in range(3):
        for cls in range(len(LABELS)):
            best_score = -1.0
            best_pair = (scale[cls], bias[cls])
            for slope in np.arange(0.5, 1.501, 0.05):
                for offset in np.arange(bias[cls] - 0.3, bias[cls] + 0.301, 0.025):
                    trial_scale, trial_bias = scale.copy(), bias.copy()
                    trial_scale[cls], trial_bias[cls] = slope, offset
                    score = f1_score(y, (z * trial_scale + trial_bias).argmax(1), average="macro")
                    if score > best_score:
                        best_score, best_pair = score, (slope, offset)
            scale[cls], bias[cls] = best_pair
        print(f"cycle={cycle}: OOF F1={best_score:.6f}; scales={scale}; bias={bias}", flush=True)
    eval_score = float(f1_score(y_eval, (z_eval * scale + bias).argmax(1), average="macro"))
    report = {
        "method": "Coordinate search over class-specific logit scales and offsets",
        "calibration_oof_macro_f1": float(f1_score(y, (z * scale + bias).argmax(1), average="macro")),
        "evaluation_macro_f1": eval_score,
        "scale_by_grade": dict(zip(LABELS, map(float, scale))),
        "bias_by_grade": dict(zip(LABELS, map(float, bias))),
    }
    (ROOT / "reports" / "validation_exp_100_interaction_slopes.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={eval_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
