"""Blend the two strongest interaction meta-stacks using OOF predictions."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Select a meta-stack averaging weight on calibration OOF rows."""
    normal = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    smooth = np.load(ROOT / "reports" / "stacking_holdout_097.npz", allow_pickle=True)
    if not np.array_equal(normal["calibration_labels"], smooth["calibration_labels"]):
        raise ValueError("Calibration rows do not align")
    y = np.array([LABELS.index(label) for label in normal["calibration_labels"]])
    best = None
    candidates = []
    for weight in np.linspace(0, 1, 21):
        cal = (1 - weight) * normal["calibration_oof_probabilities"] + weight * smooth[
            "calibration_oof_probabilities"
        ]
        eval_prob = (1 - weight) * normal["evaluation_probabilities"] + weight * smooth[
            "evaluation_probabilities"
        ]
        score = float(f1_score(y, cal.argmax(1), average="macro"))
        candidates.append((score, weight, cal, eval_prob))
        if best is None or score > best[0]:
            best = (score, weight, cal, eval_prob)
    calibrated = optimize_class_bias(
        best[2], normal["calibration_labels"], best[3], normal["evaluation_labels"]
    )
    report = {"method": "OOF probability blend of normal- and low-smoothing stacks",
              "selected_smoothing_stack_weight": float(best[1]),
              "calibration_oof_macro_f1_raw": best[0], "calibration": calibrated,
              "candidate_raw_oof_scores": [
                  {"weight": float(weight), "score": float(score)}
                  for score, weight, _, _ in candidates
              ]}
    (ROOT / "reports" / "validation_exp_099_meta_stack_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"selected weight={best[1]:.2f}; holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
