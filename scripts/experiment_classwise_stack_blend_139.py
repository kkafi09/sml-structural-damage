"""Optimize class-specific interpolation for complementary OOF stacks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]


def main() -> None:
    """Select per-class blend weights and offsets on calibration only."""
    base = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    alternate = np.load(ROOT / "reports" / "stacking_holdout_114.npz", allow_pickle=True)
    base_cal = np.log(np.clip(base["calibration_oof_probabilities"], 1e-8, 1))
    alt_cal = np.log(np.clip(alternate["calibration_oof_probabilities"], 1e-8, 1))
    base_eval = np.log(np.clip(base["evaluation_probabilities"], 1e-8, 1))
    alt_eval = np.log(np.clip(alternate["evaluation_probabilities"], 1e-8, 1))
    ycal = np.array([LABELS.index(label) for label in base["calibration_labels"]])
    yeval = np.array([LABELS.index(label) for label in base["evaluation_labels"]])
    alpha = np.zeros(5)
    bias = np.zeros(5)
    alpha_grid = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75)
    for _ in range(2):
        for cls in range(5):
            best = -1.0
            for value in alpha_grid:
                for offset in np.arange(bias[cls] - 0.2, bias[cls] + 0.2 + 0.025, 0.05):
                    trial_alpha, trial_bias = alpha.copy(), bias.copy()
                    trial_alpha[cls], trial_bias[cls] = value, offset
                    logits = base_cal + (alt_cal - base_cal) * trial_alpha + trial_bias
                    score = f1_score(ycal, logits.argmax(1), average="macro")
                    if score > best:
                        best, alpha[cls], bias[cls] = score, value, offset
    cal_logits = base_cal + (alt_cal - base_cal) * alpha + bias
    eval_logits = base_eval + (alt_eval - base_eval) * alpha + bias
    report = {
        "method": "class-specific logit interpolation between stack 086 and stack 114",
        "alpha_by_grade": dict(zip(LABELS, alpha.tolist())),
        "bias_by_grade": dict(zip(LABELS, bias.tolist())),
        "calibration_macro_f1": float(f1_score(ycal, cal_logits.argmax(1), average="macro")),
        "evaluation_macro_f1": float(f1_score(yeval, eval_logits.argmax(1), average="macro")),
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    (ROOT / "reports" / "validation_exp_139_classwise_stack_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
