"""Combine Grade 2/3 and Grade 3/4 classifiers as conditional log-odds corrections."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from scipy.special import expit, logit, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def correction(base, pair23, pair34, weight23, weight34):
    """Apply pairwise log-odds residuals to the corresponding adjacent grades."""
    eps = 1e-6
    p = np.clip(base, eps, 1.0)
    score = np.log(p)
    p23 = p[:, 1] / np.maximum(p[:, 1] + p[:, 2], eps)
    p34 = p[:, 2] / np.maximum(p[:, 2] + p[:, 3], eps)
    delta23 = logit(np.clip(pair23, eps, 1 - eps)) - logit(np.clip(p23, eps, 1 - eps))
    delta34 = logit(np.clip(pair34, eps, 1 - eps)) - logit(np.clip(p34, eps, 1 - eps))
    score[:, 1] += 0.5 * weight23 * delta23
    score[:, 2] -= 0.5 * weight23 * delta23
    score[:, 2] += 0.5 * weight34 * delta34
    score[:, 3] -= 0.5 * weight34 * delta34
    return softmax(score, axis=1)


def main() -> None:
    """Select both pairwise correction weights from OOF calibration Macro F1."""
    stack = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    pair23 = np.load(ROOT / "reports" / "pairwise_grade23_holdout_076.npz", allow_pickle=True)
    pair34 = np.load(ROOT / "reports" / "pairwise_grade34_holdout_078.npz", allow_pickle=True)
    y = np.array([LABELS.index(label) for label in stack["calibration_labels"]])
    eval_y = stack["evaluation_labels"]
    best = None
    for w23 in np.arange(0.0, 1.51, 0.1):
        for w34 in np.arange(0.0, 1.51, 0.1):
            cal = correction(stack["calibration_oof_probabilities"],
                             pair23["calibration_pair_probability"],
                             pair34["calibration_pair_probability"], w23, w34)
            score = float(f1_score(y, cal.argmax(1), average="macro"))
            if best is None or score > best["score"]:
                best = {"w23": w23, "w34": w34, "score": score, "cal": cal}
    eval_prob = correction(stack["evaluation_probabilities"],
                           pair23["evaluation_pair_probability"],
                           pair34["evaluation_pair_probability"], best["w23"], best["w34"])
    calibrated = optimize_class_bias(best["cal"], stack["calibration_labels"], eval_prob, eval_y)
    report = {"method": "Joint Grade 2/3 and 3/4 pairwise log-odds corrections",
              "selected_weight_grade23": float(best["w23"]),
              "selected_weight_grade34": float(best["w34"]),
              "calibration_oof_macro_f1_raw": best["score"], "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_080_pairwise_joint.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"weights=({best['w23']:.1f}, {best['w34']:.1f}); holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
