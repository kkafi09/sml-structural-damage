"""Blend pairwise Grade 3/4 probabilities into the best five-class stack."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def replace_pair_conditional(probabilities, pair_probability, weight):
    """Replace part of the Grade 2/3 conditional distribution, preserving pair mass."""
    result = probabilities.copy()
    grade3, grade4 = LABELS.index("Grade 3"), LABELS.index("Grade 4")
    pair_mass = result[:, grade3] + result[:, grade4]
    base_conditional = result[:, grade3] / np.maximum(pair_mass, 1e-12)
    conditional = (1 - weight) * base_conditional + weight * pair_probability
    result[:, grade3] = pair_mass * conditional
    result[:, grade4] = pair_mass * (1 - conditional)
    return result


def main() -> None:
    """Select pairwise mixture strength from OOF calibration labels."""
    stack = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    pair = np.load(ROOT / "reports" / "pairwise_grade34_holdout_078.npz", allow_pickle=True)
    for key in ("calibration_labels", "evaluation_labels"):
        if not np.array_equal(stack[key], pair[key]):
            raise ValueError("Pairwise and stack predictions use different rows")
    cal_y = stack["calibration_labels"]
    eval_y = stack["evaluation_labels"]
    cal_codes = np.array([LABELS.index(label) for label in cal_y])
    results = []
    best = None
    for weight in np.linspace(0, 1, 21):
        cal_prob = replace_pair_conditional(
            stack["calibration_oof_probabilities"], pair["calibration_pair_probability"], weight
        )
        eval_prob = replace_pair_conditional(
            stack["evaluation_probabilities"], pair["evaluation_pair_probability"], weight
        )
        cal_score = float(f1_score(cal_codes, cal_prob.argmax(1), average="macro"))
        calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
        row = {"pairwise_weight": float(weight), "calibration_f1_raw": cal_score,
               "calibration_f1_after_bias": calibrated["calibration_macro_f1"],
               "evaluation_macro_f1": calibrated["evaluation_macro_f1"],
               "bias_by_grade": calibrated["bias_by_grade"]}
        results.append(row)
        if best is None or row["calibration_f1_after_bias"] > best["calibration_f1_after_bias"]:
            best = row
    report = {"method": "Pairwise Grade 3/4 conditional blend",
              "selected_weight": best["pairwise_weight"],
              "selected_evaluation_macro_f1": best["evaluation_macro_f1"],
              "weights": results}
    (ROOT / "reports" / "validation_exp_079_pairwise_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Selected pair weight={best['pairwise_weight']:.2f}; holdout={best['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
