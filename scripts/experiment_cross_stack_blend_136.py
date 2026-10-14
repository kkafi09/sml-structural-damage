"""Cross-validate blends of saved OOF stacking candidates from the same split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]
CANDIDATES = ["089", "097", "102", "104", "105", "106", "111", "113", "114", "115"]


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select per-class offsets on calibration rows only."""
    y = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(5)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for cls in range(5):
                best = f1_score(y, (logits + bias).argmax(1), average="macro")
                chosen = bias[cls]
                for value in np.arange(bias[cls] - radius, bias[cls] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[cls] = value
                    score = f1_score(y, (logits + trial).argmax(1), average="macro")
                    if score > best:
                        best, chosen = score, value
                bias[cls] = chosen
    return bias, float(f1_score(y, (logits + bias).argmax(1), average="macro"))


def main() -> None:
    """Try each independently produced stack as a calibration-selected add-on."""
    base = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    ycal, yeval = base["calibration_labels"], base["evaluation_labels"]
    base_cal = np.log(np.clip(base["calibration_oof_probabilities"], 1e-8, 1))
    base_eval = np.log(np.clip(base["evaluation_probabilities"], 1e-8, 1))
    results = []
    for number in CANDIDATES:
        candidate = np.load(ROOT / "reports" / f"stacking_holdout_{number}.npz", allow_pickle=True)
        cal = np.log(np.clip(candidate["calibration_oof_probabilities"], 1e-8, 1))
        ev = np.log(np.clip(candidate["evaluation_probabilities"], 1e-8, 1))
        for weight in (0.05, 0.10, 0.20, 0.30):
            cal_logits = (1 - weight) * base_cal + weight * cal
            bias, cal_score = tune_bias(cal_logits, ycal)
            eval_logits = (1 - weight) * base_eval + weight * ev + bias
            score = float(f1_score(yeval, np.array(LABELS)[eval_logits.argmax(1)], average="macro"))
            results.append({"candidate_stack": number, "weight": weight,
                            "calibration_macro_f1": cal_score,
                            "evaluation_macro_f1": score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "baseline stack 086 plus one saved OOF stack at a time; calibration-selected weight and bias",
        "candidate_stacks": CANDIDATES,
        "selected_by_calibration": selected,
        "all_results": results,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    (ROOT / "reports" / "validation_exp_136_cross_stack_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "top_by_calibration": sorted(
        results, key=lambda x: x["calibration_macro_f1"], reverse=True)[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
