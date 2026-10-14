"""Search class decision offsets from stacked out-of-fold calibration scores."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    data = np.load(ROOT / "reports" / "stacking_holdout_040.npz")
    cal_y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    eval_y = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    cal_logits = np.log(np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0))
    eval_logits = np.log(np.clip(data["evaluation_probabilities"], 1e-8, 1.0))
    rng = np.random.default_rng(712)
    starts = [np.zeros(len(LABELS))]
    starts.extend(rng.uniform(-0.8, 0.8, len(LABELS)) for _ in range(12))
    best_bias = None
    best_score = -1.0
    for start_index, initial in enumerate(starts):
        biases = initial - initial.mean()
        for step in (0.1, 0.025, 0.01):
            for _ in range(3):
                for grade in range(len(LABELS)):
                    candidates = np.arange(
                        biases[grade] - 0.2, biases[grade] + 0.2 + step / 2,
                        step,
                    )
                    scores = []
                    for candidate in candidates:
                        trial = biases.copy()
                        trial[grade] = candidate
                        trial -= trial.mean()
                        scores.append(f1_score(
                            cal_y, (cal_logits + trial).argmax(axis=1),
                            average="macro",
                        ))
                    biases[grade] = candidates[int(np.argmax(scores))]
                    biases -= biases.mean()
        score = f1_score(cal_y, (cal_logits + biases).argmax(axis=1), average="macro")
        if score > best_score:
            best_score = float(score)
            best_bias = biases.copy()
        print(f"start={start_index}: OOF F1={score:.6f}", flush=True)
    evaluation_predictions = (eval_logits + best_bias).argmax(axis=1)
    evaluation_f1 = f1_score(eval_y, evaluation_predictions, average="macro")
    report = {
        "method": "Multi-start coordinate search on stacked OOF calibration rows",
        "calibration_oof_macro_f1": best_score,
        "evaluation_macro_f1": float(evaluation_f1),
        "bias_by_grade": dict(zip(LABELS, map(float, best_bias))),
    }
    path = ROOT / "reports" / "validation_exp_043_stack_bias.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best evaluation Macro F1={evaluation_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
