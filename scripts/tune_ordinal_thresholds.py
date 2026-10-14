"""Optimize ordered grade cutpoints over the best stack's OOF probabilities."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def decode(values, thresholds):
    """Convert an expected grade value into an ordered class index."""
    return np.digitize(values, thresholds)


def main() -> None:
    """Tune grade cutpoints on OOF calibration rows and score evaluation rows."""
    data = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    values = data["calibration_oof_probabilities"] @ np.arange(1, 6)
    eval_values = data["evaluation_probabilities"] @ np.arange(1, 6)
    thresholds = np.array([1.5, 2.5, 3.5, 4.5])
    for step, radius, cycles in ((0.1, 0.8, 3), (0.025, 0.2, 5), (0.01, 0.05, 3)):
        for _ in range(cycles):
            for index in range(len(thresholds)):
                low = thresholds[index] - radius
                high = thresholds[index] + radius
                candidates = np.arange(low, high + step / 2, step)
                best_score, best_cut = -1.0, thresholds[index]
                for candidate in candidates:
                    trial = thresholds.copy()
                    trial[index] = candidate
                    if np.any(np.diff(trial) <= 0):
                        continue
                    score = f1_score(y, decode(values, trial), average="macro")
                    if score > best_score:
                        best_score, best_cut = score, candidate
                thresholds[index] = best_cut
        print(f"step={step}: OOF F1={best_score:.6f}; thresholds={thresholds}", flush=True)
    base = float(f1_score(y_eval, decode(eval_values, [1.5, 2.5, 3.5, 4.5]), average="macro"))
    score = float(f1_score(y_eval, decode(eval_values, thresholds), average="macro"))
    report = {"method": "OOF-optimized ordinal thresholds on stack expected grade",
              "calibration_oof_macro_f1": float(f1_score(y, decode(values, thresholds), average="macro")),
              "evaluation_macro_f1_default_thresholds": base,
              "evaluation_macro_f1": score, "thresholds": thresholds.tolist()}
    (ROOT / "reports" / "validation_exp_073_ordinal_thresholds.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
