"""Optimize ordinal decision cutpoints for the strongest stack's OOF predictions."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    data = np.load(ROOT / "reports" / "stacking_holdout_086.npz")
    y = data["calibration_labels"].astype(str)
    y_eval = data["evaluation_labels"].astype(str)
    y_code = np.array([int(label.rsplit(" ", 1)[1]) - 1 for label in y])
    y_eval_code = np.array([int(label.rsplit(" ", 1)[1]) - 1 for label in y_eval])
    values = data["calibration_oof_probabilities"] @ np.arange(1.0, 6.0)
    eval_values = data["evaluation_probabilities"] @ np.arange(1.0, 6.0)
    starts = [np.array([1.5, 2.5, 3.5, 4.5]),
              np.array([1.3, 2.4, 3.6, 4.7]),
              np.array([1.7, 2.6, 3.4, 4.3])]
    best_score, best_thresholds = -1.0, None
    for start_id, thresholds in enumerate(starts):
        for step, radius, cycles in ((0.05, 0.4, 3), (0.01, 0.08, 3),
                                     (0.002, 0.02, 2)):
            for _ in range(cycles):
                for index in range(4):
                    candidates = np.arange(thresholds[index] - radius,
                                           thresholds[index] + radius + step / 2, step)
                    local_score, local_threshold = -1.0, thresholds[index]
                    for candidate in candidates:
                        trial = thresholds.copy()
                        trial[index] = candidate
                        if np.any(np.diff(trial) <= 0):
                            continue
                        score = f1_score(y_code, np.digitize(values, trial), average="macro")
                        if score > local_score:
                            local_score, local_threshold = score, candidate
                    thresholds[index] = local_threshold
        score = float(f1_score(y_code, np.digitize(values, thresholds), average="macro"))
        print(f"start={start_id}: OOF F1={score:.6f}; cuts={thresholds}", flush=True)
        if score > best_score:
            best_score, best_thresholds = score, thresholds.copy()
    eval_score = float(f1_score(y_eval_code, np.digitize(eval_values, best_thresholds),
                                average="macro"))
    report = {
        "method": "Three-start ordinal expected-grade cutpoint search on best-stack OOF",
        "calibration_oof_macro_f1": best_score,
        "evaluation_macro_f1": eval_score,
        "thresholds": best_thresholds.tolist(),
    }
    (ROOT / "reports" / "validation_exp_109_best_ordinal_decoder.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={eval_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
