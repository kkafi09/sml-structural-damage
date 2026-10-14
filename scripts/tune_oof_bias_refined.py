"""Refine class-offset decisions on OOF probabilities with broader restarts."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    data = np.load(ROOT / "reports" / "stacking_holdout_086.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    z = np.log(np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0))
    z_eval = np.log(np.clip(data["evaluation_probabilities"], 1e-8, 1.0))
    rng = np.random.default_rng(20261002)
    center = np.array([-0.176, 0.174, 0.149, 0.049, -0.196])
    starts = [center.copy()]
    starts.extend(center + rng.normal(0, 0.3, len(LABELS)) for _ in range(24))
    best_score, best_bias = -1.0, None
    for start_id, bias in enumerate(starts):
        bias -= bias.mean()
        for step, radius, cycles in ((0.02, 0.30, 2), (0.005, 0.05, 2),
                                     (0.001, 0.01, 1)):
            for _ in range(cycles):
                for cls in range(len(LABELS)):
                    candidates = np.arange(bias[cls] - radius,
                                           bias[cls] + radius + step / 2, step)
                    best_local_score, best_local_bias = -1.0, bias[cls]
                    for candidate in candidates:
                        trial = bias.copy()
                        trial[cls] = candidate
                        trial -= trial.mean()
                        score = f1_score(y, (z + trial).argmax(axis=1), average="macro")
                        if score > best_local_score:
                            best_local_score, best_local_bias = score, candidate
                    bias[cls] = best_local_bias
                    bias -= bias.mean()
        score = float(f1_score(y, (z + bias).argmax(axis=1), average="macro"))
        print(f"start={start_id}: OOF F1={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_bias = score, bias.copy()

    eval_score = float(f1_score(y_eval, (z_eval + best_bias).argmax(axis=1),
                                average="macro"))
    report = {
        "method": "25-start refined OOF class-offset coordinate search",
        "calibration_oof_macro_f1": best_score,
        "evaluation_macro_f1": eval_score,
        "bias_by_grade": dict(zip(LABELS, map(float, best_bias))),
    }
    (ROOT / "reports" / "validation_exp_107_refined_oof_bias.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={eval_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
