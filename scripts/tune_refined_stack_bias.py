"""Search class offsets for the location-interaction stack's OOF probabilities."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def main() -> None:
    """Tune offsets using calibration OOF rows, then score the fixed evaluation set."""
    data = np.load(ROOT / "reports" / "stacking_holdout_091.npz")
    y_cal = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    cal_logits = np.log(np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0))
    eval_logits = np.log(np.clip(data["evaluation_probabilities"], 1e-8, 1.0))
    rng = np.random.default_rng(20261001)
    starts = [np.array([-0.35, 0.0, 0.025, -0.125, -0.30])]
    starts.extend(starts[0] + rng.uniform(-0.4, 0.4, len(LABELS)) for _ in range(11))
    best = (-1.0, None)
    for i, initial in enumerate(starts):
        bias = initial - initial.mean()
        for step, radius, cycles in ((0.05, 0.4, 2), (0.01, 0.1, 2)):
            for _ in range(cycles):
                for grade in range(len(LABELS)):
                    candidates = np.arange(bias[grade] - radius, bias[grade] + radius + step / 2, step)
                    scores = []
                    for candidate in candidates:
                        trial = bias.copy()
                        trial[grade] = candidate
                        trial -= trial.mean()
                        scores.append(f1_score(y_cal, (cal_logits + trial).argmax(1), average="macro"))
                    bias[grade] = candidates[int(np.argmax(scores))]
                    bias -= bias.mean()
        score = float(f1_score(y_cal, (cal_logits + bias).argmax(1), average="macro"))
        print(f"start={i}: OOF F1={score:.6f}", flush=True)
        if score > best[0]:
            best = (score, bias.copy())
    base = float(f1_score(y_eval, eval_logits.argmax(1), average="macro"))
    tuned = float(f1_score(y_eval, (eval_logits + best[1]).argmax(1), average="macro"))
    report = {
        "method": "12-start class-offset search on location-interaction stack OOF probabilities",
        "calibration_oof_macro_f1": best[0],
        "evaluation_macro_f1_uncalibrated": base,
        "evaluation_macro_f1_tuned": tuned,
        "bias_by_grade": dict(zip(LABELS, map(float, best[1]))),
    }
    (ROOT / "reports" / "validation_exp_092_refined_bias.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout base={base:.6f}, tuned={tuned:.6f}", flush=True)


if __name__ == "__main__":
    main()
