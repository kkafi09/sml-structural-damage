"""Cross-validate class-bias calibration before applying it to evaluation rows."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS


def fit_bias(logits, y):
    """Fit class offsets by coordinate ascent on an in-sample calibration fold."""
    bias = np.zeros(len(LABELS))
    for step, radius in ((0.05, 0.5), (0.01, 0.1)):
        for _ in range(3):
            for cls in range(len(LABELS)):
                candidates = np.arange(bias[cls] - radius, bias[cls] + radius + step / 2, step)
                best_score, best_value = -1.0, bias[cls]
                for value in candidates:
                    trial = bias.copy()
                    trial[cls] = value
                    score = f1_score(y, (logits + trial).argmax(1), average="macro")
                    if score > best_score:
                        best_score, best_value = score, value
                bias[cls] = best_value
    return bias


def main() -> None:
    """Score fold-specific offsets, then fit a final offset on all OOF rows."""
    data = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    y = np.array([LABELS.index(label) for label in data["calibration_labels"]])
    y_eval = np.array([LABELS.index(label) for label in data["evaluation_labels"]])
    logits = np.log(np.clip(data["calibration_oof_probabilities"], 1e-8, 1.0))
    eval_logits = np.log(np.clip(data["evaluation_probabilities"], 1e-8, 1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=712)
    oof_pred = np.zeros(len(y), dtype=int)
    biases = []
    for fold, (fit_i, val_i) in enumerate(cv.split(logits, y)):
        bias = fit_bias(logits[fit_i], y[fit_i])
        biases.append(bias)
        oof_pred[val_i] = (logits[val_i] + bias).argmax(1)
        print(f"fold={fold}: bias={bias}; F1={f1_score(y[val_i], oof_pred[val_i], average='macro'):.6f}", flush=True)
    cv_score = float(f1_score(y, oof_pred, average="macro"))
    final_bias = fit_bias(logits, y)
    holdout_score = float(f1_score(y_eval, (eval_logits + final_bias).argmax(1), average="macro"))
    mean_bias = np.mean(biases, axis=0)
    mean_score = float(f1_score(y_eval, (eval_logits + mean_bias).argmax(1), average="macro"))
    report = {"method": "5-fold cross-fitted class-bias calibration",
              "calibration_oof_macro_f1": cv_score,
              "evaluation_macro_f1_full_bias": holdout_score,
              "evaluation_macro_f1_mean_fold_bias": mean_score,
              "full_bias": dict(zip(LABELS, map(float, final_bias))),
              "mean_fold_bias": dict(zip(LABELS, map(float, mean_bias)))}
    (ROOT / "reports" / "validation_exp_074_cv_bias.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"CV OOF={cv_score:.6f}; holdout full={holdout_score:.6f}; mean={mean_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
