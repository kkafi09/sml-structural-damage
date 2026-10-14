"""Test cross-fitted CatBoost stacking on the saved baseline probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose per-class logit offsets from cross-fitted calibration predictions."""
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
    """Cross-fit a nonlinear meta-model, then evaluate once on the held-out rows."""
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    cal_p = np.clip(stack["calibration_oof_probabilities"], 1e-7, 1)
    eval_p = np.clip(stack["evaluation_probabilities"], 1e-7, 1)
    cal_x, eval_x = np.log(cal_p), np.log(eval_p)
    cal_y = np.array([LABELS.index(label) for label in stack["calibration_labels"]])
    eval_y = np.array([LABELS.index(label) for label in stack["evaluation_labels"]])
    oof = np.zeros((len(cal_y), 5))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7134)
    for fold, (fit_i, held_i) in enumerate(splitter.split(cal_x, cal_y)):
        model = CatBoostClassifier(
            loss_function="MultiClass", iterations=350, depth=3,
            learning_rate=0.04, l2_leaf_reg=20, random_strength=0.5,
            thread_count=2, random_seed=7134 + fold, verbose=False,
        )
        model.fit(cal_x[fit_i], cal_y[fit_i])
        oof[held_i] = model.predict_proba(cal_x[held_i])
    bias, cal_score = tune_bias(np.log(np.clip(oof, 1e-7, 1)), stack["calibration_labels"])
    full = CatBoostClassifier(
        loss_function="MultiClass", iterations=350, depth=3,
        learning_rate=0.04, l2_leaf_reg=20, random_strength=0.5,
        thread_count=2, random_seed=7139, verbose=False,
    )
    full.fit(cal_x, cal_y)
    eval_logits = np.log(np.clip(full.predict_proba(eval_x), 1e-7, 1)) + bias
    score = float(f1_score(eval_y, eval_logits.argmax(1), average="macro"))
    report = {
        "method": "5-fold cross-fitted CatBoost depth-3 meta-model on baseline log-probabilities",
        "calibration_oof_macro_f1": cal_score,
        "evaluation_macro_f1": score,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
        "bias": bias.tolist(),
    }
    (ROOT / "reports" / "validation_exp_134_catboost_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
