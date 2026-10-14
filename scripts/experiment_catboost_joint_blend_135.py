"""Test whether complementary CatBoost objectives improve the saved stack."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimize class offsets using calibration labels only."""
    y = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(5)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for cls in range(5):
                best = f1_score(y, (logits + bias).argmax(1), average="macro")
                chosen = bias[cls]
                for value in np.arange(bias[cls] - radius, bias[cls] + radius + step / 2, step):
                    candidate = bias.copy()
                    candidate[cls] = value
                    score = f1_score(y, (logits + candidate).argmax(1), average="macro")
                    if score > best:
                        best, chosen = score, value
                bias[cls] = chosen
    return bias, float(f1_score(y, (logits + bias).argmax(1), average="macro"))


def main() -> None:
    """Grid-search a joint two-model blend on calibration and score holdout."""
    base = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    rf = np.load(ROOT / "reports" / "validation_exp_132_lgb_rf.npz")
    multi = np.load(ROOT / "reports" / "validation_exp_133_catboost_multiclass.npz")
    bcal, beval = (np.log(np.clip(base[key], 1e-8, 1)) for key in
                   ("calibration_oof_probabilities", "evaluation_probabilities"))
    ocal, oe = (np.log(np.clip(rf[key], 1e-8, 1)) for key in
                ("calibration_probabilities", "evaluation_probabilities"))
    mcal, me = (np.log(np.clip(multi[key], 1e-8, 1)) for key in
                ("calibration_probabilities", "evaluation_probabilities"))
    results = []
    for w_rf, w_multi in itertools.product((0.0, 0.05, 0.10, 0.15, 0.20), repeat=2):
        if w_rf + w_multi > 0.30:
            continue
        cal_logits = (1 - w_rf - w_multi) * bcal + w_rf * ocal + w_multi * mcal
        bias, cal_score = tune_bias(cal_logits, base["calibration_labels"])
        eval_logits = ((1 - w_rf - w_multi) * beval + w_rf * oe + w_multi * me + bias)
        eval_y = np.array([LABELS.index(label) for label in base["evaluation_labels"]])
        score = float(f1_score(eval_y, eval_logits.argmax(1), average="macro"))
        results.append({"rf_weight": w_rf, "multiclass_weight": w_multi,
                        "calibration_macro_f1": cal_score, "evaluation_macro_f1": score,
                        "bias": bias.tolist()})
    selected = max(results, key=lambda r: r["calibration_macro_f1"])
    report = {"method": "joint LightGBM RF + CatBoost multiclass + baseline log-probability blend",
              "selected_by_calibration": selected, "all_blends": results,
              "baseline_evaluation_macro_f1": 0.5985512292352484}
    (ROOT / "reports" / "validation_exp_135_rf_catboost_joint_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
