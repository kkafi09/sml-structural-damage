"""Search class-weighted LightGBM variants using calibration-only stacking."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def tune_calibration_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimize class offsets using calibration labels only."""
    codes = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.10, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                best_score = f1_score(codes, (logits + bias).argmax(axis=1), average="macro")
                best_value = bias[grade]
                for value in np.arange(bias[grade] - radius,
                                       bias[grade] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[grade] = value
                    score = f1_score(codes, (logits + trial).argmax(axis=1), average="macro")
                    if score > best_score:
                        best_score, best_value = score, value
                bias[grade] = best_value
    score = float(f1_score(codes, (logits + bias).argmax(axis=1), average="macro"))
    return bias, score


def main() -> None:
    """Train candidate class weights and select their blend without eval labels."""
    cache = ROOT / "reports" / "catboost_encoded_data_101.pkl"
    with cache.open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    if not np.array_equal(np.asarray(cal_y), stack["calibration_labels"]):
        raise ValueError("Calibration rows are not aligned with the cached stack")
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    sample = StratifiedShuffleSplit(n_splits=1, train_size=180_000, random_state=2026)
    positions, _ = next(sample.split(np.zeros(len(train_y)), train_y))
    fit_x = train_x.iloc[positions]
    fit_y = train_y.iloc[positions]
    del train_x, train_y
    gc.collect()
    for column in CATEGORICAL:
        if column in fit_x and isinstance(fit_x[column].dtype, pd.CategoricalDtype):
            categories = fit_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)

    weight_sets = {
        "baseline": {"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                     "Grade 4": 0.95, "Grade 5": 0.80},
        "grade23_up": {"Grade 1": 0.90, "Grade 2": 1.55, "Grade 3": 1.45,
                       "Grade 4": 0.95, "Grade 5": 0.75},
        "grade34_up": {"Grade 1": 0.90, "Grade 2": 1.25, "Grade 3": 1.45,
                       "Grade 4": 1.15, "Grade 5": 0.75},
    }
    candidates = []
    evaluation_labels = np.asarray(stack["evaluation_labels"])
    for index, (name, weights) in enumerate(weight_sets.items()):
        model = lgb.LGBMClassifier(
            objective="multiclass", metric="None", n_estimators=74,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, max_bin=127, class_weight=weights,
            n_jobs=2, random_state=3121 + index, verbosity=-1,
        )
        model.fit(fit_x, fit_y, categorical_feature=CATEGORICAL)
        cal_prob = model.predict_proba(cal_x)
        eval_prob = model.predict_proba(eval_x)
        del model
        gc.collect()
        best = None
        for blend_weight in (0.10, 0.20, 0.30, 0.40):
            cal_logits = (1.0 - blend_weight) * base_cal + blend_weight * np.log(
                np.clip(cal_prob, 1e-8, 1.0)
            )
            bias, cal_score = tune_calibration_bias(cal_logits, np.asarray(cal_y))
            if best is None or cal_score > best["calibration_macro_f1"]:
                best = {"blend_weight": blend_weight, "bias": bias, "calibration_macro_f1": cal_score}
        eval_logits = (1.0 - best["blend_weight"]) * base_eval + best["blend_weight"] * np.log(
            np.clip(eval_prob, 1e-8, 1.0)
        ) + best["bias"]
        eval_pred = np.asarray(LABELS)[eval_logits.argmax(axis=1)]
        eval_score = float(f1_score(evaluation_labels, eval_pred, average="macro"))
        row = {
            "name": name, "weights": weights,
            "selected_blend_weight": best["blend_weight"],
            "calibration_macro_f1": best["calibration_macro_f1"],
            "evaluation_macro_f1": eval_score,
            "bias": best["bias"].tolist(),
        }
        candidates.append(row)
        print(json.dumps(row), flush=True)
    selected = max(candidates, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "LightGBM class-weight candidates; blend and offsets selected on calibration only",
        "fit_sample_size": len(fit_y),
        "selected_by_calibration": selected["name"],
        "baseline_holdout_macro_f1": 0.5985512292352484,
        "candidates": candidates,
    }
    path = ROOT / "reports" / "validation_exp_121_class_weights.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected by calibration: {selected['name']}; holdout={selected['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
