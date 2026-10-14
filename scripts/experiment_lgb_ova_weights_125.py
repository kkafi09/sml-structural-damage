"""Compare weighted LightGBM One-vs-All learners in the OOF stack."""

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


def optimize_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose class offsets against calibration predictions only."""
    y = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                best = f1_score(y, (logits + bias).argmax(1), average="macro")
                value_best = bias[grade]
                for value in np.arange(bias[grade] - radius,
                                       bias[grade] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[grade] = value
                    score = f1_score(y, (logits + trial).argmax(1), average="macro")
                    if score > best:
                        best, value_best = score, value
                bias[grade] = value_best
    return bias, float(f1_score(y, (logits + bias).argmax(1), average="macro"))


def main() -> None:
    """Fit two class-weighted OVA models and choose the stack on calibration."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    if not np.array_equal(np.asarray(cal_y), stack["calibration_labels"]):
        raise ValueError("Calibration rows are not aligned")
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=150_000, random_state=2026)
    positions, _ = next(splitter.split(np.zeros(len(train_y)), train_y))
    fit_x, fit_y = train_x.iloc[positions], train_y.iloc[positions]
    del train_x, train_y
    gc.collect()
    for column in CATEGORICAL:
        if column in fit_x and isinstance(fit_x[column].dtype, pd.CategoricalDtype):
            categories = fit_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    counts = fit_y.value_counts().reindex(LABELS).to_numpy(dtype=float)
    priors = counts / counts.sum()
    variants = {
        "sqrt_balanced": 1.0 / np.sqrt(priors),
        "balanced": 1.0 / priors,
    }
    candidates = []
    for index, (name, factors) in enumerate(variants.items()):
        sample_weights = fit_y.map(dict(zip(LABELS, factors))).to_numpy(dtype=float).copy()
        sample_weights /= sample_weights.mean()
        model = lgb.LGBMClassifier(
            objective="multiclassova", metric="None", n_estimators=99,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, max_bin=127, n_jobs=2,
            random_state=5125 + index, verbosity=-1,
        )
        model.fit(fit_x, fit_y, sample_weight=sample_weights,
                  categorical_feature=CATEGORICAL)
        cal_prob = model.predict_proba(cal_x)
        eval_prob = model.predict_proba(eval_x)
        del model
        gc.collect()
        best = None
        for blend_weight in (0.05, 0.10, 0.15, 0.20, 0.30):
            cal_logits = (1.0 - blend_weight) * base_cal + blend_weight * np.log(
                np.clip(cal_prob, 1e-8, 1.0)
            )
            bias, cal_score = optimize_bias(cal_logits, np.asarray(cal_y))
            if best is None or cal_score > best["calibration_macro_f1"]:
                eval_logits = (1.0 - blend_weight) * base_eval + blend_weight * np.log(
                    np.clip(eval_prob, 1e-8, 1.0)
                ) + bias
                pred = np.asarray(LABELS)[eval_logits.argmax(axis=1)]
                eval_score = float(f1_score(stack["evaluation_labels"], pred, average="macro"))
                best = {"blend_weight": blend_weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": eval_score, "bias": bias.tolist()}
        row = {"name": name, "class_weight_factors": dict(zip(LABELS, factors.tolist())), **best}
        candidates.append(row)
        print(json.dumps(row), flush=True)
    selected = max(candidates, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "LightGBM multiclassova with per-row inverse-frequency sample weights; calibration-selected",
        "fit_sample_size": len(fit_y), "selected_by_calibration": selected,
        "candidates": candidates, "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_125_lgb_ova_weights.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected evaluation Macro F1={selected['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
