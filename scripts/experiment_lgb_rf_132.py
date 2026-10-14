"""Evaluate LightGBM random-forest boosting as a diverse stack member."""

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


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Tune class offsets with calibration labels only."""
    y = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                best = f1_score(y, (logits + bias).argmax(1), average="macro")
                selected = bias[grade]
                for value in np.arange(bias[grade] - radius,
                                       bias[grade] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[grade] = value
                    score = f1_score(y, (logits + trial).argmax(1), average="macro")
                    if score > best:
                        best, selected = score, value
                bias[grade] = selected
    return bias, float(f1_score(y, (logits + bias).argmax(1), average="macro"))


def main() -> None:
    """Fit LightGBM RF mode on a stratified sample and calibration-select blending."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    split = StratifiedShuffleSplit(n_splits=1, train_size=150_000, random_state=2026)
    positions, _ = next(split.split(np.zeros(len(train_y)), train_y))
    fit_x, fit_y = train_x.iloc[positions], train_y.iloc[positions]
    del train_x, train_y
    gc.collect()
    for column in CATEGORICAL:
        if column in fit_x and isinstance(fit_x[column].dtype, pd.CategoricalDtype):
            categories = fit_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    model = lgb.LGBMClassifier(
        boosting_type="rf", objective="multiclass", metric="None",
        n_estimators=240, learning_rate=1.0, num_leaves=63,
        min_child_samples=100, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, reg_lambda=3.0,
        max_bin=127, class_weight={"Grade 1": 0.95, "Grade 2": 1.30,
                                  "Grade 3": 1.25, "Grade 4": 0.95,
                                  "Grade 5": 0.80},
        n_jobs=2, random_state=7132, verbosity=-1,
    )
    model.fit(fit_x, fit_y, categorical_feature=CATEGORICAL)
    cal_prob, eval_prob = model.predict_proba(cal_x), model.predict_proba(eval_x)
    del model, fit_x, fit_y, cal_x, eval_x
    gc.collect()
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    results = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        cal_logits = (1 - weight) * base_cal + weight * np.log(np.clip(cal_prob, 1e-8, 1.0))
        bias, cal_score = tune_bias(cal_logits, np.asarray(stack["calibration_labels"]))
        eval_logits = (1 - weight) * base_eval + weight * np.log(np.clip(eval_prob, 1e-8, 1.0)) + bias
        pred = np.asarray(LABELS)[eval_logits.argmax(1)]
        eval_score = float(f1_score(stack["evaluation_labels"], pred, average="macro"))
        results.append({"blend_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": eval_score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "LightGBM random-forest boosting mode; calibration-selected stack blend",
        "fit_sample_size": len(positions), "selected_by_calibration": selected,
        "blends": results, "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_132_lgb_rf.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), calibration_probabilities=cal_prob,
                        evaluation_probabilities=eval_prob)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
