"""Evaluate an XGBoost DART base learner in the OOF probability stack."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS


def optimize_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select class offsets using calibration labels only."""
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
    """Fit DART on a stratified sample, then calibration-select its stack weight."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    drop = ["id_prefix6", "id_prefix8", "id_prefix9", "id_prefix10"]
    train_x = train_x.drop(columns=drop, errors="ignore")
    cal_x = cal_x.drop(columns=drop, errors="ignore")
    eval_x = eval_x.drop(columns=drop, errors="ignore")
    split = StratifiedShuffleSplit(n_splits=1, train_size=100_000, random_state=2026)
    positions, _ = next(split.split(np.zeros(len(train_y)), train_y))
    fit_x = train_x.iloc[positions].copy()
    fit_y = np.array([LABELS.index(label) for label in train_y.iloc[positions]])
    del train_x, train_y
    gc.collect()
    categories = list(fit_x.select_dtypes(include="category").columns)
    for frame in (cal_x, eval_x):
        for column in categories:
            frame[column] = frame[column].cat.set_categories(fit_x[column].cat.categories)
    model = XGBClassifier(
        booster="dart", objective="multi:softprob", num_class=5,
        eval_metric="mlogloss", n_estimators=100, learning_rate=0.05,
        max_depth=7, min_child_weight=20, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=6.0, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=4, max_cat_threshold=16,
        rate_drop=0.1, skip_drop=0.5, sample_type="uniform", normalize_type="tree",
        n_jobs=2, random_state=8128,
    )
    model.fit(fit_x, fit_y, verbose=False)
    cal_prob, eval_prob = model.predict_proba(cal_x), model.predict_proba(eval_x)
    del model, fit_x, fit_y, cal_x, eval_x
    gc.collect()
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    results = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        cal_logits = (1.0 - weight) * base_cal + weight * np.log(np.clip(cal_prob, 1e-8, 1.0))
        bias, cal_score = optimize_bias(cal_logits, stack["calibration_labels"])
        eval_logits = (1.0 - weight) * base_eval + weight * np.log(np.clip(eval_prob, 1e-8, 1.0)) + bias
        prediction = np.asarray(LABELS)[eval_logits.argmax(1)]
        eval_score = float(f1_score(stack["evaluation_labels"], prediction, average="macro"))
        results.append({"blend_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": eval_score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "XGBoost DART with native categorical splits, calibration-selected stack blend",
        "fit_sample_size": len(positions), "selected_by_calibration": selected,
        "blends": results, "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_128_xgb_dart.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), calibration_probabilities=cal_prob,
                        evaluation_probabilities=eval_prob)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
