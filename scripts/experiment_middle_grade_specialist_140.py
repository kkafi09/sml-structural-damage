"""Test a dedicated LightGBM classifier for damage grades 2, 3, and 4."""

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
    """Select five class offsets from calibration labels only."""
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


def expand_middle_logits(probabilities: np.ndarray) -> np.ndarray:
    """Center conditional grade 2-4 logits for addition to five-grade scores."""
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    return logits - logits.mean(axis=1, keepdims=True)


def main() -> None:
    """Fit the middle-grade specialist and calibrate its correction on OOF rows."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    middle = train_y.isin(["Grade 2", "Grade 3", "Grade 4"])
    middle_x = train_x.loc[middle]
    middle_y = train_y.loc[middle]
    split = StratifiedShuffleSplit(n_splits=1, train_size=180_000, random_state=7140)
    positions, _ = next(split.split(np.zeros(len(middle_y)), middle_y))
    fit_x = middle_x.iloc[positions].copy()
    fit_y = middle_y.iloc[positions].map({"Grade 2": 0, "Grade 3": 1, "Grade 4": 2})
    fit_sample_size = len(fit_y)
    del train_x, train_y, middle_x, middle_y
    gc.collect()
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400,
        learning_rate=0.045, num_leaves=31, min_child_samples=120,
        colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
        reg_lambda=5.0, max_bin=127, class_weight={0: 1.2, 1: 1.15, 2: 0.9},
        n_jobs=2, random_state=7140, verbosity=-1,
    )
    model.fit(fit_x, fit_y, categorical_feature=CATEGORICAL)
    cal_middle = expand_middle_logits(model.predict_proba(cal_x))
    eval_middle = expand_middle_logits(model.predict_proba(eval_x))
    del model, fit_x, fit_y, cal_x, eval_x
    gc.collect()
    baseline = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    cal_base = np.log(np.clip(baseline["calibration_oof_probabilities"], 1e-8, 1.0))
    eval_base = np.log(np.clip(baseline["evaluation_probabilities"], 1e-8, 1.0))
    results = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        cal_logits, eval_logits = cal_base.copy(), eval_base.copy()
        cal_logits[:, 1:4] += weight * cal_middle
        eval_logits[:, 1:4] += weight * eval_middle
        bias, cal_score = tune_bias(cal_logits, np.asarray(baseline["calibration_labels"]))
        score = float(f1_score(
            baseline["evaluation_labels"], np.asarray(LABELS)[(eval_logits + bias).argmax(1)],
            average="macro"))
        results.append({"specialist_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "LightGBM specialist trained only on grades 2-4, centered correction to baseline stack",
        "fit_sample_size": fit_sample_size, "selected_by_calibration": selected,
        "all_blends": results, "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_140_middle_grade_specialist.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        calibration_middle_logits=cal_middle,
                        evaluation_middle_logits=eval_middle)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
