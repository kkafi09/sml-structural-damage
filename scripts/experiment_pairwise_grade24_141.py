"""Test an OvO Grade 2 versus Grade 4 specialist as a margin correction."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select per-class offsets using the calibration labels only."""
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


def replace_pair_margin(logits: np.ndarray, pair_margin: np.ndarray,
                        weight: float) -> np.ndarray:
    """Blend the baseline Grade 2/4 logit gap with the pairwise model margin."""
    corrected = logits.copy()
    midpoint = (logits[:, 1] + logits[:, 3]) / 2
    margin = (1 - weight) * (logits[:, 3] - logits[:, 1]) + weight * pair_margin
    corrected[:, 1] = midpoint - margin / 2
    corrected[:, 3] = midpoint + margin / 2
    return corrected


def main() -> None:
    """Train the Grade 2/4 binary model and calibration-select its blend."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    mask = train_y.isin(["Grade 2", "Grade 4"])
    pair_x, pair_y = train_x.loc[mask], train_y.loc[mask].map({"Grade 2": 0, "Grade 4": 1})
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=min(140_000, len(pair_y)),
                                     random_state=7141)
    positions, _ = next(sampler.split(np.zeros(len(pair_y)), pair_y))
    fit_x, fit_y = pair_x.iloc[positions].copy(), pair_y.iloc[positions]
    fit_sample_size = len(fit_y)
    del train_x, train_y, pair_x, pair_y
    gc.collect()
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.04,
        num_leaves=31, min_child_samples=100, colsample_bytree=0.85,
        subsample=0.85, subsample_freq=1, reg_lambda=5.0, max_bin=127,
        class_weight={0: 1.1, 1: 1.0}, n_jobs=2, random_state=7141, verbosity=-1,
    )
    model.fit(fit_x, fit_y, categorical_feature=CATEGORICAL)
    cal_p, eval_p = model.predict_proba(cal_x), model.predict_proba(eval_x)
    cal_margin, eval_margin = np.log(np.clip(cal_p[:, 1], 1e-8, 1) /
                                     np.clip(cal_p[:, 0], 1e-8, 1)), np.log(
                                         np.clip(eval_p[:, 1], 1e-8, 1) /
                                         np.clip(eval_p[:, 0], 1e-8, 1))
    del model, fit_x, fit_y, cal_x, eval_x
    gc.collect()
    base = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    cal_logits = np.log(np.clip(base["calibration_oof_probabilities"], 1e-8, 1))
    eval_logits = np.log(np.clip(base["evaluation_probabilities"], 1e-8, 1))
    results = []
    for weight in (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0):
        cal_corrected = replace_pair_margin(cal_logits, cal_margin, weight)
        bias, cal_score = tune_bias(cal_corrected, base["calibration_labels"])
        eval_corrected = replace_pair_margin(eval_logits, eval_margin, weight) + bias
        score = float(f1_score(base["evaluation_labels"],
                               np.asarray(LABELS)[eval_corrected.argmax(1)], average="macro"))
        results.append({"pairwise_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {"method": "LightGBM OvO Grade 2 vs Grade 4 logit-margin correction",
              "fit_sample_size": fit_sample_size, "selected_by_calibration": selected,
              "all_blends": results, "baseline_evaluation_macro_f1": 0.5985512292352484}
    path = ROOT / "reports" / "validation_exp_141_pairwise_grade24.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), calibration_margin=cal_margin,
                        evaluation_margin=eval_margin)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
