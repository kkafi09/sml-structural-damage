"""Test CatBoost MultiClass as a diverse base learner."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select class offsets on the calibration predictions only."""
    codes = np.array([LABELS.index(label) for label in labels])
    bias = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.10, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                best = f1_score(codes, (logits + bias).argmax(1), average="macro")
                selected = bias[grade]
                for value in np.arange(bias[grade] - radius,
                                       bias[grade] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[grade] = value
                    score = f1_score(codes, (logits + trial).argmax(1), average="macro")
                    if score > best:
                        best, selected = score, value
                bias[grade] = selected
    score = float(f1_score(codes, (logits + bias).argmax(1), average="macro"))
    return bias, score


def main() -> None:
    """Fit CatBoost OVA on native structural/location features and stack it."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    candidates = [
        "count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
        "area_per_floor", "height_per_floor", "age_building_log1p", "district_id",
        "vdcmun_id", "ward_id", "land_surface_condition", "foundation_type", "roof_type",
        "ground_floor_type", "other_floor_type", "position", "plan_configuration",
        "superstructure", "pred_intensity", "intensity_band", "intensity_x_age",
        "intensity_x_floors", "intensity_x_height_per_floor", "intensity_x_area_per_floor",
    ]
    columns = [column for column in candidates if column in train_x.columns]
    categories = [column for column in CATEGORICAL if column in columns]
    sampler = train_x.sample(n=180_000, random_state=2026).index
    fit_x = train_x.loc[sampler, columns].copy()
    fit_y = train_y.loc[sampler].astype(str)
    cal_features = cal_x[columns].copy()
    eval_features = eval_x[columns].copy()
    del train_x, train_y, cal_x, eval_x
    gc.collect()
    for frame in (fit_x, cal_features, eval_features):
        for column in categories:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    cat_indices = [fit_x.columns.get_loc(column) for column in categories]
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=100, depth=8,
        learning_rate=0.045, l2_leaf_reg=12, random_strength=0.8,
        class_weights=[0.95, 1.30, 1.25, 0.95, 0.80],
        max_ctr_complexity=2, cat_features=cat_indices,
        thread_count=2, random_seed=2026, verbose=50,
    )
    model.fit(fit_x, fit_y, verbose=50)
    cal_prob = model.predict_proba(cal_features)
    eval_prob = model.predict_proba(eval_features)
    model_labels = list(model.classes_)
    order = [model_labels.index(label) for label in LABELS]
    cal_prob, eval_prob = cal_prob[:, order], eval_prob[:, order]
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    results = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        cal_logits = (1.0 - weight) * base_cal + weight * np.log(np.clip(cal_prob, 1e-8, 1.0))
        bias, cal_score = tune_bias(cal_logits, np.asarray(stack["calibration_labels"]))
        eval_logits = (1.0 - weight) * base_eval + weight * np.log(np.clip(eval_prob, 1e-8, 1.0)) + bias
        prediction = np.asarray(LABELS)[eval_logits.argmax(axis=1)]
        score = float(f1_score(stack["evaluation_labels"], prediction, average="macro"))
        results.append({"blend_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": score, "bias": bias.tolist()})
        print(f"blend={weight:.2f}: calibration={cal_score:.6f}, evaluation={score:.6f}", flush=True)
    best = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {
        "method": "CatBoost MultiClass over native feature subset, calibration-selected blend",
        "fit_sample_size": len(fit_x), "features": columns, "categorical_features": categories,
        "best_iteration": int(model.get_best_iteration() or 100),
        "selected_by_calibration": best,
        "all_blends": results,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_133_catboost_multiclass.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        calibration_probabilities=cal_prob,
                        evaluation_probabilities=eval_prob)
    print(f"Selected blend evaluation={best['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
