"""Test distance-weighted building-ID neighbor features with cross-fitting."""

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL

BANDWIDTHS = (2.0, 8.0, 25.0)
MAX_OFFSET = 80


def weighted_neighbor_features(source_ids: np.ndarray, source_y: np.ndarray,
                               query_ids: np.ndarray) -> pd.DataFrame:
    """Return rank-distance weighted label distributions around each query ID."""
    source_order = np.argsort(source_ids)
    ids = source_ids[source_order]
    codes = np.array([LABELS.index(label) for label in source_y[source_order]])
    one_hot = np.eye(len(LABELS), dtype=np.float32)[codes]
    positions = np.searchsorted(ids, query_ids)
    output = {}
    for bandwidth in BANDWIDTHS:
        sums = np.zeros((len(query_ids), len(LABELS)), dtype=np.float32)
        total = np.zeros(len(query_ids), dtype=np.float32)
        for offset in range(MAX_OFFSET):
            weight = float(np.exp(-(offset + 0.5) / bandwidth))
            for indices in (positions - 1 - offset, positions + offset):
                valid = (indices >= 0) & (indices < len(ids))
                safe = np.clip(indices, 0, len(ids) - 1)
                sums += one_hot[safe] * (valid * weight)[:, None]
                total += valid * weight
        distributions = sums / np.maximum(total[:, None], 1e-8)
        prefix = f"id_weighted_bw{int(bandwidth)}"
        for class_index, label in enumerate(LABELS):
            output[f"{prefix}_{label.lower().replace(' ', '')}"] = distributions[:, class_index]
        output[f"{prefix}_mean_grade"] = distributions @ np.arange(1, 6)
    return pd.DataFrame(output)


def tune_bias(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Tune per-grade offsets using calibration labels only."""
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
    """Build cross-fitted features, fit a diverse LightGBM model, and score its blend."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    raw_ids = pd.read_csv(ROOT / "data" / "train.csv", usecols=["building_id"])["building_id"]
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=180_000, random_state=7137)
    positions, _ = next(sampler.split(np.zeros(len(train_y)), train_y))
    fit_x, fit_y = train_x.iloc[positions].copy(), train_y.iloc[positions].copy()
    fit_ids = raw_ids.iloc[fit_x.index].to_numpy(dtype=np.int64)
    cal_ids = raw_ids.iloc[cal_x.index].to_numpy(dtype=np.int64)
    eval_ids = raw_ids.iloc[eval_x.index].to_numpy(dtype=np.int64)
    del train_x, train_y
    gc.collect()

    oof = np.zeros((len(fit_x), len(BANDWIDTHS) * 6), dtype=np.float32)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7137)
    for fold, (source_pos, query_pos) in enumerate(folds.split(fit_x, fit_y), start=1):
        block = weighted_neighbor_features(fit_ids[source_pos], fit_y.iloc[source_pos].to_numpy(),
                                           fit_ids[query_pos])
        oof[query_pos] = block.to_numpy(dtype=np.float32)
        print(f"weighted neighbor fold {fold}/5 ready", flush=True)
        del block
        gc.collect()
    neighbor_columns = [f"id_weighted_bw{int(bw)}_{label.lower().replace(' ', '')}"
                        for bw in BANDWIDTHS for label in LABELS]
    neighbor_columns += [f"id_weighted_bw{int(bw)}_mean_grade" for bw in BANDWIDTHS]
    oof_frame = pd.DataFrame(oof, columns=neighbor_columns, index=fit_x.index)
    fit_x = pd.concat([fit_x, oof_frame], axis=1)
    cal_block = weighted_neighbor_features(fit_ids, fit_y.to_numpy(), cal_ids)
    eval_block = weighted_neighbor_features(fit_ids, fit_y.to_numpy(), eval_ids)
    cal_x = pd.concat([cal_x, cal_block.set_axis(cal_x.index)], axis=1)
    eval_x = pd.concat([eval_x, eval_block.set_axis(eval_x.index)], axis=1)
    del oof, oof_frame, cal_block, eval_block
    gc.collect()

    for column in CATEGORICAL:
        if column in fit_x and isinstance(fit_x[column].dtype, pd.CategoricalDtype):
            categories = fit_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=200, learning_rate=0.035,
        num_leaves=31, max_depth=-1, min_child_samples=100,
        colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
        reg_lambda=5.0, max_bin=127,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=2, random_state=7137, verbosity=-1,
    )
    model.fit(fit_x, fit_y, categorical_feature=CATEGORICAL)
    cal_prob, eval_prob = model.predict_proba(cal_x), model.predict_proba(eval_x)
    fit_sample_size = len(fit_y)
    del model, fit_x, fit_y, cal_x, eval_x
    gc.collect()
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    base_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    results = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        cal_logits = (1 - weight) * base_cal + weight * np.log(np.clip(cal_prob, 1e-8, 1))
        bias, cal_score = tune_bias(cal_logits, np.asarray(stack["calibration_labels"]))
        eval_logits = ((1 - weight) * base_eval
                       + weight * np.log(np.clip(eval_prob, 1e-8, 1)) + bias)
        score = float(f1_score(stack["evaluation_labels"],
                               np.asarray(LABELS)[eval_logits.argmax(1)], average="macro"))
        results.append({"blend_weight": weight, "calibration_macro_f1": cal_score,
                        "evaluation_macro_f1": score, "bias": bias.tolist()})
    selected = max(results, key=lambda item: item["calibration_macro_f1"])
    report = {"method": "LightGBM with 3 cross-fitted exponential rank-distance ID neighborhoods",
              "fit_sample_size": fit_sample_size, "bandwidths": BANDWIDTHS,
              "max_neighbor_offset": MAX_OFFSET, "selected_by_calibration": selected,
              "blends": results, "baseline_evaluation_macro_f1": 0.5985512292352484}
    path = ROOT / "reports" / "validation_exp_137_weighted_id_neighbors.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), calibration_probabilities=cal_prob,
                        evaluation_probabilities=eval_prob)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
