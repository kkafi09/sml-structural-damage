"""Refit LightGBM objectives on 90% data while keeping the fixed eval split untouched."""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import (
    CATEGORICAL, LABELS, PROFILE_COLUMNS, add_neighbor_statistics,
    add_region_neighbor_statistics, add_target_statistics, load_data,
)


def main():
    """Run the experiment and save its report and prediction artifacts."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)

    features, labels = load_data()
    raw_train, raw_holdout, y_train, y_holdout = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    raw_cal, raw_eval, y_cal, y_eval = train_test_split(
        raw_holdout, y_holdout, test_size=0.5, random_state=2027, stratify=y_holdout
    )
    if not np.array_equal(y_cal.to_numpy(), cal_y.to_numpy()) or not np.array_equal(
        y_eval.to_numpy(), eval_y.to_numpy()
    ):
        raise ValueError("Cached split and raw feature split do not align")

    dev_raw = pd.concat([raw_train, raw_cal], axis=0)
    dev_y = pd.concat([y_train, y_cal], axis=0)
    train90 = pd.concat([train_x, cal_x], axis=0)

    evaluation_te = add_target_statistics(dev_raw, dev_y, raw_eval)
    evaluation_neighbors = add_neighbor_statistics(dev_raw, dev_y, raw_eval)
    evaluation_region = add_region_neighbor_statistics(dev_raw, dev_y, raw_eval)
    eval90 = evaluation_te.join(evaluation_neighbors).join(evaluation_region)
    eval90 = eval90.drop(columns=PROFILE_COLUMNS + ["building_id_order"], errors="ignore")
    train90 = train90.drop(columns=["building_id_order"], errors="ignore")
    if not train90.columns.equals(eval90.columns):
        raise ValueError("90% training and evaluation feature columns do not align")

    weights = {"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
               "Grade 4": 0.95, "Grade 5": 0.80}
    models = [
        lgb.LGBMClassifier(objective="multiclass", metric="None", class_weight=weights,
                           n_estimators=120, learning_rate=0.06, num_leaves=63,
                           min_child_samples=100, colsample_bytree=0.9,
                           subsample=0.85, subsample_freq=1, reg_lambda=3.0,
                           n_jobs=4, random_state=2026, verbosity=-1),
        lgb.LGBMClassifier(objective="multiclassova", metric="None", n_estimators=120,
                           learning_rate=0.06, num_leaves=63, min_child_samples=100,
                           colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
                           reg_lambda=3.0, n_jobs=4, random_state=2027, verbosity=-1),
    ]
    eval_probabilities, trained_iterations = [], []
    for model in models:
        model.fit(train90, dev_y, categorical_feature=CATEGORICAL)
        eval_probabilities.append(model.predict_proba(eval90))
        trained_iterations.append(int(model.n_estimators_))
        print(f"{model.objective_}: fitted {model.n_estimators_} iterations", flush=True)
    np.savez_compressed(
        ROOT / "reports" / "lightgbm_90pct_116.npz",
        evaluation_multiclass=eval_probabilities[0],
        evaluation_ova=eval_probabilities[1],
        evaluation_labels=y_eval.to_numpy(),
        trained_iterations=np.asarray(trained_iterations),
    )
    (ROOT / "reports" / "validation_exp_116_refit_90pct.json").write_text(
        json.dumps({"method": "LightGBM refit on train+calibration (90%)",
                    "trained_iterations": trained_iterations,
                    "eval_rows": int(len(y_eval))}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
