"""Add cross-fitted ID-prefix by structure target encodings to the cached stack."""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import (
    CATEGORICAL, LABELS, macro_f1_metric, optimize_class_bias,
)

SPECS = [
    (("id_prefix8", "foundation_type"), 80),
    (("id_prefix8", "roof_type"), 60),
    (("id_prefix8", "ground_floor_type"), 80),
    (("id_prefix8", "position"), 60),
    (("id_prefix8", "plan_configuration"), 60),
    (("id_prefix8", "land_surface_condition"), 60),
]


def target_interactions(source_x, source_y, query_x):
    """Build domain-driven target encodings for ID and structural feature interactions."""
    classes = sorted(source_y.unique())
    prior = source_y.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    output = {}
    y_codes = pd.Categorical(source_y, categories=classes).codes
    for columns, smoothing in SPECS:
        source_key = source_x[columns[0]].astype("string").fillna("__MISSING__") + "|" + \
            source_x[columns[1]].astype("string").fillna("__MISSING__")
        query_key = query_x[columns[0]].astype("string").fillna("__MISSING__") + "|" + \
            query_x[columns[1]].astype("string").fillna("__MISSING__")
        table = pd.DataFrame({"key": source_key.to_numpy(), "grade": y_codes})
        counts = table.groupby("key", sort=False, observed=True).size()
        sums = table.groupby(["key", "grade"], sort=False, observed=True).size().unstack(
            fill_value=0
        ).reindex(
            columns=np.arange(len(classes)), fill_value=0
        )
        rates = sums.add(prior * smoothing, axis=1).div(counts + smoothing, axis=0)
        mapped = rates.reindex(query_key.to_numpy()).fillna(
            {index: prior[index] for index in range(len(classes))}
        )
        values = mapped.to_numpy(dtype=float)
        prefix = "te_extra_" + "_".join(columns)
        for index, grade in enumerate(classes):
            output[f"{prefix}_{grade.lower().replace(' ', '')}"] = values[:, index]
        output[f"{prefix}_mean_grade"] = values @ np.arange(1, len(classes) + 1)
        output[f"{prefix}_log_count"] = np.log1p(
            counts.reindex(query_key.to_numpy()).fillna(0).to_numpy(dtype=float)
        )
    return pd.DataFrame(output, index=query_x.index)


def crossfit_interactions(train_x, train_y, cal_x, eval_x):
    """Generate out-of-fold interaction encodings for training rows."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2029)
    fit_features = pd.DataFrame(index=train_x.index)
    for fold_id, (fit_idx, held_idx) in enumerate(
        folds.split(np.zeros(len(train_y)), train_y), start=1
    ):
        fit_rows = train_x.iloc[fit_idx]
        held_rows = train_x.iloc[held_idx]
        features = target_interactions(fit_rows, train_y.iloc[fit_idx], held_rows)
        print(f"Completed target interaction fold {fold_id}/5", flush=True)
        if fit_features.empty:
            fit_features = pd.DataFrame(index=train_x.index, columns=features.columns)
        fit_features.iloc[held_idx] = features.to_numpy()
    cal_features = target_interactions(train_x, train_y, cal_x)
    eval_features = target_interactions(train_x, train_y, eval_x)
    return fit_features.astype(float), cal_features, eval_features


def main():
    """Run the experiment and save its report and prediction artifacts."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    extra_train, extra_cal, extra_eval = crossfit_interactions(
        train_x, train_y, cal_x, eval_x
    )
    train_x = train_x.join(extra_train)
    cal_x = cal_x.join(extra_cal)
    eval_x = eval_x.join(extra_eval)
    weights = {"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
               "Grade 4": 0.95, "Grade 5": 0.80}
    models = [
        lgb.LGBMClassifier(objective="multiclass", metric="None", class_weight=weights,
                           n_estimators=1000, learning_rate=0.06, num_leaves=63,
                           min_child_samples=100, colsample_bytree=0.9,
                           subsample=0.85, subsample_freq=1, reg_lambda=3,
                           n_jobs=4, random_state=2029, verbosity=-1),
        lgb.LGBMClassifier(objective="multiclassova", metric="None", n_estimators=1000,
                           learning_rate=0.06, num_leaves=63, min_child_samples=100,
                           colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
                           reg_lambda=3, n_jobs=4, random_state=2030, verbosity=-1),
    ]
    cal_predictions, eval_predictions, iterations = [], [], []
    for model in models:
        model.fit(train_x, train_y, eval_set=[(cal_x, cal_y)],
                  eval_metric=macro_f1_metric,
                  callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                             lgb.log_evaluation(100)],
                  categorical_feature=CATEGORICAL)
        cal_predictions.append(model.predict_proba(cal_x))
        eval_predictions.append(model.predict_proba(eval_x))
        iterations.append(int(model.best_iteration_))
        print(f"{model.objective_}: best_iteration={model.best_iteration_}", flush=True)
    cal_prob = 0.6 * cal_predictions[0] + 0.4 * cal_predictions[1]
    eval_prob = 0.6 * eval_predictions[0] + 0.4 * eval_predictions[1]
    calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
    report = {"method": "Cross-fitted ID-prefix by structure target encodings",
              "specifications": SPECS, "best_iterations": iterations,
              "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_110_id_interactions.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "lightgbm_id_interactions_110.npz",
                        calibration_multiclass=cal_predictions[0],
                        evaluation_multiclass=eval_predictions[0],
                        calibration_ova=cal_predictions[1], evaluation_ova=eval_predictions[1],
                        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy())
    print(f"ID interaction holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
