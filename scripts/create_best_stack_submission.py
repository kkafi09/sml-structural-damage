"""Refit the strongest validated stack recipe and create a Kaggle submission."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.feature_engineering import engineer_features
from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import (
    CATEGORICAL, LABELS, PROFILE_COLUMNS, add_neighbor_statistics,
    add_region_neighbor_statistics, add_target_statistics,
    cross_fit_neighbor_statistics, cross_fit_region_neighbor_statistics,
    cross_fit_target_statistics, load_data,
)


def build_test_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same row level features used by the validated models."""
    x = engineer_features(frame.drop(columns=["building_id"]))
    geo = pd.read_csv(ROOT / "data" / "ward_level_pred_intensity.csv")
    geo = geo.groupby("ward_id", as_index=False).mean(numeric_only=True)
    x = x.merge(geo, on="ward_id", how="left", validate="many_to_one")
    x["geo_features_missing"] = x["latitude"].isna().astype("int8")
    x["intensity_band"] = pd.cut(
        x["pred_intensity"], bins=[-np.inf, 7.0, 7.25, 7.5, 7.75, np.inf],
        labels=False,
    ).astype("string")
    x["intensity_x_age"] = x["pred_intensity"] * np.log1p(
        x["age_building"].fillna(0).clip(lower=0)
    )
    x["intensity_x_floors"] = x["pred_intensity"] * x["count_floors_pre_eq"]
    x["intensity_x_height_per_floor"] = x["pred_intensity"] * x["height_per_floor"]
    x["intensity_x_area_per_floor"] = x["pred_intensity"] * x["area_per_floor"]
    ids = frame["building_id"].astype("string")
    x["building_id_order"] = ids.astype("int64")
    x["id_prefix6"] = ids.str.slice(0, 6)
    x["id_prefix8"] = ids.str.slice(0, 8)
    x["id_prefix9"] = ids.str.slice(0, 9)
    x["id_prefix10"] = ids.str.slice(0, 10)
    profile = x["foundation_type"].astype("string").fillna("__MISSING__")
    for col in ["roof_type", "ground_floor_type", "other_floor_type", "superstructure"]:
        profile = profile.str.cat(x[col].astype("string").fillna("__MISSING__"), sep="|")
    x["structure_profile"] = profile
    x["id_profile"] = x["id_prefix8"].astype("string").str.cat(profile, sep="|")
    x["ward_profile"] = x["ward_id"].astype("string").str.cat(profile, sep="|")
    x["age_band"] = pd.cut(
        x["age_building"].fillna(-1), bins=[-2, 0, 5, 10, 20, 40, 60, 100, 1000],
        labels=False,
    ).astype("string")
    x["floor_band"] = x["count_floors_pre_eq"].clip(upper=6).astype("string")
    x["area_band"] = pd.cut(
        x["plinth_area_sq_ft"].clip(lower=0),
        bins=[0, 100, 200, 300, 400, 500, 700, 1000, 1500, 3000, 10000],
        labels=False,
    ).astype("string")
    x["height_band"] = pd.cut(
        x["height_ft_pre_eq"].clip(lower=0),
        bins=[0, 6, 10, 12, 15, 18, 21, 24, 30, 40, 60, 100], labels=False,
    ).astype("string")
    x["building_profile"] = (
        profile.str.cat(x["floor_band"], sep="|").str.cat(x["age_band"], sep="|")
        .str.cat(x["area_band"], sep="|").str.cat(x["height_band"], sep="|")
    )
    for col in CATEGORICAL:
        x[col] = x[col].fillna("__MISSING__").astype("category")
    return x


def make_matrices(test_raw: pd.DataFrame):
    """Build leakage-safe training and test matrices for the submission refit."""
    features, labels = load_data()
    fit_x, _, fit_y, _ = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    fit_te, test_te = cross_fit_target_statistics(fit_x, fit_y, test_raw, seed=2026)
    fit_n, test_n = cross_fit_neighbor_statistics(fit_x, fit_y, test_raw, seed=2027)
    fit_r, test_r = cross_fit_region_neighbor_statistics(fit_x, fit_y, test_raw, seed=2028)
    ncols = [c for c in fit_n if c.startswith("id_neighbor_")]
    rcols = [c for c in fit_r if c.startswith("region_neighbor_")]
    train = fit_te.drop(columns=PROFILE_COLUMNS).join(fit_n[ncols]).join(fit_r[rcols])
    test = test_te.drop(columns=PROFILE_COLUMNS).join(
        add_neighbor_statistics(fit_x, fit_y, test_raw)
    ).join(add_region_neighbor_statistics(fit_x, fit_y, test_raw)[rcols])
    train = train.drop(columns=["building_id_order"])
    test = test.drop(columns=["building_id_order"])
    return train, fit_y, test


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    parser = argparse.ArgumentParser(description="Legacy refit; not an identical reconstruction of experiment 087.")
    parser.add_argument("--external-data-approved", action="store_true",
                        help="confirm that competition rules permit the external intensity dataset")
    arguments = parser.parse_args()
    if not arguments.external_data_approved:
        parser.error("Legacy refit blocked: external-data permission is unverified. Use python -m scripts.workflow for the official-only recipe.")
    required = [ROOT / "data" / "ward_level_pred_intensity.csv"] + [
        ROOT / "reports" / name for name in (
            "stacking_holdout_086.npz", "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
            "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz", "lightgbm_holdout_041.npz",
            "lightgbm_holdout_085.npz", "validation_exp_086_location_interaction_stack.json",
            "validation_exp_087_interaction_bias.json",
        )
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        parser.error("Legacy refit blocked; missing prerequisites: " + ", ".join(missing))
    test_frame = pd.read_csv(ROOT / "data" / "test.csv")
    raw_test = build_test_features(test_frame)
    train_x, train_y, test_x = make_matrices(raw_test)
    for col in CATEGORICAL:
        test_x[col] = pd.Categorical(test_x[col], categories=train_x[col].cat.categories)
    y = np.array([LABELS.index(v) for v in train_y])

    # Keep the 80% fitting split used by the validated score; calibration rows
    # train the meta model, while the untouched 10% remains outside both.
    xgb_x = train_x.drop(
        columns=["id_prefix6", "id_prefix8", "id_prefix9", "id_prefix10"],
        errors="ignore",
    )
    xgb_test = test_x.reindex(columns=xgb_x.columns)
    xgb_specs = [
        (300, 0.10, 7, 30, 0.85, 0.90, 8.0, 2026),
        (500, 0.08, 6, 20, 0.90, 0.90, 6.0, 2026),
        (400, 0.08, 7, 20, 0.90, 0.90, 6.0, 42),
    ]
    predictions = []
    for index, (trees, lr, depth, child, subsample, columns, reg, seed) in enumerate(xgb_specs):
        model = XGBClassifier(
            objective="multi:softprob", num_class=5, eval_metric="mlogloss",
            n_estimators=trees, learning_rate=lr, max_depth=depth,
            min_child_weight=child, subsample=subsample, colsample_bytree=columns,
            reg_lambda=reg, tree_method="hist", enable_categorical=True,
            max_cat_to_onehot=4, max_cat_threshold=16, n_jobs=4,
            random_state=seed,
        )
        model.fit(xgb_x, y, verbose=False)
        predictions.append(model.predict_proba(xgb_test))
        print(f"XGBoost {index + 1}/3 selesai", flush=True)

    lgb_specs = [
        (59, 0.20, "multiclass", 2026),
        (81, 0.20, "multiclassova", 2027),
        (74, 0.55, "multiclass", 2026),
        (81, 0.55, "multiclassova", 2027),
        (74, 0.80, "multiclass", 2026),
        (99, 0.80, "multiclassova", 2027),
    ]
    lgb_preds = []
    for index, (trees, _, objective, seed) in enumerate(lgb_specs):
        weight = {"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                  "Grade 4": 0.95, "Grade 5": 0.80}
        model = lgb.LGBMClassifier(
            objective=objective, metric="None", n_estimators=trees,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, class_weight=weight if objective == "multiclass" else None,
            n_jobs=4, random_state=seed, verbosity=-1,
        )
        model.fit(train_x, train_y, categorical_feature=CATEGORICAL)
        lgb_preds.append(model.predict_proba(test_x))
        print(f"LightGBM {index + 1}/6 selesai", flush=True)

    # Refit the learned meta classifier from the saved OOF calibration stack.
    saved = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    cal_labels = saved["calibration_labels"]
    y_cal = np.array([LABELS.index(v) for v in cal_labels])
    base = [
        np.load(ROOT / "reports" / name, allow_pickle=True)
        for name in ("xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
                     "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
                     "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz")
    ]
    lgb_cal = [0.6 * base[i]["calibration_multiclass"] +
               0.4 * base[i]["calibration_ova"] for i in range(3, 6)]
    cal_x = stack_features(*(d["calibration_probabilities"] for d in base[:3]), *lgb_cal)
    c_value = float(json.loads((ROOT / "reports" / "validation_exp_086_location_interaction_stack.json").read_text())["selected_C"])
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=500))
    meta.fit(cal_x, y_cal)
    # LGB test predictions are paired as the same fixed 60/40 blend.
    test_lgb = [0.6 * lgb_preds[i] + 0.4 * lgb_preds[i + 1] for i in (0, 2, 4)]
    test_meta_x = stack_features(*predictions, *test_lgb)
    calibrated = json.loads((ROOT / "reports" / "validation_exp_087_interaction_bias.json").read_text())
    bias = np.array([calibrated["bias_by_grade"][label] for label in LABELS])
    test_codes = (np.log(np.clip(meta.predict_proba(test_meta_x), 1e-8, 1.0)) + bias).argmax(axis=1)
    test_grades = np.array([int(LABELS[i].split()[-1]) for i in test_codes])

    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    if not sample["building_id"].equals(test_frame["building_id"]):
        raise ValueError("test.csv dan sample_submission.csv berbeda urutan building_id")
    sample["damage_grade"] = test_grades
    out = ROOT / "submissions" / "candidate_legacy087_refit.csv"
    sample.to_csv(out, index=False)
    report = {
        "model": "best-stack submission refit: three XGBoost and three LightGBM base learners",
        "validation_macro_f1": calibrated["evaluation_macro_f1_tuned"],
        "validation_score_source": "reports/validation_exp_087_interaction_bias.json",
        "submission_rows": len(sample),
        "grade_counts": sample["damage_grade"].value_counts().sort_index().to_dict(),
        "csv": str(out.relative_to(ROOT)),
        "note": "0.598551 is the historical local holdout score, not a Kaggle test score or a score measured on this CSV. Base learners were refit using the validated 80% split, and the meta-model and class offsets use calibration OOF artifacts. The original fitted LightGBM model files were not saved, so their refit uses the recorded iteration counts and shared feature preparation; this can differ from the exact experiment-087 base models.",
    }
    (ROOT / "reports" / "submission_legacy087_refit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
