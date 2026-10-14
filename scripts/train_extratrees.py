"""Train an ExtraTrees baseline on raw engineered tabular features."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, load_data, optimize_class_bias


def main() -> None:
    """Train, calibrate, and save predictions from ExtraTrees."""
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_x, eval_x, cal_y, eval_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027, stratify=holdout_y
    )
    drop_cols = [
        "building_id_order", "id_prefix6", "id_prefix8", "id_prefix9", "id_prefix10",
        "structure_profile", "id_profile", "ward_profile", "age_band", "floor_band",
        "area_band", "height_band", "building_profile",
    ]
    train_x, cal_x, eval_x = [frame.drop(columns=drop_cols, errors="ignore")
                              for frame in (train_x, cal_x, eval_x)]
    categorical = list(train_x.select_dtypes(include=["category", "object", "string"]).columns)
    numeric = [column for column in train_x.columns if column not in categorical]
    for frame in (train_x, cal_x, eval_x):
        for column in categorical:
            frame[column] = frame[column].astype("string").fillna("__MISSING__")
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore", min_frequency=20, sparse_output=True
            ), categorical),
        ],
        remainder="drop",
    )
    model = ExtraTreesClassifier(
        n_estimators=80, max_depth=16, max_features=0.6, min_samples_leaf=30,
        bootstrap=True, max_samples=0.4,
        class_weight="balanced", n_jobs=4, random_state=2026,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(train_x, train_y)
    cal_prob = pipeline.predict_proba(cal_x)
    eval_prob = pipeline.predict_proba(eval_x)
    class_order = list(pipeline.named_steps["model"].classes_)
    reorder = [class_order.index(label) for label in LABELS]
    cal_prob, eval_prob = cal_prob[:, reorder], eval_prob[:, reorder]
    calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
    report = {
        "method": "ExtraTrees on raw numerical and one-hot categorical features",
        "parameters": {"n_estimators": 80, "max_depth": 16, "max_features": 0.6,
                       "min_samples_leaf": 30, "bootstrap": True, "max_samples": 0.4,
                       "class_weight": "balanced"},
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(label) for label in eval_y], eval_prob.argmax(1), average="macro"
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_083_extratrees.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "extratrees_holdout_083.npz",
        calibration_probabilities=cal_prob, evaluation_probabilities=eval_prob,
        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy(),
    )
    print(f"ExtraTrees holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
