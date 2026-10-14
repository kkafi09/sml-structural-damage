"""Tune a class-balanced nonlinear XGBoost meta-learner on OOF probabilities."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


def load_stack_data():
    """Load aligned OOF and evaluation probabilities from the best six base models."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
        "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    calibration_labels = arrays[0]["calibration_labels"]
    evaluation_labels = arrays[0]["evaluation_labels"]
    calibration_inputs = [item["calibration_probabilities"] for item in arrays[:3]]
    evaluation_inputs = [item["evaluation_probabilities"] for item in arrays[:3]]
    for item in arrays[3:]:
        calibration_inputs.append(
            0.6 * item["calibration_multiclass"] + 0.4 * item["calibration_ova"]
        )
        evaluation_inputs.append(
            0.6 * item["evaluation_multiclass"] + 0.4 * item["evaluation_ova"]
        )
    for item in arrays[1:]:
        if not np.array_equal(item["calibration_labels"], calibration_labels):
            raise ValueError("Calibration labels are not aligned")
        if not np.array_equal(item["evaluation_labels"], evaluation_labels):
            raise ValueError("Evaluation labels are not aligned")
    return (
        stack_features(*calibration_inputs),
        stack_features(*evaluation_inputs),
        np.array([LABELS.index(label) for label in calibration_labels]),
        evaluation_labels,
    )


def build_model(parameters: dict, seed: int) -> XGBClassifier:
    """Create the fixed nonlinear meta-model for one CV fold."""
    return XGBClassifier(
        objective="multi:softprob", num_class=len(LABELS), eval_metric="mlogloss",
        tree_method="hist", n_jobs=3, random_state=seed,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=8.0,
        min_child_weight=20, **parameters,
    )


def main() -> None:
    """Select meta-model settings on OOF folds and report untouched evaluation."""
    calibration_x, evaluation_x, y, evaluation_labels = load_stack_data()
    evaluation_y = np.array([LABELS.index(label) for label in evaluation_labels])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    configurations = [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200},
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 400},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 400},
    ]
    results = []
    best = None
    for index, parameters in enumerate(configurations):
        oof = np.zeros((len(y), len(LABELS)), dtype=float)
        for fold, (fit_indices, held_indices) in enumerate(folds.split(calibration_x, y), 1):
            model = build_model(parameters, 2026 + fold)
            weights = compute_sample_weight("balanced", y[fit_indices])
            model.fit(calibration_x[fit_indices], y[fit_indices], sample_weight=weights)
            oof[held_indices] = model.predict_proba(calibration_x[held_indices])
            print(f"config={index + 1} fold={fold} complete", flush=True)
        score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        results.append({"parameters": parameters, "calibration_oof_macro_f1": score})
        print(f"config={index + 1} OOF Macro F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"parameters": parameters, "score": score, "oof": oof}

    final = build_model(best["parameters"], 2026)
    final.fit(calibration_x, y, sample_weight=compute_sample_weight("balanced", y))
    evaluation_probabilities = final.predict_proba(evaluation_x)
    calibrated = optimize_class_bias(
        best["oof"], np.asarray([LABELS[index] for index in y]),
        evaluation_probabilities, evaluation_labels,
    )
    report = {
        "method": "class-balanced XGBoost nonlinear meta-model over six best base OOF probability vectors",
        "cv_folds": 3,
        "configurations": results,
        "selected_parameters": best["parameters"],
        "calibration_oof_macro_f1": best["score"],
        "calibration": calibrated,
        "baseline_holdout_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_119_xgb_meta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_119.npz",
        calibration_oof_probabilities=best["oof"],
        evaluation_probabilities=evaluation_probabilities,
        calibration_labels=np.asarray([LABELS[index] for index in y]),
        evaluation_labels=evaluation_labels,
    )
    print(f"Evaluation Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
