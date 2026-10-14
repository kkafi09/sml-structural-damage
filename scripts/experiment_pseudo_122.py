"""Cross-fit pseudo-label students and select their blend out of fold."""

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


def tune_bias(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find class offsets using only the supplied calibration rows."""
    bias = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                best = f1_score(y, (logits + bias).argmax(axis=1), average="macro")
                selected = bias[grade]
                for value in np.arange(bias[grade] - radius,
                                       bias[grade] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[grade] = value
                    score = f1_score(y, (logits + trial).argmax(axis=1), average="macro")
                    if score > best:
                        best, selected = score, value
                bias[grade] = selected
    return bias


def fit_student(features: pd.DataFrame, labels: np.ndarray,
                weights: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    """Fit a single fixed LightGBM student with weighted pseudo-label rows."""
    model = lgb.LGBMClassifier(
        objective="multiclass", metric="None", n_estimators=64,
        learning_rate=0.06, num_leaves=63, min_child_samples=100,
        colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
        reg_lambda=3.0, max_bin=127,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=2, random_state=seed, verbosity=-1,
    )
    model.fit(features, labels, sample_weight=weights, categorical_feature=CATEGORICAL)
    return model


def main() -> None:
    """Generate OOF student predictions, tune a blend on OOF, score evaluation."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    teacher_oof = stack["calibration_oof_probabilities"]
    if not np.array_equal(np.asarray(cal_y), stack["calibration_labels"]):
        raise ValueError("Cached calibration labels are not aligned")
    selected = teacher_oof.max(axis=1) >= 0.75
    pseudo_y = np.asarray(LABELS)[teacher_oof.argmax(axis=1)]
    sample = StratifiedShuffleSplit(n_splits=1, train_size=150_000, random_state=2026)
    positions, _ = next(sample.split(np.zeros(len(train_y)), train_y))
    source_x, source_y = train_x.iloc[positions], train_y.iloc[positions]
    del train_x, train_y
    gc.collect()
    for column in CATEGORICAL:
        if column in source_x and isinstance(source_x[column].dtype, pd.CategoricalDtype):
            categories = source_x[column].cat.categories
            cal_x[column] = cal_x[column].cat.set_categories(categories)
            eval_x[column] = eval_x[column].cat.set_categories(categories)

    labels_cal = np.array([LABELS.index(label) for label in cal_y])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    student_cal = np.zeros((len(cal_y), len(LABELS)), dtype=float)
    student_eval_sum = np.zeros((len(eval_y), len(LABELS)), dtype=float)
    for fold, (fit_idx, held_idx) in enumerate(folds.split(cal_x, labels_cal), 1):
        pseudo_fit = selected[fit_idx]
        pseudo_x = cal_x.iloc[fit_idx].loc[pseudo_fit]
        x = pd.concat([source_x, pseudo_x], ignore_index=True)
        y = np.concatenate([source_y.to_numpy(), pseudo_y[fit_idx][pseudo_fit]])
        weights = np.concatenate([
            np.ones(len(source_y), dtype=float),
            np.full(int(pseudo_fit.sum()), 0.35, dtype=float),
        ])
        model = fit_student(x, y, weights, 4122 + fold)
        student_cal[held_idx] = model.predict_proba(cal_x.iloc[held_idx])
        student_eval_sum += model.predict_proba(eval_x) / 3.0
        del model, pseudo_x, x, y, weights
        gc.collect()
        print(f"student fold={fold} complete", flush=True)

    base_cal = np.log(np.clip(stack["calibration_oof_probabilities"], 1e-8, 1.0))
    student_cal_logits = np.log(np.clip(student_cal, 1e-8, 1.0))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2027)
    blend_scores = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        mix = (1.0 - weight) * base_cal + weight * student_cal_logits
        fold_scores = []
        for fit_idx, held_idx in cv.split(mix, labels_cal):
            bias = tune_bias(mix[fit_idx], labels_cal[fit_idx])
            fold_scores.append(float(f1_score(
                labels_cal[held_idx], (mix[held_idx] + bias).argmax(axis=1), average="macro"
            )))
        blend_scores.append({"weight": weight, "cv_macro_f1": float(np.mean(fold_scores))})
        print(f"blend={weight:.2f} OOF CV={np.mean(fold_scores):.6f}", flush=True)
    best = max(blend_scores, key=lambda item: item["cv_macro_f1"])
    selected_mix = (1.0 - best["weight"]) * base_cal + best["weight"] * student_cal_logits
    bias = tune_bias(selected_mix, labels_cal)
    baseline_eval = np.log(np.clip(stack["evaluation_probabilities"], 1e-8, 1.0))
    evaluation_mix = (1.0 - best["weight"]) * baseline_eval + best["weight"] * np.log(
        np.clip(student_eval_sum, 1e-8, 1.0)
    ) + bias
    predictions = np.asarray(LABELS)[evaluation_mix.argmax(axis=1)]
    evaluation_labels = np.asarray(stack["evaluation_labels"])
    score = float(f1_score(evaluation_labels, predictions, average="macro"))
    report = {
        "method": "3-fold OOF pseudo-label students; blend selected by nested calibration CV",
        "pseudo_confidence_threshold": 0.75,
        "pseudo_weight": 0.35,
        "source_training_sample": len(source_y),
        "pseudo_rows_total": int(selected.sum()),
        "blend_cv_scores": blend_scores,
        "selected_blend_weight": best["weight"],
        "evaluation_macro_f1": score,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
        "bias": bias.tolist(),
    }
    path = ROOT / "reports" / "validation_exp_122_pseudo_oof.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        calibration_student_oof=student_cal,
                        evaluation_student_mean=student_eval_sum)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
