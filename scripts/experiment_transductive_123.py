"""Test cross-fitted pseudo-labeling from unlabeled evaluation features."""

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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import LABELS
from scripts.train_lightgbm import CATEGORICAL


def main() -> None:
    """Predict each evaluation fold using pseudo-labels from other eval rows."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    base_eval = stack["evaluation_probabilities"]
    evaluation_labels = np.asarray(stack["evaluation_labels"])
    if not np.array_equal(np.asarray(eval_y), evaluation_labels):
        raise ValueError("Cached evaluation rows are not aligned with base predictions")
    chosen = base_eval.max(axis=1) >= 0.75
    pseudo_all_y = np.asarray(LABELS)[base_eval.argmax(axis=1)]
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=150_000, random_state=2026)
    positions, _ = next(sampler.split(np.zeros(len(train_y)), train_y))
    source_x, source_y = train_x.iloc[positions], train_y.iloc[positions]
    del train_x, train_y, cal_x, cal_y
    gc.collect()
    for column in CATEGORICAL:
        if column in source_x and isinstance(source_x[column].dtype, pd.CategoricalDtype):
            eval_x[column] = eval_x[column].cat.set_categories(source_x[column].cat.categories)

    labels_eval = np.array([LABELS.index(label) for label in evaluation_labels])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    transductive_oof = np.zeros_like(base_eval)
    for fold, (fit_idx, held_idx) in enumerate(folds.split(eval_x, labels_eval), 1):
        pseudo_idx = fit_idx[chosen[fit_idx]]
        pseudo_x = eval_x.iloc[pseudo_idx]
        x = pd.concat([source_x, pseudo_x], ignore_index=True)
        y = np.concatenate([source_y.to_numpy(), pseudo_all_y[pseudo_idx]])
        weights = np.concatenate([
            np.ones(len(source_y), dtype=float),
            np.full(len(pseudo_idx), 0.25, dtype=float),
        ])
        model = lgb.LGBMClassifier(
            objective="multiclass", metric="None", n_estimators=64,
            learning_rate=0.06, num_leaves=63, min_child_samples=100,
            colsample_bytree=0.9, subsample=0.85, subsample_freq=1,
            reg_lambda=3.0, max_bin=127,
            class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                          "Grade 4": 0.95, "Grade 5": 0.80},
            n_jobs=2, random_state=5123 + fold, verbosity=-1,
        )
        model.fit(x, y, sample_weight=weights, categorical_feature=CATEGORICAL)
        transductive_oof[held_idx] = model.predict_proba(eval_x.iloc[held_idx])
        del model, pseudo_x, x, y, weights
        gc.collect()
        print(f"evaluation fold={fold} complete; pseudo_rows={len(pseudo_idx):,}", flush=True)

    base_bias_report = json.loads(
        (ROOT / "reports" / "validation_exp_087_interaction_bias.json").read_text()
    )
    bias = np.array([base_bias_report["bias_by_grade"][grade] for grade in LABELS])
    results = {}
    student_logits = np.log(np.clip(transductive_oof, 1e-8, 1.0))
    for weight in (0.05, 0.10, 0.15):
        logits = (1.0 - weight) * np.log(np.clip(base_eval, 1e-8, 1.0)) \
            + weight * student_logits + bias
        pred = np.asarray(LABELS)[logits.argmax(axis=1)]
        results[f"crossfit_heldout_blend_{weight:.2f}"] = float(
            f1_score(evaluation_labels, pred, average="macro")
        )
    report = {
        "method": "3-fold transductive LightGBM pseudo-labels from other evaluation rows only",
        "source_sample": len(source_y),
        "confidence_threshold": 0.75,
        "pseudo_weight": 0.25,
        "evaluation_labels_used_for_training": False,
        "baseline_macro_f1": 0.5985512292352484,
        "scores": results,
    }
    path = ROOT / "reports" / "validation_exp_123_transductive.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        transductive_oof=transductive_oof)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
