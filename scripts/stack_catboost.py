"""Cross-validate a CatBoost meta-classifier over OOF model predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, load_data, optimize_class_bias


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    xgb7 = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    xgb6 = np.load(ROOT / "reports" / "xgboost_holdout_039.npz")
    xgb_profile = np.load(ROOT / "reports" / "xgboost_holdout_045.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    lgbm_profile = np.load(ROOT / "reports" / "lightgbm_holdout_041.npz", allow_pickle=True)
    lgbm_region = np.load(ROOT / "reports" / "lightgbm_holdout_044.npz", allow_pickle=True)
    catboost = np.load(ROOT / "reports" / "catboost_holdout_047.npz", allow_pickle=True)
    cal_labels = xgb7["calibration_labels"]
    eval_labels = xgb7["evaluation_labels"]
    lgb_cal = [
        0.6 * data["calibration_multiclass"] + 0.4 * data["calibration_ova"]
        for data in (lgbm, lgbm_profile, lgbm_region)
    ]
    lgb_eval = [
        0.6 * data["evaluation_multiclass"] + 0.4 * data["evaluation_ova"]
        for data in (lgbm, lgbm_profile, lgbm_region)
    ]
    base_cal = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        xgb_profile["calibration_probabilities"], *lgb_cal,
        catboost["calibration_probabilities"],
    )
    base_eval = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        xgb_profile["evaluation_probabilities"], *lgb_eval,
        catboost["evaluation_probabilities"],
    )
    features, labels = load_data()
    _, holdout_x, _, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_raw, eval_raw, cal_raw_y, eval_raw_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027,
        stratify=holdout_y,
    )
    if not np.array_equal(cal_raw_y.to_numpy(), cal_labels) or not np.array_equal(
        eval_raw_y.to_numpy(), eval_labels
    ):
        raise ValueError("Raw data rows do not align with base-model probabilities")
    numeric_columns = [
        column for column in features.select_dtypes(include=np.number)
        if column != "building_id_order"
    ]
    raw_cal = cal_raw[numeric_columns].to_numpy(dtype=float)
    raw_eval = eval_raw[numeric_columns].to_numpy(dtype=float)
    cal_x = np.nan_to_num(np.column_stack([base_cal, raw_cal]), nan=-999.0)
    eval_x = np.nan_to_num(np.column_stack([base_eval, raw_eval]), nan=-999.0)
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros((len(y_cal), len(LABELS)), dtype=float)
    iterations = []
    for fold, (fit_i, held_i) in enumerate(cv.split(cal_x, y_cal)):
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10,
            random_seed=2026 + fold, thread_count=4,
            allow_writing_files=False, verbose=False,
        )
        model.fit(
            cal_x[fit_i], y_cal[fit_i], eval_set=(cal_x[held_i], y_cal[held_i]),
            early_stopping_rounds=50, use_best_model=True, verbose=False,
        )
        oof[held_i] = model.predict_proba(cal_x[held_i])
        iterations.append(int(model.get_best_iteration()))
        print(f"fold={fold}: best_iteration={iterations[-1]}", flush=True)
    oof_f1 = float(f1_score(y_cal, oof.argmax(axis=1), average="macro"))
    final_iterations = max(1, int(np.median(iterations)))
    final_model = CatBoostClassifier(
        loss_function="MultiClass", iterations=final_iterations, depth=6,
        learning_rate=0.08, l2_leaf_reg=10, random_seed=2026,
        thread_count=4, allow_writing_files=False, verbose=False,
    )
    final_model.fit(cal_x, y_cal, verbose=False)
    eval_prob = final_model.predict_proba(eval_x)
    calibrated = optimize_class_bias(oof, cal_labels, eval_prob, eval_labels)
    report = {
        "method": "OOF CatBoost stack over base predictions and physical features",
        "fold_best_iterations": iterations,
        "final_iterations": final_iterations,
        "calibration_oof_macro_f1": oof_f1,
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(label) for label in eval_labels], eval_prob.argmax(axis=1),
            average="macro",
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_048_catboost_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"CatBoost stack evaluation F1={calibrated['evaluation_macro_f1']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
