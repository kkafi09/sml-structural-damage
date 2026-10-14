"""Add leakage-safe within-ward ID neighbor priors to probability stacking."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import (
    LABELS, add_region_neighbor_statistics, load_data, optimize_class_bias,
)


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    xgb7 = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    xgb6 = np.load(ROOT / "reports" / "xgboost_holdout_039.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    lgbm_profile = np.load(ROOT / "reports" / "lightgbm_holdout_041.npz", allow_pickle=True)
    cal_y = xgb7["calibration_labels"]
    eval_y = xgb7["evaluation_labels"]
    lgb_cal = 0.6 * lgbm["calibration_multiclass"] + 0.4 * lgbm["calibration_ova"]
    lgb_eval = 0.6 * lgbm["evaluation_multiclass"] + 0.4 * lgbm["evaluation_ova"]
    lgb_profile_cal = 0.6 * lgbm_profile["calibration_multiclass"] + 0.4 * lgbm_profile["calibration_ova"]
    lgb_profile_eval = 0.6 * lgbm_profile["evaluation_multiclass"] + 0.4 * lgbm_profile["evaluation_ova"]
    base_cal = stack_features(
        xgb7["calibration_probabilities"], xgb6["calibration_probabilities"],
        lgb_cal, lgb_profile_cal,
    )
    base_eval = stack_features(
        xgb7["evaluation_probabilities"], xgb6["evaluation_probabilities"],
        lgb_eval, lgb_profile_eval,
    )
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_raw, eval_raw, cal_raw_y, eval_raw_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027,
        stratify=holdout_y,
    )
    if not np.array_equal(cal_raw_y.to_numpy(), cal_y) or not np.array_equal(
        eval_raw_y.to_numpy(), eval_y
    ):
        raise ValueError("Raw rows do not align with model predictions")
    cal_neighbors = add_region_neighbor_statistics(train_x, train_y, cal_raw)
    eval_neighbors = add_region_neighbor_statistics(train_x, train_y, eval_raw)
    neighbor_columns = [c for c in cal_neighbors if c.startswith("region_neighbor_")]
    x_cal = np.column_stack([base_cal, cal_neighbors[neighbor_columns].to_numpy()])
    x_eval = np.column_stack([base_eval, eval_neighbors[neighbor_columns].to_numpy()])
    y = np.array([LABELS.index(value) for value in cal_y])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, held_i in cv.split(x_cal, y):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c_value, max_iter=500)
            )
            model.fit(x_cal[fit_i], y[fit_i])
            oof[held_i] = model.predict_proba(x_cal[held_i])
        score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        print(f"C={c_value}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c_value, "score": score, "oof": oof}
    final = make_pipeline(
        StandardScaler(), LogisticRegression(C=best["C"], max_iter=500)
    )
    final.fit(x_cal, y)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], cal_y, eval_prob, eval_y)
    report = {
        "method": "OOF logistic stack plus within-prefix ID neighbor grade distributions",
        "selected_C": best["C"],
        "calibration_oof_macro_f1": best["score"],
        "evaluation_macro_f1_uncalibrated": float(f1_score(
            [LABELS.index(value) for value in eval_y], eval_prob.argmax(axis=1),
            average="macro",
        )),
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_050_region_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Region-prior stack evaluation F1={calibrated['evaluation_macro_f1']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
