"""Select an XGBoost/LightGBM probability blend on calibration rows only."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    xgb = np.load(ROOT / "reports" / "xgboost_holdout_036.npz")
    lgbm = np.load(ROOT / "reports" / "lightgbm_holdout_030.npz", allow_pickle=True)
    if not np.array_equal(xgb["calibration_labels"], lgbm["calibration_labels"]):
        raise ValueError("Calibration labels do not match across model files")
    if not np.array_equal(xgb["evaluation_labels"], lgbm["evaluation_labels"]):
        raise ValueError("Evaluation labels do not match across model files")
    calibration_labels = xgb["calibration_labels"]
    evaluation_labels = xgb["evaluation_labels"]
    xgb_cal = xgb["calibration_probabilities"]
    xgb_eval = xgb["evaluation_probabilities"]
    lgb_cal = 0.6 * lgbm["calibration_multiclass"] + 0.4 * lgbm["calibration_ova"]
    lgb_eval = 0.6 * lgbm["evaluation_multiclass"] + 0.4 * lgbm["evaluation_ova"]
    results = []
    for xgb_weight in np.linspace(0.0, 1.0, 11):
        calibration_probabilities = (
            xgb_weight * xgb_cal + (1.0 - xgb_weight) * lgb_cal
        )
        evaluation_probabilities = xgb_weight * xgb_eval + (1.0 - xgb_weight) * lgb_eval
        calibrated = optimize_class_bias(
            calibration_probabilities, calibration_labels,
            evaluation_probabilities, evaluation_labels,
        )
        base_f1 = f1_score(
            evaluation_labels,
            [LABELS[index] for index in evaluation_probabilities.argmax(axis=1)],
            average="macro",
        )
        results.append({
            "xgboost_weight": float(xgb_weight),
            "calibration_macro_f1_after_bias": calibrated["calibration_macro_f1"],
            "evaluation_macro_f1": float(base_f1),
            "evaluation_macro_f1_after_bias": calibrated["evaluation_macro_f1"],
            "bias_by_grade": calibrated["bias_by_grade"],
        })
    selected = max(results, key=lambda item: item["calibration_macro_f1_after_bias"])
    report = {
        "method": "XGBoost + 0.6 multiclass/0.4 OVA LightGBM; 80/10/10 split",
        "selected_xgboost_weight_by_calibration": selected["xgboost_weight"],
        "selected_evaluation_macro_f1_after_bias":
            selected["evaluation_macro_f1_after_bias"],
        "blends": results,
    }
    path = ROOT / "reports" / "validation_exp_037_xgb_lgbm_blend.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Selected XGBoost weight={selected['xgboost_weight']:.2f}; "
        f"evaluation Macro F1={selected['evaluation_macro_f1_after_bias']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
