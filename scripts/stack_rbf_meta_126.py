"""Fit an RBF kernel meta-learner on leakage-safe base-model probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


def load_stack_data():
    """Load aligned OOF and holdout probabilities from six selected base models."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
        "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    cal_probs = [arrays[i]["calibration_probabilities"] for i in range(3)]
    eval_probs = [arrays[i]["evaluation_probabilities"] for i in range(3)]
    for item in arrays[3:]:
        cal_probs.append(0.6 * item["calibration_multiclass"] + 0.4 * item["calibration_ova"])
        eval_probs.append(0.6 * item["evaluation_multiclass"] + 0.4 * item["evaluation_ova"])
    for item in arrays[1:]:
        if not np.array_equal(item["calibration_labels"], arrays[0]["calibration_labels"]):
            raise ValueError("Calibration rows are not aligned")
        if not np.array_equal(item["evaluation_labels"], arrays[0]["evaluation_labels"]):
            raise ValueError("Evaluation rows are not aligned")
    return (
        stack_features(*cal_probs), stack_features(*eval_probs),
        np.array([LABELS.index(label) for label in arrays[0]["calibration_labels"]]),
        arrays[0]["calibration_labels"], arrays[0]["evaluation_labels"],
    )


def build_model(gamma: float, components: int, regularization: float, seed: int):
    """Create a scaled approximate RBF mapping followed by balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=gamma, n_components=components, random_state=seed),
        LogisticRegression(C=regularization, max_iter=400),
    )


def main() -> None:
    """Choose RBF meta-model settings from OOF CV and evaluate the untouched split."""
    x, x_eval, y, y_labels, eval_labels = load_stack_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    configurations = [
        {"gamma": gamma, "components": components, "C": c}
        for gamma in (0.01, 0.05)
        for components in (128, 256)
        for c in (0.1, 1.0)
    ]
    results, best = [], None
    for index, config in enumerate(configurations):
        oof = np.zeros((len(y), len(LABELS)), dtype=float)
        for fold, (fit_idx, held_idx) in enumerate(cv.split(x, y), 1):
            model = build_model(config["gamma"], config["components"], config["C"],
                                6126 + fold)
            weights = compute_sample_weight("balanced", y[fit_idx])
            model.fit(x[fit_idx], y[fit_idx], logisticregression__sample_weight=weights)
            oof[held_idx] = model.predict_proba(x[held_idx])
            print(f"config={index + 1} fold={fold} complete", flush=True)
        score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        results.append({"parameters": config, "calibration_oof_macro_f1": score})
        print(f"config={index + 1} OOF Macro F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"parameters": config, "score": score, "oof": oof}
    final = build_model(best["parameters"]["gamma"], best["parameters"]["components"],
                        best["parameters"]["C"], 6126)
    final.fit(x, y, logisticregression__sample_weight=compute_sample_weight("balanced", y))
    evaluation_probabilities = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(best["oof"], y_labels,
                                     evaluation_probabilities, eval_labels)
    report = {
        "method": "Nystroem RBF meta-model on six base learners; class-balanced logistic regression",
        "cv_folds": 3, "configurations": results,
        "selected_parameters": best["parameters"],
        "calibration_oof_macro_f1": best["score"],
        "calibration": calibrated,
        "baseline_evaluation_macro_f1": 0.5985512292352484,
    }
    path = ROOT / "reports" / "validation_exp_126_rbf_meta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "stacking_holdout_126.npz",
                        calibration_oof_probabilities=best["oof"],
                        evaluation_probabilities=evaluation_probabilities,
                        calibration_labels=y_labels, evaluation_labels=eval_labels)
    print(f"Evaluation Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
