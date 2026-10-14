"""Test encoded CatBoost and spatial LightGBM as stack additions."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import stack_features
from scripts.train_lightgbm import LABELS, optimize_class_bias


def main() -> None:
    """Compare additions using calibration OOF and an untouched evaluation split."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "lightgbm_holdout_030.npz",
        "lightgbm_holdout_041.npz", "lightgbm_holdout_085.npz",
        "catboost_encoded_holdout_101.npz", "lightgbm_spatial_holdout_103.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    cal_labels = arrays[0]["calibration_labels"]
    eval_labels = arrays[0]["evaluation_labels"]
    for array in arrays[1:]:
        if not np.array_equal(array["calibration_labels"], cal_labels):
            raise ValueError("Calibration rows do not align")
        if not np.array_equal(array["evaluation_labels"], eval_labels):
            raise ValueError("Evaluation rows do not align")

    cal_probs, eval_probs = [], []
    for index, array in enumerate(arrays):
        if index in (3, 4, 5, 7):
            cal_probs.append(
                0.6 * array["calibration_multiclass"] + 0.4 * array["calibration_ova"]
            )
            eval_probs.append(
                0.6 * array["evaluation_multiclass"] + 0.4 * array["evaluation_ova"]
            )
        else:
            cal_probs.append(array["calibration_probabilities"])
            eval_probs.append(array["evaluation_probabilities"])

    configurations = {
        "base": list(range(6)),
        "base_plus_catboost": list(range(7)),
        "base_plus_spatial": list(range(6)) + [7],
        "base_plus_catboost_spatial": list(range(8)),
    }
    y_cal = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    reports, best_global = {}, None
    for name, indices in configurations.items():
        x_cal = stack_features(*(cal_probs[i] for i in indices))
        x_eval = stack_features(*(eval_probs[i] for i in indices))
        best = None
        for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            oof = np.zeros((len(y_cal), len(LABELS)))
            for fit_idx, held_idx in cv.split(x_cal, y_cal):
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=c_value, max_iter=500)
                )
                model.fit(x_cal[fit_idx], y_cal[fit_idx])
                oof[held_idx] = model.predict_proba(x_cal[held_idx])
            score = float(f1_score(y_cal, oof.argmax(axis=1), average="macro"))
            print(f"{name} C={c_value}: OOF F1={score:.6f}", flush=True)
            if best is None or score > best["score"]:
                best = {"C": c_value, "score": score, "oof": oof}
        final = make_pipeline(
            StandardScaler(), LogisticRegression(C=best["C"], max_iter=500)
        )
        final.fit(x_cal, y_cal)
        eval_prob = final.predict_proba(x_eval)
        calibrated = optimize_class_bias(best["oof"], cal_labels, eval_prob, eval_labels)
        reports[name] = {
            "selected_C": best["C"],
            "calibration_oof_macro_f1": best["score"],
            "evaluation_macro_f1_uncalibrated": float(f1_score(
                [LABELS.index(label) for label in eval_labels], eval_prob.argmax(axis=1),
                average="macro",
            )),
            "calibration": calibrated,
        }
        print(f"{name} holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)
        candidate = {"name": name, "score": best["score"], "oof": best["oof"],
                     "eval": eval_prob, "C": best["C"]}
        if best_global is None or candidate["score"] > best_global["score"]:
            best_global = candidate

    (ROOT / "reports" / "validation_exp_104_spatial_stack.json").write_text(
        json.dumps({"configurations": reports, "selected_by_oof": best_global["name"]},
                   indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_104.npz",
        configuration=best_global["name"],
        calibration_oof_probabilities=best_global["oof"],
        evaluation_probabilities=best_global["eval"],
        calibration_labels=cal_labels, evaluation_labels=eval_labels,
        selected_C=best_global["C"],
    )


if __name__ == "__main__":
    main()
