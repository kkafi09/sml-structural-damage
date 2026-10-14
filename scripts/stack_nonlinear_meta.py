"""Try nonlinear meta-learners on leakage-safe base-model probabilities."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias
from scripts.stack_validation_models import stack_features


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    names = [
        "xgboost_holdout_036.npz", "xgboost_holdout_039.npz",
        "xgboost_holdout_053.npz", "xgboost_holdout_094.npz",
        "lightgbm_holdout_030.npz", "lightgbm_holdout_041.npz",
        "lightgbm_holdout_085.npz", "lightgbm_holdout_096.npz",
        "catboost_encoded_holdout_101.npz", "lightgbm_spatial_holdout_103.npz",
    ]
    arrays = [np.load(ROOT / "reports" / name, allow_pickle=True) for name in names]
    cal_labels, eval_labels = arrays[0]["calibration_labels"], arrays[0]["evaluation_labels"]
    probabilities = []
    for data in arrays:
        if "calibration_probabilities" in data:
            probabilities.append((data["calibration_probabilities"],
                                  data["evaluation_probabilities"]))
        else:
            probabilities.append((
                0.6 * data["calibration_multiclass"] + 0.4 * data["calibration_ova"],
                0.6 * data["evaluation_multiclass"] + 0.4 * data["evaluation_ova"],
            ))
    y = np.array([LABELS.index(label) for label in cal_labels])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    configs = {
        "base": list(range(6)),
        "all_diverse": list(range(10)),
    }
    candidates = []
    for config_name, indices in configs.items():
        x = stack_features(*(probabilities[i][0] for i in indices))
        x_eval = stack_features(*(probabilities[i][1] for i in indices))
        for leaves in (7, 15):
            for min_leaf in (150,):
                for iterations in (100, 200):
                    oof = np.zeros((len(y), len(LABELS)))
                    for fit_idx, held_idx in cv.split(x, y):
                        model = HistGradientBoostingClassifier(
                            learning_rate=0.06, max_iter=iterations,
                            max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                            l2_regularization=2.0, early_stopping=False,
                            random_state=2026,
                        )
                        model.fit(x[fit_idx], y[fit_idx])
                        oof[held_idx] = model.predict_proba(x[held_idx])
                    score = float(f1_score(y, oof.argmax(axis=1), average="macro"))
                    print(f"{config_name} leaves={leaves} minleaf={min_leaf} "
                          f"iterations={iterations}: OOF F1={score:.6f}", flush=True)
                    candidates.append((score, config_name, indices, leaves, min_leaf,
                                       iterations, x, x_eval, oof))
    best = max(candidates, key=lambda item: item[0])
    _, config_name, indices, leaves, min_leaf, iterations, x, x_eval, oof = best
    final = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=iterations, max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf, l2_regularization=2.0,
        early_stopping=False, random_state=2026,
    )
    final.fit(x, y)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(oof, cal_labels, eval_prob, eval_labels)
    report = {
        "method": "HistGradientBoosting nonlinear meta learner; reduced grid, 3-fold CV",
        "base_configuration": config_name,
        "selected_parameters": {"max_leaf_nodes": leaves,
                                 "min_samples_leaf": min_leaf,
                                 "max_iter": iterations},
        "calibration_oof_macro_f1": best[0],
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_118_nonlinear_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "stacking_holdout_118.npz",
        calibration_oof_probabilities=oof, evaluation_probabilities=eval_prob,
        calibration_labels=cal_labels, evaluation_labels=eval_labels,
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
