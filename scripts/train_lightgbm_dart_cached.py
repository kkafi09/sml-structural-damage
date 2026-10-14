"""Train a DART LightGBM on cached leakage-safe target and neighbor features."""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import CATEGORICAL, LABELS, macro_f1_metric, optimize_class_bias


def main():
    """Run the experiment and save its report and prediction artifacts."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    model = lgb.LGBMClassifier(
        objective="multiclass", metric="None", boosting_type="dart",
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=100, colsample_bytree=0.9,
        subsample=0.85, subsample_freq=1, reg_lambda=3.0,
        drop_rate=0.1, skip_drop=0.5, max_drop=50,
        class_weight={"Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
                      "Grade 4": 0.95, "Grade 5": 0.80},
        n_jobs=4, random_state=2040, verbosity=-1,
    )
    def save_checkpoint(env):
        """Save a LightGBM booster checkpoint for later continuation."""
        iteration = env.iteration + 1
        if iteration in (100, 200, 300, 400):
            env.model.save_model(
                str(ROOT / "reports" / f"dart_checkpoint_112_{iteration}.txt"),
                num_iteration=iteration,
            )
    save_checkpoint.order = 99
    save_checkpoint.before_iteration = False
    model.fit(train_x, train_y, categorical_feature=CATEGORICAL,
              eval_set=[(cal_x, cal_y)], eval_metric=macro_f1_metric,
              callbacks=[lgb.log_evaluation(100), save_checkpoint])
    best = None
    for iteration in (100, 200, 300, 400):
        cal_prob = model.predict_proba(cal_x, num_iteration=iteration)
        score = float(f1_score(cal_y, [LABELS[i] for i in cal_prob.argmax(axis=1)],
                               average="macro"))
        print(f"iteration={iteration}: calibration F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"iteration": iteration, "score": score,
                    "cal": cal_prob,
                    "eval": model.predict_proba(eval_x, num_iteration=iteration)}
    calibrated = optimize_class_bias(best["cal"], cal_y, best["eval"], eval_y)
    report = {"method": "DART LightGBM on cached target/neighbor encodings",
              "selected_iteration": best["iteration"],
              "calibration_macro_f1_uncalibrated": best["score"],
              "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_112_dart_cached.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "reports" / "lightgbm_dart_cached_112.npz",
                        calibration_probabilities=best["cal"],
                        evaluation_probabilities=best["eval"],
                        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy())
    print(f"DART holdout={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
