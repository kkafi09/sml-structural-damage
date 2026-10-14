"""Train a dedicated Grade 3 versus Grade 4 LightGBM classifier."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import CATEGORICAL, LABELS
from scripts.train_lightgbm_ensemble import prepare_data


def main() -> None:
    """Fit the pair classifier using only the training split and save probabilities."""
    train_x, train_y, cal_x, cal_y, eval_x, eval_y = prepare_data(2026)
    pair_train = train_y.isin(["Grade 3", "Grade 4"])
    pair_cal = cal_y.isin(["Grade 3", "Grade 4"])
    binary_train_y = train_y.loc[pair_train].eq("Grade 3").astype("int8")
    binary_cal_y = cal_y.loc[pair_cal].eq("Grade 3").astype("int8")
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.05,
        num_leaves=31, min_child_samples=100, colsample_bytree=0.9,
        subsample=0.85, subsample_freq=1, reg_lambda=3.0,
        n_jobs=4, random_state=2026, verbosity=-1,
    )
    model.fit(
        train_x.loc[pair_train], binary_train_y,
        eval_set=[(cal_x.loc[pair_cal], binary_cal_y)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
        categorical_feature=CATEGORICAL,
    )
    cal_pair = model.predict_proba(cal_x)[:, 1]
    eval_pair = model.predict_proba(eval_x)[:, 1]
    np.savez_compressed(
        ROOT / "reports" / "pairwise_grade34_holdout_078.npz",
        calibration_pair_probability=cal_pair, evaluation_pair_probability=eval_pair,
        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy(),
    )
    report = {
        "method": "Pairwise binary LightGBM for Grade 3 versus Grade 4",
        "best_iteration": int(model.best_iteration_),
        "calibration_pair_f1": float(f1_score(
            binary_cal_y, (cal_pair[pair_cal] >= 0.5).astype("int8"), average="binary"
        )),
    }
    (ROOT / "reports" / "validation_exp_078_pairwise_grade34.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"best_iteration={model.best_iteration_}; pairwise F1={report['calibration_pair_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
