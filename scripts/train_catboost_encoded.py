"""Train CatBoost on leakage-safe target and ID-neighbor features."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias
from scripts.train_lightgbm_ensemble import prepare_data


def prepare_cat_features(train_x, *query_frames):
    """Drop extreme-cardinality raw IDs and normalize CatBoost categories."""
    drop = ["id_prefix9", "id_prefix10"]
    frames = [frame.drop(columns=drop, errors="ignore").copy()
              for frame in (train_x, *query_frames)]
    categorical = list(frames[0].select_dtypes(include="category").columns)
    for frame in frames:
        for column in categorical:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    return frames, categorical


def main() -> None:
    """Fit a multiclass CatBoost model and save its held-out probabilities."""
    cache_path = ROOT / "reports" / "catboost_encoded_data_101.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(cache_file)
        print(f"Loaded prepared data from {cache_path.name}", flush=True)
    else:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = prepare_data(2026)
        with cache_path.open("wb") as cache_file:
            pickle.dump((train_x, train_y, cal_x, cal_y, eval_x, eval_y), cache_file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved prepared data to {cache_path.name}", flush=True)
    frames, categorical = prepare_cat_features(train_x, cal_x, eval_x)
    train_x, cal_x, eval_x = frames
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10,
        random_strength=0.5, max_ctr_complexity=1,
        cat_features=categorical, random_seed=2026, thread_count=4,
        train_dir=str(ROOT / "reports" / "catboost_info_101"), verbose=100,
    )
    model.fit(
        train_x, train_y, eval_set=(cal_x, cal_y),
        early_stopping_rounds=100, use_best_model=True, verbose=100,
        save_snapshot=True,
        snapshot_file=str(ROOT / "reports" / "catboost_snapshot_101"),
        snapshot_interval=600,
    )
    cal_prob = model.predict_proba(cal_x)
    eval_prob = model.predict_proba(eval_x)
    order = [list(model.classes_).index(label) for label in LABELS]
    cal_prob, eval_prob = cal_prob[:, order], eval_prob[:, order]
    calibrated = optimize_class_bias(cal_prob, cal_y, eval_prob, eval_y)
    base_score = float(f1_score(
        [LABELS.index(label) for label in eval_y], eval_prob.argmax(1), average="macro"
    ))
    report = {
        "method": "CatBoost on cross-fitted target encodings and ID-neighbor features",
        "best_iteration": int(model.get_best_iteration()),
        "categorical_features": categorical,
        "evaluation_macro_f1_uncalibrated": base_score,
        "calibration": calibrated,
    }
    (ROOT / "reports" / "validation_exp_101_catboost_encoded.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "reports" / "catboost_encoded_holdout_101.npz",
        calibration_probabilities=cal_prob, evaluation_probabilities=eval_prob,
        calibration_labels=cal_y.to_numpy(), evaluation_labels=eval_y.to_numpy(),
    )
    print(f"CatBoost best_iteration={model.get_best_iteration()}; base={base_score:.6f}; "
          f"calibrated={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
