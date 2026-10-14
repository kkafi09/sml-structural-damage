"""Use cross-fitted location and structure priors to correct stack probabilities."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import LABELS


def make_meta_features(frame, probabilities: np.ndarray, columns: list[str]) -> np.ndarray:
    """Join stack log-probabilities with numeric cross-fitted prior features."""
    prior = frame[columns].to_numpy(dtype=np.float32)
    return np.column_stack([
        np.log(np.clip(probabilities, 1e-7, 1)).astype(np.float32), prior,
    ])


def tune_bias(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimize class offsets with cross-fitted calibration probabilities."""
    y = np.array([LABELS.index(label) for label in labels])
    logits = np.log(np.clip(probabilities, 1e-7, 1))
    bias = np.zeros(5)
    for step, radius in ((0.1, 0.8), (0.025, 0.1)):
        for _ in range(2):
            for cls in range(5):
                best = f1_score(y, (logits + bias).argmax(1), average="macro")
                chosen = bias[cls]
                for value in np.arange(bias[cls] - radius, bias[cls] + radius + step / 2, step):
                    trial = bias.copy()
                    trial[cls] = value
                    score = f1_score(y, (logits + trial).argmax(1), average="macro")
                    if score > best:
                        best, chosen = score, value
                bias[cls] = chosen
    return bias, float(f1_score(y, (logits + bias).argmax(1), average="macro"))


def main() -> None:
    """Cross-fit a context-aware meta-model and score its final held-out predictions."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    del train_x, train_y, cal_y
    gc.collect()
    columns = [column for column in cal_x.columns
               if column.startswith(("te_", "id_neighbor_", "region_neighbor_", "spatial_"))
               and np.issubdtype(cal_x[column].dtype, np.number)]
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    cal_prob = stack["calibration_oof_probabilities"]
    eval_prob = stack["evaluation_probabilities"]
    if len(cal_x) != len(cal_prob) or len(eval_x) != len(eval_prob):
        raise ValueError("Cached feature rows and baseline stack rows do not match")
    xcal = make_meta_features(cal_x, cal_prob, columns)
    xeval = make_meta_features(eval_x, eval_prob, columns)
    ycal = np.array([LABELS.index(label) for label in stack["calibration_labels"]])
    yeval = np.array([LABELS.index(label) for label in stack["evaluation_labels"]])
    del cal_x, eval_x
    gc.collect()
    oof = np.zeros((len(ycal), 5), dtype=np.float32)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7142)
    for fold, (fit_idx, held_idx) in enumerate(cv.split(xcal, ycal), start=1):
        model = CatBoostClassifier(
            loss_function="MultiClass", iterations=180, depth=4,
            learning_rate=0.05, l2_leaf_reg=12, random_strength=0.5,
            thread_count=2, random_seed=7142 + fold, verbose=False,
        )
        model.fit(xcal[fit_idx], ycal[fit_idx])
        oof[held_idx] = model.predict_proba(xcal[held_idx])
        print(f"context meta fold {fold}/3 complete", flush=True)
        del model
        gc.collect()
    bias, cal_score = tune_bias(oof, stack["calibration_labels"])
    final = CatBoostClassifier(
        loss_function="MultiClass", iterations=180, depth=4,
        learning_rate=0.05, l2_leaf_reg=12, random_strength=0.5,
        thread_count=2, random_seed=7150, verbose=False,
    )
    final.fit(xcal, ycal)
    eval_meta = final.predict_proba(xeval)
    score = float(f1_score(yeval, (np.log(np.clip(eval_meta, 1e-7, 1)) + bias).argmax(1),
                           average="macro"))
    report = {
        "method": "3-fold cross-fitted CatBoost meta-model using stack log-probabilities and numeric target-prior, neighbor, and spatial features",
        "context_feature_count": len(columns), "context_features": columns,
        "meta_feature_count": xcal.shape[1], "calibration_oof_macro_f1": cal_score,
        "evaluation_macro_f1": score, "baseline_evaluation_macro_f1": 0.5985512292352484,
        "bias": bias.tolist(),
    }
    path = ROOT / "reports" / "validation_exp_142_context_meta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), calibration_oof_probabilities=oof,
                        evaluation_probabilities=eval_meta)
    print(json.dumps({k: v for k, v in report.items() if k != "context_features"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
