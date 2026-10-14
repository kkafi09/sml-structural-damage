"""Test regularized linear meta-stacking with leakage-safe context priors."""

from __future__ import annotations

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stack_validation_models import LABELS


def make_meta_features(frame, probabilities: np.ndarray, columns: list[str]) -> np.ndarray:
    """Combine baseline log-probabilities and numeric target-prior features."""
    return np.column_stack([
        np.log(np.clip(probabilities, 1e-7, 1)).astype(np.float32),
        frame[columns].to_numpy(dtype=np.float32),
    ])


def tune_bias(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select per-class decision offsets using out-of-fold calibration predictions."""
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


def make_model(regularization: float):
    """Create an imputed, scaled, regularized multiclass logistic pipeline."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=regularization, max_iter=600, solver="lbfgs"),
    )


def main() -> None:
    """Choose regularization with calibration cross-validation, then score holdout."""
    with (ROOT / "reports" / "catboost_encoded_data_101.pkl").open("rb") as file:
        train_x, train_y, cal_x, cal_y, eval_x, eval_y = pickle.load(file)
    del train_x, train_y, cal_y
    gc.collect()
    columns = [column for column in cal_x.columns
               if column.startswith(("te_", "id_neighbor_", "region_neighbor_", "spatial_"))
               and np.issubdtype(cal_x[column].dtype, np.number)]
    stack = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    xcal = make_meta_features(cal_x, stack["calibration_oof_probabilities"], columns)
    xeval = make_meta_features(eval_x, stack["evaluation_probabilities"], columns)
    ycal = np.array([LABELS.index(label) for label in stack["calibration_labels"]])
    yeval = np.array([LABELS.index(label) for label in stack["evaluation_labels"]])
    del cal_x, eval_x
    gc.collect()
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=7143).split(xcal, ycal))
    candidates = []
    for c_value in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        oof = np.zeros((len(ycal), 5), dtype=np.float32)
        for fit_idx, held_idx in folds:
            model = make_model(c_value)
            model.fit(xcal[fit_idx], ycal[fit_idx])
            oof[held_idx] = model.predict_proba(xcal[held_idx])
        score = float(f1_score(ycal, oof.argmax(1), average="macro"))
        candidates.append({"C": c_value, "oof_macro_f1": score, "oof_probabilities": oof})
        print(f"C={c_value:g}: calibration OOF={score:.6f}", flush=True)
    selected = max(candidates, key=lambda item: item["oof_macro_f1"])
    bias, cal_score = tune_bias(selected["oof_probabilities"], stack["calibration_labels"])
    final = make_model(selected["C"])
    final.fit(xcal, ycal)
    eval_prob = final.predict_proba(xeval)
    score = float(f1_score(yeval, (np.log(np.clip(eval_prob, 1e-7, 1)) + bias).argmax(1),
                           average="macro"))
    report = {
        "method": "3-fold OOF regularized logistic meta-model on stack probabilities plus context priors",
        "context_feature_count": len(columns), "selected_C": selected["C"],
        "candidate_oof_scores": [{"C": item["C"], "macro_f1": item["oof_macro_f1"]}
                                 for item in candidates],
        "calibration_oof_macro_f1_uncalibrated": selected["oof_macro_f1"],
        "calibration_oof_macro_f1_calibrated": cal_score,
        "evaluation_macro_f1": score, "baseline_evaluation_macro_f1": 0.5985512292352484,
        "bias": bias.tolist(),
    }
    path = ROOT / "reports" / "validation_exp_143_context_logistic_meta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"),
                        calibration_oof_probabilities=selected["oof_probabilities"],
                        evaluation_probabilities=eval_prob)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
