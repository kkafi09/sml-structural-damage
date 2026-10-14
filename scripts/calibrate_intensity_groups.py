"""Cross-validate class-bias calibration by seismic intensity group."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lightgbm import LABELS, load_data, optimize_class_bias


def intensity_groups(values) -> np.ndarray:
    """Map intensity bands to low, middle, high, or missing groups."""
    numeric = np.asarray(values.astype("string").fillna("__MISSING__"))
    groups = np.full(len(numeric), -1, dtype=np.int8)
    for index, value in enumerate(numeric):
        if value in {"0.0", "1.0"}:
            groups[index] = 0
        elif value == "2.0":
            groups[index] = 1
        elif value in {"3.0", "4.0"}:
            groups[index] = 2
    return groups


def fit_bias(probabilities, labels):
    """Fit class offsets on one calibration subset."""
    result = optimize_class_bias(probabilities, labels, probabilities, labels)
    return np.array([result["bias_by_grade"][label] for label in LABELS])


def main() -> None:
    """Run the experiment and save its report and prediction artifacts."""
    saved = np.load(ROOT / "reports" / "stacking_holdout_054.npz", allow_pickle=True)
    features, labels = load_data()
    _, holdout_x, _, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_x, eval_x, cal_y, eval_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027,
        stratify=holdout_y,
    )
    if not np.array_equal(cal_y.to_numpy(), saved["calibration_labels"]):
        raise ValueError("Calibration labels are not aligned with stack predictions")
    if not np.array_equal(eval_y.to_numpy(), saved["evaluation_labels"]):
        raise ValueError("Evaluation labels are not aligned with stack predictions")
    cal_groups = intensity_groups(cal_x["intensity_band"])
    eval_groups = intensity_groups(eval_x["intensity_band"])
    cal_prob = saved["calibration_oof_probabilities"]
    eval_prob = saved["evaluation_probabilities"]
    y = np.array([LABELS.index(value) for value in cal_y])
    eval_codes = np.array([LABELS.index(value) for value in eval_y])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    shrinkages = (0.0, 0.25, 0.5, 0.75, 1.0)
    fold_scores = {amount: [] for amount in shrinkages}
    for fit_i, held_i in cv.split(cal_prob, y):
        global_bias = fit_bias(cal_prob[fit_i], cal_y.iloc[fit_i])
        local_biases = {}
        for group in (0, 1, 2):
            group_fit = fit_i[cal_groups[fit_i] == group]
            if len(group_fit) < 1000:
                local_biases[group] = global_bias
            else:
                local_biases[group] = fit_bias(cal_prob[group_fit], cal_y.iloc[group_fit])
        held_logits = np.log(np.clip(cal_prob[held_i], 1e-8, 1.0))
        for shrinkage in shrinkages:
            prediction_codes = np.empty(len(held_i), dtype=np.int64)
            for local_index, row_index in enumerate(held_i):
                group = int(cal_groups[row_index])
                group_bias = local_biases.get(group, global_bias)
                bias = global_bias + shrinkage * (group_bias - global_bias)
                prediction_codes[local_index] = (held_logits[local_index] + bias).argmax()
            fold_scores[shrinkage].append(float(f1_score(
                y[held_i], prediction_codes, average="macro"
            )))
    mean_scores = {
        str(amount): float(np.mean(scores)) for amount, scores in fold_scores.items()
    }
    selected_shrinkage = max(shrinkages, key=lambda amount: np.mean(fold_scores[amount]))
    global_bias = fit_bias(cal_prob, cal_y)
    local_biases = {}
    for group in (0, 1, 2):
        group_fit = np.flatnonzero(cal_groups == group)
        local_biases[group] = (
            fit_bias(cal_prob[group_fit], cal_y.iloc[group_fit])
            if len(group_fit) >= 1000 else global_bias
        )
    eval_logits = np.log(np.clip(eval_prob, 1e-8, 1.0))
    eval_predictions = np.empty(len(eval_y), dtype=np.int64)
    for row_index, group in enumerate(eval_groups):
        local = local_biases.get(int(group), global_bias)
        bias = global_bias + selected_shrinkage * (local - global_bias)
        eval_predictions[row_index] = (eval_logits[row_index] + bias).argmax()
    report = {
        "method": "OOF stack with intensity-group class-bias calibration",
        "cross_validated_macro_f1_by_shrinkage": mean_scores,
        "selected_shrinkage": float(selected_shrinkage),
        "evaluation_macro_f1": float(f1_score(eval_codes, eval_predictions, average="macro")),
        "evaluation_macro_f1_global_bias": float(f1_score(
            eval_codes,
            (eval_logits + global_bias).argmax(axis=1),
            average="macro",
        )),
        "global_bias": dict(zip(LABELS, map(float, global_bias))),
        "local_bias_by_intensity_group": {
            str(group): dict(zip(LABELS, map(float, bias)))
            for group, bias in local_biases.items()
        },
    }
    (ROOT / "reports" / "validation_exp_061_intensity_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
