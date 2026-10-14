"""Cross-validate shrunk district-specific class biases for the best stack."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, load_data, optimize_class_bias


def fit_bias(probabilities, labels):
    """Tune class offsets using only one district or calibration fold."""
    report = optimize_class_bias(probabilities, labels, probabilities, labels)
    return np.array([report["bias_by_grade"][label] for label in LABELS])


def main() -> None:
    """Select regional shrinkage by cross-validating rows within calibration data."""
    saved = np.load(ROOT / "reports" / "stacking_holdout_086.npz", allow_pickle=True)
    features, labels = load_data()
    _, holdout_x, _, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_x, eval_x, cal_y, eval_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027, stratify=holdout_y
    )
    if not np.array_equal(cal_y.to_numpy(), saved["calibration_labels"]):
        raise ValueError("Calibration labels are not aligned")
    groups = cal_x["district_id"].to_numpy()
    eval_groups = eval_x["district_id"].to_numpy()
    probs, eval_probs = saved["calibration_oof_probabilities"], saved["evaluation_probabilities"]
    y = np.array([LABELS.index(value) for value in cal_y])
    eval_codes = np.array([LABELS.index(value) for value in eval_y])
    shrinkages = (0.0, 0.25, 0.5, 0.75, 1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    fold_scores = {value: [] for value in shrinkages}
    for fold, (fit_i, held_i) in enumerate(cv.split(probs, y)):
        global_bias = fit_bias(probs[fit_i], cal_y.iloc[fit_i])
        local_biases = {}
        for district in np.unique(groups[fit_i]):
            rows = fit_i[groups[fit_i] == district]
            local_biases[district] = (
                fit_bias(probs[rows], cal_y.iloc[rows]) if len(rows) >= 1000 else global_bias
            )
        held_logits = np.log(np.clip(probs[held_i], 1e-8, 1.0))
        for shrinkage in shrinkages:
            bias_rows = np.vstack([
                global_bias + shrinkage * (local_biases.get(groups[row], global_bias) - global_bias)
                for row in held_i
            ])
            pred = (held_logits + bias_rows).argmax(1)
            fold_scores[shrinkage].append(float(f1_score(y[held_i], pred, average="macro")))
        print(f"fold={fold} complete", flush=True)
    means = {str(value): float(np.mean(scores)) for value, scores in fold_scores.items()}
    selected = max(shrinkages, key=lambda value: np.mean(fold_scores[value]))
    global_bias = fit_bias(probs, cal_y)
    local_biases = {}
    for district in np.unique(groups):
        rows = np.flatnonzero(groups == district)
        local_biases[district] = (
            fit_bias(probs[rows], cal_y.iloc[rows]) if len(rows) >= 1000 else global_bias
        )
    eval_logits = np.log(np.clip(eval_probs, 1e-8, 1.0))
    eval_biases = np.vstack([
        global_bias + selected * (local_biases.get(district, global_bias) - global_bias)
        for district in eval_groups
    ])
    score = float(f1_score(eval_codes, (eval_logits + eval_biases).argmax(1), average="macro"))
    baseline = float(f1_score(eval_codes, (eval_logits + global_bias).argmax(1), average="macro"))
    report = {"method": "Five-fold shrunk district class-bias calibration",
              "cross_validated_macro_f1_by_shrinkage": means,
              "selected_shrinkage": selected, "evaluation_macro_f1": score,
              "evaluation_macro_f1_global_bias": baseline}
    (ROOT / "reports" / "validation_exp_090_interaction_district_bias.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"selected shrinkage={selected}; holdout={score:.6f}; global={baseline:.6f}", flush=True)


if __name__ == "__main__":
    main()
