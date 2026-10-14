"""Blend direct ID-neighborhood grade distributions into the best OOF stack."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, add_neighbor_statistics, load_data, optimize_class_bias


def main() -> None:
    """Select a neighbor mixture on calibration OOF predictions."""
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    cal_x, eval_x, cal_y, eval_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=2027, stratify=holdout_y
    )
    base = np.load(ROOT / "reports" / "stacking_holdout_054.npz")
    if not np.array_equal(cal_y.to_numpy(), base["calibration_labels"]) or not np.array_equal(
        eval_y.to_numpy(), base["evaluation_labels"]
    ):
        raise ValueError("Data rows do not match saved model predictions")
    y = np.array([LABELS.index(value) for value in cal_y])
    eval_codes = np.array([LABELS.index(value) for value in eval_y])
    stats_cal = add_neighbor_statistics(train_x, train_y, cal_x)
    stats_eval = add_neighbor_statistics(train_x, train_y, eval_x)
    candidates = []
    for radius in (5, 10, 25, 50, 100):
        columns = [f"id_neighbor_{radius}_grade{grade}" for grade in range(1, 6)]
        local = stats_cal[columns].to_numpy()
        local_eval = stats_eval[columns].to_numpy()
        for weight in (0.05, 0.1, 0.2, 0.3):
            cal_prob = (1 - weight) * base["calibration_oof_probabilities"] + weight * local
            eval_prob = (1 - weight) * base["evaluation_probabilities"] + weight * local_eval
            score = float(f1_score(y, cal_prob.argmax(1), average="macro"))
            candidates.append((score, radius, weight, cal_prob, eval_prob))
        print(f"radius={radius}: candidates evaluated", flush=True)
    best = max(candidates, key=lambda item: item[0])
    cal_bias = optimize_class_bias(best[3], cal_y, best[4], eval_y)
    report = {"method": "Direct ID-neighbor probability blend with best-stack OOF predictions",
              "selected_radius": best[1], "selected_weight": best[2],
              "calibration_oof_macro_f1_uncalibrated": best[0],
              "calibration": cal_bias}
    (ROOT / "reports" / "validation_exp_075_local_id_blend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Selected radius={best[1]}, weight={best[2]}, holdout={cal_bias['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
