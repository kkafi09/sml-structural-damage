"""Let an OOF logistic meta-model learn when pairwise grade signals help."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lightgbm import LABELS, optimize_class_bias


def make_features(base, pair23, pair34):
    """Combine stack class scores and pairwise log-odds into meta-features."""
    base = np.log(np.clip(base, 1e-6, 1.0))
    pair = np.column_stack([
        logit(np.clip(pair23, 1e-6, 1 - 1e-6)),
        logit(np.clip(pair34, 1e-6, 1 - 1e-6)),
    ])
    return np.column_stack([base, pair])


def main() -> None:
    """Cross-validate logistic stacking and evaluate the calibrated holdout."""
    stack = np.load(ROOT / "reports" / "stacking_holdout_054.npz", allow_pickle=True)
    pair23 = np.load(ROOT / "reports" / "pairwise_grade23_holdout_076.npz", allow_pickle=True)
    pair34 = np.load(ROOT / "reports" / "pairwise_grade34_holdout_078.npz", allow_pickle=True)
    x = make_features(stack["calibration_oof_probabilities"],
                      pair23["calibration_pair_probability"],
                      pair34["calibration_pair_probability"])
    x_eval = make_features(stack["evaluation_probabilities"],
                           pair23["evaluation_pair_probability"],
                           pair34["evaluation_pair_probability"])
    y = np.array([LABELS.index(label) for label in stack["calibration_labels"]])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best = None
    for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros((len(y), len(LABELS)))
        for fit_i, val_i in cv.split(x, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500))
            model.fit(x[fit_i], y[fit_i])
            oof[val_i] = model.predict_proba(x[val_i])
        score = float(f1_score(y, oof.argmax(1), average="macro"))
        print(f"C={c}: OOF F1={score:.6f}", flush=True)
        if best is None or score > best["score"]:
            best = {"C": c, "score": score, "oof": oof}
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=500))
    final.fit(x, y)
    eval_prob = final.predict_proba(x_eval)
    calibrated = optimize_class_bias(
        best["oof"], stack["calibration_labels"], eval_prob, stack["evaluation_labels"]
    )
    report = {"method": "OOF logistic meta-stack with Grade 2/3 and Grade 3/4 pairwise scores",
              "selected_C": best["C"], "calibration_oof_macro_f1": best["score"],
              "calibration": calibrated}
    (ROOT / "reports" / "validation_exp_082_pairwise_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Holdout Macro F1={calibrated['evaluation_macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
