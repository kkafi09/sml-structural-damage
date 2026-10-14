"""Train and evaluate the official-data-only stack without historical caches."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from scripts.preprocess import CATEGORICAL, LABELS, build_preprocessor


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def load_data(data_dir):
    """Read official training data, normalizing Grade strings to integers."""
    frame = pd.read_csv(Path(data_dir) / "train.csv")
    y = frame.pop("damage_grade").astype(str).str.extract(r"^(?:Grade )?([1-5])$", expand=False)
    if y.isna().any() or not frame["building_id"].is_unique:
        raise ValueError("Invalid target labels or duplicate building IDs.")
    return frame, y.astype(int)


def split_data(X, y):
    """Freeze an 80/10/10 stratified train/calibration/evaluation partition."""
    train_x, held_x, train_y, held_y = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cal_x, eval_x, cal_y, eval_y = train_test_split(
        held_x, held_y, test_size=0.5, random_state=SEED + 1, stratify=held_y
    )
    return train_x, cal_x, eval_x, train_y, cal_y, eval_y


def evaluate(y, predictions):
    """Return Macro F1, per-class metrics, and a consistently ordered confusion matrix."""
    return {
        "macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist(),
        "labels": LABELS.tolist(),
    }


def probability_features(probabilities):
    """Concatenate clipped log probabilities in fixed model/class order."""
    return np.concatenate([np.log(np.clip(p, 1e-6, 1.0)) for p in probabilities], axis=1)


def select_bias(probabilities, y):
    """Choose class logit offsets using only cross-validated calibration predictions."""
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    codes = np.asarray(y, dtype=int) - 1
    bias = np.zeros(5)
    best = f1_score(codes, logits.argmax(axis=1), average="macro")
    for step in (0.20, 0.05):
        for grade in range(5):
            selected = bias.copy()
            for offset in np.arange(-4, 5) * step:
                trial = bias.copy()
                trial[grade] += offset
                score = f1_score(codes, (logits + trial).argmax(axis=1), average="macro")
                if score > best:
                    best, selected = score, trial
            bias = selected
    return bias


class GradeXGBoost(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost's zero-based targets while preprocessing retains grades 1–5."""

    def __init__(self, estimator):
        """Store the cloneable underlying XGBoost estimator."""
        self.estimator = estimator

    def fit(self, X, y):
        """Convert labels only at the estimator boundary, after target encoding."""
        self.classes_ = LABELS.copy()
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - 1)
        return self

    def predict_proba(self, X):
        """Return probabilities in the original grade order."""
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """Restore integer grades after XGBoost prediction."""
        return self.estimator_.predict(X).astype(int) + 1


class ReproducibleStack(ClassifierMixin, BaseEstimator):
    """Fit base pipelines on training and the logistic stack on calibration only."""

    def __init__(self, seed=SEED, cache_dir=None):
        """Store deterministic training settings and an optional preprocessing cache."""
        self.seed = seed
        self.cache_dir = cache_dir

    def fit(self, X, y, calibration_X, calibration_y):
        """Fit official-data pipelines, tune meta C, and calibrate without evaluation labels."""
        self._progress("Training base pipelines")
        self.classes_ = LABELS.copy()
        self.pipelines_ = []
        self.recipe_ = []
        memory = joblib.Memory(self.cache_dir, verbose=0) if self.cache_dir else None
        xgb_specs = [(300, .10, 7, 30, 8., 2026),
                     (500, .08, 6, 20, 6., 2026),
                     (400, .08, 7, 20, 6., 42)]
        smoothing_values = [30., 100., 300.]
        for i, (trees, lr, depth, child, reg, seed) in enumerate(xgb_specs):
            estimator = XGBClassifier(
                n_estimators=trees, learning_rate=lr, max_depth=depth,
                min_child_weight=child, reg_lambda=reg, subsample=.85,
                colsample_bytree=.9, tree_method="hist", enable_categorical=True,
                max_cat_to_onehot=4, max_cat_threshold=16, n_jobs=4,
                random_state=seed, objective="multi:softprob", num_class=5,
            )
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(smoothing_values[i], self.seed)),
                ("model", GradeXGBoost(estimator)),
            ], memory=memory)
            pipeline.fit(X, np.asarray(y))
            self.pipelines_.append(pipeline)
            self.recipe_.append({"family": "xgboost", "smoothing": smoothing_values[i],
                                 "parameters": estimator.get_params()})
            self._progress(f"Base XGBoost {i + 1}/3 fitted")
        iterations = [(59, 81), (74, 81), (74, 99)]
        for i, strength in enumerate(smoothing_values):
            for objective, trees, seed in zip(
                ["multiclass", "multiclassova"], iterations[i], [2026, 2027]
            ):
                estimator = lgb.LGBMClassifier(
                    objective=objective, n_estimators=trees, learning_rate=.06,
                    num_leaves=63, min_child_samples=100, colsample_bytree=.9,
                    subsample=.85, subsample_freq=1, reg_lambda=3., n_jobs=4,
                    random_state=seed, verbosity=-1,
                )
                pipeline = Pipeline([
                    ("preprocess", build_preprocessor(strength, self.seed)),
                    ("model", estimator),
                ], memory=memory)
                pipeline.fit(X, np.asarray(y))
                self.pipelines_.append(pipeline)
                self.recipe_.append({"family": "lightgbm", "smoothing": strength,
                                     "parameters": estimator.get_params()})
                self._progress(f"Base LightGBM {i + 1}/3 {objective} fitted")
        self._progress("Selecting logistic regularization and calibration offsets")
        calibration = self._meta_features(calibration_X)
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        self.meta_search_ = []
        best = None
        for c_value in (.03, .3, 3.):
            meta = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=500))
            oof = cross_val_predict(meta, calibration, calibration_y, cv=cv, method="predict_proba")
            score = float(f1_score(calibration_y, LABELS[oof.argmax(axis=1)], average="macro"))
            self.meta_search_.append({"C": c_value, "calibration_oof_macro_f1": score})
            if best is None or score > best[0]:
                best = (score, meta, oof)
        self.bias_ = select_bias(best[2], calibration_y)
        self.meta_ = best[1].fit(calibration, calibration_y)
        for pipeline in self.pipelines_:
            pipeline.memory = None
        return self

    def _progress(self, message):
        """Expose progress for notebook execution without changing training decisions."""
        print(message, flush=True)
        if self.cache_dir:
            path = Path(self.cache_dir).parent / "workflow_progress.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"stage": message}), encoding="utf-8")

    def _meta_features(self, X):
        """Construct three XGBoost and three blended LightGBM probability blocks."""
        probabilities = [p.predict_proba(X) for p in self.pipelines_]
        blocks = probabilities[:3] + [
            .6 * probabilities[i] + .4 * probabilities[i + 1] for i in (3, 5, 7)
        ]
        return probability_features(blocks)

    def predict_proba(self, X):
        """Return normalized class probabilities after the stored logit offsets."""
        raw = self.meta_.predict_proba(self._meta_features(X))
        logits = np.log(np.clip(raw, 1e-8, 1.0)) + self.bias_
        adjusted = np.exp(logits - logits.max(axis=1, keepdims=True))
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict integer damage grades in the same order as input rows."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def train_baseline(train_x, train_y, cal_x, cal_y):
    """Select one of two CatBoost pipelines using calibration Macro F1 only."""
    results, best = [], None
    for depth, trees in ((5, 200), (6, 300)):
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(statistics=False)),
            ("model", CatBoostClassifier(
                depth=depth, iterations=trees, learning_rate=.1, l2_leaf_reg=6,
                cat_features=CATEGORICAL, random_seed=SEED, thread_count=4,
                allow_writing_files=False, verbose=False,
            )),
        ])
        pipeline.fit(train_x, train_y)
        score = f1_score(cal_y, pipeline.predict(cal_x).ravel(), average="macro")
        results.append({"depth": depth, "iterations": trees, "calibration_macro_f1": float(score)})
        if best is None or score > best[0]:
            best = (score, pipeline)
        print(f"CatBoost depth {depth}: calibration F1={score:.6f}", flush=True)
    return best[1], results


def compare_strategies(train_x, train_y, eval_x, eval_y):
    """Compare complete OvA/OvO on one fixed stratified subset and identical features."""
    fit_x, _, fit_y, _ = train_test_split(
        train_x, train_y, train_size=min(30000, len(train_x) - 5),
        random_state=SEED, stratify=train_y,
    )
    numeric = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
    categorical = [c for c in CATEGORICAL if c in train_x.columns]
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    estimator = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=SEED,
                              max_iter=1500, tol=1e-3)
    results = {}
    for name, wrapper in (("OvA", OneVsRestClassifier), ("OvO", OneVsOneClassifier)):
        pipeline = Pipeline([("preprocess", clone(preprocess)), ("model", wrapper(clone(estimator)))])
        pipeline.fit(fit_x, fit_y)
        results[name] = {**evaluate(eval_y, pipeline.predict(eval_x)), "training_rows": len(fit_y),
                         "evaluation_rows": len(eval_y), "binary_classifiers": 5 if name == "OvA" else 10}
    return results


def run_workflow(data_dir=ROOT / "data", output_dir=ROOT):
    """Train from raw CSVs, persist estimators, and evaluate the frozen stack once."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    X, y = load_data(data_dir)
    train_x, cal_x, eval_x, train_y, cal_y, eval_y = split_data(X, y)
    for folder in ("models", "reports"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    progress = output_dir / "reports" / "workflow_progress.json"
    progress.write_text(json.dumps({"stage": "Training and selecting CatBoost baseline"}), encoding="utf-8")
    baseline, baseline_search = train_baseline(train_x, train_y, cal_x, cal_y)
    stack = ReproducibleStack(cache_dir=str(output_dir / "reports" / "workflow_cache"))
    stack.fit(train_x, train_y, calibration_X=cal_x, calibration_y=cal_y)
    baseline_predictions = baseline.predict(eval_x).ravel().astype(int)
    stack_predictions = stack.predict(eval_x)
    model_path = output_dir / "models" / "reproducible_stack.joblib"
    joblib.dump(stack, model_path, compress=3)
    joblib.dump(baseline, output_dir / "models" / "reproducible_baseline.joblib", compress=3)
    probe = eval_x.iloc[:200]
    reloaded = joblib.load(model_path)
    if not np.array_equal(stack.predict(probe), reloaded.predict(probe)):
        raise AssertionError("Serialized stack changed predictions.")
    progress.write_text(json.dumps({"stage": "Comparing complete OvA and OvO"}), encoding="utf-8")
    report = {
        "recipe": "official-data-stack-v1", "seed": SEED, "split": "80/10/10 stratified",
        "rows": {"train": len(train_y), "calibration": len(cal_y), "evaluation": len(eval_y)},
        "data_sha256": hashlib.sha256((data_dir / "train.csv").read_bytes()).hexdigest(),
        "split_id_sha256": {name: hashlib.sha256(part.building_id.to_numpy().tobytes()).hexdigest()
                            for name, part in (("train", train_x), ("calibration", cal_x), ("evaluation", eval_x))},
        "baseline_search": baseline_search,
        "baseline": evaluate(eval_y, baseline_predictions),
        "stack": evaluate(eval_y, stack_predictions),
        "meta_search": stack.meta_search_, "bias": stack.bias_.tolist(),
        "base_models": stack.recipe_, "serialization_predictions_equal": True,
        "strategies": compare_strategies(train_x, train_y, eval_x, eval_y),
        "external_data": {"used": False, "legacy_087_status": "blocked: intensity file absent and permission unverified"},
        "note": "New official-only recipe; not a reconstruction or remeasurement of experiment 087. Meta C and offsets use calibration OOF; evaluation never selects a candidate. Base models retain the 80% training fit.",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "joblib")},
    }
    (output_dir / "reports" / "workflow_validation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame({"building_id": eval_x.building_id, "actual": eval_y,
                  "baseline": baseline_predictions, "stack": stack_predictions}).to_csv(
        output_dir / "reports" / "workflow_evaluation.csv", index=False
    )
    print(f"Stack evaluation Macro F1: {report['stack']['macro_f1']:.6f}", flush=True)
    progress.write_text(json.dumps({"stage": "Training and evaluation complete"}), encoding="utf-8")
    return report


def main():
    """Expose the cache-independent training workflow as a module command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    run_workflow(args.data_dir, args.output_dir)


if __name__ == "__main__":
    # Use canonical module names in joblib even when launched with python -m.
    from scripts.workflow import main as module_main
    module_main()
