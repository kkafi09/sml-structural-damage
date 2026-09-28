"""Validate CatBoost configurations and train the selected final pipeline."""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feature_engineering import engineer_features, stringify_categories


DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET = "damage_grade"
LABELS = [f"Grade {grade}" for grade in range(1, 6)]
NUMERIC_FEATURES = [
    "count_floors_pre_eq", "age_building", "plinth_area_sq_ft",
    "height_ft_pre_eq", "age_is_unknown", "area_per_floor",
    "height_per_floor", "area_height_ratio",
]
CATEGORICAL_FEATURES = [
    "district_id", "vdcmun_id", "ward_id", "land_surface_condition",
    "foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
    "position", "plan_configuration", "superstructure",
]


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the training CSV, separate the label, and exclude row identifiers."""
    data = pd.read_csv(path)
    labels = data[TARGET].astype(str)
    features = data.drop(columns=[TARGET, "building_id"])
    return features, labels


def build_pipeline(parameters: dict, seed: int) -> Pipeline:
    """Build a preprocessing and CatBoost pipeline with native categories."""
    feature_builder = FunctionTransformer(engineer_features, validate=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("categorical", FunctionTransformer(
                stringify_categories, validate=False, feature_names_out="one-to-one"
            ),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    classifier = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        cat_features=CATEGORICAL_FEATURES,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
        **parameters,
    )
    return Pipeline([
        ("features", feature_builder),
        ("preprocess", preprocessor),
        ("model", classifier),
    ])


def evaluate_candidates(features: pd.DataFrame, labels: pd.Series, seed: int,
                        candidates: list | None = None) -> dict:
    """Compare a small set of CatBoost settings on one stratified holdout."""
    train_x, valid_x, train_y, valid_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    if candidates is None:
        candidates = [
            {"depth": 5, "learning_rate": 0.12, "iterations": 350, "l2_leaf_reg": 5},
            {"depth": 6, "learning_rate": 0.10, "iterations": 300, "l2_leaf_reg": 6},
            {"depth": 7, "learning_rate": 0.08, "iterations": 280, "l2_leaf_reg": 8},
        ]
    candidate_summaries = []
    best = None
    for number, parameters in enumerate(candidates, start=1):
        pipeline = build_pipeline(parameters, seed)
        train_features = pipeline.named_steps["features"].fit_transform(train_x, train_y)
        valid_features = pipeline.named_steps["features"].transform(valid_x)
        train_matrix = pipeline.named_steps["preprocess"].fit_transform(train_features, train_y)
        valid_matrix = pipeline.named_steps["preprocess"].transform(valid_features)
        pipeline.named_steps["model"].fit(
            train_matrix, train_y, eval_set=(valid_matrix, valid_y),
            early_stopping_rounds=50, verbose=50
        )
        predictions = pipeline.predict(valid_x).ravel()
        score = f1_score(valid_y, predictions, average="macro")
        best_iteration = pipeline.named_steps["model"].get_best_iteration()
        result = {
            "candidate": number,
            "parameters": parameters,
            "macro_f1": float(score),
            "best_iteration": int(best_iteration),
            "pipeline": pipeline,
            "predictions": predictions,
        }
        candidate_summaries.append(
            {key: result[key] for key in ("candidate", "parameters", "macro_f1", "best_iteration")}
        )
        if best is None or score > best["macro_f1"]:
            best = result
        print(f"Candidate {number}: Macro F1={score:.6f}, best_iteration={best_iteration}")

    report = classification_report(
        valid_y, best["predictions"], labels=LABELS, output_dict=True, zero_division=0
    )
    confusion = confusion_matrix(valid_y, best["predictions"], labels=LABELS).tolist()
    summary = {
        "seed": seed,
        "validation_size": int(len(valid_y)),
        "best_candidate": best["candidate"],
        "best_parameters": best["parameters"],
        "best_iteration": best["best_iteration"],
        "macro_f1": best["macro_f1"],
        "classification_report": report,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion,
        "candidates": candidate_summaries,
    }
    return {"best": best, "summary": summary}


def train_final(features: pd.DataFrame, labels: pd.Series, parameters: dict,
                best_iteration: int, seed: int) -> Pipeline:
    """Fit the selected pipeline on all labeled data for test predictions."""
    final_parameters = dict(parameters)
    final_parameters["iterations"] = max(best_iteration + 1, 1)
    pipeline = build_pipeline(final_parameters, seed)
    pipeline.fit(features, labels, model__verbose=100)
    return pipeline


def main() -> None:
    """Run validation, save its report, then optionally train the final model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final", action="store_true", help="also train and save final model")
    parser.add_argument("--force-search", action="store_true", help="rerun candidate validation")
    parser.add_argument("--balanced-only", action="store_true",
                        help="evaluate a class-balanced candidate on the standard holdout")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="unique ID for versioned validation/model/submission artifacts")
    parser.add_argument("--candidate", action="append", default=[],
                        help="CatBoost parameter JSON; repeat to compare multiple candidates")
    arguments = parser.parse_args()

    if arguments.experiment_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", arguments.experiment_id
    ):
        parser.error("--experiment-id may contain only letters, numbers, '_' and '-'.")
    if arguments.balanced_only and arguments.candidate:
        parser.error("Use --balanced-only or --candidate, not both.")

    features, labels = load_training_data(DATA_DIR / "train.csv")
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    if arguments.experiment_id:
        report_path = REPORT_DIR / f"validation_{arguments.experiment_id}.json"
    else:
        report_name = "validation_balanced.json" if arguments.balanced_only else "validation.json"
        report_path = REPORT_DIR / report_name
    if arguments.balanced_only:
        candidates = [{
            "depth": 6, "learning_rate": 0.10, "iterations": 300,
            "l2_leaf_reg": 6, "auto_class_weights": "Balanced",
        }]
        outcome = evaluate_candidates(features, labels, arguments.seed, candidates)
        summary = outcome["summary"]
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        selected_report_path = report_path
    elif arguments.final and report_path.exists() and not arguments.force_search:
        if arguments.experiment_id:
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            selected_report_path = report_path
            print(f"Using validation for {arguments.experiment_id}.")
        else:
            summaries = [json.loads(report_path.read_text(encoding="utf-8"))]
            selected_report_path = report_path
            balanced_path = REPORT_DIR / "validation_balanced.json"
            if balanced_path.exists():
                balanced_summary = json.loads(balanced_path.read_text(encoding="utf-8"))
                summaries.append(balanced_summary)
                if balanced_summary["macro_f1"] > summaries[0]["macro_f1"]:
                    selected_report_path = balanced_path
            summary = max(summaries, key=lambda item: item["macro_f1"])
            print("Using the best existing validation selection.")
    elif arguments.final and arguments.experiment_id and not arguments.force_search:
        parser.error(
            f"No validation report for {arguments.experiment_id}; run validation first "
            "or pass --force-search."
        )
    else:
        candidates = None
        if arguments.candidate:
            try:
                candidates = [json.loads(candidate) for candidate in arguments.candidate]
            except json.JSONDecodeError as error:
                parser.error(f"Invalid --candidate JSON: {error}")
            if not all(isinstance(candidate, dict) for candidate in candidates):
                parser.error("Each --candidate must be a JSON object of CatBoost parameters.")
        outcome = evaluate_candidates(features, labels, arguments.seed, candidates)
        summary = outcome["summary"]
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        selected_report_path = report_path
    print(f"Best holdout Macro F1: {summary['macro_f1']:.6f}")
    print(f"Validation report: {selected_report_path.relative_to(ROOT)}")

    if arguments.final:
        from joblib import dump

        experiment_id = arguments.experiment_id or datetime.now(timezone.utc).strftime(
            "run_%Y%m%d_%H%M%S"
        )
        final_model = train_final(
            features, labels, summary["best_parameters"],
            summary["best_iteration"], arguments.seed
        )
        model_path = MODEL_DIR / f"{experiment_id}.joblib"
        dump(final_model, model_path, compress=3)
        best_path = REPORT_DIR / "best_model.json"
        best_summary = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
        is_new_best = best_summary is None or summary["macro_f1"] > best_summary["macro_f1"]
        if is_new_best:
            shutil.copy2(model_path, MODEL_DIR / "best_pipeline.joblib")
            best_path.write_text(json.dumps({
                "experiment_id": experiment_id,
                "model_file": model_path.name,
                "macro_f1": summary["macro_f1"],
                "parameters": summary["best_parameters"],
                "best_iteration": summary["best_iteration"],
                "validation_report": selected_report_path.relative_to(ROOT).as_posix(),
            }, indent=2), encoding="utf-8")
        print(f"Saved experiment model: {model_path.relative_to(ROOT)}")
        print("New local best." if is_new_best else "Existing best model retained.")
        print(f"Model used for best-pointer notebook: {(MODEL_DIR / 'best_pipeline.joblib').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
