"""Generate a separately named submission from a saved reproducible stack."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def validate_submission(submission, sample, test):
    """Reject invalid columns, IDs, labels, row order, or missing predictions."""
    if submission.columns.tolist() != ["building_id", "damage_grade"]:
        raise ValueError("Expected building_id,damage_grade in that order.")
    if not (len(submission) == len(sample) == len(test)):
        raise ValueError("Submission, sample, and test row counts differ.")
    if submission.isna().any().any() or not submission["building_id"].is_unique:
        raise ValueError("Missing values or duplicate building IDs.")
    if not submission["building_id"].equals(sample["building_id"]) or not submission[
        "building_id"
    ].equals(test["building_id"]):
        raise ValueError("Building IDs or their order differ from the template/test.")
    if not pd.api.types.is_integer_dtype(submission["damage_grade"]) or not submission[
        "damage_grade"
    ].isin(range(1, 6)).all():
        raise ValueError("Damage grades must be integers 1 through 5.")


def create_submission(model_path, data_dir, output_path):
    """Load the fitted stack and predict in chunks while preserving template IDs."""
    model_path, data_dir, output_path = map(Path, (model_path, data_dir, output_path))
    protected = {ROOT / "submissions" / name for name in ("submission.csv", "best_stack_087.csv")}
    if output_path.resolve() in {p.resolve() for p in protected}:
        raise ValueError("Use a new candidate filename; the original final CSV is protected.")
    model = joblib.load(model_path)
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    predictions = [model.predict(test.iloc[start:start + 25000])
                   for start in range(0, len(test), 25000)]
    submission = sample.copy()
    submission["damage_grade"] = np.concatenate(predictions).astype(int)
    validate_submission(submission, sample, test)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main():
    """Run standalone prediction using a saved stack and a new output filename."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "reproducible_stack.joblib")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions" / "candidate_reproducible.csv")
    args = parser.parse_args()
    result = create_submission(args.model, args.data_dir, args.output)
    print(f"Validated {len(result):,} predictions: {args.output}")


if __name__ == "__main__":
    main()
