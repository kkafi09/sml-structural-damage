"""Check leakage boundaries, inference behavior, and submission invariants."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from scripts.predict import create_submission, validate_submission
from scripts.preprocess import BuildingFeatures, CrossFitStatistics, build_preprocessor
from scripts.workflow import ReproducibleStack, split_data


def example_data(rows=200):
    """Make schema-compatible rows with all five classes and nonconsecutive indices."""
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame({
        "building_id": np.arange(122207000000, 122207000000 + rows),
        "district_id": rng.integers(1, 4, rows),
        "vdcmun_id": rng.integers(1, 6, rows),
        "ward_id": rng.integers(1, 10, rows),
        "count_floors_pre_eq": rng.integers(1, 4, rows),
        "age_building": rng.integers(1, 50, rows),
        "plinth_area_sq_ft": rng.integers(100, 1000, rows),
        "height_ft_pre_eq": rng.integers(8, 30, rows),
    }, index=np.arange(rows) * 3 + 7)
    for column in ("land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
                   "other_floor_type", "position", "plan_configuration", "superstructure"):
        frame[column] = rng.choice(["a", "b"], rows)
    y = pd.Series(np.tile(np.arange(1, 6), rows // 5), index=frame.index)
    return frame, y


class LeakageTests(unittest.TestCase):
    """Exercise preprocessing with held labels and unseen validation values."""

    def test_held_labels_cannot_change_own_encoding(self):
        """Perturb a fixed held fold and verify its encoded values stay identical."""
        X, y = example_data()
        features = BuildingFeatures().fit_transform(X)
        held, source = np.arange(40), np.arange(40, len(X))
        folds = [(source, held), (held, source)]
        altered = y.copy()
        altered.iloc[held] = altered.iloc[held] % 5 + 1
        with patch("scripts.preprocess.StratifiedKFold.split", return_value=iter(folds)):
            original = CrossFitStatistics(n_splits=2).fit_transform(features, y)
        with patch("scripts.preprocess.StratifiedKFold.split", return_value=iter(folds)):
            perturbed = CrossFitStatistics(n_splits=2).fit_transform(features, altered)
        columns = [c for c in original if c.startswith(("te_", "neighbor_"))]
        np.testing.assert_array_equal(original.iloc[held][columns], perturbed.iloc[held][columns])

    def test_validation_does_not_refit_imputation_or_categories(self):
        """New categories and extreme query ages must not alter fitted training state."""
        X, y = example_data()
        pipeline = build_preprocessor()
        pipeline.fit_transform(X, y)
        columns = pipeline.named_steps["columns"]
        imputer = columns.named_transformers_["numeric"]
        statistics = imputer.statistics_.copy()
        categories = columns.named_transformers_["categorical"].categories_.copy()
        query = X.iloc[:5].copy()
        query["foundation_type"] = "unseen"
        query["age_building"] = 999
        transformed = pipeline.transform(query)
        self.assertTrue(transformed.foundation_type.isna().all())
        self.assertTrue(np.isfinite(transformed.select_dtypes(include="number")).all().all())
        np.testing.assert_array_equal(imputer.statistics_, statistics)
        self.assertEqual(columns.named_transformers_["categorical"].categories_, categories)

    def test_partition_ids_do_not_overlap(self):
        """Verify the frozen stratified partitions cover all rows exactly once."""
        X, y = example_data(500)
        train, cal, evaluation, *_ = split_data(X, y)
        self.assertEqual((len(train), len(cal), len(evaluation)), (400, 50, 50))
        self.assertFalse(set(train.index) & set(cal.index))
        self.assertFalse(set(train.index) & set(evaluation.index))
        self.assertFalse(set(cal.index) & set(evaluation.index))


class InferenceTests(unittest.TestCase):
    """Check model persistence and strict submission validation."""

    def test_stack_roundtrip_and_prediction(self):
        """Train the complete architecture on small data and compare reloaded probabilities."""
        X, y = example_data(500)
        train, cal, evaluation, ty, cy, _ = split_data(X, y)
        model = ReproducibleStack().fit(train, ty, calibration_X=cal, calibration_y=cy)
        probabilities = model.predict_proba(evaluation)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        self.assertTrue(np.isin(model.predict(evaluation), range(1, 6)).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict(evaluation), restored.predict(evaluation))
            np.testing.assert_allclose(probabilities, restored.predict_proba(evaluation), rtol=0, atol=0)

    def test_submission_rejects_invalid_files(self):
        """Reject reordered IDs, duplicates, floating labels, and missing predictions."""
        sample = pd.DataFrame({"building_id": [10, 20], "damage_grade": [1, 1]})
        test = sample.drop(columns="damage_grade")
        validate_submission(sample, sample, test)
        invalid = [sample.iloc[::-1], sample.assign(building_id=[10, 10]),
                   sample.assign(damage_grade=[1., 2.]), sample.assign(damage_grade=[1, 6]),
                   sample.assign(damage_grade=[1, np.nan])]
        for frame in invalid:
            with self.assertRaises(ValueError):
                validate_submission(frame, sample, test)

    def test_original_submission_is_protected(self):
        """Refuse to overwrite either original CSV before loading any estimator."""
        root = Path(__file__).resolve().parents[1]
        for name in ("submission.csv", "best_stack_087.csv"):
            with self.assertRaises(ValueError):
                create_submission("missing.joblib", root / "data", root / "submissions" / name)


if __name__ == "__main__":
    unittest.main()
