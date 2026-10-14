"""Preprocessing fitted only on training rows for the reproducible workflow."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scripts.feature_engineering import engineer_features


LABELS = np.arange(1, 6)
CATEGORICAL = [
    "district_id", "vdcmun_id", "ward_id", "land_surface_condition",
    "foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
    "position", "plan_configuration", "superstructure", "id_prefix6",
    "id_prefix8", "structure_profile",
]
GROUPS = ["district_id", "vdcmun_id", "ward_id", "id_prefix8", "structure_profile"]


class BuildingFeatures(TransformerMixin, BaseEstimator):
    """Create deterministic building features without inspecting labels."""

    def fit(self, X, y=None):
        """Remember the required raw feature columns."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        """Create ratios, survey prefixes, and the structural profile."""
        check_is_fitted(self, "feature_names_in_")
        if "damage_grade" in X:
            raise ValueError("Target must be excluded from model features.")
        frame = engineer_features(X.copy())
        ids = frame.pop("building_id").astype("int64")
        frame["building_id_order"] = ids
        frame["id_prefix6"] = ids.astype(str).str[:6]
        frame["id_prefix8"] = ids.astype(str).str[:8]
        profile_columns = ["foundation_type", "roof_type", "ground_floor_type",
                           "other_floor_type", "superstructure"]
        profile = frame[profile_columns].fillna("__MISSING__").astype(str)
        frame["structure_profile"] = profile.agg("|".join, axis=1)
        for column in CATEGORICAL:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(object)
        return frame


class CrossFitStatistics(TransformerMixin, BaseEstimator):
    """Encode target groups and sorted-ID neighbors with out-of-fold training values."""

    def __init__(self, smoothing=100.0, n_splits=5, seed=2026):
        """Store cloneable parameters without fitting data."""
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, y):
        """Store smoothed group rates and neighbor labels from training rows only."""
        codes = np.asarray(y, dtype=int)
        if not np.isin(codes, LABELS).all():
            raise ValueError("Labels must be integers 1 through 5.")
        self.prior_ = np.bincount(codes, minlength=6)[1:] / len(codes)
        self.tables_ = {}
        for group in GROUPS:
            table = pd.crosstab(X[group].to_numpy(), codes).reindex(columns=LABELS, fill_value=0)
            counts = table.sum(axis=1)
            self.tables_[group] = (table + self.smoothing * self.prior_).div(
                counts + self.smoothing, axis=0
            )
        order = np.argsort(X["building_id_order"].to_numpy())
        self.ids_ = X["building_id_order"].to_numpy(dtype=np.int64)[order]
        one_hot = np.eye(5, dtype=np.float32)[codes[order] - 1]
        self.cumulative_ = np.vstack([np.zeros(5), one_hot.cumsum(axis=0)])
        return self

    def _encoded(self, X):
        """Map queries using stored training statistics with prior fallback."""
        check_is_fitted(self, "tables_")
        values = {}
        for group, table in self.tables_.items():
            mapped = table.reindex(X[group].to_numpy()).to_numpy()
            mapped = np.where(np.isnan(mapped), self.prior_, mapped)
            for k in range(5):
                values[f"te_{group}_grade{k + 1}"] = mapped[:, k]
        positions = np.searchsorted(self.ids_, X["building_id_order"].to_numpy(dtype=np.int64))
        for radius in (5, 25, 100):
            lower = np.maximum(0, positions - radius)
            upper = np.minimum(len(self.ids_), positions + radius)
            rates = (self.cumulative_[upper] - self.cumulative_[lower]) / np.maximum(
                upper - lower, 1
            )[:, None]
            for k in range(5):
                values[f"neighbor_{radius}_grade{k + 1}"] = rates[:, k]
        return pd.DataFrame(values, index=X.index, dtype=np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        """Exclude every held fold's labels when producing training encodings."""
        if y is None:
            raise ValueError("Training labels are required for cross-fitting.")
        codes = np.asarray(y, dtype=int)
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        encoded = None
        for source, held in cv.split(X, codes):
            encoder = CrossFitStatistics(self.smoothing, self.n_splits, self.seed)
            encoder.fit(X.iloc[source], codes[source])
            block = encoder._encoded(X.iloc[held])
            if encoded is None:
                encoded = pd.DataFrame(index=X.index, columns=block.columns, dtype=np.float32)
            encoded.iloc[held] = block.to_numpy()
        self.fit(X, codes)
        return pd.concat([X.drop(columns="building_id_order"), encoded], axis=1)

    def transform(self, X):
        """Transform new rows without accepting or learning their labels."""
        return pd.concat([X.drop(columns="building_id_order"), self._encoded(X)], axis=1)


class NativeCategories(TransformerMixin, BaseEstimator):
    """Preserve named categorical columns with vocabularies learned on training data."""

    def fit(self, X, y=None):
        """Freeze each training column's category vocabulary."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in X}
        return self

    def transform(self, X):
        """Map unknown categories to missing without changing fitted vocabularies."""
        check_is_fitted(self, "categories_")
        return pd.DataFrame({c: pd.Categorical(X[c].where(X[c].isin(self.categories_[c])),
                                              categories=self.categories_[c])
                             for c in self.feature_names_in_}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Return stable names for ColumnTransformer pandas output."""
        return self.feature_names_in_


def build_preprocessor(smoothing=100.0, seed=2026, statistics=True):
    """Build feature engineering, cross-fitting, and a typed ColumnTransformer."""
    steps = [("features", BuildingFeatures())]
    if statistics:
        steps.append(("statistics", CrossFitStatistics(smoothing=smoothing, seed=seed)))
    else:
        # The baseline excludes the raw survey ID as well as target-derived features.
        steps.append(("drop_id", ColumnTransformer(
            [("keep", "passthrough", make_column_selector(pattern="^(?!building_id_order$).*$"))],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")))
    columns = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
         make_column_selector(dtype_include=np.number)),
        ("categorical", NativeCategories(), CATEGORICAL),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    steps.append(("columns", columns))
    return Pipeline(steps)
