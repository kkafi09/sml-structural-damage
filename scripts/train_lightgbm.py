"""Run a stratified LightGBM search for the structural damage task."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feature_engineering import engineer_features


TARGET = "damage_grade"
LABELS = [f"Grade {grade}" for grade in range(1, 6)]
CATEGORICAL = [
    "district_id", "vdcmun_id", "ward_id", "land_surface_condition",
    "foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
    "position", "plan_configuration", "superstructure",
    "id_prefix6", "id_prefix8", "id_prefix9", "id_prefix10",
    "intensity_band",
]
PROFILE_COLUMNS = [
    "structure_profile", "id_profile", "ward_profile", "age_band", "floor_band",
    "area_band", "height_band", "building_profile",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load competition training data and apply shared feature engineering."""
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    labels = frame[TARGET]
    features = engineer_features(frame.drop(columns=[TARGET]))
    geo_path = ROOT / "data" / "ward_level_pred_intensity.csv"
    if not geo_path.exists():
        raise FileNotFoundError(
            f"Missing {geo_path}; download ward_level_pred_intensity.csv first."
        )
    geo_features = pd.read_csv(geo_path)
    geo_features = geo_features.groupby("ward_id", as_index=False).mean(numeric_only=True)
    features = features.merge(
        geo_features, on="ward_id", how="left", validate="many_to_one"
    )
    features["geo_features_missing"] = features["latitude"].isna().astype("int8")
    features["intensity_band"] = pd.cut(
        features["pred_intensity"],
        bins=[-np.inf, 7.0, 7.25, 7.5, 7.75, np.inf],
        labels=False,
    ).astype("string")
    features["intensity_x_age"] = features["pred_intensity"] * np.log1p(
        features["age_building"].fillna(0).clip(lower=0)
    )
    features["intensity_x_floors"] = (
        features["pred_intensity"] * features["count_floors_pre_eq"]
    )
    features["intensity_x_height_per_floor"] = (
        features["pred_intensity"] * features["height_per_floor"]
    )
    features["intensity_x_area_per_floor"] = (
        features["pred_intensity"] * features["area_per_floor"]
    )
    building_ids = features.pop("building_id").astype("string")
    features["building_id_order"] = building_ids.astype("int64")
    features["id_prefix6"] = building_ids.str.slice(0, 6)
    features["id_prefix8"] = building_ids.str.slice(0, 8)
    features["id_prefix9"] = building_ids.str.slice(0, 9)
    features["id_prefix10"] = building_ids.str.slice(0, 10)
    structure_profile = features["foundation_type"].astype("string").fillna("__MISSING__")
    for column in ["roof_type", "ground_floor_type", "other_floor_type", "superstructure"]:
        structure_profile = structure_profile.str.cat(
            features[column].astype("string").fillna("__MISSING__"), sep="|"
        )
    features["structure_profile"] = structure_profile
    features["id_profile"] = features["id_prefix8"].astype("string").str.cat(
        structure_profile, sep="|"
    )
    features["ward_profile"] = features["ward_id"].astype("string").str.cat(
        structure_profile, sep="|"
    )
    features["age_band"] = pd.cut(
        features["age_building"].fillna(-1),
        bins=[-2, 0, 5, 10, 20, 40, 60, 100, 1000], labels=False,
    ).astype("string")
    features["floor_band"] = features["count_floors_pre_eq"].clip(upper=6).astype("string")
    features["area_band"] = pd.cut(
        features["plinth_area_sq_ft"].clip(lower=0),
        bins=[0, 100, 200, 300, 400, 500, 700, 1000, 1500, 3000, 10000],
        labels=False,
    ).astype("string")
    features["height_band"] = pd.cut(
        features["height_ft_pre_eq"].clip(lower=0),
        bins=[0, 6, 10, 12, 15, 18, 21, 24, 30, 40, 60, 100],
        labels=False,
    ).astype("string")
    features["building_profile"] = (
        structure_profile.str.cat(features["floor_band"], sep="|")
        .str.cat(features["age_band"], sep="|")
        .str.cat(features["area_band"], sep="|")
        .str.cat(features["height_band"], sep="|")
    )
    for column in CATEGORICAL:
        features[column] = features[column].fillna("__MISSING__").astype("category")
    return features, labels


def add_target_statistics(train_x: pd.DataFrame, train_y: pd.Series,
                          query_x: pd.DataFrame) -> pd.DataFrame:
    """Add smoothed regional damage distributions learned only from train rows."""
    classes = sorted(train_y.unique())
    priors = train_y.value_counts(normalize=True).reindex(classes, fill_value=0)
    specifications = [
        (("district_id",), 30), (("vdcmun_id",), 40), (("ward_id",), 60),
        (("id_prefix6",), 50), (("id_prefix8",), 30),
        (("id_prefix10",), 20),
        (("ward_id", "foundation_type"), 100),
        (("ward_id", "ground_floor_type"), 100),
        (("ward_id", "superstructure"), 100),
        (("structure_profile",), 80),
        (("id_profile",), 120), (("ward_profile",), 150),
        (("ward_id", "age_band"), 100), (("ward_id", "floor_band"), 100),
        (("ward_id", "superstructure", "floor_band"), 150),
        (("id_prefix8", "age_band"), 50), (("id_prefix8", "floor_band"), 50),
        (("id_prefix10", "age_band"), 30),
        (("ward_id", "area_band"), 100), (("ward_id", "height_band"), 100),
        (("id_prefix8", "area_band"), 60), (("id_prefix8", "height_band"), 60),
        (("building_profile",), 80),
        (("id_prefix8", "building_profile"), 200),
        (("id_prefix10", "structure_profile"), 100),
        (("id_prefix10", "building_profile"), 180),
        (("id_prefix10", "superstructure", "floor_band"), 120),
        (("structure_profile", "intensity_band"), 120),
        (("building_profile", "intensity_band"), 200),
        (("id_prefix8", "intensity_band"), 80),
        (("ward_id", "foundation_type", "intensity_band"), 150),
        (("ward_id", "building_profile"), 300),
        (("ward_id", "roof_type"), 50),
        (("ward_id", "position"), 50),
        (("ward_id", "plan_configuration"), 50),
        (("ward_id", "land_surface_condition"), 50),
        (("vdcmun_id", "foundation_type"), 60),
        (("vdcmun_id", "roof_type"), 60),
        (("district_id", "superstructure"), 150),
        (("district_id", "foundation_type"), 150),
        (("id_prefix8", "superstructure"), 40),
    ]
    encoded_values = {}
    for columns, smoothing in specifications:
        key_name = "__target_key"
        source = train_x[columns[0]].astype("string").fillna("__MISSING__")
        query = query_x[columns[0]].astype("string").fillna("__MISSING__")
        for column in columns[1:]:
            source = source + "|" + train_x[column].astype("string").fillna("__MISSING__")
            query = query + "|" + query_x[column].astype("string").fillna("__MISSING__")
        target_columns = [f"__grade_{index}" for index in range(len(classes))]
        table = pd.DataFrame({key_name: source}, index=train_x.index)
        for index, grade in enumerate(classes):
            table[target_columns[index]] = train_y.eq(grade).astype("int8")
        grouped = table.groupby(key_name, observed=True, sort=False)
        counts = grouped.size()
        sums = grouped[target_columns].sum()
        smoothed = sums.add(priors.to_numpy() * smoothing, axis=1).div(
            counts + smoothing, axis=0
        )
        smoothed.columns = classes
        smoothed["__count"] = counts
        mapped = pd.DataFrame({key_name: query}, index=query_x.index).merge(
            smoothed, left_on=key_name, right_index=True, how="left", sort=False
        )
        values = mapped[classes].fillna(priors).to_numpy(dtype=float)
        prefix = "te_" + "_".join(columns)
        for index, grade in enumerate(classes):
            encoded_values[f"{prefix}_{grade.lower().replace(' ', '')}"] = values[:, index]
        encoded_values[f"{prefix}_mean_grade"] = values @ np.arange(1, len(classes) + 1)
        encoded_values[f"{prefix}_log_count"] = np.log1p(
            mapped["__count"].fillna(0).to_numpy(dtype=float)
        )
    encoded = pd.DataFrame(encoded_values, index=query_x.index)
    return pd.concat([query_x, encoded], axis=1)


def cross_fit_target_statistics(train_x: pd.DataFrame, train_y: pd.Series,
                                query_x: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build out-of-fold encodings for fit data and full-fit encodings for query data."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_encoded = None
    for fit_positions, held_positions in folds.split(train_x, train_y):
        fit_indices = train_x.index[fit_positions]
        held_indices = train_x.index[held_positions]
        block = add_target_statistics(
            train_x.loc[fit_indices], train_y.loc[fit_indices], train_x.loc[held_indices]
        )
        encoded_columns = [column for column in block if column.startswith("te_")]
        if fit_encoded is None:
            fit_encoded = pd.DataFrame(index=train_x.index, columns=encoded_columns, dtype=float)
        fit_encoded.loc[held_indices, encoded_columns] = block.loc[:, encoded_columns]
    query_encoded = add_target_statistics(train_x, train_y, query_x)
    return pd.concat([train_x, fit_encoded], axis=1), query_encoded


def add_neighbor_statistics(source_x: pd.DataFrame, source_y: pd.Series,
                            query_x: pd.DataFrame) -> pd.DataFrame:
    """Add local damage distributions from nearby survey IDs in sorted order."""
    classes = sorted(source_y.unique())
    source_ids = source_x["building_id_order"].to_numpy(dtype=np.int64)
    query_ids = query_x["building_id_order"].to_numpy(dtype=np.int64)
    order = np.argsort(source_ids)
    source_ids = source_ids[order]
    source_codes = pd.Categorical(source_y.iloc[order], categories=classes).codes
    one_hot = np.eye(len(classes), dtype=np.float64)[source_codes]
    cumulative = np.vstack([np.zeros(len(classes)), one_hot.cumsum(axis=0)])
    positions = np.searchsorted(source_ids, query_ids)
    columns = {}
    for neighbors in (2, 5, 10, 25, 50, 100):
        lower = np.maximum(0, positions - neighbors)
        upper = np.minimum(len(source_ids), positions + neighbors)
        counts = np.maximum(upper - lower, 1)
        distributions = (cumulative[upper] - cumulative[lower]) / counts[:, None]
        prefix = f"id_neighbor_{neighbors}"
        for index, grade in enumerate(classes):
            columns[f"{prefix}_{grade.lower().replace(' ', '')}"] = distributions[:, index]
        columns[f"{prefix}_mean_grade"] = distributions @ np.arange(1, len(classes) + 1)
    return pd.DataFrame(columns, index=query_x.index)


def add_region_neighbor_statistics(source_x: pd.DataFrame, source_y: pd.Series,
                                   query_x: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nearby building labels within each six-digit survey prefix."""
    classes = sorted(source_y.unique())
    source = pd.DataFrame({
        "prefix": source_x["id_prefix6"].astype("string").to_numpy(),
        "building_id": source_x["building_id_order"].to_numpy(dtype=np.int64),
        "label": source_y.to_numpy(),
    }, index=source_x.index).sort_values(["prefix", "building_id"])
    query = pd.DataFrame({
        "prefix": query_x["id_prefix6"].astype("string").to_numpy(),
        "building_id": query_x["building_id_order"].to_numpy(dtype=np.int64),
    }, index=query_x.index)
    # Index query rows by prefix once. Filtering the full query frame for every
    # source prefix made cross-fitting scale as O(prefixes * query_rows).
    query_groups = {
        prefix: block.index.to_numpy()
        for prefix, block in query.groupby("prefix", observed=True, sort=False)
    }
    values = {}
    for prefix, block in source.groupby("prefix", observed=True, sort=False):
        query_index = query_groups.get(prefix)
        if query_index is None or not len(query_index):
            continue
        ids = block["building_id"].to_numpy()
        codes = pd.Categorical(block["label"], categories=classes).codes
        one_hot = np.eye(len(classes), dtype=np.float64)[codes]
        cumulative = np.vstack([np.zeros(len(classes)), one_hot.cumsum(axis=0)])
        locations = np.searchsorted(ids, query.loc[query_index, "building_id"].to_numpy())
        for radius in (2, 5, 10, 25, 50, 100):
            lower = np.maximum(0, locations - radius)
            upper = np.minimum(len(ids), locations + radius)
            distribution = (cumulative[upper] - cumulative[lower]) / np.maximum(
                upper - lower, 1
            )[:, None]
            for class_index, grade in enumerate(classes):
                values.setdefault(f"region_neighbor_{radius}_{grade.lower().replace(' ', '')}", {})
                values[f"region_neighbor_{radius}_{grade.lower().replace(' ', '')}"].update(
                    dict(zip(query_index, distribution[:, class_index]))
                )
            values.setdefault(f"region_neighbor_{radius}_mean_grade", {}).update(
                dict(zip(query_index, distribution @ np.arange(1, len(classes) + 1)))
            )
    return pd.DataFrame({
        column: pd.Series(mapping).reindex(query_x.index)
        for column, mapping in values.items()
    }, index=query_x.index)


def cross_fit_region_neighbor_statistics(train_x: pd.DataFrame, train_y: pd.Series,
                                         query_x: pd.DataFrame, seed: int):
    """Build leakage-safe within-prefix neighbor features."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    encoded = None
    for source_positions, held_positions in folds.split(train_x, train_y):
        source_index = train_x.index[source_positions]
        held_index = train_x.index[held_positions]
        block = add_region_neighbor_statistics(
            train_x.loc[source_index], train_y.loc[source_index], train_x.loc[held_index]
        )
        if encoded is None:
            encoded = pd.DataFrame(index=train_x.index, columns=block.columns, dtype=float)
        encoded.loc[held_index, block.columns] = block
    query_encoded = add_region_neighbor_statistics(train_x, train_y, query_x)
    return pd.concat([train_x, encoded], axis=1), query_encoded


def cross_fit_neighbor_statistics(train_x: pd.DataFrame, train_y: pd.Series,
                                  query_x: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe neighbor features for train and external query rows."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_encoded = pd.DataFrame(index=train_x.index)
    for source_positions, held_positions in folds.split(train_x, train_y):
        source_indices = train_x.index[source_positions]
        held_indices = train_x.index[held_positions]
        block = add_neighbor_statistics(
            train_x.loc[source_indices], train_y.loc[source_indices],
            train_x.loc[held_indices],
        )
        if fit_encoded.empty:
            fit_encoded = pd.DataFrame(index=train_x.index, columns=block.columns, dtype=float)
        fit_encoded.loc[held_indices, block.columns] = block
    query_encoded = add_neighbor_statistics(train_x, train_y, query_x)
    return pd.concat([train_x, fit_encoded], axis=1), pd.concat(
        [query_x, query_encoded], axis=1
    )


def add_spatial_target_statistics(source_x: pd.DataFrame, source_y: pd.Series,
                                  query_x: pd.DataFrame) -> pd.DataFrame:
    """Add smoothed damage rates from nearby ward centroids."""
    classes = sorted(source_y.unique())
    source = source_x[["ward_id", "latitude", "longitude"]].copy()
    source["label"] = source_y
    source = source.dropna(subset=["latitude", "longitude"])
    ward_labels = pd.crosstab(source["ward_id"], source["label"]).reindex(
        columns=classes, fill_value=0
    )
    ward_locations = source.groupby("ward_id", sort=False)[
        ["latitude", "longitude"]
    ].mean().reindex(ward_labels.index)
    counts = ward_labels.sum(axis=1).to_numpy(dtype=float)
    prior = source_y.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    rates = (ward_labels.to_numpy(dtype=float) + 60.0 * prior) / (counts[:, None] + 60.0)
    reference_latitude = ward_locations["latitude"].mean()
    coordinates = np.column_stack([
        ward_locations["latitude"].to_numpy() * 111.32,
        ward_locations["longitude"].to_numpy() * 111.32
        * np.cos(np.deg2rad(reference_latitude)),
    ])
    tree = cKDTree(coordinates)
    query_locations = query_x[["ward_id", "latitude", "longitude"]].drop_duplicates(
        "ward_id"
    ).dropna(subset=["latitude", "longitude"])
    query_coordinates = np.column_stack([
        query_locations["latitude"].to_numpy() * 111.32,
        query_locations["longitude"].to_numpy() * 111.32
        * np.cos(np.deg2rad(reference_latitude)),
    ])
    spatial = pd.DataFrame({"ward_id": query_locations["ward_id"].to_numpy()})
    for neighbors in (5, 15):
        distances, indices = tree.query(
            query_coordinates, k=min(neighbors, len(coordinates))
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = np.exp(-distances / 12.0)
        weights /= weights.sum(axis=1, keepdims=True)
        distribution = (rates[indices] * weights[:, :, None]).sum(axis=1)
        prefix = f"spatial_{neighbors}"
        for index, grade in enumerate(classes):
            spatial[f"{prefix}_{grade.lower().replace(' ', '')}"] = distribution[:, index]
        spatial[f"{prefix}_mean_grade"] = distribution @ np.arange(1, len(classes) + 1)
    spatial_columns = [column for column in spatial if column != "ward_id"]
    query = query_x.copy()
    indexed = spatial.set_index("ward_id")
    for column in spatial_columns:
        query[f"{column}"] = query["ward_id"].map(indexed[column])
    return query


def cross_fit_spatial_target_statistics(train_x: pd.DataFrame, train_y: pd.Series,
                                        query_x: pd.DataFrame,
                                        seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create out-of-fold spatial target rates for fit data and query rows."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_encoded = None
    for source_positions, held_positions in folds.split(train_x, train_y):
        source_indices = train_x.index[source_positions]
        held_indices = train_x.index[held_positions]
        block = add_spatial_target_statistics(
            train_x.loc[source_indices], train_y.loc[source_indices],
            train_x.loc[held_indices],
        )
        spatial_columns = [column for column in block if column.startswith("spatial_")]
        if fit_encoded is None:
            fit_encoded = pd.DataFrame(
                index=train_x.index, columns=spatial_columns, dtype=float
            )
        fit_encoded.loc[held_indices, spatial_columns] = block[spatial_columns]
    query_encoded = add_spatial_target_statistics(train_x, train_y, query_x)
    query_columns = [column for column in query_encoded if column.startswith("spatial_")]
    return pd.concat([train_x, fit_encoded], axis=1), query_encoded


def macro_f1_metric(y_true, y_pred):
    """Return Macro F1 from LightGBM multiclass probability predictions."""
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(len(LABELS), -1).T
    return "macro_f1", f1_score(y_true, y_pred.argmax(axis=1), average="macro"), True


def optimize_class_bias(calibration_probabilities: np.ndarray,
                        calibration_labels: pd.Series,
                        evaluation_probabilities: np.ndarray,
                        evaluation_labels: pd.Series) -> dict:
    """Tune class decision biases on calibration data and score a separate holdout."""
    calibration_logits = np.log(np.clip(calibration_probabilities, 1e-8, 1.0))
    evaluation_logits = np.log(np.clip(evaluation_probabilities, 1e-8, 1.0))
    calibration_codes = np.array([LABELS.index(label) for label in calibration_labels])
    evaluation_codes = np.array([LABELS.index(label) for label in evaluation_labels])
    biases = np.zeros(len(LABELS), dtype=float)
    for step, radius in ((0.10, 1.0), (0.025, 0.10)):
        for _ in range(2):
            for grade in range(len(LABELS)):
                candidates = np.arange(biases[grade] - radius,
                                       biases[grade] + radius + step / 2, step)
                best_bias = biases[grade]
                best_score = f1_score(
                    calibration_codes,
                    (calibration_logits + biases).argmax(axis=1),
                    average="macro",
                )
                for candidate in candidates:
                    trial = biases.copy()
                    trial[grade] = candidate
                    score = f1_score(
                        calibration_codes,
                        (calibration_logits + trial).argmax(axis=1),
                        average="macro",
                    )
                    if score > best_score:
                        best_score, best_bias = score, candidate
                biases[grade] = best_bias
    base_predictions = evaluation_logits.argmax(axis=1)
    calibrated_predictions = (evaluation_logits + biases).argmax(axis=1)
    return {
        "bias_by_grade": {grade: float(bias) for grade, bias in zip(LABELS, biases)},
        "calibration_macro_f1": float(f1_score(
            calibration_codes, (calibration_logits + biases).argmax(axis=1),
            average="macro",
        )),
        "evaluation_macro_f1": float(f1_score(
            evaluation_codes, calibrated_predictions, average="macro"
        )),
        "evaluation_classification_report": classification_report(
            evaluation_labels,
            [LABELS[index] for index in calibrated_predictions],
            labels=LABELS, output_dict=True, zero_division=0,
        ),
        "evaluation_confusion_matrix": confusion_matrix(
            evaluation_labels,
            [LABELS[index] for index in calibrated_predictions], labels=LABELS
        ).tolist(),
        "uncalibrated_evaluation_macro_f1": float(f1_score(
            evaluation_codes, base_predictions, average="macro"
        )),
    }


def location_prior(frame: pd.DataFrame, source: str) -> np.ndarray:
    """Get a normalized grade distribution from one cross-fitted ID prefix."""
    columns = [f"te_id_{source}_grade{grade}" for grade in range(1, 6)]
    return frame[columns].to_numpy(dtype=float)


def evaluate() -> dict:
    """Fit the selected LightGBM model and evaluate calibrated decisions."""
    seed = 2026
    features, labels = load_data()
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    calibration_x, evaluation_x, calibration_y, evaluation_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=seed + 1, stratify=holdout_y
    )
    raw_train_x = train_x
    raw_calibration_x = calibration_x
    raw_evaluation_x = evaluation_x
    train_target_x, calibration_x = cross_fit_target_statistics(
        raw_train_x, train_y, raw_calibration_x, seed=seed
    )
    evaluation_x = add_target_statistics(raw_train_x, train_y, raw_evaluation_x)
    train_neighbor_x, calibration_neighbor_x = cross_fit_neighbor_statistics(
        raw_train_x, train_y, raw_calibration_x, seed=seed + 1
    )
    evaluation_neighbor_x = add_neighbor_statistics(
        raw_train_x, train_y, raw_evaluation_x
    )
    train_region_x, calibration_region_x = cross_fit_region_neighbor_statistics(
        raw_train_x, train_y, raw_calibration_x, seed=seed + 3
    )
    evaluation_region_x = add_region_neighbor_statistics(
        raw_train_x, train_y, raw_evaluation_x
    )
    train_spatial_x, calibration_spatial_x = cross_fit_spatial_target_statistics(
        raw_train_x, train_y, raw_calibration_x, seed=seed + 2
    )
    evaluation_spatial_x = add_spatial_target_statistics(
        raw_train_x, train_y, raw_evaluation_x
    )
    neighbor_columns = [
        column for column in train_neighbor_x if column.startswith("id_neighbor_")
    ]
    region_neighbor_columns = [
        column for column in train_region_x if column.startswith("region_neighbor_")
    ]
    spatial_columns = [
        column for column in train_spatial_x if column.startswith("spatial_")
    ]
    train_x = pd.concat([
        train_target_x, train_neighbor_x[neighbor_columns],
        train_region_x[region_neighbor_columns],
        train_spatial_x[spatial_columns],
    ], axis=1)
    calibration_x = pd.concat([
        calibration_x, calibration_neighbor_x[neighbor_columns],
        calibration_region_x[region_neighbor_columns],
        calibration_spatial_x[spatial_columns]
    ], axis=1)
    evaluation_x = pd.concat([
        evaluation_x, evaluation_neighbor_x, evaluation_region_x,
        evaluation_spatial_x[spatial_columns]
    ], axis=1)
    train_x = train_x.drop(columns=PROFILE_COLUMNS)
    calibration_x = calibration_x.drop(columns=PROFILE_COLUMNS)
    evaluation_x = evaluation_x.drop(columns=PROFILE_COLUMNS)
    train_x = train_x.drop(columns=["building_id_order"])
    calibration_x = calibration_x.drop(columns=["building_id_order"])
    evaluation_x = evaluation_x.drop(columns=["building_id_order"])
    candidate = {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.06,
                 "min_child_samples": 100, "class_weight": None}
    candidate["class_weight"] = {
        "Grade 1": 0.95, "Grade 2": 1.30, "Grade 3": 1.25,
        "Grade 4": 0.95, "Grade 5": 0.80,
    }
    model = lgb.LGBMClassifier(
        objective="multiclass", metric="None", n_estimators=1000, verbosity=-1,
        n_jobs=4, random_state=seed, colsample_bytree=0.9,
        subsample=0.85, subsample_freq=1, reg_lambda=3.0,
        num_leaves=candidate["num_leaves"], max_depth=candidate["max_depth"],
        learning_rate=candidate["learning_rate"],
        min_child_samples=candidate["min_child_samples"],
        class_weight=candidate["class_weight"],
    )
    model.fit(
        train_x, train_y, eval_set=[(calibration_x, calibration_y)],
        eval_metric=macro_f1_metric,
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(100)],
        categorical_feature=CATEGORICAL,
    )
    calibration_probabilities = model.predict_proba(calibration_x)
    evaluation_probabilities = model.predict_proba(evaluation_x)
    candidate_sources = {
        "prefix6": (location_prior(calibration_x, "prefix6"),
                    location_prior(evaluation_x, "prefix6")),
        "prefix8": (location_prior(calibration_x, "prefix8"),
                    location_prior(evaluation_x, "prefix8")),
    }
    best_blend = None
    calibration_codes = np.array([LABELS.index(label) for label in calibration_y])
    for source, (cal_prior, eval_prior) in candidate_sources.items():
        for weight in np.linspace(0.0, 1.0, 41):
            cal_blend = weight * calibration_probabilities + (1.0 - weight) * cal_prior
            score = f1_score(calibration_codes, cal_blend.argmax(axis=1), average="macro")
            if best_blend is None or score > best_blend["calibration_macro_f1"]:
                best_blend = {
                    "source": source,
                    "model_weight": float(weight),
                    "calibration_macro_f1": float(score),
                    "calibration_probabilities": cal_blend,
                    "evaluation_probabilities": (
                        weight * evaluation_probabilities + (1.0 - weight) * eval_prior
                    ),
                }
    calibration_probabilities = best_blend["calibration_probabilities"]
    evaluation_probabilities = best_blend["evaluation_probabilities"]
    calibration_base = [LABELS[index] for index in calibration_probabilities.argmax(axis=1)]
    evaluation_base = [LABELS[index] for index in evaluation_probabilities.argmax(axis=1)]
    calibration_score = f1_score(calibration_y, calibration_base, average="macro")
    evaluation_score = f1_score(evaluation_y, evaluation_base, average="macro")
    calibration = optimize_class_bias(
        calibration_probabilities, calibration_y,
        evaluation_probabilities, evaluation_y,
    )
    print(f"Base Macro F1 calibration/evaluation: {calibration_score:.6f}/"
          f"{evaluation_score:.6f}; calibrated evaluation: "
          f"{calibration['evaluation_macro_f1']:.6f}", flush=True)
    summary = {
        "seed": seed,
        "method": "LightGBM; 80/10/10 stratified train/calibration/evaluation",
        "parameters": candidate,
        "location_blend": {
            key: value for key, value in best_blend.items()
            if key not in {"calibration_probabilities", "evaluation_probabilities"}
        },
        "best_iteration": int(model.best_iteration_),
        "calibration_macro_f1_uncalibrated": float(calibration_score),
        "calibration": calibration,
        "evaluation_macro_f1": float(evaluation_score),
        "evaluation_classification_report": classification_report(
            evaluation_y, evaluation_base, labels=LABELS, output_dict=True,
            zero_division=0,
        ),
    }
    report_path = ROOT / "reports" / "validation_exp_035_prefix10.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {report_path.relative_to(ROOT)}", flush=True)
    return summary


if __name__ == "__main__":
    evaluate()
