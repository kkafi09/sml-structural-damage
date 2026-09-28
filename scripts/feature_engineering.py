"""Feature engineering helpers for structural damage classification."""

import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add robust ratios and an explicit indicator for unknown building age."""
    features = frame.copy()
    unknown_age = features["age_building"].eq(999)
    features["age_is_unknown"] = unknown_age.astype("int8")
    features.loc[unknown_age, "age_building"] = pd.NA

    floors = features["count_floors_pre_eq"].clip(lower=1)
    features["area_per_floor"] = features["plinth_area_sq_ft"] / floors
    features["height_per_floor"] = features["height_ft_pre_eq"] / floors
    features["area_height_ratio"] = features["plinth_area_sq_ft"] / features[
        "height_ft_pre_eq"].clip(lower=1)
    return features


def stringify_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to strings and give missing values a token."""
    return frame.fillna("__MISSING__").astype(str)
