from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "class"

REQUIRED_COLUMNS = [
    "obj_ID",
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    TARGET_COLUMN,
    "redshift",
    "plate",
    "MJD",
    "fiber_ID",
]

IDENTIFIER_COLUMNS = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID",
]

BASE_FEATURE_COLUMNS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
ENGINEERED_FEATURE_COLUMNS = ["u_g", "g_r", "r_i", "i_z"]


@dataclass(frozen=True)
class PreprocessingConfig:
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load CSV from disk into a pandas DataFrame."""
    return pd.read_csv(
        path,
        dtype={
            "obj_ID": "string",
            "spec_obj_ID": "string",
        },
    )


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate that required columns exist in the dataframe."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run lightweight, deterministic data cleaning."""
    cleaned = df.copy()

    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.replace('"', "", regex=False).str.strip()
    )

    numeric_columns = [
        column
        for column in cleaned.columns
        if column not in {TARGET_COLUMN, *IDENTIFIER_COLUMNS}
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    required_non_null = [TARGET_COLUMN, *BASE_FEATURE_COLUMNS]
    existing_required = [column for column in required_non_null if column in cleaned.columns]
    cleaned = cleaned.dropna(subset=existing_required)

    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop technical identifiers and keep model-ready columns."""
    selected_columns = [column for column in df.columns if column not in IDENTIFIER_COLUMNS]
    return df[selected_columns].copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple color-index features used in astronomy."""
    engineered = df.copy()

    required = ["u", "g", "r", "i", "z"]
    missing = [column for column in required if column not in engineered.columns]
    if missing:
        raise ValueError(f"Cannot create engineered features, missing: {missing}")

    engineered["u_g"] = engineered["u"] - engineered["g"]
    engineered["g_r"] = engineered["g"] - engineered["r"]
    engineered["r_i"] = engineered["r"] - engineered["i"]
    engineered["i_z"] = engineered["i"] - engineered["z"]

    return engineered


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test split."""
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[target_column],
        random_state=random_state,
    )

    test_fraction_from_temp = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction_from_temp,
        stratify=temp_df[target_column],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    """Return feature columns only."""
    return [column for column in df.columns if column != target_column]


def fit_clip_bounds(
    train_df: pd.DataFrame,
    feature_columns: Iterable[str],
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> Dict[str, Tuple[float, float]]:
    """Estimate quantile clipping bounds on train data only."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(train_df[column]):
            lower = float(train_df[column].quantile(lower_quantile))
            upper = float(train_df[column].quantile(upper_quantile))
            bounds[column] = (lower, upper)
    return bounds


def apply_clip_bounds(df: pd.DataFrame, clip_bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Apply precomputed clipping bounds to a dataframe."""
    clipped = df.copy()
    for column, (lower, upper) in clip_bounds.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower=lower, upper=upper)
    return clipped


def save_processed_splits(
    output_dir: str | Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clip_bounds: Dict[str, Tuple[float, float]],
) -> None:
    """Save processed splits and preprocessing metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    metadata = {
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "columns": list(train_df.columns),
        "target_column": TARGET_COLUMN,
        "clip_bounds": {column: [lower, upper] for column, (lower, upper) in clip_bounds.items()},
    }

    with (output_path / "preprocessing_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=True, indent=2)


def prepare_dataset(
    raw_data_path: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> Dict[str, pd.DataFrame]:
    """Run the full preprocessing pipeline and return processed splits."""
    raw_df = load_raw_data(raw_data_path)
    validate_columns(raw_df)

    cleaned_df = clean_data(raw_df)
    model_df = select_model_columns(cleaned_df)
    model_df = add_engineered_features(model_df)

    train_df, val_df, test_df = split_data(
        model_df,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    feature_columns = get_feature_columns(train_df)
    clip_bounds = fit_clip_bounds(
        train_df,
        feature_columns=feature_columns,
        lower_quantile=config.lower_quantile,
        upper_quantile=config.upper_quantile,
    )

    train_df = apply_clip_bounds(train_df, clip_bounds)
    val_df = apply_clip_bounds(val_df, clip_bounds)
    test_df = apply_clip_bounds(test_df, clip_bounds)

    save_processed_splits(output_dir, train_df, val_df, test_df, clip_bounds)

    return {"train": train_df, "val": val_df, "test": test_df}


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(description="Preprocess stellar classification data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/star_classification.csv"),
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed splits.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PreprocessingConfig(random_state=args.random_state)
    prepare_dataset(raw_data_path=args.input, output_dir=args.output, config=config)


if __name__ == "__main__":
    main()
