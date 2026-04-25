import pandas as pd

from pathlib import Path

from src.modeling import run_cp1_experiments
from src.preprocessing import (
    BASE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    clean_data,
    split_data,
)


def _make_raw_row(
    obj_id: int,
    target: str,
    u: float = 20.0,
    g: float = 19.5,
    r: float = 19.0,
    i: float = 18.8,
    z: float = 18.7,
) -> dict:
    return {
        "obj_ID": obj_id,
        "alpha": 130.0 + obj_id,
        "delta": 20.0 + obj_id,
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "run_ID": 1,
        "rerun_ID": 1,
        "cam_col": 1,
        "field_ID": 1,
        "spec_obj_ID": float(100000 + obj_id),
        "class": target,
        "redshift": 0.1 + obj_id * 0.0001,
        "plate": 1,
        "MJD": 1,
        "fiber_ID": 1,
    }


def test_clean_data_removes_missing_and_exact_duplicates_only() -> None:
    row_a = _make_raw_row(obj_id=1, target="STAR")
    row_b = _make_raw_row(obj_id=1, target="STAR", u=21.0)
    row_c = dict(row_a)
    row_d = _make_raw_row(obj_id=2, target="QSO", u=None)

    raw_df = pd.DataFrame([row_a, row_b, row_c, row_d])
    cleaned_df = clean_data(raw_df)

    assert len(cleaned_df) == 2
    assert cleaned_df["obj_ID"].nunique() == 1
    assert cleaned_df.duplicated().sum() == 0
    assert cleaned_df[BASE_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0


def test_add_engineered_features_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "u": [21.0, 22.5],
            "g": [20.0, 21.0],
            "r": [19.0, 20.0],
            "i": [18.0, 19.5],
            "z": [17.0, 18.5],
            "class": ["STAR", "GALAXY"],
        }
    )

    result = add_engineered_features(df)

    assert "u_g" in result.columns
    assert "g_r" in result.columns
    assert "r_i" in result.columns
    assert "i_z" in result.columns
    assert result.loc[0, "u_g"] == 1.0
    assert result.loc[1, "i_z"] == 1.0


def test_split_data_is_stratified_and_complete() -> None:
    rows = []
    labels = ["STAR", "GALAXY", "QSO"]
    obj_id = 1
    for label in labels:
        for _ in range(20):
            row = _make_raw_row(obj_id=obj_id, target=label)
            rows.append(row)
            obj_id += 1

    df = pd.DataFrame(rows)
    train_df, val_df, test_df = split_data(df, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert 40 <= len(train_df) <= 42
    assert 8 <= len(val_df) <= 10
    assert 8 <= len(test_df) <= 10

    assert set(train_df["class"].unique()) == set(labels)
    assert set(val_df["class"].unique()) == set(labels)
    assert set(test_df["class"].unique()) == set(labels)


def _make_modeling_dataset(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    idx = 0
    for label, redshift_base, mag_base in [
        ("STAR", 0.02, 17.0),
        ("GALAXY", 0.70, 20.0),
        ("QSO", 1.80, 22.0),
    ]:
        for j in range(n_per_class):
            idx += 1
            rows.append(
                {
                    "alpha": 100.0 + idx * 0.01,
                    "delta": 20.0 + idx * 0.01,
                    "u": mag_base + 0.50 + (j % 5) * 0.05,
                    "g": mag_base + 0.20 + (j % 5) * 0.05,
                    "r": mag_base + 0.05 + (j % 5) * 0.05,
                    "i": mag_base + 0.00 + (j % 5) * 0.05,
                    "z": mag_base - 0.05 + (j % 5) * 0.05,
                    "redshift": redshift_base + (j % 7) * 0.01,
                    "class": label,
                }
            )
    return add_engineered_features(pd.DataFrame(rows))


def test_run_cp1_experiments_smoke(tmp_path) -> None:
    df = _make_modeling_dataset(n_per_class=30)
    train_df, val_df, test_df = split_data(df, random_state=42)

    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    output = run_cp1_experiments(
        processed_dir=processed_dir,
        models_dir=models_dir,
        random_state=42,
    )

    results_df = output["results_df"]
    assert len(results_df) == 2
    assert set(results_df["model_id"]) == {"baseline_logreg", "random_forest_exp"}
    assert output["best_model_id"] in {"baseline_logreg", "random_forest_exp"}

    assert 0.0 <= output["best_val_metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= output["best_test_metrics"]["macro_f1"] <= 1.0

    for artifact_path in output["artifacts"].values():
        assert pd.notna(artifact_path)
        assert Path(artifact_path).exists()
