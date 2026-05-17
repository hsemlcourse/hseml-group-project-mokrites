from pathlib import Path

import pandas as pd

from src.cp2_experiments import run_cp2_experiments
from src.preprocessing import add_engineered_features, split_data


def _make_cp2_dataset(n_per_class: int = 30) -> pd.DataFrame:
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


def test_run_cp2_experiments_smoke(tmp_path) -> None:
    df = _make_cp2_dataset(n_per_class=30)
    train_df, val_df, test_df = split_data(df, random_state=42)

    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    output = run_cp2_experiments(
        processed_dir=processed_dir,
        models_dir=models_dir,
        random_state=42,
        tune_max_samples=40,
        tune_n_iter=2,
    )

    results_df = output["results_df"]
    assert len(results_df) >= 9
    assert set(results_df["phase"]).issuperset({"part1", "part2_tuning"})
    assert output["best_model_id"] in set(results_df["model_id"])

    # CP2 should be additive and avoid CP1 model ids.
    assert "baseline_logreg" not in set(results_df["model_id"])
    assert "random_forest_exp" not in set(results_df["model_id"])

    assert 0.0 <= output["best_val_metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= output["best_test_metrics"]["macro_f1"] <= 1.0

    for artifact_path in output["artifacts"].values():
        assert pd.notna(artifact_path)
        assert Path(artifact_path).exists()
