from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "class"

BASE_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
ENGINEERED_FEATURES = ["u_g", "g_r", "r_i", "i_z"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    feature_set: str
    estimator: Any
    note: str


def load_processed_splits(processed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits from disk."""
    root = Path(processed_dir)
    splits = {}
    for split_name in ("train", "val", "test"):
        path = root / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split_name] = pd.read_csv(path)
    return splits


def resolve_feature_sets(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Resolve available feature sets from columns in processed data."""
    available = set(train_df.columns)
    base = [column for column in BASE_FEATURES if column in available]
    with_engineering = [column for column in [*BASE_FEATURES, *ENGINEERED_FEATURES] if column in available]

    if len(base) != len(BASE_FEATURES):
        missing = sorted(set(BASE_FEATURES) - set(base))
        raise ValueError(f"Base features are missing in processed splits: {missing}")
    if len(with_engineering) < len(base):
        raise ValueError("Engineered feature set is unexpectedly smaller than base feature set.")

    return {
        "base": base,
        "with_engineering": with_engineering,
    }


def build_model_specs(random_state: int) -> List[ModelSpec]:
    """Create baseline and one lightweight experiment model."""
    baseline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
        ]
    )
    random_forest_exp = RandomForestClassifier(
        n_estimators=220,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return [
        ModelSpec(
            model_id="baseline_logreg",
            model_name="LogisticRegression",
            feature_set="base",
            estimator=baseline,
            note="CP1 baseline without using engineered features.",
        ),
        ModelSpec(
            model_id="random_forest_exp",
            model_name="RandomForestClassifier",
            feature_set="with_engineering",
            estimator=random_forest_exp,
            note="Lightweight CP1 experiment with a tree model.",
        ),
    ]


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute core multiclass metrics."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def run_cp1_experiments(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train baseline and one experiment model, then persist artifacts."""
    splits = load_processed_splits(processed_dir)
    feature_sets = resolve_feature_sets(splits["train"])
    specs = build_model_specs(random_state=random_state)

    y_train = splits["train"][TARGET_COLUMN]
    y_val = splits["val"][TARGET_COLUMN]
    y_test = splits["test"][TARGET_COLUMN]

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        features = feature_sets[spec.feature_set]
        model = spec.estimator
        model.fit(splits["train"][features], y_train)

        val_pred = model.predict(splits["val"][features])
        val_metrics = evaluate_predictions(y_val, val_pred)

        result_row = {
            "model_id": spec.model_id,
            "model_name": spec.model_name,
            "feature_set": spec.feature_set,
            "num_features": len(features),
            "macro_f1_val": val_metrics["macro_f1"],
            "weighted_f1_val": val_metrics["weighted_f1"],
            "accuracy_val": val_metrics["accuracy"],
            "note": spec.note,
        }
        results.append(result_row)
        trained_models[spec.model_id] = {
            "model": model,
            "features": features,
            "model_name": spec.model_name,
            "feature_set": spec.feature_set,
        }

    results_df = pd.DataFrame(results).sort_values("macro_f1_val", ascending=False).reset_index(drop=True)
    best_model_id = str(results_df.loc[0, "model_id"])
    best_info = trained_models[best_model_id]
    best_model = best_info["model"]
    best_features = best_info["features"]

    val_pred_best = best_model.predict(splits["val"][best_features])
    test_pred_best = best_model.predict(splits["test"][best_features])
    best_val_metrics = evaluate_predictions(y_val, val_pred_best)
    best_test_metrics = evaluate_predictions(y_test, test_pred_best)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    baseline_artifact = trained_models["baseline_logreg"]
    joblib.dump(
        {
            "model_id": "baseline_logreg",
            "model_name": baseline_artifact["model_name"],
            "feature_set": baseline_artifact["feature_set"],
            "features": baseline_artifact["features"],
            "target_column": TARGET_COLUMN,
            "estimator": baseline_artifact["model"],
        },
        models_path / "cp1_baseline.joblib",
    )
    joblib.dump(
        {
            "model_id": best_model_id,
            "model_name": best_info["model_name"],
            "feature_set": best_info["feature_set"],
            "features": best_features,
            "target_column": TARGET_COLUMN,
            "estimator": best_model,
        },
        models_path / "cp1_best_model.joblib",
    )

    results_df.to_csv(models_path / "cp1_results.csv", index=False)

    detailed_metrics = {
        "best_model": {
            "model_id": best_model_id,
            "model_name": best_info["model_name"],
            "feature_set": best_info["feature_set"],
            "features": best_features,
        },
        "validation_metrics_best_model": best_val_metrics,
        "test_metrics_best_model": best_test_metrics,
        "validation_classification_report_best_model": classification_report(
            y_val,
            val_pred_best,
            output_dict=True,
            zero_division=0,
        ),
        "test_classification_report_best_model": classification_report(
            y_test,
            test_pred_best,
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix_best_model": confusion_matrix(y_val, val_pred_best).tolist(),
        "test_confusion_matrix_best_model": confusion_matrix(y_test, test_pred_best).tolist(),
        "feature_sets": feature_sets,
    }
    with (models_path / "cp1_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(detailed_metrics, file, ensure_ascii=True, indent=2)

    return {
        "results_df": results_df,
        "best_model_id": best_model_id,
        "best_model_name": best_info["model_name"],
        "best_feature_set": best_info["feature_set"],
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "baseline_model_path": str(models_path / "cp1_baseline.joblib"),
            "best_model_path": str(models_path / "cp1_best_model.joblib"),
            "results_path": str(models_path / "cp1_results.csv"),
            "metrics_path": str(models_path / "cp1_metrics.json"),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for CP1 experiments."""
    parser = argparse.ArgumentParser(description="Run CP1 baseline and lightweight experiments.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with train.csv / val.csv / test.csv from preprocessing stage.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for model artifacts and experiment tables.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    output = run_cp1_experiments(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        random_state=args.random_state,
    )

    print("CP1 experiments completed.")
    print(f"Best model: {output['best_model_name']} ({output['best_model_id']})")
    print(
        "Validation metrics (best): "
        f"macro_f1={output['best_val_metrics']['macro_f1']:.4f}, "
        f"accuracy={output['best_val_metrics']['accuracy']:.4f}, "
        f"weighted_f1={output['best_val_metrics']['weighted_f1']:.4f}"
    )
    print(
        "Test metrics (best): "
        f"macro_f1={output['best_test_metrics']['macro_f1']:.4f}, "
        f"accuracy={output['best_test_metrics']['accuracy']:.4f}, "
        f"weighted_f1={output['best_test_metrics']['weighted_f1']:.4f}"
    )
    print(f"Results table: {output['artifacts']['results_path']}")
    print(f"Metrics JSON: {output['artifacts']['metrics_path']}")


if __name__ == "__main__":
    main()
