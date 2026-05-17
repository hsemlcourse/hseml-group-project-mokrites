from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "class"
BASE_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
ENGINEERED_FEATURES = ["u_g", "g_r", "r_i", "i_z"]
CP2_FEATURES = [*BASE_FEATURES, *ENGINEERED_FEATURES]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    phase: str
    hypothesis: str
    check_method: str
    estimator: Any


@dataclass(frozen=True)
class TuneSpec:
    model_id: str
    model_name: str
    estimator: Any
    param_distributions: Dict[str, List[Any]]
    n_iter: int
    hypothesis: str
    check_method: str


def load_processed_splits(processed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    root = Path(processed_dir)
    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_path = root / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        splits[split_name] = pd.read_csv(split_path)
    return splits


def resolve_cp2_features(train_df: pd.DataFrame) -> List[str]:
    missing = [column for column in CP2_FEATURES if column not in train_df.columns]
    if missing:
        raise ValueError(
            "Processed splits are missing required CP2 features: "
            f"{missing}. Run preprocessing with engineered features first."
        )
    return CP2_FEATURES.copy()


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _knn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=35, weights="distance")),
        ]
    )


def build_cp2_part1_specs(random_state: int) -> List[ModelSpec]:
    return [
        ModelSpec(
            model_id="knn_exp",
            model_name="KNeighborsClassifier",
            phase="part1",
            hypothesis="KNN on color-index features should capture local class neighborhoods.",
            check_method="Fixed hyperparameters on train, evaluate on val/test.",
            estimator=_knn_pipeline(),
        ),
        ModelSpec(
            model_id="extra_trees_exp",
            model_name="ExtraTreesClassifier",
            phase="part1",
            hypothesis="Highly randomized trees should improve robustness against noisy photometric features.",
            check_method="Fixed hyperparameters on train, evaluate on val/test.",
            estimator=ExtraTreesClassifier(
                n_estimators=320,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
        ModelSpec(
            model_id="hist_gb_exp",
            model_name="HistGradientBoostingClassifier",
            phase="part1",
            hypothesis="Histogram boosting should model non-linear interactions better than neighborhood models.",
            check_method="Fixed hyperparameters on train, evaluate on val/test.",
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            model_id="gaussian_nb_exp",
            model_name="GaussianNB",
            phase="part1",
            hypothesis="Naive Bayes provides a fast probabilistic baseline among non-linear CP2 models.",
            check_method="Fixed hyperparameters on train, evaluate on val/test.",
            estimator=GaussianNB(var_smoothing=1e-9),
        ),
        ModelSpec(
            model_id="adaboost_exp",
            model_name="AdaBoostClassifier",
            phase="part1",
            hypothesis="Adaptive boosting over shallow trees should focus on hard-to-classify examples.",
            check_method="Fixed hyperparameters on train, evaluate on val/test.",
            estimator=AdaBoostClassifier(
                n_estimators=240,
                learning_rate=0.6,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            model_id="voting_soft_exp",
            model_name="VotingClassifier",
            phase="part1",
            hypothesis="Soft voting across diverse estimators should reduce variance and improve macro-F1.",
            check_method="Train ensemble on same split, evaluate on val/test.",
            estimator=VotingClassifier(
                estimators=[
                    ("knn", _knn_pipeline()),
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=220,
                            min_samples_leaf=2,
                            random_state=random_state,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_leaf_nodes=31,
                            min_samples_leaf=30,
                            random_state=random_state,
                        ),
                    ),
                    ("gnb", GaussianNB(var_smoothing=1e-9)),
                ],
                voting="soft",
                n_jobs=1,
            ),
        ),
        ModelSpec(
            model_id="stacking_exp",
            model_name="StackingClassifier",
            phase="part1",
            hypothesis="Stacking should combine complementary decision patterns from neighborhood and tree models.",
            check_method="3-fold stacking on train, then evaluate on val/test.",
            estimator=StackingClassifier(
                estimators=[
                    ("knn", _knn_pipeline()),
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=220,
                            min_samples_leaf=2,
                            random_state=random_state,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_leaf_nodes=31,
                            min_samples_leaf=30,
                            random_state=random_state,
                        ),
                    ),
                    (
                        "dt",
                        DecisionTreeClassifier(
                            max_depth=8,
                            min_samples_leaf=12,
                            random_state=random_state,
                            class_weight="balanced",
                        ),
                    ),
                ],
                final_estimator=GaussianNB(var_smoothing=1e-9),
                cv=3,
                n_jobs=1,
            ),
        ),
    ]


def build_cp2_tune_specs(random_state: int, n_iter: int) -> List[TuneSpec]:
    return [
        TuneSpec(
            model_id="extra_trees_tuned",
            model_name="ExtraTreesClassifier",
            estimator=ExtraTreesClassifier(
                random_state=random_state,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
            param_distributions={
                "n_estimators": [180, 260, 340, 420],
                "max_depth": [None, 10, 16, 24],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            },
            n_iter=n_iter,
            hypothesis="Tuning tree depth and leaf constraints should improve macro-F1 for minority class QSO.",
            check_method="RandomizedSearchCV (3-fold) on train subset, retrain best on full train.",
        ),
        TuneSpec(
            model_id="hist_gb_tuned",
            model_name="HistGradientBoostingClassifier",
            estimator=HistGradientBoostingClassifier(random_state=random_state),
            param_distributions={
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_leaf_nodes": [15, 31, 63],
                "max_depth": [None, 6, 10, 14],
                "min_samples_leaf": [10, 20, 35, 60],
                "l2_regularization": [0.0, 0.1, 0.5, 1.0],
            },
            n_iter=n_iter,
            hypothesis="Tuning learning rate and leaf complexity should improve class-separation balance.",
            check_method="RandomizedSearchCV (3-fold) on train subset, retrain best on full train.",
        ),
    ]


def _sample_for_tuning(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    max_samples: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(x_train) <= max_samples:
        return x_train, y_train

    x_subset, _, y_subset, _ = train_test_split(
        x_train,
        y_train,
        train_size=max_samples,
        stratify=y_train,
        random_state=random_state,
    )
    return x_subset, y_subset


def _safe_params(estimator: Any) -> str:
    params = estimator.get_params(deep=False)
    key_candidates = [
        "n_neighbors",
        "weights",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "learning_rate",
        "max_leaf_nodes",
        "var_smoothing",
    ]
    compact = {key: params[key] for key in key_candidates if key in params}
    if not compact:
        compact = {"type": type(estimator).__name__}
    return json.dumps(compact, ensure_ascii=True, sort_keys=True)


def _run_single_experiment(
    spec: ModelSpec,
    splits: Dict[str, pd.DataFrame],
    feature_columns: List[str],
) -> Dict[str, Any]:
    x_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]
    x_val = splits["val"][feature_columns]
    y_val = splits["val"][TARGET_COLUMN]
    x_test = splits["test"][feature_columns]
    y_test = splits["test"][TARGET_COLUMN]

    model = clone(spec.estimator)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    row = {
        "phase": spec.phase,
        "model_id": spec.model_id,
        "model_name": spec.model_name,
        "feature_set": "with_engineering",
        "num_features": len(feature_columns),
        "hypothesis": spec.hypothesis,
        "check_method": spec.check_method,
        "params": _safe_params(model),
        "best_params": "",
        "macro_f1_val": val_metrics["macro_f1"],
        "weighted_f1_val": val_metrics["weighted_f1"],
        "accuracy_val": val_metrics["accuracy"],
        "macro_f1_test": test_metrics["macro_f1"],
        "weighted_f1_test": test_metrics["weighted_f1"],
        "accuracy_test": test_metrics["accuracy"],
    }
    return {
        "row": row,
        "model": model,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_cp2_experiments(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    random_state: int = 42,
    tune_max_samples: int = 25000,
    tune_n_iter: int = 8,
) -> Dict[str, Any]:
    splits = load_processed_splits(processed_dir)
    feature_columns = resolve_cp2_features(splits["train"])

    y_train = splits["train"][TARGET_COLUMN]
    y_val = splits["val"][TARGET_COLUMN]
    y_test = splits["test"][TARGET_COLUMN]

    results: List[Dict[str, Any]] = []
    trained: Dict[str, Dict[str, Any]] = {}

    for spec in build_cp2_part1_specs(random_state=random_state):
        output = _run_single_experiment(spec=spec, splits=splits, feature_columns=feature_columns)
        results.append(output["row"])
        trained[spec.model_id] = output

    tuning_summaries: List[Dict[str, Any]] = []
    tune_specs = build_cp2_tune_specs(random_state=random_state, n_iter=tune_n_iter)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for tune_spec in tune_specs:
        x_train = splits["train"][feature_columns]
        x_tune, y_tune = _sample_for_tuning(
            x_train=x_train,
            y_train=y_train,
            random_state=random_state,
            max_samples=tune_max_samples,
        )

        search = RandomizedSearchCV(
            estimator=clone(tune_spec.estimator),
            param_distributions=tune_spec.param_distributions,
            n_iter=tune_spec.n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            random_state=random_state,
            refit=True,
        )
        search.fit(x_tune, y_tune)

        tuned_model = clone(search.best_estimator_)
        tuned_model.fit(x_train, y_train)

        val_pred = tuned_model.predict(splits["val"][feature_columns])
        test_pred = tuned_model.predict(splits["test"][feature_columns])
        val_metrics = evaluate_predictions(y_val, val_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)

        results.append(
            {
                "phase": "part2_tuning",
                "model_id": tune_spec.model_id,
                "model_name": tune_spec.model_name,
                "feature_set": "with_engineering",
                "num_features": len(feature_columns),
                "hypothesis": tune_spec.hypothesis,
                "check_method": tune_spec.check_method,
                "params": _safe_params(tuned_model),
                "best_params": json.dumps(search.best_params_, ensure_ascii=True, sort_keys=True),
                "macro_f1_val": val_metrics["macro_f1"],
                "weighted_f1_val": val_metrics["weighted_f1"],
                "accuracy_val": val_metrics["accuracy"],
                "macro_f1_test": test_metrics["macro_f1"],
                "weighted_f1_test": test_metrics["weighted_f1"],
                "accuracy_test": test_metrics["accuracy"],
            }
        )
        trained[tune_spec.model_id] = {
            "model": tuned_model,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        tuning_summaries.append(
            {
                "model_id": tune_spec.model_id,
                "model_name": tune_spec.model_name,
                "tune_samples": int(len(x_tune)),
                "cv_splits": 3,
                "n_iter": tune_spec.n_iter,
                "best_cv_macro_f1": float(search.best_score_),
                "best_params": search.best_params_,
            }
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1_val", ascending=False).reset_index(drop=True)
    best_model_id = str(results_df.loc[0, "model_id"])
    best_info = trained[best_model_id]
    best_val_metrics = best_info["val_metrics"]
    best_test_metrics = best_info["test_metrics"]

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    results_path = models_path / "cp2_results.csv"
    metrics_path = models_path / "cp2_metrics.json"
    best_model_path = models_path / "cp2_best_model.joblib"

    results_df.to_csv(results_path, index=False)
    joblib.dump(
        {
            "model_id": best_model_id,
            "target_column": TARGET_COLUMN,
            "features": feature_columns,
            "estimator": best_info["model"],
        },
        best_model_path,
    )

    metrics = {
        "best_model": {
            "model_id": best_model_id,
            "model_name": str(results_df.loc[0, "model_name"]),
            "phase": str(results_df.loc[0, "phase"]),
            "feature_set": "with_engineering",
            "features": feature_columns,
        },
        "validation_metrics_best_model": best_val_metrics,
        "test_metrics_best_model": best_test_metrics,
        "validation_classification_report_best_model": classification_report(
            y_val,
            best_info["val_pred"],
            output_dict=True,
            zero_division=0,
        ),
        "test_classification_report_best_model": classification_report(
            y_test,
            best_info["test_pred"],
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix_best_model": confusion_matrix(y_val, best_info["val_pred"]).tolist(),
        "test_confusion_matrix_best_model": confusion_matrix(y_test, best_info["test_pred"]).tolist(),
        "tuning_summaries": tuning_summaries,
        "results_table_path": str(results_path),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)

    return {
        "results_df": results_df,
        "best_model_id": best_model_id,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "results_path": str(results_path),
            "metrics_path": str(metrics_path),
            "best_model_path": str(best_model_path),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CP2 model experiments and hyperparameter tuning.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with processed train/val/test splits.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where CP2 artifacts will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tune-max-samples",
        type=int,
        default=25000,
        help="Maximum number of train rows used in RandomizedSearchCV (0 = full train).",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=8,
        help="Number of RandomizedSearch iterations for each tuned model.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output = run_cp2_experiments(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        random_state=args.random_state,
        tune_max_samples=args.tune_max_samples,
        tune_n_iter=args.tune_n_iter,
    )

    print("CP2 experiments completed.")
    print(f"Best model id: {output['best_model_id']}")
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
