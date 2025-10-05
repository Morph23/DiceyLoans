from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np
from pathlib import Path

from src.counterfactuals.dice_explainer import DiceExplainer
from src.models.tabular_predictor import TabularPredictor
from src.utils.data_loader import DatasetBundle, load_tabular_data


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "loan_approval_dataset.csv"
DEFAULT_TARGET_COLUMN = "loan_status"
DEFAULT_POSITIVE_LABEL = "Approved"
DEFAULT_DICE_METHOD = "random"

def _select_candidate_index(
    predictor: TabularPredictor,
    bundle: DatasetBundle,
    desired_class: int,
    threshold: float,
) -> int:
    scores = predictor.predict_proba(bundle.X_test)
    labels = (scores >= threshold).astype(int)
    for idx, label in zip(bundle.X_test.index, labels):
        if label != desired_class:
            return idx
    pool = bundle.X_test.index if desired_class == 0 else bundle.X_train.index
    return next(iter(pool))


def _resolve_desired_class(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    original_label: int,
) -> int:
    if args.desired is not None:
        return int(args.desired)
    if args.desired_label is not None:
        label_key = args.desired_label.strip()
        if label_key not in bundle.target_inverse_mapping:
            available = ", ".join(map(str, bundle.target_mapping.values()))
            raise ValueError(
                f"Desired label '{label_key}' not found in target values: {available}"
            )
        return bundle.target_inverse_mapping[label_key]
    return 1 - original_label


def _to_python(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate minimal counterfactual explanations for tabular datasets.",
    )
    parser.add_argument("--csv", default=str(DEFAULT_DATASET_PATH), help="Path to the input CSV file")
    parser.add_argument("--target", default=DEFAULT_TARGET_COLUMN, help="Target column in the dataset")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="Label treated as the positive outcome")
    parser.add_argument("--index", type=int, help="Row index to explain from the dataset")
    parser.add_argument("--desired", type=int, help="Desired target class identifier (default: flip original)")
    parser.add_argument("--desired-label", help="Desired target label (overrides --desired; default: flip original)")
    parser.add_argument("--total", type=int, default=5, help="Number of counterfactual candidates")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument(
        "--method",
        default=DEFAULT_DICE_METHOD,
        choices=["random", "kdtree", "genetic"],
        help="DiCE generation method",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning for the logistic regression model",
    )
    parser.add_argument(
        "--c-grid",
        help="Comma-separated list of C values for tuning (requires --tune)",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of folds for hyperparameter tuning")
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        help="Scoring metric for hyperparameter tuning (default: roc_auc)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel jobs for tuning (default: scikit-learn default)",
    )
    args = parser.parse_args()

    bundle = load_tabular_data(args.csv, args.target, positive_label=args.positive_label)
    c_grid = None
    if args.c_grid:
        try:
            c_grid = [float(value.strip()) for value in args.c_grid.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("--c-grid must be a comma-separated list of numeric values") from exc
        if not c_grid:
            raise ValueError("--c-grid requires at least one numeric value")

    predictor = TabularPredictor(
        tune_hyperparameters=args.tune,
        c_grid=c_grid,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
    )
    predictor.fit(bundle)
    accuracy = float(predictor.evaluate(bundle, threshold=args.threshold))
    explainer = DiceExplainer(predictor, bundle, method=args.method, threshold=args.threshold)

    def score_index(row_index: int) -> tuple[float, int]:
        row = bundle.data.loc[[row_index]]
        probability = float(predictor.predict_proba(row)[0])
        label = int(probability >= args.threshold)
        return probability, label

    if args.index is not None:
        index = args.index
    else:
        index = next(iter(bundle.X_test.index))

    original_probability, original_label = score_index(index)
    desired_numeric = _resolve_desired_class(args, bundle, original_label)

    if args.index is None and original_label == desired_numeric:
        index = _select_candidate_index(predictor, bundle, desired_numeric, args.threshold)
        original_probability, original_label = score_index(index)
        desired_numeric = _resolve_desired_class(args, bundle, original_label)

    explanation = explainer.generate(
        index=index,
        desired_class=desired_numeric,
        total_cfs=args.total,
    )

    changes_payload = [
        {
            "feature": change.name,
            "original": _to_python(change.original),
            "counterfactual": _to_python(change.counterfactual),
            "delta": _to_python(change.delta),
        }
        for change in explanation.changes
    ]

    original_features = bundle.get_original_row(index).to_dict()
    original_payload = {key: _to_python(value) for key, value in original_features.items()}
    counterfactual_payload = {
        key: _to_python(value)
        for key, value in explanation.counterfactual_original.items()
    }

    summary: Dict[str, Any] = {
        "csv_path": str(args.csv),
        "target_column": args.target,
        "index": index,
        "model_accuracy": accuracy,
        "dice_method": args.method,
        "generation_method": explanation.generation_method,
        "desired_label": bundle.target_mapping[desired_numeric],
    "tuned": args.tune,
    "model_best_params": predictor.best_params_,
        "original_probability": explanation.original_probability,
        "original_label": explanation.original_label,
        "original_label_name": explanation.original_label_name,
        "counterfactual_probability": explanation.counterfactual_probability,
        "counterfactual_label": explanation.counterfactual_label,
        "counterfactual_label_name": explanation.counterfactual_label_name,
        "original_features": original_payload,
        "counterfactual_features": counterfactual_payload,
        "changes": changes_payload,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
