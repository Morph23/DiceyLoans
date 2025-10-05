from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    target: pd.Series
    target_original: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    categorical_map: Dict[str, Dict[str, str]]
    target_column: str
    source_path: Path
    original_features: pd.DataFrame
    target_mapping: Dict[int, Any]
    target_inverse_mapping: Dict[Any, int]
    positive_label: Any
    positive_class: int

    def get_processed_row(self, index: int) -> pd.Series:
        return self.data.iloc[index]

    def get_original_row(self, index: int) -> pd.Series:
        return self.original_features.iloc[index]


def _infer_categorical_columns(
    data: pd.DataFrame, provided: Optional[Iterable[str]] = None
) -> List[str]:
    if provided:
        return [col for col in provided if col in data.columns]

    categorical_cols: List[str] = []
    for column in data.columns:
        dtype = data[column].dtype
        if dtype.name in {"object", "category", "bool"}:
            categorical_cols.append(column)
            continue

        if np.issubdtype(dtype, np.integer):
            unique_count = data[column].nunique(dropna=True)
            if unique_count <= 10:
                categorical_cols.append(column)

    return categorical_cols


def load_tabular_data(
    csv_path: str | Path,
    target_column: str,
    *,
    categorical_columns: Optional[Iterable[str]] = None,
    positive_label: Optional[Any] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetBundle:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw = pd.read_csv(csv_path, skipinitialspace=True)
    if target_column not in raw.columns:
        available = ", ".join(raw.columns)
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {available}"
        )

    target_series = raw[target_column]
    if target_series.dtype == object:
        target_clean = target_series.astype(str).str.strip()
    else:
        target_clean = target_series

    unique_values = list(pd.unique(target_clean))
    if len(unique_values) != 2:
        raise ValueError("Target column must contain exactly two unique values")

    if positive_label is None:
        if np.issubdtype(target_clean.dtype, np.number):
            positive_label_value = max(unique_values)
        else:
            raise ValueError("positive_label must be provided for non-numeric target columns")
    else:
        positive_label_value = positive_label if not isinstance(positive_label, str) else positive_label.strip()
        if positive_label_value not in unique_values:
            available = ", ".join(map(str, unique_values))
            raise ValueError(f"positive_label '{positive_label_value}' not found in target values: {available}")

    negative_label_value = next(value for value in unique_values if value != positive_label_value)
    value_to_int = {negative_label_value: 0, positive_label_value: 1}
    target_numeric = target_clean.map(value_to_int).astype(int)
    target_original = target_clean

    features = raw.drop(columns=[target_column])
    categorical_cols = _infer_categorical_columns(features, categorical_columns)
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found in the dataset.")

    preprocessor = ColumnTransformer(transformers=transformers)
    transformed = preprocessor.fit_transform(features)

    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"]["encoder"]
        categorical_feature_names = encoder.get_feature_names_out(categorical_cols)

        categorical_map: Dict[str, Dict[str, str]] = {}
        cursor = 0
        for column, categories in zip(categorical_cols, encoder.categories_):
            width = len(categories)
            column_names = categorical_feature_names[cursor : cursor + width]
            categorical_map[column] = {
                column_name: category
                for column_name, category in zip(column_names, categories)
            }
            cursor += width
    else:
        categorical_feature_names = np.array([])
        categorical_map = {}

    feature_names = list(numeric_cols) + categorical_feature_names.tolist()

    processed = pd.DataFrame(transformed, columns=feature_names, index=raw.index)

    stratify_target = target_numeric if stratify and target_numeric.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        processed,
        target_numeric,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    target_mapping = {0: negative_label_value, 1: positive_label_value}
    target_inverse_mapping = {value: key for key, value in target_mapping.items()}

    bundle = DatasetBundle(
        data=processed,
        target=target_numeric,
        target_original=target_original,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_features=numeric_cols,
        categorical_features=categorical_feature_names.tolist(),
        categorical_map=categorical_map,
        target_column=target_column,
        source_path=csv_path,
        original_features=features,
        target_mapping=target_mapping,
        target_inverse_mapping=target_inverse_mapping,
        positive_label=positive_label_value,
        positive_class=1,
    )

    return bundle