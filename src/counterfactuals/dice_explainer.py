from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import dice_ml

from src.models.tabular_predictor import TabularPredictor
from src.utils.data_loader import DatasetBundle


@dataclass
class FeatureChange:
    name: str
    original: object
    counterfactual: object
    delta: float


@dataclass
class CounterfactualExplanation:
    index: int
    original_probability: float
    original_label: int
    original_label_name: object
    counterfactual_probability: float
    counterfactual_label: int
    counterfactual_label_name: object
    generation_method: str
    counterfactual_processed: pd.Series
    counterfactual_original: Dict[str, object]
    changes: List[FeatureChange]


class DiceExplainer:
    def __init__(
        self,
        predictor: TabularPredictor,
        dataset: DatasetBundle,
        method: str = "random",
        threshold: float = 0.5,
    ):
        self.predictor = predictor
        self.dataset = dataset
        self.threshold = threshold
        self.method = method
        training_df = dataset.data.copy()
        training_df[dataset.target_column] = dataset.target
        self.data_interface = dice_ml.Data(
            dataframe=training_df,
            continuous_features=dataset.feature_names,
            outcome_name=dataset.target_column,
        )
        self.model_interface = dice_ml.Model(
            model=predictor.pipeline,
            backend="sklearn",
        )
        self.dice = self._build_dice(method)

    def generate(
        self,
        index: int,
        desired_class: Optional[Union[int, str]] = None,
        total_cfs: int = 5,
    ) -> CounterfactualExplanation:
        query_processed = self.dataset.data.loc[[index]]
        original_probability = float(self.predictor.predict_proba(query_processed)[0])
        original_label = int(self.predictor.predict_label(query_processed)[0])
        target_class = (
            self._to_numeric_label(desired_class)
            if desired_class is not None
            else 1 - original_label
        )
        processed_original = query_processed.iloc[0]
        candidate_records = self._score_candidates(
            processed_original,
            query_processed,
            target_class,
            total_cfs,
            self.dice,
        )

        if not self._has_valid(candidate_records, target_class) and self.method != "genetic":
            fallback_dice = self._build_dice("genetic")
            candidate_records.extend(
                self._score_candidates(
                    processed_original,
                    query_processed,
                    target_class,
                    total_cfs,
                    fallback_dice,
                    method="genetic",
                )
            )

        valid_records = [record for record in candidate_records if record["label"] == target_class]
        if not valid_records:
            raise RuntimeError("No counterfactual satisfied the desired class")

        best_record = min(
            valid_records,
            key=lambda item: (item["distance"], -item["probability"]),
        )
        best_row = best_record["row"]
        counterfactual_probability = best_record["probability"]
        counterfactual_label = best_record["label"]
        generation_method = best_record["method"]
        original_raw = self.dataset.get_original_row(index)
        original_categorical = self._decode_categorical(processed_original)
        counterfactual_categorical = self._decode_categorical(best_row)
        changes = self._collect_changes(
            original_raw,
            processed_original,
            best_row,
            original_categorical,
            counterfactual_categorical,
        )
        counterfactual_original = self._compose_original(counterfactual_categorical, best_row)
        return CounterfactualExplanation(
            index=index,
            original_probability=original_probability,
            original_label=original_label,
            original_label_name=self.dataset.target_mapping[original_label],
            counterfactual_probability=counterfactual_probability,
            counterfactual_label=counterfactual_label,
            counterfactual_label_name=self.dataset.target_mapping[counterfactual_label],
            generation_method=generation_method,
            counterfactual_processed=best_row,
            counterfactual_original=counterfactual_original,
            changes=changes,
        )

    def _to_numeric_label(self, value: Union[int, str]) -> int:
        if isinstance(value, str):
            key = value.strip()
            if key not in self.dataset.target_inverse_mapping:
                available = ", ".join(map(str, self.dataset.target_mapping.values()))
                raise ValueError(f"Unknown target label '{key}'. Available labels: {available}")
            return self.dataset.target_inverse_mapping[key]
        return int(value)

    def _build_dice(self, method: str) -> dice_ml.Dice:
        return dice_ml.Dice(
            self.data_interface,
            self.model_interface,
            method=method,
        )

    def _score_candidates(
        self,
        processed_original: pd.Series,
        query_processed: pd.DataFrame,
        target_class: int,
        total_cfs: int,
        dice_engine: dice_ml.Dice,
        method: Optional[str] = None,
    ) -> List[Dict[str, Union[float, int, pd.Series, str]]]:
        cf_result = dice_engine.generate_counterfactuals(
            query_processed,
            total_CFs=total_cfs,
            desired_class=target_class,
            features_to_vary="all",
            verbose=False,
        )
        candidates: List[pd.Series] = []
        for item in cf_result.cf_examples_list:
            frame = item.final_cfs_df
            if self.dataset.target_column in frame.columns:
                frame = frame.drop(columns=[self.dataset.target_column])
            frame = frame[self.dataset.feature_names]
            for _, row in frame.iterrows():
                candidates.append(row.astype(float))
        records: List[Dict[str, Union[float, int, pd.Series, str]]] = []
        for candidate in candidates:
            normalized = self._normalize_candidate(candidate.copy())
            probability = float(self.predictor.predict_proba(normalized.to_frame().T)[0])
            label = int(probability >= self.threshold)
            distance = self._distance(processed_original, normalized)
            records.append(
                {
                    "row": normalized,
                    "probability": probability,
                    "label": label,
                    "distance": distance,
                    "method": method or self.method,
                }
            )
        return records

    @staticmethod
    def _has_valid(records: List[Dict[str, Union[float, int, pd.Series, str]]], target_class: int) -> bool:
        return any(record["label"] == target_class for record in records)

    def _normalize_candidate(self, row: pd.Series) -> pd.Series:
        for column in self.dataset.numeric_features:
            if column in row.index:
                minimum = float(self.dataset.data[column].min())
                maximum = float(self.dataset.data[column].max())
                row.loc[column] = float(np.clip(row.loc[column], minimum, maximum))
        for mapping in self.dataset.categorical_map.values():
            columns = [col for col in mapping.keys() if col in row.index]
            if not columns:
                continue
            values = row[columns].to_numpy()
            max_index = int(np.argmax(values))
            for i, column in enumerate(columns):
                row.loc[column] = 1.0 if i == max_index else 0.0
        return row

    def _distance(self, original: pd.Series, candidate: pd.Series) -> float:
        numeric_distance = 0.0
        if self.dataset.numeric_features:
            numeric_distance = float(
                np.abs(
                    candidate[self.dataset.numeric_features].values
                    - original[self.dataset.numeric_features].values
                ).sum()
            )
        categorical_distance = 0.0
        if self.dataset.categorical_map:
            original_map = self._decode_categorical(original)
            candidate_map = self._decode_categorical(candidate)
            for key, value in original_map.items():
                if candidate_map.get(key, value) != value:
                    categorical_distance += 1.0
        return numeric_distance + categorical_distance

    def _decode_categorical(self, row: pd.Series) -> Dict[str, object]:
        decoded: Dict[str, object] = {}
        for feature, mapping in self.dataset.categorical_map.items():
            columns = [col for col in mapping.keys() if col in row.index]
            if not columns:
                continue
            values = row[columns].to_numpy()
            max_index = int(np.argmax(values))
            decoded[feature] = mapping[columns[max_index]]
        return decoded

    def _collect_changes(
        self,
        original_raw: pd.Series,
        original_processed: pd.Series,
        candidate: pd.Series,
        original_categorical: Dict[str, object],
        counterfactual_categorical: Dict[str, object],
    ) -> List[FeatureChange]:
        changes: List[FeatureChange] = []
        for feature in self.dataset.numeric_features:
            if feature not in candidate.index:
                continue
            original_value = float(original_raw.get(feature, original_processed.get(feature, 0.0)))
            candidate_value = float(candidate[feature])
            delta = candidate_value - original_value
            if abs(delta) > 1e-6:
                changes.append(
                    FeatureChange(
                        name=feature,
                        original=original_value,
                        counterfactual=candidate_value,
                        delta=delta,
                    )
                )
        for feature, original_value in original_categorical.items():
            candidate_value = counterfactual_categorical.get(feature, original_value)
            if feature in original_raw.index and pd.notna(original_raw[feature]):
                original_value = original_raw[feature]
            if candidate_value != original_value:
                changes.append(
                    FeatureChange(
                        name=feature,
                        original=original_value,
                        counterfactual=candidate_value,
                        delta=1.0,
                    )
                )
        return changes

    def _compose_original(
        self,
        categorical_values: Dict[str, object],
        candidate: pd.Series,
    ) -> Dict[str, object]:
        combined: Dict[str, object] = {}
        for feature in self.dataset.numeric_features:
            if feature in candidate.index:
                combined[feature] = float(candidate[feature])
        combined.update(categorical_values)
        return combined