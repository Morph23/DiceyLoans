from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import DatasetBundle


@dataclass
class PredictionResult:
    probabilities: np.ndarray
    labels: np.ndarray


class TabularPredictor:
    def __init__(
        self,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        c_grid: Optional[Sequence[float]] = None,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: Optional[int] = None,
    ):
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.tune_hyperparameters = tune_hyperparameters
        self.c_grid = c_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.feature_names: Optional[Iterable[str]] = None
        self.best_params_: Optional[dict[str, Any]] = None
        self.cv_results_: Optional[dict[str, Any]] = None
        self.fitted = False

    def fit(self, bundle: DatasetBundle) -> "TabularPredictor":
        if self.tune_hyperparameters:
            param_grid = {
                "model__C": self.c_grid
                if self.c_grid is not None
                else [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__class_weight": [None, "balanced"],
            }
            search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit=True,
            )
            search.fit(bundle.X_train, bundle.y_train)
            self.pipeline = search.best_estimator_
            self.best_params_ = search.best_params_
            self.cv_results_ = search.cv_results_
        else:
            self.pipeline.fit(bundle.X_train, bundle.y_train)
            model: LogisticRegression = self.pipeline.named_steps["model"]
            self.best_params_ = {"model__C": model.C, "model__class_weight": model.class_weight}
            self.cv_results_ = None
        self.feature_names = bundle.feature_names
        self.fitted = True
        return self

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        if self.feature_names is None:
            raise RuntimeError("Model is not fitted")
        return pd.DataFrame(X, columns=self.feature_names)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model is not fitted")
        data = self._ensure_dataframe(X)
        probabilities = self.pipeline.predict_proba(data)
        return probabilities[:, 1]

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def evaluate(self, bundle: DatasetBundle, threshold: float = 0.5) -> float:
        probabilities = self.predict_proba(bundle.X_test)
        labels = (probabilities >= threshold).astype(int)
        return accuracy_score(bundle.y_test, labels)

    def predict_with_result(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> PredictionResult:
        probabilities = self.predict_proba(X)
        labels = (probabilities >= threshold).astype(int)
        return PredictionResult(probabilities=probabilities, labels=labels)