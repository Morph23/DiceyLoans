from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.counterfactuals.dice_explainer import DiceExplainer
from src.models.tabular_predictor import TabularPredictor
from src.utils.data_loader import load_tabular_data


def test_counterfactual_generation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_samples = 300
    income = rng.normal(55000, 8000, n_samples)
    age = rng.integers(21, 65, n_samples)
    credit_score = rng.integers(480, 820, n_samples)
    employment_status = rng.choice(
        ["employed", "self-employed", "unemployed"],
        n_samples,
        p=[0.6, 0.25, 0.15],
    )
    marital_status = rng.choice(["single", "married", "divorced"], n_samples)
    loan_amount = rng.normal(18000, 4000, n_samples)
    status_boost = pd.Series(employment_status).map(
        {"employed": 1.5, "self-employed": 0.8, "unemployed": -1.2}
    ).to_numpy()
    marital_boost = pd.Series(marital_status).map(
        {"married": 0.6, "single": 0.2, "divorced": -0.4}
    ).to_numpy()
    score = (
        0.00004 * income
        + 0.015 * credit_score
        - 0.03 * age
        - 0.00006 * loan_amount
        + status_boost
        + marital_boost
        - 7
    )
    probability = 1 / (1 + np.exp(-score))
    loan_approved = (probability > rng.random(n_samples)).astype(int)
    frame = pd.DataFrame(
        {
            "income": income,
            "age": age,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_status": employment_status,
            "marital_status": marital_status,
            "loan_approved": loan_approved,
        }
    )
    csv_path = tmp_path / "loan.csv"
    frame.to_csv(csv_path, index=False)
    bundle = load_tabular_data(csv_path, target_column="loan_approved")
    predictor = TabularPredictor().fit(bundle)
    explainer = DiceExplainer(predictor, bundle)
    labels = predictor.predict_label(bundle.X_test)
    candidates = [idx for idx, label in zip(bundle.X_test.index, labels) if label == 0]
    if not candidates:
        candidates = [bundle.X_test.index[0]]
    explanation = explainer.generate(candidates[0], desired_class=1, total_cfs=3)
    assert explanation.counterfactual_label == 1
    assert explanation.changes


def test_predictor_tuning(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_samples = 200
    income = rng.normal(60000, 5000, n_samples)
    debt = rng.normal(15000, 3000, n_samples)
    credit_score = rng.integers(500, 820, n_samples)
    approvals = (
        0.00003 * income
        - 0.00005 * debt
        + 0.012 * credit_score
        - 8.5
    )
    probability = 1 / (1 + np.exp(-approvals))
    labels = (probability > rng.random(n_samples)).astype(int)
    frame = pd.DataFrame(
        {
            "income": income,
            "debt": debt,
            "credit_score": credit_score,
            "loan_status": labels,
        }
    )
    csv_path = tmp_path / "loan_tuning.csv"
    frame.to_csv(csv_path, index=False)
    bundle = load_tabular_data(csv_path, target_column="loan_status")
    predictor = TabularPredictor(
        tune_hyperparameters=True,
        c_grid=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        n_jobs=1,
    )
    predictor.fit(bundle)
    assert predictor.best_params_ is not None
    assert "model__C" in predictor.best_params_
    preds = predictor.predict_label(bundle.X_test)
    assert set(preds).issubset({0, 1})
