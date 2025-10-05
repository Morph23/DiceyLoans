"""
DiCE: Diverse Counterfactual Explanations for Tabular Data
A comprehensive framework for generating actionable counterfactual explanations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .src.models.tabular_predictor import TabularPredictor
from .src.counterfactuals.dice_explainer import DiceExplainer
from .src.utils.data_loader import DatasetBundle, load_tabular_data

__all__ = [
    "TabularPredictor",
    "DiceExplainer",
    "DatasetBundle",
    "load_tabular_data",
]