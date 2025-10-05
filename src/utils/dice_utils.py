"""
Utilities for DiCE Counterfactual Explanations
Data processing, evaluation metrics, and helper functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union


class DiCEEvaluator:
    """Evaluator for DiCE counterfactual explanations."""
    
    def __init__(self, model, data):
        """Initialize evaluator.
        
        Args:
            model: Trained model
            data: Training/reference data
        """
        self.model = model
        self.data = data
    
    def evaluate_counterfactuals(self, query_instances, counterfactuals_list):
        """Evaluate quality of counterfactual explanations.
        
        Args:
            query_instances: Original instances
            counterfactuals_list: List of counterfactuals for each query
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'validity': [],      # Proportion of CFs with desired prediction
            'proximity': [],     # Average distance to original
            'diversity': [],     # Diversity among CFs
            'sparsity': [],      # Number of changed features
            'realism': []        # How realistic the CFs are
        }
        
        for i, (query, cfs) in enumerate(zip(query_instances, counterfactuals_list)):
            if isinstance(query, pd.DataFrame):
                query = query.iloc[0]
            elif isinstance(query, np.ndarray):
                query = pd.Series(query.flatten(), index=self.data.columns)
            
            # Get original prediction
            original_pred = self.model.predict(query.values.reshape(1, -1))[0]
            original_class = 1 if original_pred > 0.5 else 0
            desired_class = 1 - original_class
            
            # Evaluate each set of counterfactuals
            cf_metrics = self._evaluate_cf_set(query, cfs, desired_class)
            
            for metric, value in cf_metrics.items():
                metrics[metric].append(value)
        
        # Calculate summary statistics
        summary_metrics = {}
        for metric, values in metrics.items():
            summary_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary_metrics
    
    def _evaluate_cf_set(self, query, counterfactuals, desired_class):
        """Evaluate a set of counterfactuals for a single query."""
        if not counterfactuals:
            return {
                'validity': 0.0,
                'proximity': float('inf'),
                'diversity': 0.0,
                'sparsity': 0.0,
                'realism': 0.0
            }
        
        # Validity: proportion of CFs with desired prediction
        valid_cfs = 0
        proximities = []
        sparsities = []
        
        for cf in counterfactuals:
            cf_pred = self.model.predict(cf.values)[0]
            cf_class = 1 if cf_pred > 0.5 else 0
            
            if cf_class == desired_class:
                valid_cfs += 1
            
            # Proximity: L2 distance
            proximity = np.sqrt(np.sum((cf.iloc[0].values - query.values) ** 2))
            proximities.append(proximity)
            
            # Sparsity: number of changed features
            changed_features = np.sum(np.abs(cf.iloc[0].values - query.values) > 1e-6)
            sparsities.append(changed_features)
        
        validity = valid_cfs / len(counterfactuals)
        avg_proximity = np.mean(proximities)
        avg_sparsity = np.mean(sparsities)
        
        # Diversity: average pairwise distance between CFs
        diversity = self._calculate_diversity(counterfactuals)
        
        # Realism: how close CFs are to training data distribution
        realism = self._calculate_realism(counterfactuals)
        
        return {
            'validity': validity,
            'proximity': avg_proximity,
            'diversity': diversity,
            'sparsity': avg_sparsity,
            'realism': realism
        }
    
    def _calculate_diversity(self, counterfactuals):
        """Calculate diversity among counterfactuals."""
        if len(counterfactuals) < 2:
            return 0.0
        
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                cf1 = counterfactuals[i].iloc[0].values
                cf2 = counterfactuals[j].iloc[0].values
                distance = np.sqrt(np.sum((cf1 - cf2) ** 2))
                distances.append(distance)
        
        return np.mean(distances)
    
    def _calculate_realism(self, counterfactuals):
        """Calculate how realistic counterfactuals are."""
        # Use training data to define realistic ranges
        realism_scores = []
        
        for cf in counterfactuals:
            cf_values = cf.iloc[0]
            realism_score = 0.0
            
            for feature in self.data.columns:
                if feature not in cf_values.index:
                    continue
                
                cf_val = cf_values[feature]
                
                # Check if value is within reasonable range of training data
                train_min = self.data[feature].min()
                train_max = self.data[feature].max()
                train_mean = self.data[feature].mean()
                train_std = self.data[feature].std()
                
                # Check if within 3 standard deviations
                if abs(cf_val - train_mean) <= 3 * train_std:
                    realism_score += 1.0
                elif train_min <= cf_val <= train_max:
                    realism_score += 0.5
                # else: 0 points for this feature
            
            realism_scores.append(realism_score / len(cf_values))
        
        return np.mean(realism_scores)


class DataProcessor:
    """Data processing utilities for tabular data."""
    
    @staticmethod
    def detect_feature_types(data):
        """Automatically detect continuous and categorical features.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Tuple of (continuous_features, categorical_features)
        """
        continuous_features = []
        categorical_features = []
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Check if it's likely categorical (few unique values)
                unique_ratio = data[column].nunique() / len(data)
                if unique_ratio < 0.05 and data[column].nunique() < 20:
                    categorical_features.append(column)
                else:
                    continuous_features.append(column)
            else:
                categorical_features.append(column)
        
        return continuous_features, categorical_features
    
    @staticmethod
    def prepare_adult_dataset():
        """Prepare the Adult dataset for DiCE experiments.
        
        Returns:
            Processed DataFrame
        """
        # Create synthetic adult-like dataset
        np.random.seed(42)
        n_samples = 2000
        
        # Generate features
        age = np.random.normal(39, 13, n_samples).clip(17, 80)
        workclass = np.random.choice(['Private', 'Self-emp', 'Gov', 'Without-pay'], 
                                   n_samples, p=[0.7, 0.15, 0.1, 0.05])
        education_num = np.random.choice(range(1, 17), n_samples)
        marital_status = np.random.choice(['Married', 'Never-married', 'Divorced'], 
                                        n_samples, p=[0.5, 0.3, 0.2])
        occupation = np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 
                                     'Sales', 'Exec-managerial'], n_samples)
        relationship = np.random.choice(['Husband', 'Not-in-family', 'Own-child', 
                                       'Unmarried', 'Wife'], n_samples)
        race = np.random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Other'], 
                              n_samples, p=[0.85, 0.1, 0.03, 0.02])
        sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.67, 0.33])
        capital_gain = np.random.exponential(500, n_samples).clip(0, 99999)
        capital_loss = np.random.exponential(100, n_samples).clip(0, 4356)
        hours_per_week = np.random.normal(40, 12, n_samples).clip(1, 99)
        native_country = np.random.choice(['United-States', 'Other'], 
                                        n_samples, p=[0.9, 0.1])
        
        # Create target based on features (simplified logic)
        income_score = (age * 0.1 + 
                       education_num * 0.2 + 
                       hours_per_week * 0.05 + 
                       capital_gain * 0.0001 + 
                       (sex == 'Male') * 0.3 +
                       np.random.normal(0, 1, n_samples))
        
        income = (income_score > np.percentile(income_score, 75)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'workclass': workclass,
            'education-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': native_country,
            'income': income
        })
        
        return df
    
    @staticmethod
    def create_credit_dataset():
        """Create synthetic credit approval dataset.
        
        Returns:
            DataFrame with credit features
        """
        np.random.seed(42)
        n_samples = 1500
        
        # Generate correlated features
        income = np.random.lognormal(10, 0.5, n_samples)
        age = np.random.normal(35, 10, n_samples).clip(18, 70)
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        debt_ratio = np.random.beta(2, 5, n_samples)
        employment_years = np.random.exponential(5, n_samples).clip(0, 40)
        
        # Credit approval based on features
        approval_score = (
            np.log(income) * 0.3 +
            age * 0.01 +
            credit_score * 0.002 +
            employment_years * 0.05 +
            (1 - debt_ratio) * 2 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        approved = (approval_score > np.percentile(approval_score, 70)).astype(int)
        
        df = pd.DataFrame({
            'income': income,
            'age': age,
            'credit_score': credit_score,
            'debt_ratio': debt_ratio,
            'employment_years': employment_years,
            'approved': approved
        })
        
        return df


def plot_dice_results(evaluator_results, save_path=None):
    """Plot DiCE evaluation results.
    
    Args:
        evaluator_results: Results from DiCEEvaluator
        save_path: Optional path to save the plot
    """
    metrics = ['validity', 'proximity', 'diversity', 'sparsity', 'realism']
    means = [evaluator_results[m]['mean'] for m in metrics]
    stds = [evaluator_results[m]['std'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of mean values
    bars = ax1.bar(metrics, means, yerr=stds, capsize=5, 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    ax1.set_title('DiCE Evaluation Metrics (Mean ± Std)')
    ax1.set_ylabel('Metric Value')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    # Box plot showing distribution
    metric_data = []
    for metric in metrics:
        # Create synthetic data points for visualization
        mean_val = evaluator_results[metric]['mean']
        std_val = evaluator_results[metric]['std']
        data_points = np.random.normal(mean_val, std_val, 50)
        metric_data.append(data_points)
    
    ax2.boxplot(metric_data, labels=metrics)
    ax2.set_title('Distribution of Evaluation Metrics')
    ax2.set_ylabel('Metric Value')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample data
    processor = DataProcessor()
    data = processor.create_credit_dataset()
    
    print("DiCE utilities ready!")
    print(f"Sample dataset shape: {data.shape}")
    
    continuous_features, categorical_features = processor.detect_feature_types(data)
    print(f"Continuous features: {continuous_features}")
    print(f"Categorical features: {categorical_features}")