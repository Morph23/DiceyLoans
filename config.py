"""
DiCE Configuration Settings
"""

# DiCE Algorithm Parameters
DICE_CONFIG = {
    'total_CFs': 5,              # Number of counterfactuals to generate
    'proximity_weight': 1.0,      # Weight for proximity loss (closeness to original)
    'diversity_weight': 1.0,      # Weight for diversity loss (variety among CFs)
    'categorical_penalty': 2.0,   # Penalty for changing categorical features
    'max_iterations': 1000,       # Maximum optimization iterations
    'learning_rate': 0.01,        # Learning rate for gradient descent
    'desired_class': None,        # Target class (None for opposite of original)
}

# Model Training Parameters
MODEL_CONFIG = {
    'hidden_layers': [128, 64, 32],  # Neural network architecture
    'dropout_rate': 0.3,             # Dropout rate for regularization
    'epochs': 100,                   # Training epochs
    'batch_size': 32,                # Batch size
    'validation_split': 0.2,         # Validation split ratio
    'random_state': 42,              # Random seed for reproducibility
}

# Data Processing Parameters
DATA_CONFIG = {
    'test_size': 0.2,               # Test set size
    'categorical_threshold': 0.05,   # Threshold for detecting categorical features
    'max_categorical_unique': 20,    # Max unique values for categorical detection
}

# Visualization Parameters
VIS_CONFIG = {
    'figure_size': (12, 8),         # Default figure size
    'dpi': 300,                     # Image resolution
    'style': 'seaborn-v0_8',        # Plot style
    'color_palette': 'Set3',        # Color palette for plots
}

# File Paths
PATHS = {
    'data_dir': 'data/',
    'models_dir': 'models/',
    'results_dir': 'results/',
    'notebooks_dir': 'notebooks/',
}

# Evaluation Thresholds
EVAL_CONFIG = {
    'validity_threshold': 0.8,      # Minimum validity score
    'proximity_threshold': 1.0,     # Maximum acceptable proximity
    'diversity_threshold': 0.1,     # Minimum diversity score
    'realism_threshold': 0.7,       # Minimum realism score
}