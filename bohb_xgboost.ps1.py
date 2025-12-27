import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # For real data loading

# Check GPU availability
import torch
use_gpu = torch.cuda.is_available()
print(f"GPU available: {use_gpu}")

# Load real data (Iris dataset for example; replace with your data file)
data = load_iris()
X = data.data
y = data.target
print(f"Loaded real data: {X.shape[0]} samples, {X.shape[1]} features")

# Universal objective function (supports XGBoost or custom models)
def objective(trial, model_type='xgboost'):
    # Common hyperparameters (adjust for your model)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    min_child_weight = trial.suggest_float('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0, 5)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
    
    # GPU acceleration if available
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    
    # Create model based on type
    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            random_state=42,
            device='gpu' if use_gpu else 'cpu'
        )
    elif model_type == 'custom':  # Example custom model - replace with your own
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Cross-validation with progress
    score = cross_val_score(model, X, y, n_jobs=-1, cv=3).mean()
    print(f'Trial {trial.number}: score={score:.4f}')
    return score

# Create study with storage for persistence
storage = 'sqlite:///optuna_study.db'  # Saves progress
study = optuna.create_study(direction='maximize', study_name='xgboost_opt', storage=storage, load_if_exists=True)

# Progress tracker callback
def callback(study, trial):
    print(f'Trial {trial.number}: value={trial.value:.4f}, params={trial.params}')
    if trial.number % 10 == 0:
        print(f"Progress: {trial.number} trials complete")

# Optimize with GPU (50 trials - adjust as needed)
study.optimize(lambda trial: objective(trial, model_type='xgboost'), n_trials=50, callbacks=[callback])

# Print best results
print(f'Best trial: value={study.best_value:.4f}, params={study.best_params}')

# Visualize results
fig1 = plot_optimization_history(study)
fig1.show()
fig2 = plot_param_importances(study)
fig2.show()
print('Optimization complete! Best model ready.')
