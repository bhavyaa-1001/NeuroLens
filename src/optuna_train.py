"""
Optuna Hyperparameter Tuning Module

This module handles hyperparameter optimization for XGBoost models
using Optuna for three different loss types: box_loss, dfl_loss, and class_loss.
"""

import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import json


def run_optuna(df_features: pd.DataFrame, target_column: str, n_trials: int = 200) -> dict:
    """
    Run Optuna hyperparameter optimization for XGBoost model.
    
    Args:
        df_features: DataFrame with features and target column
        target_column: Name of the target column (box_loss, dfl_loss, or class_loss)
        n_trials: Number of Optuna trials to run
        
    Returns:
        Dictionary containing best parameters and performance metrics
    """
    # Define features
    features = [
        "laplacian_variance",
        "snr",
        "mean_brightness",
        "rms_contrast",
        "entropy",
        "edge_density",
        "colorfulness_index"
    ]
    
    X = df_features[features]
    y = df_features[target_column]
    
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        return mae  # Minimize MAE
    
    # Run Optuna study
    print(f"🔍 Running Optuna Hyperparameter Tuning for {target_column}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n🏆 Best Parameters Found for {target_column}:")
    for key, val in best_params.items():
        print(f"{key}: {val}")
    print(f"Best MAE: {best_value:.6f}")
    
    # Train final model with best parameters
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate final model
    y_pred = final_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    print(f"\n📊 Final Model Performance for {target_column}:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'best_params': best_params,
        'best_mae': best_value,
        'final_mae': mae,
        'r2_score': r2,
        'features': features,
        'X_train': X_train,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_valid': y_valid
    }


def run_optuna_for_all_losses(df_features: pd.DataFrame, n_trials: int = 200) -> dict:
    """
    Run Optuna optimization for all three loss types.
    
    Args:
        df_features: DataFrame with features and all loss columns
        n_trials: Number of Optuna trials to run for each loss type
        
    Returns:
        Dictionary containing results for all loss types
    """
    results = {}
    
    loss_types = ['box_loss', 'dfl_loss', 'class_loss']
    
    for loss_type in loss_types:
        if loss_type in df_features.columns:
            print(f"\n{'='*50}")
            print(f"Processing {loss_type.upper()}")
            print(f"{'='*50}")
            
            results[loss_type] = run_optuna(df_features, loss_type, n_trials)
        else:
            print(f"Warning: {loss_type} column not found in DataFrame")
            results[loss_type] = None
    
    return results


def save_optuna_results(results: dict, output_dir: str) -> None:
    """
    Save Optuna results to JSON files.
    
    Args:
        results: Dictionary containing Optuna results
        output_dir: Directory to save the results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for loss_type, result in results.items():
        if result is not None:
            # Save only the parameters and metrics (not the data)
            save_data = {
                'best_params': result['best_params'],
                'best_mae': result['best_mae'],
                'final_mae': result['final_mae'],
                'r2_score': result['r2_score'],
                'features': result['features']
            }
            
            output_file = output_dir / f"optuna_results_{loss_type}.json"
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Optuna results for {loss_type} saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    # This would be called from the main pipeline
    print("Optuna training module loaded successfully")
