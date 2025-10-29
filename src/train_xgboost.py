"""
XGBoost Training Module

This module handles training XGBoost models for predicting YOLO losses
using optimized hyperparameters from Optuna.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import json
import pickle


def train_xgboost(features: pd.DataFrame, params: dict, target_column: str) -> str:
    """
    Train XGBoost model with given parameters.
    
    Args:
        features: DataFrame with features and target column
        params: Dictionary of hyperparameters
        target_column: Name of the target column
        
    Returns:
        Path to the saved model file
    """
    # Define features
    feature_columns = [
        "laplacian_variance",
        "snr",
        "mean_brightness",
        "rms_contrast",
        "entropy",
        "edge_density",
        "colorfulness_index"
    ]
    
    X = features[feature_columns]
    y = features[target_column]
    
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    # Evaluate model
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    print(f"📊 Model Performance for {target_column}:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    model_path = f"models/xgb_optuna_{target_column}.json"
    model.save_model(model_path)
    
    # Also save as pickle for easier loading
    pickle_path = f"models/xgb_optuna_{target_column}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved as {model_path} and {pickle_path}")
    
    return model_path


def train_all_models(features: pd.DataFrame, optuna_results: dict) -> dict:
    """
    Train XGBoost models for all loss types using Optuna results.
    
    Args:
        features: DataFrame with features and all loss columns
        optuna_results: Dictionary containing Optuna results for each loss type
        
    Returns:
        Dictionary containing model paths for each loss type
    """
    model_paths = {}
    
    loss_types = ['box_loss', 'dfl_loss', 'class_loss']
    
    for loss_type in loss_types:
        if loss_type in features.columns and optuna_results.get(loss_type) is not None:
            print(f"\n{'='*50}")
            print(f"Training XGBoost model for {loss_type.upper()}")
            print(f"{'='*50}")
            
            best_params = optuna_results[loss_type]['best_params']
            model_path = train_xgboost(features, best_params, loss_type)
            model_paths[loss_type] = model_path
        else:
            print(f"Warning: Skipping {loss_type} - no data or Optuna results available")
            model_paths[loss_type] = None
    
    return model_paths


def load_model(model_path: str) -> XGBRegressor:
    """
    Load a trained XGBoost model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded XGBoost model
    """
    model = XGBRegressor()
    model.load_model(model_path)
    return model


def get_feature_importance(model: XGBRegressor, feature_names: list) -> dict:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    importance_scores = model.feature_importances_
    return dict(zip(feature_names, importance_scores))


def save_model_metadata(model_paths: dict, optuna_results: dict, output_file: str) -> None:
    """
    Save model metadata including paths and performance metrics.
    
    Args:
        model_paths: Dictionary of model paths
        optuna_results: Dictionary of Optuna results
        output_file: Path to save metadata
    """
    metadata = {
        'model_paths': model_paths,
        'performance_metrics': {}
    }
    
    for loss_type in ['box_loss', 'dfl_loss', 'class_loss']:
        if optuna_results.get(loss_type) is not None:
            metadata['performance_metrics'][loss_type] = {
                'best_mae': optuna_results[loss_type]['best_mae'],
                'final_mae': optuna_results[loss_type]['final_mae'],
                'r2_score': optuna_results[loss_type]['r2_score'],
                'best_params': optuna_results[loss_type]['best_params']
            }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("XGBoost training module loaded successfully")
