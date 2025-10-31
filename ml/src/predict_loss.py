"""
Loss Prediction Module

This module handles predicting YOLO losses (box_loss, dfl_loss, class_loss)
using trained XGBoost models.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path
import json


def predict_loss(model: XGBRegressor, features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict losses using a trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        features: DataFrame with feature columns
        
    Returns:
        DataFrame with predictions
    """
    # Define canonical feature columns expected by the model
    feature_columns = [
        "laplacian_variance",
        "snr",
        "mean_brightness",
        "rms_contrast",
        "entropy",
        "edge_density",
        "colorfulness_index"
    ]

    # Allow common alternative/alias names from different extractors
    # If a canonical column is missing but an alias exists, copy it under the expected name
    alias_map = {
        "laplacian_variance": [
            "laplacian",
            "variance_of_laplacian",
            "lap_var",
        ],
        "snr": [
            "snr_db",
            "signal_to_noise_ratio",
        ],
        "mean_brightness": [
            "avg_brightness",
            "brightness_mean",
            "mean_intensity",
        ],
        "rms_contrast": [
            "contrast_rms",
        ],
        "entropy": [
            "image_entropy",
        ],
        "edge_density": [
            "edges_density",
        ],
        "colorfulness_index": [
            "colorfulness",
            "colorfulness_metric",
        ],
    }

    # Create a working copy so we can add/rename without mutating the caller's df
    features = features.copy()

    for canonical_name, alternatives in alias_map.items():
        if canonical_name not in features.columns:
            # Find the first available alias and copy it to the canonical column name
            for alt in alternatives:
                if alt in features.columns:
                    features[canonical_name] = features[alt]
                    break
    
    # Ensure all required features are present after alias backfilling
    missing_features = [col for col in feature_columns if col not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features
    X = features[feature_columns]
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create result DataFrame
    result_df = features.copy()
    result_df['predicted_loss'] = predictions
    
    return result_df


def predict_all_losses(features: pd.DataFrame, model_paths: dict) -> pd.DataFrame:
    """
    Predict all three types of losses using trained models.
    
    Args:
        features: DataFrame with feature columns
        model_paths: Dictionary mapping loss types to model paths
        
    Returns:
        DataFrame with predictions for all loss types
    """
    result_df = features.copy()
    
    loss_types = ['box_loss', 'dfl_loss', 'class_loss']
    
    for loss_type in loss_types:
        model_path = model_paths.get(loss_type)
        if model_path and Path(model_path).exists():
            try:
                # Load model
                model = XGBRegressor()
                model.load_model(model_path)
                
                # Make predictions
                predictions = predict_loss(model, features)
                result_df[f'predicted_{loss_type}'] = predictions['predicted_loss']
                
                print(f"✅ Predictions completed for {loss_type}")
                
            except Exception as e:
                print(f"❌ Error predicting {loss_type}: {e}")
                result_df[f'predicted_{loss_type}'] = np.nan
        else:
            print(f"⚠️  Model not found for {loss_type}")
            result_df[f'predicted_{loss_type}'] = np.nan
    
    return result_df


def load_models_from_directory(models_dir: str) -> dict:
    """
    Load all trained models from a directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary mapping loss types to loaded models
    """
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"❌ Models directory does not exist: {models_path}")
        return models
        
    print(f"🔍 Looking for models in: {models_path.absolute()}")
    
    # First try to load .json models
    model_files = list(models_path.glob("xgb_optuna_*.json"))
    
    # If no .json files found, try .pkl files
    if not model_files:
        model_files = list(models_path.glob("xgb_optuna_*.pkl"))
    
    if not model_files:
        print(f"ℹ️ No model files found in {models_path}")
        return models
    
    for model_file in model_files:
        loss_type = model_file.stem.replace("xgb_optuna_", "")
        try:
            if model_file.suffix == '.json':
                model = XGBRegressor()
                model.load_model(str(model_file))
            else:  # .pkl file
                import pickle
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            # Verify the model is properly loaded
            if hasattr(model, 'predict'):
                models[loss_type] = model
                print(f"✅ Successfully loaded model for {loss_type} from {model_file}")
                print(f"   Model parameters: {model.get_params()}")
            else:
                print(f"❌ Invalid model in {model_file}: missing predict method")
                
        except Exception as e:
            print(f"❌ Failed to load model {model_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not models:
        print("⚠️ No valid models were loaded")
    else:
        print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    return models


def predict_with_loaded_models(features: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Predict losses using pre-loaded models.
    
    Args:
        features: DataFrame with feature columns
        models: Dictionary mapping loss types to loaded models
        
    Returns:
        DataFrame with predictions for all loss types
    """
    result_df = features.copy()
    
    for loss_type, model in models.items():
        try:
            # Make predictions
            predictions = predict_loss(model, features)
            result_df[f'predicted_{loss_type}'] = predictions['predicted_loss']
            
            print(f"✅ Predictions completed for {loss_type}")
            
        except Exception as e:
            print(f"❌ Error predicting {loss_type}: {e}")
            result_df[f'predicted_{loss_type}'] = np.nan
    
    return result_df


def save_predictions(predictions_df: pd.DataFrame, output_path: str) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save the predictions
    """
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def get_prediction_summary(predictions_df: pd.DataFrame) -> dict:
    """
    Get summary statistics for predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    prediction_columns = [col for col in predictions_df.columns if col.startswith('predicted_')]
    
    for col in prediction_columns:
        if col in predictions_df.columns:
            summary[col] = {
                'mean': predictions_df[col].mean(),
                'std': predictions_df[col].std(),
                'min': predictions_df[col].min(),
                'max': predictions_df[col].max(),
                'count': predictions_df[col].count()
            }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Loss prediction module loaded successfully")
