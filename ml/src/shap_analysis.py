"""
SHAP Analysis Module

This module handles SHAP (SHapley Additive exPlanations) analysis
to explain model predictions and find the most impactful features.
"""

import pandas as pd
import numpy as np
import shap
import xgboost
from xgboost import XGBRegressor
from pathlib import Path
import json


def shap_explain(model: XGBRegressor, features: pd.DataFrame, image_names: list = None) -> list:
    """
    Perform SHAP analysis on a trained model and return JSON-formatted results.
    
    Args:
        model: Trained XGBoost model
        features: DataFrame with feature columns
        image_names: List of image names (optional)
        
    Returns:
        List of dictionaries with SHAP analysis results in JSON format
    """
    # Define feature columns
    feature_columns = [
        "laplacian_variance",
        "snr",
        "mean_brightness",
        "rms_contrast",
        "entropy",
        "edge_density",
        "colorfulness_index"
    ]

    # Support common alias names so SHAP works across different extractors
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

    features = features.copy()
    for canonical_name, alternatives in alias_map.items():
        if canonical_name not in features.columns:
            for alt in alternatives:
                if alt in features.columns:
                    features[canonical_name] = features[alt]
                    break
    
    # Ensure all required features are present
    missing_features = [col for col in feature_columns if col not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features
    X = features[feature_columns]
    
    # Create explainer
    def predict_fn(X_input):
        return model.predict(X_input)
    
    explainer = shap.Explainer(predict_fn, masker=shap.maskers.Independent(X))
    
    # Generate SHAP values
    shap_values = explainer(X)
    
    # Convert to DataFrame for easier processing
    shap_df = pd.DataFrame(shap_values.values, columns=[f'{col}_SHAP' for col in feature_columns])
    
    # Calculate feature means for comparison
    feature_means = X.mean()
    
    # Process results for each image
    results = []
    
    for index, row in shap_df.iterrows():
        # Find the most impactful feature
        abs_shap_values = row.abs()
        
        if abs_shap_values.sum() == 0:
            most_impactful_feature = "None"
            impact_percent = 0.0
        else:
            most_impactful_feature_shap_col = abs_shap_values.idxmax()
            max_abs_shap_value = abs_shap_values.max()
            
            # Calculate impact percentage
            impact_percent = (max_abs_shap_value / abs_shap_values.sum()) * 100
            
            # Clean up feature name
            most_impactful_feature = most_impactful_feature_shap_col.replace('_SHAP', '')
        
        # Get image name
        image_name = image_names[index] if image_names and index < len(image_names) else f"image_{index}"
        
        # Get predictions for all loss types (if available)
        predictions = {}
        for loss_type in ['box_loss', 'dfl_loss', 'class_loss']:
            pred_col = f'predicted_{loss_type}'
            if pred_col in features.columns:
                predictions[f'loss_{loss_type.replace("_", "")}'] = float(features.iloc[index][pred_col])
            else:
                predictions[f'loss_{loss_type.replace("_", "")}'] = 0.0
        
        # Create result dictionary
        result = {
            "image": image_name,
            "loss_box": predictions.get('loss_box', 0.0),
            "loss_dfl": predictions.get('loss_dfl', 0.0),
            "loss_class": predictions.get('loss_class', 0.0),
            "most_impactful_feature": most_impactful_feature,
            "impact_percent": round(impact_percent, 1)
        }
        
        results.append(result)
    
    return results


def analyze_all_models(features: pd.DataFrame, model_paths: dict, image_names: list = None) -> dict:
    """
    Perform SHAP analysis for all trained models.
    
    Args:
        features: DataFrame with feature columns
        model_paths: Dictionary mapping loss types to model paths
        image_names: List of image names (optional)
        
    Returns:
        Dictionary containing SHAP results for each loss type
    """
    results = {}
    
    loss_types = ['box_loss', 'dfl_loss', 'class_loss']
    
    for loss_type in loss_types:
        model_path = model_paths.get(loss_type)
        if model_path and Path(model_path).exists():
            try:
                # Load model
                model = XGBRegressor()
                model.load_model(model_path)
                
                # Perform SHAP analysis
                shap_results = shap_explain(model, features, image_names)
                results[loss_type] = shap_results
                
                print(f"✅ SHAP analysis completed for {loss_type}")
                
            except Exception as e:
                print(f"❌ Error in SHAP analysis for {loss_type}: {e}")
                results[loss_type] = []
        else:
            print(f"⚠️  Model not found for {loss_type}")
            results[loss_type] = []
    
    return results


def combine_shap_results(shap_results: dict) -> list:
    """
    Combine SHAP results from all models into a single list.
    
    Args:
        shap_results: Dictionary containing SHAP results for each loss type
        
    Returns:
        Combined list of SHAP results
    """
    if not shap_results:
        return []
    
    # Get the first loss type to determine the number of images
    first_loss_type = list(shap_results.keys())[0]
    if not shap_results[first_loss_type]:
        return []
    
    num_images = len(shap_results[first_loss_type])
    combined_results = []
    
    for i in range(num_images):
        # Start with the first result as base
        combined_result = shap_results[first_loss_type][i].copy()
        
        # Update with results from other loss types
        key_map = {
            'box_loss': 'loss_box',
            'dfl_loss': 'loss_dfl',
            'class_loss': 'loss_class',
        }
        for loss_type, results in shap_results.items():
            if i < len(results):
                result = results[i]
                src_key = key_map.get(loss_type)
                if src_key and src_key in result:
                    combined_result[src_key] = result[src_key]
                
                # Use the most impactful feature from the loss type with highest impact
                if result['impact_percent'] > combined_result.get('impact_percent', 0):
                    combined_result['most_impactful_feature'] = result['most_impactful_feature']
                    combined_result['impact_percent'] = result['impact_percent']
        
        combined_results.append(combined_result)
    
    return combined_results


def save_shap_results(shap_results: list, output_path: str) -> None:
    """
    Save SHAP results to JSON file.
    
    Args:
        shap_results: List of SHAP results
        output_path: Path to save the results
    """
    with open(output_path, 'w') as f:
        json.dump(shap_results, f, indent=2)
    
    print(f"SHAP results saved to {output_path}")


def get_feature_importance_summary(shap_results: list) -> dict:
    """
    Get summary of feature importance across all images.
    
    Args:
        shap_results: List of SHAP results
        
    Returns:
        Dictionary with feature importance summary
    """
    if not shap_results:
        return {}
    
    # Count occurrences of each feature as most impactful
    feature_counts = {}
    total_impact = {}
    
    for result in shap_results:
        feature = result['most_impactful_feature']
        impact = result['impact_percent']
        
        if feature != "None":
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
            total_impact[feature] = total_impact.get(feature, 0) + impact
    
    # Calculate average impact for each feature
    summary = {}
    for feature in feature_counts:
        summary[feature] = {
            'count': feature_counts[feature],
            'percentage': (feature_counts[feature] / len(shap_results)) * 100,
            'avg_impact': total_impact[feature] / feature_counts[feature]
        }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("SHAP analysis module loaded successfully")
