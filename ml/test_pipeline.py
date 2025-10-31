#!/usr/bin/env python3
"""
Test script for the YOLO Loss Prediction Pipeline

This script tests the complete pipeline without requiring the FastAPI server.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from feature_extraction import extract_features
from optuna_train import run_optuna_for_all_losses
from train_xgboost import train_all_models
from predict_loss import predict_all_losses
from shap_analysis import analyze_all_models, combine_shap_results


def create_test_data():
    """Create dummy test data for testing the pipeline."""
    print("📊 Creating test data...")
    
    # Create dummy features
    np.random.seed(42)
    n_images = 10
    
    test_data = {
        'S.No': range(n_images),
        'image_name': [f'test_image_{i:03d}.jpg' for i in range(n_images)],
        'laplacian_variance': np.random.uniform(0.1, 5.0, n_images),
        'snr': np.random.uniform(10, 50, n_images),
        'mean_brightness': np.random.uniform(0.1, 0.9, n_images),
        'rms_contrast': np.random.uniform(0.05, 0.3, n_images),
        'entropy': np.random.uniform(6, 8, n_images),
        'edge_density': np.random.uniform(0.01, 0.1, n_images),
        'colorfulness_index': np.random.uniform(5, 50, n_images),
        'box_loss': np.random.uniform(0.1, 2.0, n_images),
        'dfl_loss': np.random.uniform(0.1, 1.5, n_images),
        'class_loss': np.random.uniform(0.1, 3.0, n_images)
    }
    
    return pd.DataFrame(test_data)


def test_pipeline():
    """Test the complete pipeline."""
    print("🚀 Starting YOLO Loss Prediction Pipeline Test")
    print("=" * 60)
    
    try:
        # Step 1: Create test data
        test_df = create_test_data()
        print(f"✅ Created test data with {len(test_df)} samples")
        
        # Step 2: Run Optuna optimization
        print("\n🎯 Running Optuna hyperparameter tuning...")
        optuna_results = run_optuna_for_all_losses(test_df, n_trials=10)  # Reduced for testing
        print("✅ Optuna optimization completed")
        
        # Step 3: Train XGBoost models
        print("\n🤖 Training XGBoost models...")
        model_paths = train_all_models(test_df, optuna_results)
        print("✅ Model training completed")
        
        # Step 4: Make predictions
        print("\n🔮 Making predictions...")
        predictions_df = predict_all_losses(test_df, model_paths)
        print("✅ Predictions completed")
        
        # Step 5: Perform SHAP analysis
        print("\n🔍 Performing SHAP analysis...")
        image_names = test_df['image_name'].tolist()
        shap_results = analyze_all_models(predictions_df, model_paths, image_names)
        combined_results = combine_shap_results(shap_results)
        print("✅ SHAP analysis completed")
        
        # Step 6: Display results
        print("\n📊 Results Summary:")
        print(f"Total images processed: {len(combined_results)}")
        
        if combined_results:
            print("\nSample results:")
            for i, result in enumerate(combined_results[:3]):  # Show first 3 results
                print(f"  {i+1}. {result['image']}:")
                print(f"     - Box Loss: {result['loss_box']:.3f}")
                print(f"     - DFL Loss: {result['loss_dfl']:.3f}")
                print(f"     - Class Loss: {result['loss_class']:.3f}")
                print(f"     - Most Impactful Feature: {result['most_impactful_feature']}")
                print(f"     - Impact: {result['impact_percent']:.1f}%")
        
        print("\n🎉 Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
