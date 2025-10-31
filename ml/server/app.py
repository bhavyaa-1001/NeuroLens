"""
FastAPI Backend Server

This server provides endpoints for image analysis and YOLO loss prediction
using the modular pipeline components.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extraction import extract_features
from src.optuna_train import run_optuna_for_all_losses, save_optuna_results
from src.train_xgboost import train_all_models, save_model_metadata
from src.predict_loss import predict_all_losses, load_models_from_directory
from src.shap_analysis import analyze_all_models, combine_shap_results, save_shap_results

app = FastAPI(title="YOLO Loss Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories relative to ml/ root, not ml/server/
BASE_DIR = Path(__file__).resolve().parent.parent  # points to .../ml
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
# Models are saved in ml/server/models/ by train_xgboost when called from app.py
# Check both locations: server/models (where they actually are) and root models/ (standard location)
SERVER_MODELS_DIR = Path(__file__).resolve().parent / "models"  # ml/server/models/
MODELS_DIR = BASE_DIR / "models"  # ml/models/
SRC_MODELS_DIR = BASE_DIR / "src"  # user said models may be here
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YOLO Loss Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze uploaded images and predict YOLO losses",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


@app.post("/analyze")
async def analyze_images(files: List[UploadFile] = File(...)):
    """
    Analyze uploaded images and predict YOLO losses.
    
    This endpoint:
    1. Saves uploaded images to /data/uploads
    2. Extracts features from images
    3. Runs Optuna hyperparameter tuning
    4. Trains XGBoost models
    5. Makes predictions
    6. Performs SHAP analysis
    7. Returns JSON results
    """
    try:
        # Step 1: Save uploaded images
        print(f"📁 Saving {len(files)} uploaded images...")
        saved_files = []
        
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        print(f"✅ Saved {len(saved_files)} images to {UPLOAD_DIR}")
        
        # Step 2: Extract features
        print("🔍 Extracting features from images...")
        features_df = extract_features(str(UPLOAD_DIR))
        
        if features_df.empty:
            raise HTTPException(status_code=400, detail="No valid images found for feature extraction")
        
        # Save features to CSV
        features_path = DATA_DIR / "extracted_features.csv"
        features_df.to_csv(features_path, index=False)
        print(f"✅ Features extracted and saved to {features_path}")
        
        # Step 3: Create dummy loss data for training (since we don't have actual YOLO losses)
        # In a real scenario, this would come from your YOLO training data
        print("📊 Creating training data with dummy loss values...")
        
        # Add dummy loss columns for training
        np.random.seed(42)  # For reproducible results
        features_df['box_loss'] = np.random.uniform(0.1, 2.0, len(features_df))
        features_df['dfl_loss'] = np.random.uniform(0.1, 1.5, len(features_df))
        features_df['class_loss'] = np.random.uniform(0.1, 3.0, len(features_df))
        
        # Step 4: Run Optuna hyperparameter tuning
        print("🎯 Running Optuna hyperparameter tuning...")
        optuna_results = run_optuna_for_all_losses(features_df, n_trials=50)  # Reduced for faster response
        
        # Save Optuna results
        optuna_output_dir = DATA_DIR / "optuna_results"
        optuna_output_dir.mkdir(exist_ok=True)
        save_optuna_results(optuna_results, str(optuna_output_dir))
        
        # Step 5: Train XGBoost models
        print("🤖 Training XGBoost models...")
        model_paths = train_all_models(features_df, optuna_results)
        
        # Save model metadata
        metadata_path = DATA_DIR / "model_metadata.json"
        save_model_metadata(model_paths, optuna_results, str(metadata_path))
        
        # Step 6: Make predictions
        print("🔮 Making predictions...")
        predictions_df = predict_all_losses(features_df, model_paths)
        
        # Save predictions
        predictions_path = DATA_DIR / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        # Step 7: Perform SHAP analysis
        print("🔍 Performing SHAP analysis...")
        image_names = features_df['image_name'].tolist()
        shap_results = analyze_all_models(features_df, model_paths, image_names)
        
        # Combine SHAP results
        combined_shap_results = combine_shap_results(shap_results)
        
        # Save SHAP results
        shap_path = DATA_DIR / "shap_results.json"
        save_shap_results(combined_shap_results, str(shap_path))
        
        # Step 8: Prepare response
        response_data = {
            "status": "success",
            "message": f"Successfully analyzed {len(features_df)} images",
            "results": combined_shap_results,
            "summary": {
                "total_images": len(features_df),
                "features_extracted": len(features_df.columns) - 4,  # Excluding S.No, image_name, and dummy losses
                "models_trained": len([p for p in model_paths.values() if p is not None]),
                "predictions_made": len(combined_shap_results)
            }
        }
        
        print("✅ Analysis completed successfully!")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up uploaded files
        for file_path in saved_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")


@app.get("/models")
async def list_models():
    """List available trained models."""
    models = []
    # Check all known locations
    for models_dir in [SERVER_MODELS_DIR, MODELS_DIR, SRC_MODELS_DIR]:
        if models_dir.exists():
            for model_file in models_dir.glob("*.json"):
                models.append({
                    "name": model_file.stem,
                    "path": str(model_file),
                    "size": model_file.stat().st_size
                })
    return {"models": models}


@app.delete("/models")
async def clear_models():
    """Clear all trained models."""
    try:
        for model_file in MODELS_DIR.glob("*"):
            model_file.unlink()
        return {"message": "All models cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear models: {str(e)}")


# Lightweight endpoint to be called by frontend RunModel
@app.post("/run-xgboost")
async def run_xgboost(
    payload: dict = Body(default={})
):
    """
    Load features from shared uploads directory and run predictions using
    existing models if available. Returns data in the shape expected by the frontend.
    """
    try:
        # Features are extracted from images mirrored by Node server under ml/data/uploads
        images_dir = UPLOAD_DIR
        print(f"/run-xgboost reading images from: {images_dir}")
        if not images_dir.exists():
            return {"results": []}

        # Extract features
        features_df = extract_features(str(images_dir))
        print(f"Extracted features rows: {len(features_df)}")
        if features_df.empty:
            return {"results": []}

        # Try loading models; check server/models/, root models/, and src/
        model_paths = {}
        for loss_type in ['box_loss', 'dfl_loss', 'class_loss']:
            # First check ml/server/models/ (where models are actually saved)
            candidate = SERVER_MODELS_DIR / f"xgb_optuna_{loss_type}.json"
            if not candidate.exists():
                # Fallback to ml/models/
                candidate = MODELS_DIR / f"xgb_optuna_{loss_type}.json"
            if not candidate.exists():
                # Finally check ml/src/
                candidate = SRC_MODELS_DIR / f"xgb_optuna_{loss_type}.json"
            if candidate.exists():
                model_paths[loss_type] = str(candidate)
                print(f"✅ Found model for {loss_type} at {candidate}")

        results = []
        image_names = features_df['image_name'].tolist() if 'image_name' in features_df.columns else [f"image_{i}" for i in range(len(features_df))]

        if model_paths:
            print(f"Loaded models found for: {list(model_paths.keys())}")
            # Predict per-loss values
            predictions_df = predict_all_losses(features_df, model_paths)
            # Attach predicted columns to features for SHAP module consumption if needed
            features_with_preds = features_df.copy()
            for col in ['box_loss', 'dfl_loss', 'class_loss']:
                pred_col = f'predicted_{col}'
                if pred_col in predictions_df.columns:
                    features_with_preds[pred_col] = predictions_df[pred_col]

            # Full SHAP-style output matching the screenshot format
            # Pass features_with_preds so SHAP result includes per-loss predictions
            shap_results = analyze_all_models(features_with_preds, model_paths, image_names)
            combined = combine_shap_results(shap_results)
            # Normalize/round fields and add aliases expected by UI/CSV
            for item in combined:
                img_name = item.get('image')
                item['imageUrl'] = f"{os.environ.get('NODE_PUBLIC_URL', 'http://localhost:4000')}/uploads/{img_name}"
                # Round numeric fields
                for k in ['loss_box', 'loss_dfl', 'loss_class']:
                    val = item.get(k)
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (pd.isna(val) or np.isnan(val))):
                        item[k] = float(f"{val:.4f}")
                    else:
                        item[k] = 0.0000
                if 'impact_percent' in item and isinstance(item['impact_percent'], (int, float)):
                    item['impact_percent'] = float(f"{item['impact_percent']:.1f}")
                # CSV alias
                item['Name_in'] = img_name
            return {"results": combined}
        else:
            print("No trained models found; quickly training lightweight models with dummy targets…")
            # Create lightweight dummy targets and run a very small optuna tuning + training
            np.random.seed(42)
            df_for_training = features_df.copy()
            df_for_training['box_loss'] = np.random.uniform(0.1, 2.0, len(df_for_training))
            df_for_training['dfl_loss'] = np.random.uniform(0.1, 1.5, len(df_for_training))
            df_for_training['class_loss'] = np.random.uniform(0.1, 3.0, len(df_for_training))

            # Very few trials to keep it responsive
            optuna_results = run_optuna_for_all_losses(df_for_training, n_trials=10)
            optuna_output_dir = DATA_DIR / "optuna_results"
            optuna_output_dir.mkdir(exist_ok=True)
            save_optuna_results(optuna_results, str(optuna_output_dir))

            model_paths = train_all_models(df_for_training, optuna_results)
            metadata_path = DATA_DIR / "model_metadata.json"
            save_model_metadata(model_paths, optuna_results, str(metadata_path))

            # Predict and then return the same SHAP-style structure
            predictions_df = predict_all_losses(features_df, model_paths)
            features_with_preds = features_df.copy()
            for col in ['box_loss', 'dfl_loss', 'class_loss']:
                pred_col = f'predicted_{col}'
                if pred_col in predictions_df.columns:
                    features_with_preds[pred_col] = predictions_df[pred_col]
            shap_results = analyze_all_models(features_with_preds, model_paths, image_names)
            combined = combine_shap_results(shap_results)
            for item in combined:
                img_name = item.get('image')
                item['imageUrl'] = f"{os.environ.get('NODE_PUBLIC_URL', 'http://localhost:4000')}/uploads/{img_name}"
                for k in ['loss_box', 'loss_dfl', 'loss_class']:
                    val = item.get(k)
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (pd.isna(val) or np.isnan(val))):
                        item[k] = float(f"{val:.4f}")
                    else:
                        item[k] = 0.0000
                if 'impact_percent' in item and isinstance(item['impact_percent'], (int, float)):
                    item['impact_percent'] = float(f"{item['impact_percent']:.1f}")
                item['Name_in'] = img_name
            return {"results": combined}

        return {"results": results}
    except Exception as e:
        print(f"❌ /run-xgboost error: {e}")
        raise HTTPException(status_code=500, detail=f"run-xgboost failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
