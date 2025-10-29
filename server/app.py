"""
FastAPI Backend Server

This server provides endpoints for image analysis and YOLO loss prediction
using the modular pipeline components.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Create necessary directories
UPLOAD_DIR = Path("data/uploads")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

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
    for model_file in MODELS_DIR.glob("*.json"):
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
