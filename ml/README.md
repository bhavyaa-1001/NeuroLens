# YOLO Loss Prediction Backend

A modular backend system for predicting YOLO losses (box_loss, dfl_loss, class_loss) from image features using XGBoost and SHAP analysis.

##  Features

- **Image Feature Extraction**: Extracts 7 quality metrics from images
- **Hyperparameter Optimization**: Uses Optuna for automatic tuning
- **XGBoost Training**: Trains separate models for each loss type
- **SHAP Analysis**: Explains model predictions and finds impactful features
- **FastAPI Backend**: RESTful API for real-time analysis
- **JSON Output**: Structured results for frontend integration

## � Project Structure

```
project/
├── src/
│   ├── feature_extraction.py    # Image feature extraction
│   ├── optuna_train.py         # Hyperparameter optimization
│   ├── train_xgboost.py        # XGBoost model training
│   ├── predict_loss.py         # Loss prediction
│   └── shap_analysis.py        # SHAP analysis
├── server/
│   └── app.py                  # FastAPI backend server
├── models/                     # Trained model storage
├── data/
│   └── uploads/               # Temporary image storage
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

##  Installation

1. **Clone or download the project**
   ```bash
   cd project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p models data/uploads
   ```

##  Usage

### Start the Server

```bash
cd server
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### `POST /analyze`
Upload images for analysis and get YOLO loss predictions.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple image files (JPG, PNG, BMP, TIFF)

**Response:**
```json
{
  "status": "success",
  "message": "Successfully analyzed 5 images",
  "results": [
    {
      "image": "img_001.jpg",
      "loss_box": 0.451,
      "loss_dfl": 0.301,
      "loss_class": 0.120,
      "most_impactful_feature": "laplacian_variance",
      "impact_percent": 67.3
    }
  ],
  "summary": {
    "total_images": 5,
    "features_extracted": 7,
    "models_trained": 3,
    "predictions_made": 5
  }
}
```

#### `GET /health`
Health check endpoint.

#### `GET /models`
List available trained models.

#### `DELETE /models`
Clear all trained models.

### Example Usage with curl

```bash
# Upload images for analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Check health
curl -X GET "http://localhost:8000/health"

# List models
curl -X GET "http://localhost:8000/models"
```

##  Extracted Features

The system extracts the following image quality metrics:

1. **Laplacian Variance**: Sharpness measure
2. **SNR (Signal-to-Noise Ratio)**: Image quality indicator
3. **Mean Brightness**: Average pixel intensity
4. **RMS Contrast**: Root mean square contrast
5. **Entropy**: Information content measure
6. **Edge Density**: Proportion of edge pixels
7. **Colorfulness Index**: Color richness measure

##  Model Training Pipeline

1. **Feature Extraction**: Extract 7 quality metrics from uploaded images
2. **Optuna Optimization**: Find best hyperparameters for each loss type
3. **XGBoost Training**: Train separate models for box_loss, dfl_loss, class_loss
4. **Prediction**: Generate loss predictions for each image
5. **SHAP Analysis**: Identify most impactful features per image

##  SHAP Output Format

Each image returns:
```json
{
  "image": "filename.jpg",
  "loss_box": 0.451,
  "loss_dfl": 0.301,
  "loss_class": 0.120,
  "most_impactful_feature": "laplacian_variance",
  "impact_percent": 67.3
}
```

##  Configuration

### Environment Variables
- `UPLOAD_DIR`: Directory for uploaded images (default: `data/uploads`)
- `MODELS_DIR`: Directory for trained models (default: `models`)
- `DATA_DIR`: Directory for data storage (default: `data`)

### Model Parameters
- Optuna trials: 200 (configurable in `optuna_train.py`)
- Train/validation split: 80/20
- Random state: 42 (for reproducibility)

##  Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Permission Errors**: Check directory permissions
   ```bash
   chmod 755 models data/uploads
   ```

3. **Memory Issues**: Reduce Optuna trials in `optuna_train.py`

4. **Image Loading Errors**: Ensure images are in supported formats (JPG, PNG, BMP, TIFF)

### Logs
Check console output for detailed error messages and progress updates.

##  Performance

- **Feature Extraction**: ~0.1-0.5 seconds per image
- **Optuna Optimization**: ~2-5 minutes (50 trials)
- **Model Training**: ~10-30 seconds
- **SHAP Analysis**: ~1-3 seconds per image

##  Security Notes

- Configure CORS properly for production
- Validate file types and sizes
- Implement authentication if needed
- Clean up temporary files regularly

##  License

This project is for educational and research purposes.

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.
