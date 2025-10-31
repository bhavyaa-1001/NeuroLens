"""
Image Feature Extraction Module

This module extracts various image quality metrics including:
- Laplacian Variance
- Signal-to-Noise Ratio (SNR)
- Mean Brightness
- RMS Contrast
- Entropy
- Edge Density
- Colorfulness Index
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel
from skimage.feature import canny
from skimage.measure import shannon_entropy
from math import sqrt
import os


def calculate_laplacian_variance(image):
    """Calculates the Laplacian Variance of an image."""
    gray = rgb2gray(image)
    laplacian = cv2.Laplacian(np.uint8(gray * 255), cv2.CV_64F)
    return laplacian.var()


def calculate_snr(image):
    """Calculates the Signal-to-Noise Ratio of an image."""
    gray = rgb2gray(image)
    # Assuming noise is the difference from a smoothed image
    smoothed = gaussian(gray, sigma=1)
    noise = gray - smoothed
    signal_power = np.mean(gray**2)
    noise_power = np.mean(noise**2)
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def calculate_mean_brightness(image):
    """Calculates the mean brightness of an image."""
    gray = rgb2gray(image)
    return np.mean(gray)


def calculate_rms_contrast(image):
    """Calculates the RMS contrast of an image."""
    gray = rgb2gray(image)
    mean = np.mean(gray)
    rms = sqrt(np.mean((gray - mean)**2))
    return rms


def calculate_entropy(image):
    """Calculates the entropy of an image."""
    gray = rgb2gray(image)
    return shannon_entropy(gray)


def calculate_edge_density(image):
    """Calculates the edge density of an image using Canny edge detection."""
    gray = rgb2gray(image)
    edges = canny(gray)
    # Proportion of edge pixels
    return np.sum(edges) / edges.size


def calculate_colorfulness_index(image):
    """Calculates the colorfulness index of a color image."""
    # Based on Hasler and Susstrunk method
    # Convert to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    a = lab[:,:,1]
    b = lab[:,:,2]

    # Compute the standard deviations of a and b channels
    std_a = np.std(a)
    std_b = np.std(b)

    # Compute the mean of the square roots of a and b channel means
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    mean_ab = sqrt(mean_a**2 + mean_b**2)

    # Calculate colorfulness
    colorfulness = sqrt(std_a**2 + std_b**2) + 0.3 * mean_ab
    return colorfulness


def extract_features(images_folder: str) -> pd.DataFrame:
    """
    Extract features from all images in the specified folder.
    
    Args:
        images_folder: Path to folder containing images
        
    Returns:
        DataFrame with extracted features for each image
    """
    images_folder = Path(images_folder)
    image_files = []
    
    # Get all image files from the folder
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        image_files.extend(images_folder.glob(ext))
        image_files.extend(images_folder.glob(ext.upper()))
    
    image_metrics_list = []
    
    for image_path in image_files:
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                # Convert BGR to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Calculate metrics
                laplacian_var = calculate_laplacian_variance(image)
                snr = calculate_snr(image)
                brightness = calculate_mean_brightness(image)
                contrast = calculate_rms_contrast(image)
                entropy_val = calculate_entropy(image)
                edge_density_val = calculate_edge_density(image)
                colorfulness = calculate_colorfulness_index(image)

                # Store results
                metrics_dict = {
                    'image_name': image_path.name,
                    'laplacian_variance': laplacian_var,
                    'snr': snr,
                    'mean_brightness': brightness,
                    'rms_contrast': contrast,
                    'entropy': entropy_val,
                    'edge_density': edge_density_val,
                    'colorfulness_index': colorfulness
                }
                image_metrics_list.append(metrics_dict)
            else:
                print(f"Warning: Could not load image at {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Create DataFrame from results
    metrics_df = pd.DataFrame(image_metrics_list)
    
    # Add S.No column as index
    if not metrics_df.empty:
        metrics_df.insert(0, 'S.No', metrics_df.index)
    
    return metrics_df


def save_features_to_csv(features_df: pd.DataFrame, output_path: str) -> None:
    """
    Save extracted features to CSV file.
    
    Args:
        features_df: DataFrame with extracted features
        output_path: Path to save the CSV file
    """
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    images_folder = "data/uploads"
    features_df = extract_features(images_folder)
    print(f"Extracted features for {len(features_df)} images")
    print(features_df.head())
