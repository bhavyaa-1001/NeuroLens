#!/usr/bin/env python3
"""
Example client script for the YOLO Loss Prediction API

This script demonstrates how to use the API to analyze images.
"""

import requests
import json
from pathlib import Path


def test_api_health(base_url="http://localhost:8000"):
    """Test if the API is running."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False


def analyze_images(image_paths, base_url="http://localhost:8000"):
    """Analyze images using the API."""
    if not image_paths:
        print("❌ No image paths provided")
        return None
    
    # Check if images exist
    valid_paths = []
    for path in image_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            print(f"⚠️  Image not found: {path}")
    
    if not valid_paths:
        print("❌ No valid images found")
        return None
    
    print(f"📁 Analyzing {len(valid_paths)} images...")
    
    try:
        # Prepare files for upload
        files = []
        for path in valid_paths:
            files.append(('files', (Path(path).name, open(path, 'rb'), 'image/jpeg')))
        
        # Make API request
        response = requests.post(f"{base_url}/analyze", files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis completed successfully!")
            return result
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None


def display_results(results):
    """Display analysis results in a formatted way."""
    if not results or 'results' not in results:
        print("❌ No results to display")
        return
    
    print("\n📊 Analysis Results:")
    print("=" * 60)
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. Image: {result['image']}")
        print(f"   📦 Box Loss: {result['loss_box']:.3f}")
        print(f"   🎯 DFL Loss: {result['loss_dfl']:.3f}")
        print(f"   🏷️  Class Loss: {result['loss_class']:.3f}")
        print(f"   🔍 Most Impactful Feature: {result['most_impactful_feature']}")
        print(f"   📈 Impact: {result['impact_percent']:.1f}%")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\n📋 Summary:")
        print(f"   Total Images: {summary.get('total_images', 'N/A')}")
        print(f"   Features Extracted: {summary.get('features_extracted', 'N/A')}")
        print(f"   Models Trained: {summary.get('models_trained', 'N/A')}")
        print(f"   Predictions Made: {summary.get('predictions_made', 'N/A')}")


def main():
    """Main function to run the example client."""
    print("🚀 YOLO Loss Prediction API Client Example")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\n💡 To start the server, run: python start_server.py")
        return
    
    # Example: Analyze some images
    # Replace these paths with actual image files
    example_images = [
        "data/uploads/sample1.jpg",
        "data/uploads/sample2.jpg",
        "data/uploads/sample3.jpg"
    ]
    
    print(f"\n📁 Looking for example images...")
    
    # Check if example images exist, if not, create dummy ones
    from PIL import Image
    import numpy as np
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images if they don't exist
    for i, img_path in enumerate(example_images):
        if not Path(img_path).exists():
            # Create a simple colored image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_path)
            print(f"✅ Created dummy image: {img_path}")
    
    # Analyze images
    results = analyze_images(example_images)
    
    if results:
        display_results(results)
        
        # Save results to file
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    print("\n🎉 Example completed!")


if __name__ == "__main__":
    main()
