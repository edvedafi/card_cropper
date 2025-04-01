import cv2
import numpy as np
from pathlib import Path
import os
import torch
import shutil
import urllib.request
import socket
import ssl

# Define model paths
MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "models"
MODEL_PATH = MODEL_DIR / "yolov5s.pt"
TORCH_HUB_PATH = Path(os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master"))

# Direct download URL for YOLOv5s model
YOLO_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"

def find_cached_model():
    """
    Look for the model in the torch hub cache.
    Returns the path to the model if found, None otherwise.
    """
    if TORCH_HUB_PATH.exists():
        # Look for .pt files in the cache directory
        pt_files = list(TORCH_HUB_PATH.glob("*.pt"))
        if pt_files:
            return pt_files[0]
    return None

def test_internet_connection():
    """
    Test if we can connect to common websites.
    Returns tuple of (has_connection, error_message)
    """
    try:
        # Try to connect to Google's DNS
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True, None
    except OSError as e:
        return False, f"Network error: {e}"
    
def direct_download_model():
    """
    Download the model directly using urllib.
    """
    try:
        print("Downloading YOLOv5s model directly...")
        # Create an SSL context that doesn't verify certificates
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # Download with a custom User-Agent to avoid some blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(YOLO_MODEL_URL, headers=headers)
        
        with urllib.request.urlopen(req, context=ctx) as response:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"Direct download failed: {e}")
        return False

def download_model():
    """
    Download the YOLO model and save it locally.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"\nChecking model paths:")
        print(f"Model directory: {MODEL_DIR}")
        print(f"Model directory exists: {MODEL_DIR.exists()}")
        print(f"Model file path: {MODEL_PATH}")
        print(f"Model file exists: {MODEL_PATH.exists()}")
        print(f"Torch hub cache path: {TORCH_HUB_PATH}")
        print(f"Torch hub cache exists: {TORCH_HUB_PATH.exists()}")
        
        # First check if model exists in torch hub cache
        cached_model = find_cached_model()
        if cached_model:
            print(f"\nFound model in cache at {cached_model}")
            # Create model directory if it doesn't exist
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            # Copy from cache to our local directory
            shutil.copy(cached_model, MODEL_PATH)
            return True
        
        # Test internet connection
        print("\nTesting internet connection...")
        has_connection, error = test_internet_connection()
        if not has_connection:
            print(f"Internet connection test failed: {error}")
            return False
        print("✓ Internet connection test passed")
            
        print("\nAttempting to download model...")
        
        # Try direct download first
        try:
            print("Attempting direct download from GitHub...")
            if direct_download_model():
                print("✓ Direct download successful")
                return True
        except Exception as e:
            print(f"Direct download failed with error: {e}")
            
        print("\nDirect download failed, trying torch hub...")
        try:
            print("Loading model from torch hub...")
            # Try to load model from torch hub which will download it
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
            
            # Get the model file path from the loaded model
            src_path = model.pt[0] if isinstance(model.pt, list) else model.pt
            print(f"Model downloaded to: {src_path}")
            
            # Copy the model file to our local directory
            print(f"Copying model to: {MODEL_PATH}")
            shutil.copy(src_path, MODEL_PATH)
            print("✓ Torch hub download successful")
            
            return True
        except Exception as e:
            print(f"Torch hub download failed with error: {e}")
            
        return False
    except Exception as e:
        print(f"\nAll download attempts failed: {e}")
        print("Please try one of the following:")
        print("1. Check your internet connection")
        print("2. Try downloading the model manually from:")
        print(f"   {YOLO_MODEL_URL}")
        print(f"   and place it in: {MODEL_PATH}")
        return False

def load_model():
    """
    Load the YOLO model, downloading it if necessary.
    Returns the model or None if loading fails.
    """
    try:
        print("\nAttempting to load YOLO model...")
        # If model doesn't exist locally, try to download it
        if not MODEL_PATH.exists():
            print("Model not found locally, attempting download...")
            if not download_model():
                print("Failed to download model")
                return None
        
        # Load the model from local file
        try:
            print("Loading model from local file...")
            # Load the model using YOLOv5's custom model loading
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=True, trust_repo=True)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Attempting to download fresh model from torch hub...")
            try:
                # If loading local model fails, try downloading fresh from torch hub
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)
                print("✓ Model downloaded and loaded successfully")
                return model
            except Exception as e:
                print(f"Failed to download and load model: {e}")
                return None
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def detect_card(image_path, output_path, border_size=5):
    """
    ML-based card detection using YOLO model.
    Best for complex backgrounds or when traditional methods fail.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) where:
            - success is True if card was successfully detected and cropped
            - error_reason is None on success, or error description on failure
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            return False, "Could not read image"
        
        # Create a copy for output
        orig = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        print("Attempting YOLO detection...")
        # Load YOLO model
        model = load_model()
        if model is None:
            return False, "Could not load YOLO model - please ensure you have internet connection for first run"
        
        # Run inference
        results = model(image)
        
        # Get the first detection (highest confidence)
        if len(results.xyxy[0]) > 0:
            print("✓ YOLO detection successful")
            # Get bounding box coordinates
            x1, y1, x2, y2 = results.xyxy[0][0].cpu().numpy()
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Calculate dimensions
            w = x2 - x1
            h = y2 - y1
            
            # Ensure reasonable dimensions
            if w < 10 or h < 10 or w > width * 1.5 or h > height * 1.5:
                return False, "Invalid dimensions detected"
            
            # Add border to dimensions (ensuring we don't exceed source dimensions)
            output_width = min(w + (2 * border_size), width)
            output_height = min(h + (2 * border_size), height)
            
            # Adjust border size if needed to fit within source dimensions
            actual_border_x = (output_width - w) // 2
            actual_border_y = (output_height - h) // 2
            
            # Set up source and destination points for the perspective transformation
            src = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            
            dst = np.array([
                [actual_border_x, actual_border_y],
                [output_width - actual_border_x - 1, actual_border_y],
                [output_width - actual_border_x - 1, output_height - actual_border_y - 1],
                [actual_border_x, output_height - actual_border_y - 1]
            ], dtype=np.float32)
            
            # Compute perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(orig, transform_matrix, (output_width, output_height))
            
            # Create the output directory if needed
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            
            # Save the cropped image
            cv2.imwrite(str(output_path), warped)
            return True, None
        else:
            return False, "No detections found by YOLO"
            
    except Exception as e:
        return False, f"YOLO detection failed: {str(e)}" 