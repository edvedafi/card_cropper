import cv2
import numpy as np
from pathlib import Path
import os
from rembg import remove, new_session
import time

def detect_card(image_path, output_path, border_size=5):
    """
    Detect and crop a card using rembg for background removal.
    
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
        print("Starting rembg detection...")
        start_time = time.time()
        
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            return False, "Could not read image"
            
        # Create a copy for output
        orig = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Remove background using rembg
        print("Removing background...")
        session = new_session()
        output = remove(image, session=session)
        
        # Convert to grayscale
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        
        # Apply a very low threshold to ensure we catch all card pixels
        # and add some morphological dilation to be extra conservative
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, "No contours found after background removal"
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add border while ensuring we don't exceed image bounds
        x = max(0, x - border_size)
        y = max(0, y - border_size)
        w = min(width - x, w + (2 * border_size))
        h = min(height - y, h + (2 * border_size))
        
        # Crop the image
        cropped = orig[y:y+h, x:x+w]
        
        # Save the cropped image
        cv2.imwrite(str(output_path), cropped)
        
        end_time = time.time()
        print(f"rembg detection completed in {end_time - start_time:.2f} seconds")
        
        return True, None
        
    except Exception as e:
        return False, f"rembg detection failed: {str(e)}" 