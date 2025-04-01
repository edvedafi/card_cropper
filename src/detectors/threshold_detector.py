import cv2
import numpy as np
import imutils
from pathlib import Path
import os

def detect_card(image_path, output_path, border_size=5):
    """
    Card detection using adaptive thresholding.
    Best for images with good contrast between card and background.
    
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
        
        print("Applying adaptive thresholding...")
        # Convert to grayscale and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if not contours:
            return False, "No contours found"
        
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try each contour until we find a valid one
        for contour in contours:
            # Calculate the minimum area needed for a valid contour (0.5% of the image)
            min_contour_area = height * width * 0.005
            
            # Check if the contour is too small
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If the approximation doesn't have 4 points, try to use a bounding rectangle
            if len(approx) != 4:
                x, y, w, h = cv2.boundingRect(contour)
                approx = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ])
            
            # Get the 4 corners in the correct order
            approx = approx.reshape(len(approx), 2)
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype=np.float32)
            
            # Sum of coordinates - smallest is top-left, largest is bottom-right
            s = approx.sum(axis=1)
            rect[0] = approx[np.argmin(s)]  # Top-left
            rect[2] = approx[np.argmax(s)]  # Bottom-right
            
            # Diff of coordinates - smallest is top-right, largest is bottom-left
            diff = np.diff(approx, axis=1)
            rect[1] = approx[np.argmin(diff)]  # Top-right
            rect[3] = approx[np.argmax(diff)]  # Bottom-left
            
            # Calculate width and height of the new image
            width_1 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_2 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            max_width = max(int(width_1), int(width_2))
            
            height_1 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            height_2 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            max_height = max(int(height_1), int(height_2))
            
            # Ensure reasonable dimensions
            if max_width < 10 or max_height < 10 or max_width > width * 1.5 or max_height > height * 1.5:
                continue
            
            # Add border to dimensions (ensuring we don't exceed source dimensions)
            output_width = min(max_width + (2 * border_size), width)
            output_height = min(max_height + (2 * border_size), height)
            
            # Adjust border size if needed to fit within source dimensions
            actual_border_x = (output_width - max_width) // 2
            actual_border_y = (output_height - max_height) // 2
            
            # Set up destination points for the perspective transformation with added border
            dst = np.array([
                [actual_border_x, actual_border_y],
                [output_width - actual_border_x - 1, actual_border_y],
                [output_width - actual_border_x - 1, output_height - actual_border_y - 1],
                [actual_border_x, output_height - actual_border_y - 1]
            ], dtype=np.float32)
            
            # Compute perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(rect, dst)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(orig, transform_matrix, (output_width, output_height))
            
            # Create the output directory if needed
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            
            # Save the cropped image
            cv2.imwrite(str(output_path), warped)
            print("✓ Adaptive thresholding successful")
            return True, None
        
        return False, "No valid contours found"
        
    except Exception as e:
        return False, f"Error in threshold detection: {str(e)}" 