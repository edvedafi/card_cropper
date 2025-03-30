import cv2
import numpy as np
import imutils
from pathlib import Path
import os

def is_solid_color_image(image_path, threshold=50):
    """
    Detect if an image is mostly a solid color (or very close to it).
        
    Args:
        image_path: Path to the image file
        threshold: Color variance threshold (higher = more lenient)
        
    Returns:
        bool: True if the image is mostly a solid color, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            return True  # Can't read image, consider it a solid color
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check if the image is very small
        if img.shape[0] < 20 or img.shape[1] < 20:
            return True
        
        # Calculate standard deviation of pixel values
        std_dev = np.std(gray)
        
        # Check histogram distribution - more lenient for photos
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_max = np.max(hist)
        hist_sum = np.sum(hist)
        
        # For photos, we'll be more lenient - if most pixels are in a small range
        # To handle natural variations in lighting and camera sensors
        if hist_max / hist_sum > 0.6:  # 60% of pixels in one intensity bin
            return True
            
        # Check if most pixels are concentrated in a small range of the histogram
        # Calculate how many bins contain 90% of the pixels
        hist_normalized = hist / hist_sum
        hist_cumulative = np.cumsum(hist_normalized)
        bins_90_percent = np.searchsorted(hist_cumulative, 0.9)
        
        # If 90% of pixels are in less than 20% of possible bins, likely a near-solid color
        if bins_90_percent < 51:  # 20% of 256 bins
            return True
        
        # More lenient threshold for photos with lighting variations
        return std_dev < threshold
        
    except Exception as e:
        print(f"Error checking if image is solid color: {e}")
        return False  # Default to False on error

def is_card_cut_off(cropped_image_path, original_image_path, border_check_size=10):
    """
    Detect if a card is cut off by comparing the cropped image with the original.
    A properly cropped card should be completely contained within the original image
    with some background margins.
        
    Args:
        cropped_image_path: Path to the cropped card image
        original_image_path: Path to the original image
        border_check_size: Size of border to check
            
    Returns:
        bool: True if the card appears to be cut off, False otherwise
    """
    try:
        # Read the images
        cropped_img = cv2.imread(str(cropped_image_path))
        original_img = cv2.imread(str(original_image_path))
        
        if cropped_img is None or original_img is None:
            return False
            
        # Get image dimensions
        cropped_h, cropped_w = cropped_img.shape[:2]
        original_h, original_w = original_img.shape[:2]
        
        # If the image is too small to check effectively
        if cropped_h < 50 or cropped_w < 50:
            return False
        
        # Instead of template matching, let's use a different approach to detect cut-off cards
        # We'll analyze the border pixels of the cropped image

        # First check border pixels for consistency
        # Extract borders
        top_border = cropped_img[0:border_check_size, :]
        bottom_border = cropped_img[-border_check_size:, :]
        left_border = cropped_img[:, 0:border_check_size]
        right_border = cropped_img[:, -border_check_size:]
        
        # Calculate color statistics for each border
        # A cut-off card would likely have inconsistent borders with the card content extending to the edge
        
        def calc_border_stats(border):
            # Convert to HSV for better color analysis
            hsv_border = cv2.cvtColor(border, cv2.COLOR_BGR2HSV)
            # Calculate standard deviation of each channel
            h_std = np.std(hsv_border[:,:,0])
            s_std = np.std(hsv_border[:,:,1])
            v_std = np.std(hsv_border[:,:,2])
            # Higher std dev means more color variation (inconsistent border)
            return (h_std, s_std, v_std)
        
        top_stats = calc_border_stats(top_border)
        bottom_stats = calc_border_stats(bottom_border)
        left_stats = calc_border_stats(left_border)
        right_stats = calc_border_stats(right_border)
        
        # Calculate average std dev across all channels for each border
        top_std = np.mean(top_stats)
        bottom_std = np.mean(bottom_stats)
        left_std = np.mean(left_stats)
        right_std = np.mean(right_stats)
        
        # Print border statistics
        print(f"Border std dev - Top: {top_std:.2f}, Bottom: {bottom_std:.2f}, Left: {left_std:.2f}, Right: {right_std:.2f}")
        
        # Set thresholds for what constitutes a consistent border
        # Lower std dev means more consistent color (likely background)
        # Higher std dev means more variation (likely card content reaching the edge)
        std_threshold = 25.0  # This may need adjustment based on testing
        
        # Check if any border has high color variation (suggesting card content at the edge)
        if (top_std > std_threshold or 
            bottom_std > std_threshold or 
            left_std > std_threshold or 
            right_std > std_threshold):
            
            # Further verify with color continuity check
            # For a border that seems to have high variation, check if it's similar to the 
            # adjacent inner pixels (which would suggest content extends to the edge)
            
            # Get the adjacent inner pixels for each border
            inner_top = cropped_img[border_check_size:border_check_size*2, :]
            inner_bottom = cropped_img[-border_check_size*2:-border_check_size, :]
            inner_left = cropped_img[:, border_check_size:border_check_size*2]
            inner_right = cropped_img[:, -border_check_size*2:-border_check_size]
            
            # Compare border to inner pixels (using average color)
            top_similarity = np.mean(np.abs(np.mean(top_border, axis=(0,1)) - np.mean(inner_top, axis=(0,1))))
            bottom_similarity = np.mean(np.abs(np.mean(bottom_border, axis=(0,1)) - np.mean(inner_bottom, axis=(0,1))))
            left_similarity = np.mean(np.abs(np.mean(left_border, axis=(0,1)) - np.mean(inner_left, axis=(0,1))))
            right_similarity = np.mean(np.abs(np.mean(right_border, axis=(0,1)) - np.mean(inner_right, axis=(0,1))))
            
            print(f"Border-inner similarity - Top: {top_similarity:.2f}, Bottom: {bottom_similarity:.2f}, Left: {left_similarity:.2f}, Right: {right_similarity:.2f}")
            
            # If border is similar to adjacent inner pixels in color AND has high std dev,
            # it likely means card content extends to the edge (cut off)
            similarity_threshold = 20.0  # Lower value means more similar (adjust as needed)
            
            cutoff_borders = []
            
            if top_std > std_threshold and top_similarity < similarity_threshold:
                cutoff_borders.append("top")
            if bottom_std > std_threshold and bottom_similarity < similarity_threshold:
                cutoff_borders.append("bottom")
            if left_std > std_threshold and left_similarity < similarity_threshold:
                cutoff_borders.append("left")
            if right_std > std_threshold and right_similarity < similarity_threshold:
                cutoff_borders.append("right")
            
            if cutoff_borders:
                print(f"Card appears to be cut off at: {', '.join(cutoff_borders)}")
                return True
        
        # If we got here, the card likely has consistent borders (not cut off)
        return False
        
    except Exception as e:
        print(f"Error checking if card is cut off: {e}")
        return False  # Default to False on error

def detect_sports_card(image):
    """
    Detect if an image looks like a sports card with visible borders
    Sports cards typically have well-defined borders, text, and consistent color patterns
        
    Args:
        image: The image to analyze
            
    Returns:
        bool: True if the image has characteristics of a sports card, False otherwise
    """
    try:
        # 1. Check for horizontal lines (common in sports cards)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Horizontal lines will have angles close to 0 or 180 degrees
                if angle < 20 or angle > 160:
                    horizontal_lines += 1
        
        # 2. Check for text (common in sports cards)
        # Simple check for potential text areas using gradient information
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        _, sobel_thresh = cv2.threshold(sobel_normalized.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # Count potential text pixels
        text_pixels = cv2.countNonZero(sobel_thresh)
        text_percentage = text_pixels / (gray.shape[0] * gray.shape[1])
        
        # 3. Check for defined borders/rectangles
        # A sports card typically has a rectangular border
        # Use contour detection to find rectangles
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_contours = 0
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # If the contour has 4 points, it might be a rectangle
            if len(approx) == 4:
                rectangular_contours += 1
        
        # Sports card criteria
        has_horizontal_lines = horizontal_lines >= 3
        has_text = text_percentage > 0.05  # At least 5% of the image should be potential text
        has_rectangular_elements = rectangular_contours >= 2
        
        # Debug output
        print(f"Sports card detection - Horizontal lines: {horizontal_lines}, Text percentage: {text_percentage:.2f}, Rectangular elements: {rectangular_contours}")
        
        # If it meets at least 2 of the 3 criteria, it's likely a sports card
        is_sports_card = sum([has_horizontal_lines, has_text, has_rectangular_elements]) >= 2
        
        if is_sports_card:
            print("Detected potential sports card with border")
        
        return is_sports_card
        
    except Exception as e:
        print(f"Error in sports card detection: {e}")
        return False

def check_crop_quality(source_image_path, test_image_path):
    """
    Check if the test image contains only the card from the source image and nothing more.
    
    Args:
        source_image_path: Path to the original source image
        test_image_path: Path to the cropped/test image to verify
        
    Returns:
        tuple: (is_valid, issues) where:
            - is_valid is True if the test image is a valid crop of the source image
            - issues is a list of strings describing any issues found, empty if is_valid is True
    """
    issues = []
    
    # Read both images
    source_img = cv2.imread(str(source_image_path))
    test_img = cv2.imread(str(test_image_path))
    
    if source_img is None or test_img is None:
        return False, ["Could not read one or both images"]
    
    # 1. Check if test image is solid color (shouldn't be)
    if is_solid_color_image(test_image_path):
        issues.append("Test image appears to be a solid color")
    
    # 2. Check if card is cut off
    if is_card_cut_off(test_image_path, source_image_path):
        issues.append("Card appears to be cut off")
    
    # 3. Check if test image is larger than source image
    test_h, test_w = test_img.shape[:2]
    source_h, source_w = source_img.shape[:2]
    if test_h > source_h or test_w > source_w:
        issues.append("Test image is larger than source image")
    
    # 4. Check if test image has characteristics of a sports card
    if not detect_sports_card(test_img):
        issues.append("Test image does not appear to be a valid sports card")
    
    # 5. Check if test image is contained within source image
    # Convert both images to grayscale for template matching
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Try to find the test image in the source image
    result = cv2.matchTemplate(source_gray, test_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # If the best match is too low, the test image might not be from the source
    if max_val < 0.8:  # Threshold may need adjustment
        issues.append("Test image does not appear to be from the source image")
    
    return len(issues) == 0, issues 