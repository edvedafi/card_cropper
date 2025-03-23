import argparse
import zipfile
import os
import shutil
import cv2
import numpy as np
import imutils
from pathlib import Path
import time
import subprocess
import sys
import tty
import termios
from term_image.image import from_file

def get_key():
    """
    Get a single keypress from the user without requiring Enter.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def display_image(image_path):
    """
    Display an image in the terminal using term-image library.
    Returns a tuple of (is_correct, error_category) where error_category is None if is_correct is True,
    or one of "no_image", "cut_off", "skewed", "other" if is_correct is False.
    """
    try:
        # Get image info with OpenCV
        cv_image = cv2.imread(str(image_path))
        if cv_image is None:
            print(f"Error: Could not read image {image_path}")
            return False, "no_image"
            
        # Display image dimensions
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Dimensions: {cv_image.shape[1]}x{cv_image.shape[0]}")
        
        # Check if running in Cursor (can detect via environment variables)
        is_cursor = any('CURSOR' in env_var for env_var in os.environ.keys())
        
        # Use system viewer directly if in Cursor
        if is_cursor:
            print("Detected Cursor editor, opening with system viewer for better image quality...")
            if sys.platform == 'darwin':
                subprocess.run(['open', str(image_path)], check=True)
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(image_path)], check=True)
            elif sys.platform == 'win32':
                subprocess.run(['start', str(image_path)], check=True)
        else:
            # Use term-image for terminals that support it well (like iTerm2)
            try:
                # Load and render the image
                term_img = from_file(str(image_path))
                
                # Set the size to fit in terminal
                term_img.set_size(height=24)  # Fixed height instead of width to avoid Size error
                
                # Display the image
                print(term_img)
            except Exception as e:
                print(f"Could not display image with term-image: {e}")
                print("Opening with system viewer instead...")
                if sys.platform == 'darwin':
                    subprocess.run(['open', str(image_path)], check=True)
                elif sys.platform.startswith('linux'):
                    subprocess.run(['xdg-open', str(image_path)], check=True)
                elif sys.platform == 'win32':
                    subprocess.run(['start', str(image_path)], check=True)
            
        # Get user verification with multiple options
        print("\nPlease choose an option:")
        print("1. Correct - Image looks good")
        print("2. No image - Can't see the card at all")
        print("3. Cut off - Part of the card is missing")
        print("4. Skewed - Card is distorted or at wrong angle")
        print("5. Other - Other issue not covered above")
        print("\nPress a key (1-5): ", end='', flush=True)
        
        while True:
            response = get_key()
            print(response)  # Echo the key pressed
            if response in ['1', '2', '3', '4', '5']:
                if response == '1':
                    return True, None
                elif response == '2':
                    return False, "no_image"
                elif response == '3':
                    return False, "cut_off"
                elif response == '4':
                    return False, "skewed"
                elif response == '5':
                    return False, "other"
            else:
                print("Please press a key between 1 and 5: ", end='', flush=True)
            
    except Exception as e:
        print(f"Error: {e}")
        return False, "no_image"

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

def open_directory(directory_path):
    """
    Open a directory using the default file explorer.
    """
    try:
        print(f"Opening directory: {directory_path}")
        if sys.platform == 'darwin':
            subprocess.run(['open', str(directory_path)], check=True)
        elif sys.platform.startswith('linux'):
            subprocess.run(['xdg-open', str(directory_path)], check=True)
        elif sys.platform == 'win32':
            subprocess.run(['start', '', str(directory_path)], shell=True, check=True)
        return True
    except Exception as e:
        print(f"Error opening directory: {e}")
        return False

def crop_largest_object(image_path, output_path, border_size=5):
    """
    Detects and crops the largest object (card/rectangle) in the image,
    correcting for perspective if the rectangle is skewed.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) - success is True/False, error_reason is a string or None
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        error_reason = "Could not read image"
        print(f"Error: {error_reason} {image_path}")
        return False, error_reason
    
    # Create a copy for output
    orig = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple approaches to find the best card detection
    best_contour = None
    best_contour_area = 0
    
    # Approach 1: Canny edge detection with different thresholds
    for low, high in [(30, 150), (50, 200), (20, 100)]:
        edged = cv2.Canny(blurred, low, high)
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if contours:
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if area > best_contour_area:
                    best_contour_area = area
                    best_contour = contour
    
    # Approach 2: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if contours:
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area > best_contour_area:
                best_contour_area = area
                best_contour = contour
    
    # Approach 3: Various binary thresholding
    for threshold in [127, 100, 150]:
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if contours:
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if area > best_contour_area:
                    best_contour_area = area
                    best_contour = contour
    
    # Approach 4: Color-based segmentation
    # Try to find cards by color differences (white/light colored cards)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White/light colored mask
    lower_val = np.array([0, 0, 150])
    upper_val = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if contours:
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area > best_contour_area:
                best_contour_area = area
                best_contour = contour
    
    # If no good contour found
    if best_contour is None or best_contour_area < (height * width * 0.005):  # Even lower threshold: 0.5%
        error_reason = "No suitable contours found"
        print(f"{error_reason} in {image_path}, skipping...")
        cv2.imwrite(str(output_path), orig)
        return False, error_reason
    
    # Draw the contour on a debug image
    debug_image = orig.copy()
    cv2.drawContours(debug_image, [best_contour], -1, (0, 255, 0), 2)
    
    # Save debug image to the debug directory
    debug_path = str(output_path).replace('/final/', '/debug/')
    debug_path = debug_path.replace('.jpg', '_debug.jpg').replace('.jpeg', '_debug.jpeg').replace('.png', '_debug.png')
    cv2.imwrite(debug_path, debug_image)
    
    # Approximate the contour
    peri = cv2.arcLength(best_contour, True)
    
    # Try multiple epsilon values for approximation
    best_approx = None
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        approx = cv2.approxPolyDP(best_contour, epsilon_factor * peri, True)
        if len(approx) == 4:
            best_approx = approx
            break
    
    # If we couldn't find a good 4-point approximation, try convex hull
    if best_approx is None or len(best_approx) != 4:
        hull = cv2.convexHull(best_contour)
        hull_peri = cv2.arcLength(hull, True)
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            approx = cv2.approxPolyDP(hull, epsilon_factor * hull_peri, True)
            if len(approx) == 4:
                best_approx = approx
                break
    
    # If still no good approximation, create a bounding rectangle
    if best_approx is None or len(best_approx) != 4:
        x, y, w, h = cv2.boundingRect(best_contour)
        best_approx = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])
    
    # Get the 4 corners in the correct order
    best_approx = best_approx.reshape(len(best_approx), 2)
    
    # If not 4 points, just use the bounding rectangle
    if len(best_approx) != 4:
        x, y, w, h = cv2.boundingRect(best_contour)
        best_approx = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum of coordinates - smallest is top-left, largest is bottom-right
    s = best_approx.sum(axis=1)
    rect[0] = best_approx[np.argmin(s)]  # Top-left
    rect[2] = best_approx[np.argmax(s)]  # Bottom-right
    
    # Diff of coordinates - smallest is top-right, largest is bottom-left
    diff = np.diff(best_approx, axis=1)
    rect[1] = best_approx[np.argmin(diff)]  # Top-right
    rect[3] = best_approx[np.argmax(diff)]  # Bottom-left
    
    # Calculate width and height of the new image
    width_1 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    width_2 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    height_2 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    # Ensure reasonable dimensions
    if max_width < 10 or max_height < 10 or max_width > width * 1.5 or max_height > height * 1.5:
        error_reason = "Invalid dimensions detected"
        print(f"{error_reason} in {image_path}, skipping...")
        cv2.imwrite(str(output_path), debug_image)
        return False, error_reason
    
    # Add border to dimensions (2 * border_size because we add to both sides)
    output_width = max_width + (2 * border_size)
    output_height = max_height + (2 * border_size)
    
    # Set up destination points for the perspective transformation with added border
    dst = np.array([
        [border_size, border_size],                     # Top-left with border
        [output_width - border_size - 1, border_size],  # Top-right with border
        [output_width - border_size - 1, output_height - border_size - 1],  # Bottom-right with border
        [border_size, output_height - border_size - 1]  # Bottom-left with border
    ], dtype=np.float32)
    
    # Compute perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply perspective transformation to the new dimensions with border
    warped = cv2.warpPerspective(orig, transform_matrix, (output_width, output_height))
    
    # Save the cropped image
    cv2.imwrite(str(output_path), warped)
    return True, None

def process_zip_file(zip_path, border_size=5, clean_input=True, open_errors_dir=False, additional_params=None):
    """
    Process a zip file containing images.
    
    Args:
        zip_path: Path to the zip file
        border_size: Size of border to add around detected cards
        clean_input: Whether to clean the input directory before extraction
        open_errors_dir: Whether to open the errors directory after processing (always False now)
        additional_params: Dictionary of additional parameters to adjust processing
    """
    # Get the base name of the zip file without extension
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Create input and output directories if they don't exist
    input_dir = Path("input")
    final_dir = Path("output") / "final" / zip_name
    debug_dir = Path("output") / "debug" / zip_name
    
    # Create error category directories
    errors_base_dir = Path("output") / "errors" / zip_name
    errors_no_image_dir = errors_base_dir / "no_image"
    errors_cut_off_dir = errors_base_dir / "cut_off"
    errors_skewed_dir = errors_base_dir / "skewed"
    errors_other_dir = errors_base_dir / "other"
    errors_processing_dir = errors_base_dir / "processing"
    
    # Clean input directory if requested
    if clean_input and input_dir.exists():
        print(f"Cleaning input directory {input_dir}...")
        for item in input_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    input_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True, parents=True)
    debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Create all error directories
    errors_base_dir.mkdir(exist_ok=True, parents=True)
    errors_no_image_dir.mkdir(exist_ok=True, parents=True)
    errors_cut_off_dir.mkdir(exist_ok=True, parents=True)
    errors_skewed_dir.mkdir(exist_ok=True, parents=True)
    errors_other_dir.mkdir(exist_ok=True, parents=True)
    errors_processing_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing {zip_path}...")
    print(f"Extracting to {input_dir}")
    
    # Extract zip contents to input directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_dir)
    
    # Define common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Process image files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Track error images with reasons
    error_images = []  # List to store tuples of (filename, reason)
    user_rejected_images = {
        "no_image": [],
        "cut_off": [],
        "skewed": [],
        "other": []
    }
    
    start_time = time.time()
    total_images = 0
    
    # First, count total images
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1
    
    print(f"Found {total_images} images in the zip file")
    
    # Process each image
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(root, file)
                final_path = final_dir / file
                debug_path = debug_dir / file
                
                print(f"Processing image {processed_count + 1}/{total_images}: {file}")
                
                # Process the image
                success, error_reason = crop_largest_object(input_path, final_path, border_size)
                if success:
                    processed_count += 1
                    
                    # Check if the output image is a solid color (failed cropping)
                    is_solid = is_solid_color_image(final_path)
                    
                    # Auto-categorize solid color output images
                    if is_solid:
                        print(f"Automatically categorizing {file} as 'No image' (cropped image is solid color)")
                        error_path = errors_no_image_dir / file
                        os.makedirs(os.path.dirname(error_path), exist_ok=True)
                        user_rejected_images["no_image"].append(file)
                        shutil.move(str(final_path), str(error_path))
                        error_count += 1
                        continue
                    
                    # Check if the card appears to be cut off
                    if is_card_cut_off(final_path, input_path):
                        print(f"Automatically categorizing {file} as 'Cut off' (missing background border detected)")
                        error_path = errors_cut_off_dir / file
                        os.makedirs(os.path.dirname(error_path), exist_ok=True)
                        user_rejected_images["cut_off"].append(file)
                        shutil.move(str(final_path), str(error_path))
                        error_count += 1
                        continue
                    
                    # Display the image and ask for verification if it wasn't auto-categorized
                    print(f"\nVerifying image: {file}")
                    is_correct, error_category = display_image(final_path)
                    
                    if not is_correct:
                        # Move to appropriate error directory based on the category
                        if error_category == "no_image":
                            error_path = errors_no_image_dir / file
                            error_dir_name = "no_image"
                            user_rejected_images["no_image"].append(file)
                        elif error_category == "cut_off":
                            error_path = errors_cut_off_dir / file
                            error_dir_name = "cut_off"
                            user_rejected_images["cut_off"].append(file)
                        elif error_category == "skewed":
                            error_path = errors_skewed_dir / file
                            error_dir_name = "skewed"
                            user_rejected_images["skewed"].append(file)
                        elif error_category == "other":
                            error_path = errors_other_dir / file
                            error_dir_name = "other"
                            user_rejected_images["other"].append(file)
                        else:
                            # Fallback if category is not recognized
                            error_path = errors_base_dir / file
                            error_dir_name = "unspecified"
                        
                        shutil.move(str(final_path), str(error_path))
                        print(f"Moved incorrect image to: {error_path}")
                        error_count += 1
                    else:
                        print("Image verified as correct.")
                else:
                    processed_count += 1
                    
                    # Auto-categorize contour detection failures
                    if error_reason and "No suitable contours found" in error_reason:
                        print(f"Automatically categorizing {file} as 'No image' (no card detected)")
                        error_path = errors_no_image_dir / file
                        os.makedirs(os.path.dirname(error_path), exist_ok=True)
                        user_rejected_images["no_image"].append(file)
                        shutil.copy2(input_path, error_path)
                        error_count += 1
                        continue
                    
                    # For failed contour detection, copy the original to final path for verification
                    shutil.copy2(input_path, final_path)
                    
                    # Provide helpful message about contour detection failure
                    if error_reason and "No suitable contours found" in error_reason:
                        print("\nNOTE: No card contour was detected in this image.")
                    else:
                        print(f"\nNOTE: Processing error: {error_reason}")
                    
                    # Still ask for user verification
                    print(f"\nVerifying image: {file}")
                    is_correct, error_category = display_image(final_path)
                    
                    if not is_correct:
                        # Move to appropriate error directory based on the category
                        if error_category == "no_image":
                            error_path = errors_no_image_dir / file
                            error_dir_name = "no_image"
                            user_rejected_images["no_image"].append(file)
                        elif error_category == "cut_off":
                            error_path = errors_cut_off_dir / file
                            error_dir_name = "cut_off"
                            user_rejected_images["cut_off"].append(file)
                        elif error_category == "skewed":
                            error_path = errors_skewed_dir / file
                            error_dir_name = "skewed"
                            user_rejected_images["skewed"].append(file)
                        elif error_category == "other":
                            error_path = errors_other_dir / file
                            error_dir_name = "other"
                            user_rejected_images["other"].append(file)
                        else:
                            # Fallback if category is not recognized
                            error_path = errors_base_dir / file
                            error_dir_name = "unspecified"
                        
                        # Copy input image to error directory
                        os.makedirs(os.path.dirname(error_path), exist_ok=True)
                        shutil.copy2(input_path, error_path)
                        print(f"Image categorized as '{error_dir_name}', saved to {error_path}")
                        # Remove from final directory
                        if final_path.exists():
                            os.remove(final_path)
                        error_count += 1
                    else:
                        print("Image verified as correct despite processing issues.")
                        # The image is already in the final directory
                
                # Update progress
                if processed_count % 5 == 0:  # Update every 5 images
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(f"Progress: {processed_count}/{total_images} images ({rate:.2f} images/second)")
    
    # Count errors by category
    no_image_count = len(user_rejected_images["no_image"])
    cut_off_count = len(user_rejected_images["cut_off"])
    skewed_count = len(user_rejected_images["skewed"])
    other_count = len(user_rejected_images["other"])
    processing_error_count = len(error_images)
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Successfully cropped {processed_count - error_count} images with {border_size}px border to {final_dir}")
    print(f"Debug images saved to {debug_dir}")
    
    # Print error summary
    total_errors = no_image_count + cut_off_count + skewed_count + other_count + processing_error_count
    if total_errors > 0:
        print("\n==== Error Summary ====")
        
        # Report user-rejected images by category
        if no_image_count > 0:
            print(f"\nNo image detected ({no_image_count} images):")
            for idx, filename in enumerate(user_rejected_images["no_image"], 1):
                print(f"{idx}. {filename}")
                
        if cut_off_count > 0:
            print(f"\nImage cut off ({cut_off_count} images):")
            for idx, filename in enumerate(user_rejected_images["cut_off"], 1):
                print(f"{idx}. {filename}")
                
        if skewed_count > 0:
            print(f"\nImage skewed ({skewed_count} images):")
            for idx, filename in enumerate(user_rejected_images["skewed"], 1):
                print(f"{idx}. {filename}")
        
        if other_count > 0:
            print(f"\nOther issues ({other_count} images):")
            for idx, filename in enumerate(user_rejected_images["other"], 1):
                print(f"{idx}. {filename}")
        
        # Report processing errors
        if processing_error_count > 0:
            print(f"\nProcessing errors ({processing_error_count} images):")
            for idx, (filename, reason) in enumerate(error_images, 1):
                print(f"{idx}. {filename}: {reason}")
        
        # Report total errors
        if total_errors > 0:
            error_percentage = (total_errors / processed_count) * 100 if processed_count > 0 else 0
            print(f"\nTotal error images: {total_errors} ({error_percentage:.1f}% of processed images)")
            print(f"Error images saved to: {errors_base_dir}")
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images")

def main():
    parser = argparse.ArgumentParser(description='Process a zip file containing images.')
    parser.add_argument('file_path', help='Path to the zip file or individual image')
    parser.add_argument('--border', type=int, default=5, help='Size of border (in pixels) to add around detected cards. Default is 5.')
    parser.add_argument('--clean', action='store_true', default=True, help='Clean input directory before extraction (default: True)')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Do not clean input directory before extraction')
    parser.add_argument('--open-errors', action='store_true', default=False, help='Open errors directory after processing (default: False)')
    parser.add_argument('--no-open-errors', dest='open_errors', action='store_false', help='Do not open errors directory after processing')
    parser.add_argument('--single-image', action='store_true', help='Process a single image file instead of a zip')
    
    args = parser.parse_args()
    
    # Check if input is a single image
    if args.single_image or args.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
        print(f"Processing single image: {args.file_path}")
        
        # Create output directories
        final_dir = Path("output") / "final" / "single"
        debug_dir = Path("output") / "debug" / "single"
        final_dir.mkdir(exist_ok=True, parents=True)
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        # Process the image
        output_path = final_dir / os.path.basename(args.file_path)
        success, error_reason = crop_largest_object(args.file_path, output_path, args.border)
        
        if success:
            print(f"Successfully processed {args.file_path}")
            print(f"Output saved to {output_path}")
            
            # Display the image for verification
            is_correct, error_category = display_image(output_path)
            
            if not is_correct:
                print(f"Image verification failed: {error_category}")
        else:
            print(f"Failed to process {args.file_path}: {error_reason}")
    else:
        # Process as zip file
        process_zip_file(args.file_path, args.border, args.clean, args.open_errors)

if __name__ == "__main__":
    main() 