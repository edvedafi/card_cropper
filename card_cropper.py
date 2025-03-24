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
    
    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 50, 200)
    
    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # If no contours found
    if not contours:
        error_reason = "No contours found"
        print(f"{error_reason} in {image_path}, skipping...")
        cv2.imwrite(str(output_path), orig)
        return False, error_reason
    
    # Use the largest contour
    largest_contour = contours[0]
    
    # Calculate the minimum area needed for a valid contour (0.5% of the image)
    min_contour_area = height * width * 0.005
    
    # Check if the contour is too small
    if cv2.contourArea(largest_contour) < min_contour_area:
        error_reason = "No suitable contours found"
        print(f"{error_reason} in {image_path}, skipping...")
        cv2.imwrite(str(output_path), orig)
        return False, error_reason
    
    # Draw the contour on a debug image
    debug_image = orig.copy()
    cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)
    
    # Save debug image to the debug directory
    # Extract test name from the output path structure (output/{test_name}/...)
    output_path_str = str(output_path)
    parts = Path(output_path_str).parts
    
    # Find "output" in the parts to get the test name
    if "output" in parts:
        output_index = parts.index("output")
        if output_index + 1 < len(parts):
            test_name = parts[output_index + 1]  # The part right after "output"
        else:
            test_name = "unknown"
    else:
        test_name = "unknown"
    
    # Create path for the debug directory
    debug_dir = os.path.join("output", test_name, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get the debug file base name
    debug_base = os.path.basename(output_path_str)
    
    # Save the debug image in the correct debug directory
    debug_path = os.path.join(debug_dir, f"contours_{debug_base}")
    cv2.imwrite(debug_path, debug_image)
    
    # Approximate the contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    # If the approximation doesn't have 4 points, try to use a bounding rectangle
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(largest_contour)
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
    
    # Create the output directory if needed
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    
    # Save the cropped image
    cv2.imwrite(str(output_path), warped)
    return True, None

def process_zip_file(zip_path, border_size=5, clean_input=True, open_errors_dir=False):
    """
    Process a zip file containing images.
    
    Args:
        zip_path: Path to the zip file
        border_size: Number of pixels to add as border around detected cards
        clean_input: Whether to clean the input directory before extraction
        open_errors_dir: Whether to open the errors directory after processing
    """
    # Extract the name of the zip file without extension for using in output paths
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Create the new output directory structure
    # All outputs will be in output/{test_name}/...
    test_output_dir = Path("output") / zip_name
    
    # Define subdirectories
    test_input_dir = test_output_dir / "input"  # Store original images
    final_dir = test_output_dir / "final"       # Store successful crops
    debug_dir = test_output_dir / "debug"       # Store debug images
    errors_dir = test_output_dir / "errors"     # Store error cases
    
    # Create subdirectories for error categories
    errors_no_image_dir = errors_dir / "no_image"
    errors_cut_off_dir = errors_dir / "cut_off"
    errors_skewed_dir = errors_dir / "skewed"
    errors_other_dir = errors_dir / "other"
    errors_processing_dir = errors_dir / "processing"
    
    # Create all directories
    test_input_dir.mkdir(exist_ok=True, parents=True)
    final_dir.mkdir(exist_ok=True, parents=True)
    debug_dir.mkdir(exist_ok=True, parents=True)
    errors_dir.mkdir(exist_ok=True, parents=True)
    errors_no_image_dir.mkdir(exist_ok=True, parents=True)
    errors_cut_off_dir.mkdir(exist_ok=True, parents=True)
    errors_skewed_dir.mkdir(exist_ok=True, parents=True)
    errors_other_dir.mkdir(exist_ok=True, parents=True)
    errors_processing_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract the zip file directly to our new input directory
    print(f"Processing {zip_path}...")
    print(f"Extracting to {test_input_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(str(test_input_dir))
    
    # Get list of image files from our test input directory
    image_files = []
    for root, _, files in os.walk(test_input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in the zip file")
    
    # Process each image
    successful_count = 0
    error_count = 0
    user_rejected_images = {
        "no_image": [],
        "cut_off": [],
        "skewed": [],
        "other": []
    }
    
    start_time = time.time()
    
    for i, input_path in enumerate(image_files, 1):
        file = os.path.basename(input_path)
        output_path = final_dir / file
        print(f"Processing image {i}/{len(image_files)}: {file}")
        
        # Get image dimensions for diagnostic purposes
        try:
            img = cv2.imread(str(input_path))
            if img is not None:
                img_height, img_width = img.shape[:2]
                print(f"Input image dimensions: {img_width}x{img_height}")
        except Exception as e:
            print(f"Error reading image dimensions: {e}")
        
        # First pass - standard processing
        success, error_reason = crop_largest_object(input_path, output_path, border_size)
        
        # Check the dimensions of the output image
        cut_off_detected = False
        try:
            result_img = cv2.imread(str(output_path))
            if result_img is not None:
                res_height, res_width = result_img.shape[:2]
                print(f"First pass output dimensions: {res_width}x{res_height}")
                
                # Check for extremely distorted aspect ratios
                aspect_ratio = float(res_width) / res_height if res_height > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5:
                    print(f"WARNING: Extreme aspect ratio detected: {aspect_ratio:.4f}")
                    cut_off_detected = True
        except Exception as e:
            print(f"Error checking output dimensions: {e}")
        
        # Always attempt second pass for problematic files or when first pass fails
        if not success or cut_off_detected:
            reason_msg = f"First pass failed: {error_reason}" if not success else "First pass produced distorted output"
            print(f"{reason_msg}. Attempting second pass with alternative methods...")
            
            # Save a copy of the first pass result for comparison
            first_pass_output = debug_dir / ("first_pass_" + file)
            try:
                shutil.copy2(str(output_path), str(first_pass_output))
            except Exception as e:
                print(f"Error saving first pass output: {e}")
            
            # Determine the most likely failure reason and use an appropriate second method
            if error_reason and "No suitable contours found" in error_reason or cut_off_detected:
                # Try a more aggressive preprocessing approach
                second_pass_output = errors_processing_dir / ("second_pass_" + file)
                success, error_reason = second_pass_processing(input_path, second_pass_output, border_size)
                
                if success:
                    print(f"Second pass successful! Using improved result.")
                    # Copy the successful second pass result to the final output
                    shutil.copy2(str(second_pass_output), str(output_path))
                    
                    # Check the new dimensions
                    try:
                        second_img = cv2.imread(str(output_path))
                        if second_img is not None:
                            second_height, second_width = second_img.shape[:2]
                            print(f"Second pass output dimensions: {second_width}x{second_height}")
                    except Exception as e:
                        print(f"Error checking second pass output: {e}")
            
        # If auto-detection is used and we have additional info, use it for categorization
        if not success:
            # Attempt to categorize based on detected issues
            if is_solid_color_image(str(output_path)):
                print(f"Automatically categorizing {file} as 'No image' (cropped image is solid color)")
                error_path = errors_no_image_dir / file
                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                user_rejected_images["no_image"].append(file)
                shutil.copy2(input_path, error_path)
                error_count += 1
                continue
                
            if error_reason and "No suitable contours found" in error_reason:
                print(f"Automatically categorizing {file} as 'No image' (no card detected)")
                error_path = errors_no_image_dir / file
                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                user_rejected_images["no_image"].append(file)
                shutil.copy2(input_path, error_path)
                error_count += 1
                continue
                
            # Check if it's a cut-off card
            if is_card_cut_off(output_path, input_path):
                print(f"Automatically categorizing {file} as 'Cut off' (missing background border detected)")
                error_path = errors_cut_off_dir / file
                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                user_rejected_images["cut_off"].append(file)
                shutil.copy2(input_path, error_path)
                error_count += 1
                continue
                
            # For other types of failures, we'll need user verification
        
        # Show the processed image and get user's verification
        is_correct, error_category = display_image(output_path)
        
        if is_correct:
            successful_count += 1
        else:
            # Move the original image to the appropriate error directory
            if error_category == "no_image":
                error_path = errors_no_image_dir / file
                user_rejected_images["no_image"].append(file)
            elif error_category == "cut_off":
                error_path = errors_cut_off_dir / file
                user_rejected_images["cut_off"].append(file)
            elif error_category == "skewed":
                error_path = errors_skewed_dir / file 
                user_rejected_images["skewed"].append(file)
            else:
                error_path = errors_other_dir / file
                user_rejected_images["other"].append(file)
            
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            shutil.copy2(input_path, error_path)
            print(f"Moved incorrect image to: {error_path}")
            error_count += 1
        
        # Show progress every 10 images
        if i % 10 == 0 and i < len(image_files):
            elapsed_time = time.time() - start_time
            images_per_second = i / elapsed_time if elapsed_time > 0 else 0
            print(f"Progress: {i}/{len(image_files)} images ({images_per_second:.2f} images/second)")
    
    # Calculate and print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\nProcessing complete!")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Successfully cropped {successful_count} images with {border_size}px border to {final_dir}")
    print(f"Debug images saved to {debug_dir}")
    
    if error_count > 0:
        print("\n==== Error Summary ====\n")
        
        if user_rejected_images["no_image"]:
            print(f"No image detected ({len(user_rejected_images['no_image'])} images):")
            for i, img in enumerate(user_rejected_images["no_image"], 1):
                print(f"{i}. {img}")
            print()
        
        if user_rejected_images["cut_off"]:
            print(f"Image cut off ({len(user_rejected_images['cut_off'])} images):")
            for i, img in enumerate(user_rejected_images["cut_off"], 1):
                print(f"{i}. {img}")
            print()
        
        if user_rejected_images["skewed"]:
            print(f"Image skewed ({len(user_rejected_images['skewed'])} images):")
            for i, img in enumerate(user_rejected_images["skewed"], 1):
                print(f"{i}. {img}")
            print()
        
        if user_rejected_images["other"]:
            print(f"Other issues ({len(user_rejected_images['other'])} images):")
            for i, img in enumerate(user_rejected_images["other"], 1):
                print(f"{i}. {img}")
            print()
        
        total_error_percent = (error_count / len(image_files)) * 100
        print(f"Total error images: {error_count} ({total_error_percent:.1f}% of processed images)")
        print(f"Error images saved to: {errors_dir}")
    
    if open_errors_dir and error_count > 0:
        open_directory_prompt = input("Would you like to open the errors directory? (y/n): ")
        if open_directory_prompt.lower() == 'y':
            open_directory(errors_dir)

def second_pass_processing(image_path, output_path, border_size=5):
    """
    Alternative processing method for when the standard approach fails.
    This method uses multiple approaches to try to detect and crop the card.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) - success is True/False, error_reason is a string or None
    """
    try:
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
        print(f"DEBUG: Second pass - Image size: {width}x{height}")
        
        # Extract the test name from the output path
        # The structure is now: output/{test_name}/errors/processing/second_pass_file.jpg
        output_path_str = str(output_path)
        parts = Path(output_path_str).parts
        
        # Find "output" in the parts to get the test name
        if "output" in parts:
            output_index = parts.index("output")
            if output_index + 1 < len(parts):
                test_name = parts[output_index + 1]  # The part right after "output"
            else:
                test_name = "unknown"
        else:
            test_name = "unknown"
        
        # Create path for the second_pass debug directory
        second_pass_debug_dir = os.path.join("output", test_name, "debug", "second_pass")
        os.makedirs(second_pass_debug_dir, exist_ok=True)
        
        # Get the debug file base name
        debug_base = os.path.basename(output_path_str)
        
        # Print debug info for directory structure
        print(f"DEBUG: Saving second pass debug images to {second_pass_debug_dir}")
        
        # Save the original image for reference
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"original_{debug_base}"), orig)
        
        # Try multiple approaches to find the best card detection
        best_contour = None
        best_contour_area = 0
        
        # Approach -1: Try rotation correction first for skewed cards
        # This approach is specifically designed for cards that appear skewed but are otherwise visible
        # We'll try to detect lines in the image and rotate it to align with the dominant orientation
        rotation_angle = None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"rotation_canny_{debug_base}"), edges)
            
            # Use probabilistic Hough transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
            
            if lines is not None and len(lines) > 0:
                print(f"DEBUG: Found {len(lines)} lines for rotation analysis")
                
                # Draw detected lines on a copy of the image
                line_image = orig.copy()
                
                # Calculate angles of lines
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate angle of line
                    if x2 - x1 != 0:  # Avoid division by zero
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        
                        # Normalize angle to be between -45 and 45 degrees
                        # (we only care about rotation within this range for cards)
                        while angle < -45:
                            angle += 90
                        while angle > 45:
                            angle -= 90
                        
                        # Only consider angles that are close to horizontal or vertical
                        if abs(angle) < 45:
                            angles.append(angle)
                
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"rotation_lines_{debug_base}"), line_image)
                
                if angles:
                    # Calculate the most common angle using a histogram
                    hist, bins = np.histogram(angles, bins=90, range=(-45, 45))
                    dominant_angle_idx = np.argmax(hist)
                    rotation_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
                    
                    print(f"DEBUG: Dominant rotation angle: {rotation_angle:.2f} degrees")
                    
                    # Draw the rotation angle on the image
                    angle_image = orig.copy()
                    cv2.putText(angle_image, f"Rotation: {rotation_angle:.2f} degrees", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw a line showing the rotation
                    center = (width // 2, height // 2)
                    line_length = min(width, height) // 4
                    end_point = (
                        int(center[0] + line_length * np.cos(np.radians(rotation_angle))),
                        int(center[1] + line_length * np.sin(np.radians(rotation_angle)))
                    )
                    cv2.line(angle_image, center, end_point, (0, 0, 255), 3)
                    cv2.imwrite(os.path.join(second_pass_debug_dir, f"rotation_angle_{debug_base}"), angle_image)
                    
                    # If the rotation angle is significant, rotate the image
                    if abs(rotation_angle) > 1.0:  # Only rotate if angle is more than 1 degree
                        # Get rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        
                        # Determine new image dimensions
                        abs_cos = abs(rotation_matrix[0, 0])
                        abs_sin = abs(rotation_matrix[0, 1])
                        new_width = int(height * abs_sin + width * abs_cos)
                        new_height = int(height * abs_cos + width * abs_sin)
                        
                        # Adjust the rotation matrix
                        rotation_matrix[0, 2] += (new_width / 2) - center[0]
                        rotation_matrix[1, 2] += (new_height / 2) - center[1]
                        
                        # Perform the rotation
                        rotated_image = cv2.warpAffine(orig, rotation_matrix, (new_width, new_height))
                        cv2.imwrite(os.path.join(second_pass_debug_dir, f"rotated_{debug_base}"), rotated_image)
                        
                        # Use this rotated image for subsequent processing
                        image = rotated_image
                        height, width = image.shape[:2]
                        orig = image.copy()
                        print("DEBUG: Applied rotation correction")
        except Exception as e:
            print(f"DEBUG: Error in rotation correction: {e}")
        
        # Special handling for severe distortions - Check for past distortion first
        # If we had a severely cut off card in the first pass, let's try a different approach
        # by focusing on detecting large rectangular shapes
        
        # Approach 0: Specialized aspect ratio-aware detection
        # This approach is specifically designed for cards that get severely distorted
        # Focus on finding rectangular shapes with reasonable aspect ratios
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # Stronger blur to reduce noise
        
        # Apply a more aggressive edge detection
        edged = cv2.Canny(blurred, 20, 80)  # Lower thresholds to catch more edges
        
        # Save edge detection result for debugging
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"edges_{debug_base}"), edged)
        
        # Dilate to connect edge fragments that might be broken
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Save dilated edges for debugging
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"dilated_edges_{debug_base}"), closed)
        
        # Find contours in the enhanced edge map
        contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Save all contours on a debug image
        contour_img = orig.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"all_contours_{debug_base}"), contour_img)
        
        # Handle case where no contours are found
        if not contours:
            print("DEBUG: No contours found in specialized detection")
        else:
            # Sort by area but filter based on aspect ratio
            good_contours = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Draw this contour on a separate image with the aspect ratio
                contour_debug = orig.copy()
                cv2.drawContours(contour_debug, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(contour_debug, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(contour_debug, f"AR: {aspect_ratio:.2f}, Area: {cv2.contourArea(contour)}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"contour_{len(good_contours)}_{debug_base}"), contour_debug)
                
                # Most cards have aspect ratios between 0.5 and 2.0
                # (height might be 1-2x width, or width might be 1-2x height)
                # For problematic cards with extreme distortion, let's be more lenient
                if 0.25 <= aspect_ratio <= 4.0:
                    area = cv2.contourArea(contour)
                    if area > height * width * 0.05:  # Must be at least 5% of image
                        good_contours.append(contour)
                        if area > best_contour_area:
                            best_contour_area = area
                            best_contour = contour
                            print(f"DEBUG: Found good aspect ratio contour: {aspect_ratio:.2f}, area: {area}")
            
            # If we found good contours, save them for debugging
            if good_contours:
                good_contour_img = orig.copy()
                cv2.drawContours(good_contour_img, good_contours, -1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"good_contours_{debug_base}"), good_contour_img)
        
        # If we found a good contour with reasonable aspect ratio, we'll use that
        # Otherwise, continue with other approaches
        
        # Approach 1: Canny edge detection with different thresholds
        # Only run if we don't have a good contour yet
        if best_contour is None:
            for idx, (low, high) in enumerate([(30, 150), (50, 200), (20, 100)]):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, low, high)
                
                # Save edge detection result for debugging
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"canny{idx+1}_{debug_base}"), edged)
                
                contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                
                # Save contours for debugging
                if contours:
                    canny_contours = orig.copy()
                    cv2.drawContours(canny_contours, contours, -1, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(second_pass_debug_dir, f"canny{idx+1}_contours_{debug_base}"), canny_contours)
                
                if contours:
                    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                        area = cv2.contourArea(contour)
                        if area > best_contour_area:
                            best_contour_area = area
                            best_contour = contour
        
        # Approach 6: Last resort for extremely distorted images
        # Try using a fixed card-sized rectangle in the center of the image
        if best_contour is None:
            print("DEBUG: All detection methods failed, using fixed-size card rectangle in center")
            # Use a standard card aspect ratio of 2.5 x 3.5 inches
            if width > height:
                # Landscape image, card is likely in center
                card_width = int(min(width * 0.8, height * (2.5/3.5) * 1.2))
                card_height = int(card_width * (3.5/2.5))
            else:
                # Portrait image, card is likely in center
                card_height = int(min(height * 0.8, width * (3.5/2.5) * 1.2))
                card_width = int(card_height * (2.5/3.5))
            
            # Center the card in the image
            x = (width - card_width) // 2
            y = (height - card_height) // 2
            
            # Create a rectangle contour
            rect_contour = np.array([
                [[x, y]],
                [[x + card_width, y]],
                [[x + card_width, y + card_height]],
                [[x, y + card_height]]
            ], dtype=np.int32)
            
            best_contour = rect_contour
            best_contour_area = card_width * card_height
            
            # Save the guessed rectangle
            fixed_rect_img = orig.copy()
            cv2.drawContours(fixed_rect_img, [best_contour], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"fixed_rectangle_{debug_base}"), fixed_rect_img)
            
            print(f"DEBUG: Created fixed rectangle: w={card_width}, h={card_height}, area={best_contour_area}")
        
        # If still no good contour found
        if best_contour is None or best_contour_area < (height * width * 0.003):  # Even lower threshold: 0.3%
            error_reason = "No suitable contours found"
            print(f"{error_reason} in {image_path}, skipping...")
            cv2.imwrite(str(output_path), orig)
            return False, error_reason
        
        # Draw the contour on a debug image
        debug_image = orig.copy()
        cv2.drawContours(debug_image, [best_contour], -1, (0, 255, 0), 2)
        
        # If we detected a rotation angle, add it to the debug image
        if rotation_angle is not None:
            cv2.putText(debug_image, f"Rotation corrected: {rotation_angle:.2f} degrees", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save best contour for debugging
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"best_contour_{debug_base}"), debug_image)
        
        # Save debug image to the processing directory
        debug_path = str(output_path).replace('second_pass_', 'second_pass_debug_')
        cv2.imwrite(debug_path, debug_image)
        
        # Approximate the contour
        peri = cv2.arcLength(best_contour, True)
        
        # Try multiple epsilon values for approximation
        best_approx = None
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            approx = cv2.approxPolyDP(best_contour, epsilon_factor * peri, True)
            
            # Save the approximation for debugging
            approx_img = orig.copy()
            cv2.drawContours(approx_img, [approx], -1, (0, 255, 0), 2)
            cv2.putText(approx_img, f"Epsilon: {epsilon_factor}, Points: {len(approx)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"approx_{epsilon_factor}_{debug_base}"), approx_img)
            
            if len(approx) == 4:
                best_approx = approx
                break
        
        # If we couldn't find a good 4-point approximation, try convex hull
        if best_approx is None or len(best_approx) != 4:
            hull = cv2.convexHull(best_contour)
            
            # Save the hull for debugging
            hull_img = orig.copy()
            cv2.drawContours(hull_img, [hull], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"hull_{debug_base}"), hull_img)
            
            hull_peri = cv2.arcLength(hull, True)
            for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
                approx = cv2.approxPolyDP(hull, epsilon_factor * hull_peri, True)
                print(f"DEBUG: Second pass - Hull epsilon {epsilon_factor} gives {len(approx)} points")
                
                # Save hull approximation
                hull_approx_img = orig.copy()
                cv2.drawContours(hull_approx_img, [approx], -1, (0, 255, 0), 2)
                cv2.putText(hull_approx_img, f"Hull Epsilon: {epsilon_factor}, Points: {len(approx)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"hull_approx_{epsilon_factor}_{debug_base}"), hull_approx_img)
                
                if len(approx) == 4:
                    best_approx = approx
                    print(f"DEBUG: Second pass - Found good 4-point hull approximation with epsilon {epsilon_factor}")
                    break
        
        # If still no good approximation, use a bounding rectangle
        if best_approx is None or len(best_approx) != 4:
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # Check for extreme aspect ratios that indicate a problem
            aspect_ratio = float(w) / h if h > 0 else 0
            print(f"DEBUG: Bounding rect aspect ratio: {aspect_ratio:.4f}, w={w}, h={h}")
            
            # Save the bounding rectangle
            rect_img = orig.copy()
            cv2.rectangle(rect_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(rect_img, f"Bounding Rect AR: {aspect_ratio:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"bounding_rect_{debug_base}"), rect_img)
            
            # If we have an extreme aspect ratio, try to correct it
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                print("DEBUG: Extreme aspect ratio detected, attempting correction")
                # We've detected a severely distorted rectangle
                # Instead of using this bounding rect, let's use a more reasonable proportion
                
                # For a typical card, use a standard aspect ratio of 2.5/3.5 ≈ 0.714
                # This is the ratio of a standard trading card
                if w > h:
                    # Width is larger, adjust height
                    new_h = int(w * (3.5/2.5))
                    # Center it vertically
                    new_y = max(y - (new_h - h) // 2, 0)
                    h = min(new_h, height - new_y)  # Don't exceed image bounds
                else:
                    # Height is larger, adjust width
                    new_w = int(h * (2.5/3.5))
                    # Center it horizontally
                    new_x = max(x - (new_w - w) // 2, 0)
                    w = min(new_w, width - new_x)  # Don't exceed image bounds
                    
                print(f"DEBUG: Corrected to w={w}, h={h}")
                
                # Save the corrected rectangle
                corrected_rect_img = orig.copy()
                cv2.rectangle(corrected_rect_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(corrected_rect_img, f"Corrected AR: {float(w)/h:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(second_pass_debug_dir, f"corrected_rect_{debug_base}"), corrected_rect_img)
            
            best_approx = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ])
        
        # Get the 4 corners in the correct order
        best_approx = best_approx.reshape(len(best_approx), 2)
        
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
        
        # Draw the ordered points on a debug image
        ordered_points_img = orig.copy()
        for i, point in enumerate(rect):
            cv2.circle(ordered_points_img, tuple(point.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(ordered_points_img, str(i), tuple(point.astype(int) + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"ordered_points_{debug_base}"), ordered_points_img)
        
        # Calculate width and height of the new image
        width_1 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        width_2 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        max_width = max(int(width_1), int(width_2))
        
        height_1 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        height_2 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        max_height = max(int(height_1), int(height_2))
        
        print(f"DEBUG: Second pass - Final rect dimensions: width={max_width}, height={max_height}")
        
        # Check for excessive distortion in the perspective transformation
        aspect_ratio = float(max_width) / max_height if max_height > 0 else 0
        print(f"DEBUG: Final perspective aspect ratio: {aspect_ratio:.4f}")
        
        # If the aspect ratio is extreme, we likely have a bad perspective transform
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            print("DEBUG: Extreme aspect ratio in perspective transformation, attempting correction")
            
            # Standard trading card aspect ratio is 2.5" x 3.5" ≈ 0.714
            # We'll adjust to approximate this
            if max_width > max_height:
                max_height = int(max_width * (3.5/2.5))
            else:
                max_width = int(max_height * (2.5/3.5))
                
            print(f"DEBUG: Corrected dimensions: width={max_width}, height={max_height}")
            
            # Save the corrected dimensions for debugging
            corrected_dimensions_img = orig.copy()
            cv2.putText(corrected_dimensions_img, f"Corrected Size: {max_width}x{max_height}, AR: {float(max_width)/max_height:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(second_pass_debug_dir, f"corrected_dimensions_{debug_base}"), corrected_dimensions_img)
        
        # Ensure reasonable dimensions
        if max_width < 10 or max_height < 10:
            error_reason = "Invalid dimensions detected (too small)"
            print(f"Second pass: {error_reason} in {image_path}, skipping...")
            cv2.imwrite(str(output_path), debug_image)
            return False, error_reason
            
        # Don't allow excessive expansion but be more lenient for second pass
        if max_width > width * 2 or max_height > height * 2:
            error_reason = "Invalid dimensions detected (too large)"
            print(f"Second pass: {error_reason} in {image_path}, skipping...")
            cv2.imwrite(str(output_path), debug_image)
            return False, error_reason
        
        # Add border to dimensions
        output_width = max_width + (2 * border_size)
        output_height = max_height + (2 * border_size)
        
        # Set up destination points with added border
        dst = np.array([
            [border_size, border_size],
            [output_width - border_size - 1, border_size],
            [output_width - border_size - 1, output_height - border_size - 1],
            [border_size, output_height - border_size - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(orig, transform_matrix, (output_width, output_height))
        
        # Save the cropped image
        cv2.imwrite(str(output_path), warped)
        
        # Save a copy to debug dir
        cv2.imwrite(os.path.join(second_pass_debug_dir, f"final_warped_{debug_base}"), warped)
        
        print("DEBUG: Second pass successful!")
        return True, None
        
    except Exception as e:
        import traceback
        error_reason = f"Error in second pass: {str(e)}"
        print(error_reason)
        traceback.print_exc()
        return False, error_reason

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
        
        # Get the file name without extension as the test name
        file_name = os.path.basename(args.file_path)
        test_name = os.path.splitext(file_name)[0]
        
        # Create test directory structure
        test_output_dir = Path("output") / test_name
        test_input_dir = test_output_dir / "input"
        final_dir = test_output_dir / "final"
        debug_dir = test_output_dir / "debug"
        
        # Create directories
        test_input_dir.mkdir(exist_ok=True, parents=True)
        final_dir.mkdir(exist_ok=True, parents=True)
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy the image to the input directory
        input_copy_path = test_input_dir / file_name
        shutil.copy2(args.file_path, input_copy_path)
        
        # Process the image
        output_path = final_dir / file_name
        success, error_reason = crop_largest_object(str(input_copy_path), output_path, args.border)
        
        if success:
            print(f"Successfully processed {args.file_path}")
            print(f"Output saved to {output_path}")
            
            # Display the image for verification
            is_correct, error_category = display_image(output_path)
            
            if not is_correct:
                print(f"Image verification failed: {error_category}")
                
                # Create error directories
                errors_dir = test_output_dir / "errors"
                error_category_dir = errors_dir / error_category
                error_category_dir.mkdir(exist_ok=True, parents=True)
                
                # Copy the original image to the appropriate error directory
                error_path = error_category_dir / file_name
                shutil.copy2(str(input_copy_path), error_path)
                print(f"Original image saved to: {error_path}")
        else:
            print(f"Failed to process {args.file_path}: {error_reason}")
    else:
        # Process as zip file
        process_zip_file(args.file_path, args.border, args.clean, args.open_errors)

if __name__ == "__main__":
    main() 