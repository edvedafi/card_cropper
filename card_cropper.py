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
import traceback

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
    Find and crop the largest object (card) in an image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) - success is True/False, error_reason is a string or None
    """
    try:
        # Read the image
        print(f"Reading image from {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            error_reason = "Could not read image"
            print(f"Error: {error_reason} {image_path}")
            return False, error_reason
        
        # Create a debug copy
        debug_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 75, 200)
        
        # Dilate to connect edge fragments
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add a debug message
        print(f"Found {len(contours)} contours")
        
        if not contours:
            error_reason = "No suitable contours found"
            print(f"Error: {error_reason}")
            cv2.imwrite(str(output_path), image)  # Save original image for debugging
            return False, error_reason
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = cv2.contourArea(largest_contour)
        
        # Get image dimensions and calculate minimum contour size
        height, width = image.shape[:2]
        min_contour_size = height * width * 0.005  # 0.5% of image area
        
        # Add debug info
        print(f"Largest contour area: {largest_contour_area} pixels")
        print(f"Minimum contour threshold: {min_contour_size} pixels")
        
        # Draw all contours on debug image
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
        
        # Save debug image
        debug_dir = os.path.dirname(output_path)
        debug_path = os.path.join(debug_dir, f"debug_contours_{os.path.basename(output_path)}")
        cv2.imwrite(debug_path, debug_image)
        
        # Check if the largest contour is big enough
        if largest_contour_area < min_contour_size:
            error_reason = "No suitable contours found"
            print(f"Error: {error_reason} (largest contour too small)")
            cv2.imwrite(str(output_path), image)  # Save original image for debugging
            return False, error_reason
        
        # Find the rotated rectangle (min area rectangle)
        rect = cv2.minAreaRect(largest_contour)
        
        # Get the corner points
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Draw the box on the debug image
        debug_box = image.copy()
        cv2.drawContours(debug_box, [box], 0, (0, 0, 255), 2)
        
        # Save the debug image with the box
        debug_box_path = os.path.join(debug_dir, f"debug_box_{os.path.basename(output_path)}")
        cv2.imwrite(debug_box_path, debug_box)
        
        # Get width and height of the rectangle
        width_rect = rect[1][0]
        height_rect = rect[1][1]
        
        # Create a perspective transform to flatten the card
        # Get the corners in a particular order (top-left, top-right, bottom-right, bottom-left)
        rect_points = order_points(box.astype("float32"))
        
        # Calculate the width and height of the transformed image
        width_dest = max(int(width_rect), int(height_rect))
        height_dest = min(int(width_rect), int(height_rect))
        
        if width_dest < 10 or height_dest < 10:
            error_reason = "Detected card is too small"
            print(f"Error: {error_reason} ({width_dest}x{height_dest})")
            cv2.imwrite(str(output_path), image)
            return False, error_reason
        
        # Add a border around the image
        dst_width = width_dest + 2 * border_size
        dst_height = height_dest + 2 * border_size
        
        # Set up destination points with added border
        dst_points = np.array([
            [border_size, border_size],                              # Top left
            [dst_width - border_size, border_size],                  # Top right
            [dst_width - border_size, dst_height - border_size],     # Bottom right
            [border_size, dst_height - border_size]                  # Bottom left
        ], dtype="float32")
        
        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect_points, dst_points)
        
        # Apply the perspective transformation
        warped = cv2.warpPerspective(image, M, (dst_width, dst_height))
        
        # Save the cropped card
        cv2.imwrite(str(output_path), warped)
        
        print(f"Successfully cropped card to {output_path}")
        return True, None
        
    except Exception as e:
        error_reason = f"Error processing image: {str(e)}"
        print(f"Exception: {error_reason}")
        traceback.print_exc()  # Print stack trace
        return False, error_reason

def process_zip_file(zip_path, border_size=5, clean_input=True, open_errors_dir=False, auto_second_pass=False):
    """
    Process a zip file containing images.
    
    Args:
        zip_path: Path to the zip file
        border_size: Number of pixels to add as border around detected cards
        clean_input: Whether to clean the input directory before extraction
        open_errors_dir: Whether to open the errors directory after processing
        auto_second_pass: Whether to attempt a second pass for failed images
    """
    # Extract the name of the zip file without extension for using in output paths
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Clean the input directory if requested
    input_dir = "input"
    if clean_input:
        print(f"Cleaning input directory {input_dir}...")
        if os.path.exists(input_dir):
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    
    # Create directories for output
    final_dir = Path("output") / "final" / zip_name
    debug_dir = Path("output") / "debug" / zip_name
    errors_dir = Path("output") / "errors" / zip_name
    errors_no_image_dir = errors_dir / "no_image"
    errors_cut_off_dir = errors_dir / "cut_off"
    errors_skewed_dir = errors_dir / "skewed"
    errors_other_dir = errors_dir / "other"
    errors_processing_dir = errors_dir / "processing"
    
    final_dir.mkdir(exist_ok=True, parents=True)
    debug_dir.mkdir(exist_ok=True, parents=True)
    errors_dir.mkdir(exist_ok=True, parents=True)
    errors_no_image_dir.mkdir(exist_ok=True, parents=True)
    errors_cut_off_dir.mkdir(exist_ok=True, parents=True)
    errors_skewed_dir.mkdir(exist_ok=True, parents=True)
    errors_other_dir.mkdir(exist_ok=True, parents=True)
    errors_processing_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract the zip file
    print(f"Processing {zip_path}...")
    print(f"Extracting to {input_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_dir)
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(input_dir):
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
        
        # First pass - standard processing
        success, error_reason = crop_largest_object(input_path, output_path, border_size)
        
        # If first pass failed, attempt a second pass with alternative methods
        if not success and auto_second_pass:
            print(f"First pass failed: {error_reason}. Attempting second pass with alternative methods...")
            
            # Determine the most likely failure reason and use an appropriate second method
            if error_reason and "No suitable contours found" in error_reason:
                # Try a more aggressive preprocessing approach
                second_pass_output = errors_processing_dir / ("second_pass_" + file)
                success, error_reason = second_pass_processing(input_path, second_pass_output, border_size)
                
                if success:
                    print(f"Second pass successful! Using improved result.")
                    # Copy the successful second pass result to the final output
                    shutil.copy2(str(second_pass_output), str(output_path))
            
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
        
        # Try multiple methods and use the best result
        methods = [
            "enhanced_preprocessing",
            "hough_lines",
            "color_segmentation"
        ]
        
        best_result = None
        best_score = 0
        best_method = None
        
        for method in methods:
            print(f"DEBUG: Second pass - Trying method: {method}")
            
            try:
                if method == "enhanced_preprocessing":
                    result = try_enhanced_preprocessing(image, orig, border_size, height, width)
                elif method == "hough_lines":
                    result = try_hough_lines(image, orig, border_size, height, width)
                elif method == "color_segmentation":
                    result = try_color_segmentation(image, orig, border_size, height, width)
                
                if result and result.get("success"):
                    # Score the result (higher is better)
                    score = result.get("score", 0)
                    print(f"DEBUG: Second pass - Method {method} succeeded with score {score}")
                    
                    if score > best_score:
                        best_result = result
                        best_score = score
                        best_method = method
                else:
                    print(f"DEBUG: Second pass - Method {method} failed")
            except Exception as e:
                print(f"DEBUG: Second pass - Method {method} raised exception: {str(e)}")
        
        if best_result:
            print(f"DEBUG: Second pass - Using best method: {best_method} with score {best_score}")
            warped = best_result["result"]
            
            # Save debug image
            debug_dir = os.path.dirname(output_path)
            debug_path = os.path.join(debug_dir, f"debug_{best_method}_{os.path.basename(output_path)}")
            cv2.imwrite(debug_path, best_result.get("debug_image", warped))
            
            # Save the cropped image
            cv2.imwrite(str(output_path), warped)
            print("DEBUG: Second pass successful!")
            return True, None
        
        print("DEBUG: Second pass - All methods failed")
        return False, "All second-pass methods failed"
        
    except Exception as e:
        error_reason = f"Error in second pass: {str(e)}"
        print(error_reason)
        return False, error_reason

def try_enhanced_preprocessing(image, orig, border_size, height, width):
    """Enhanced preprocessing method for second pass"""
    # Convert to grayscale with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply stronger blurring to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
    
    # Use a combination of edge detection methods
    canny_edges = cv2.Canny(blurred, 30, 150)
    
    # Dilate to connect edge fragments
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=2)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours in the enhanced edge map
    contours = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Find the largest contour by area
    if not contours:
        return None
    
    best_contour = max(contours, key=cv2.contourArea)
    best_contour_area = cv2.contourArea(best_contour)
    
    # Use a more lenient threshold for the second pass
    min_contour_threshold = (height * width * 0.003)  # 0.3% instead of 0.5%
    
    if best_contour_area < min_contour_threshold:
        return None
    
    # Draw the contour on a debug image
    debug_image = orig.copy()
    cv2.drawContours(debug_image, [best_contour], -1, (0, 255, 0), 2)
    
    # Alternative approach 2: Use a more aggressive approximation strategy
    # Try convex hull first
    hull = cv2.convexHull(best_contour)
    hull_peri = cv2.arcLength(hull, True)
    
    # Try a wider range of epsilon values
    best_approx = None
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        approx = cv2.approxPolyDP(hull, epsilon_factor * hull_peri, True)
        if len(approx) == 4:
            best_approx = approx
            print(f"DEBUG: Enhanced preprocessing - Found good 4-point hull approximation with epsilon {epsilon_factor}")
            break
    
    # If no good approximation, use a simple bounding rectangle
    if best_approx is None or len(best_approx) != 4:
        x, y, w, h = cv2.boundingRect(hull)
        print(f"DEBUG: Enhanced preprocessing - Using bounding rectangle: x={x}, y={y}, w={w}, h={h}")
        best_approx = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
    
    # Reshape for perspective transform
    best_approx = np.array(best_approx).reshape(4, 2)
    
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
        return None
    
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
    
    # Calculate a score for this method - contour area and approximation quality
    contour_ratio = best_contour_area / (height * width)
    score = 50 + (contour_ratio * 100)  # Base score plus ratio-based component
    
    return {
        "success": True,
        "result": warped,
        "debug_image": debug_image,
        "score": score
    }

def try_hough_lines(image, orig, border_size, height, width):
    """Use Hough line detection to find card edges"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Use probabilistic Hough transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=max(width, height)/10, maxLineGap=20)
    
    if lines is None or len(lines) < 4:
        print("DEBUG: Hough lines - Not enough lines detected")
        return None
    
    # Create debug image
    debug_image = orig.copy()
    
    # Draw all lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Group lines into horizontal and vertical clusters
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # If line is more horizontal than vertical
        if dx > dy:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)
    
    # Need at least 2 horizontal and 2 vertical lines for a quadrilateral
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("DEBUG: Hough lines - Need at least 2 horizontal and 2 vertical lines")
        return None
    
    # Find two best horizontal and two best vertical lines
    # Sort horizontal lines by y-coordinate (top to bottom)
    horizontal_lines.sort(key=lambda line: (line[0][1] + line[0][3]) / 2)
    top_line = horizontal_lines[0][0]
    bottom_line = horizontal_lines[-1][0]
    
    # Sort vertical lines by x-coordinate (left to right)
    vertical_lines.sort(key=lambda line: (line[0][0] + line[0][2]) / 2)
    left_line = vertical_lines[0][0]
    right_line = vertical_lines[-1][0]
    
    # Draw the selected lines in a different color
    cv2.line(debug_image, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (255, 0, 0), 3)
    cv2.line(debug_image, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (255, 0, 0), 3)
    cv2.line(debug_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 3)
    cv2.line(debug_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 3)
    
    # Calculate intersections to find corners
    def line_intersection(line1, line2):
        # Convert to the form ax + by = c
        a1 = line1[3] - line1[1]
        b1 = line1[0] - line1[2]
        c1 = a1 * line1[0] + b1 * line1[1]
        
        a2 = line2[3] - line2[1]
        b2 = line2[0] - line2[2]
        c2 = a2 * line2[0] + b2 * line2[1]
        
        determinant = a1 * b2 - a2 * b1
        
        if determinant == 0:
            # Lines are parallel
            return None
        
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        return (int(x), int(y))
    
    # Find the four corners
    try:
        top_left = line_intersection(top_line, left_line)
        top_right = line_intersection(top_line, right_line)
        bottom_right = line_intersection(bottom_line, right_line)
        bottom_left = line_intersection(bottom_line, left_line)
        
        if None in [top_left, top_right, bottom_right, bottom_left]:
            print("DEBUG: Hough lines - One or more intersections failed")
            return None
        
        # Draw corner points
        for corner in [top_left, top_right, bottom_right, bottom_left]:
            cv2.circle(debug_image, corner, 10, (0, 0, 255), -1)
        
        # Create the rectangle for perspective transform
        rect = np.array([
            [top_left[0], top_left[1]],
            [top_right[0], top_right[1]],
            [bottom_right[0], bottom_right[1]],
            [bottom_left[0], bottom_left[1]]
        ], dtype=np.float32)
        
        # Calculate width and height
        width_1 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        width_2 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        max_width = max(int(width_1), int(width_2))
        
        height_1 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        height_2 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        max_height = max(int(height_1), int(height_2))
        
        # Check reasonable dimensions
        if max_width < 10 or max_height < 10 or max_width > width * 1.5 or max_height > height * 1.5:
            print("DEBUG: Hough lines - Unreasonable dimensions")
            return None
        
        # Add border
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
        
        # Calculate score based on line detection quality
        line_count_score = min(20, len(lines))
        corner_distance_score = 0
        
        # Check if corners form a reasonable quadrilateral
        # Score based on how close the rectangle is to a perfect rectangle
        width_ratio = min(width_1, width_2) / max(width_1, width_2) if max(width_1, width_2) > 0 else 0
        height_ratio = min(height_1, height_2) / max(height_1, height_2) if max(height_1, height_2) > 0 else 0
        
        corner_distance_score = (width_ratio + height_ratio) * 50
        
        score = 40 + line_count_score + corner_distance_score
        
        return {
            "success": True,
            "result": warped,
            "debug_image": debug_image,
            "score": score
        }
    
    except Exception as e:
        print(f"DEBUG: Hough lines - Error during processing: {str(e)}")
        return None

def try_color_segmentation(image, orig, border_size, height, width):
    """Use color segmentation to isolate the card from background"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create a mask using K-means clustering
    # Flatten the image and convert to floats
    pixels = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Map the labels to their respective centers
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Create debug image
    debug_image = segmented_image.copy()
    
    # Create binary masks for each cluster
    masks = []
    for i in range(k):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == i] = 255
        mask = mask.reshape(height, width)
        masks.append(mask)
    
    # Find the mask that most likely represents the card
    best_mask = None
    best_score = -1
    
    for i, mask in enumerate(masks):
        # Use morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if not contours:
            continue
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # Skip if too small
        if contour_area < (width * height * 0.05):
            continue
        
        # Score based on how rectangular the contour is
        rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(largest_contour)).astype(np.int32))
        rectangularity = contour_area / rect_area if rect_area > 0 else 0
        
        # Score based on area and rectangularity
        score = (contour_area / (width * height)) * 50 + rectangularity * 50
        
        if score > best_score:
            best_score = score
            best_mask = mask
            best_contour = largest_contour
    
    if best_mask is None:
        print("DEBUG: Color segmentation - No good mask found")
        return None
    
    # Draw the best contour on the debug image
    cv2.drawContours(debug_image, [best_contour], -1, (0, 255, 0), 3)
    
    # Try to find a quadrilateral approximation
    epsilon = 0.05 * cv2.arcLength(best_contour, True)
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    
    # If we don't get exactly 4 points, try to adjust epsilon
    if len(approx) != 4:
        # Try different epsilon values
        for factor in [0.02, 0.03, 0.07, 0.1]:
            epsilon = factor * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
            if len(approx) == 4:
                break
    
    # If we still don't have 4 points, use the bounding rectangle
    if len(approx) != 4:
        print(f"DEBUG: Color segmentation - Using bounding rectangle (approx had {len(approx)} points)")
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        approx = np.int32(box)
    
    # Draw the approximation on the debug image
    cv2.drawContours(debug_image, [approx], -1, (255, 0, 0), 3)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Convert approx to the right format
    approx = approx.reshape(len(approx), 2)
    
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
    
    # Check reasonable dimensions
    if max_width < 10 or max_height < 10 or max_width > width * 1.5 or max_height > height * 1.5:
        print(f"DEBUG: Color segmentation - Unreasonable dimensions: {max_width}x{max_height}")
        return None
    
    # Add border
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
    
    # Score based on area and rectangularity
    rectangularity = contour_area / (max_width * max_height) if (max_width * max_height) > 0 else 0
    score = 30 + (rectangularity * 80)
    
    return {
        "success": True,
        "result": warped,
        "debug_image": debug_image,
        "score": score
    }

def order_points(pts):
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: numpy array of shape (4, 2)
        
    Returns:
        numpy array of shape (4, 2) with ordered points
    """
    # Initialize result array in the same type as input
    rect = np.zeros((4, 2), dtype=pts.dtype)
    
    # Sum of coordinates - smallest is top-left, largest is bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    # Diff of coordinates - smallest is top-right, largest is bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def main():
    """Main function to process command line arguments."""
    parser = argparse.ArgumentParser(description='Crop trading cards from images')
    parser.add_argument('input', help='Path to input image file, zip file, or directory')
    parser.add_argument('--border', type=int, default=5, help='Border size in pixels, default is 5')
    parser.add_argument('--clean-input', action='store_true', help='Clean input directory before processing')
    parser.add_argument('--open-errors', action='store_true', help='Open errors directory after processing')
    parser.add_argument('--auto-second-pass', action='store_true', 
                      help='Enable second pass processing for failed images')
    
    args = parser.parse_args()
    
    # Process based on the input type
    input_path = args.input
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.zip'):
            # Process zip file
            process_zip_file(input_path, args.border, args.clean_input, args.open_errors, args.auto_second_pass)
        elif any(input_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
            # Process single image
            output_dir = Path('output/final/single')
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / os.path.basename(input_path)
            
            success, error_reason = crop_largest_object(input_path, output_path, args.border)
            
            if not success and args.auto_second_pass:
                print(f"First pass failed: {error_reason}. Attempting second pass...")
                errors_processing_dir = Path('output/errors/single/processing')
                errors_processing_dir.mkdir(exist_ok=True, parents=True)
                second_pass_output = errors_processing_dir / ("second_pass_" + os.path.basename(input_path))
                success, error_reason = second_pass_processing(input_path, second_pass_output, args.border)
                
                if success:
                    print("Second pass successful! Using improved result.")
                    shutil.copy2(str(second_pass_output), str(output_path))
            
            # Verify the output
            is_correct, error_category = display_image(output_path)
            
            if not is_correct:
                # Create appropriate error directory
                if error_category == "no_image":
                    error_dir = Path('output/errors/single/no_image')
                elif error_category == "cut_off":
                    error_dir = Path('output/errors/single/cut_off')
                elif error_category == "skewed":
                    error_dir = Path('output/errors/single/skewed')
                else:
                    error_dir = Path('output/errors/single/other')
                
                error_dir.mkdir(exist_ok=True, parents=True)
                error_path = error_dir / os.path.basename(input_path)
                
                # Copy the original image to the error directory
                shutil.copy2(input_path, error_path)
                print(f"Image marked as '{error_category}', saved to {error_path}")
            else:
                print(f"Image processed successfully: {output_path}")
    
    elif os.path.isdir(input_path):
        # Process directory
        print(f"Processing directory: {input_path}")
        # Add directory processing logic here

# Call the main function if the script is run directly
if __name__ == "__main__":
    main() 