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
import glob
import platform
import torch
from ultralytics import YOLO

print("Script started...")  # Debug print

def get_key():
    """
    Get a single keypress from the user without requiring Enter.
    """
    print("get_key() called...")  # Debug print
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

def detect_card_basic(image_path, output_path, border_size=5):
    """
    Basic card detection using simple contour detection and edge finding.
    Best for clear, well-lit images with good contrast between card and background.
    """
    # ... existing implementation of crop_largest_object ...

def detect_card_enhanced(image_path, output_path, border_size=5):
    """
    Enhanced card detection using multiple edge detection methods and rotation correction.
    Better at handling problematic images with skew or moderate lighting issues.
    """
    # ... existing implementation of second_pass_processing ...

def detect_card_with_ml(image_path, output_path, border_size=5):
    """
    Machine learning based card detection using YOLO model.
    Excellent at handling complex backgrounds and unusual card orientations.
    Falls back to adaptive thresholding if ML detection fails.
    """
    # ... existing implementation of third_pass_processing ...

def detect_card_aggressive(image_path, output_path, border_size=5):
    """
    Aggressive card detection using multiple color spaces and thresholding techniques.
    Best for severely distorted images or those with very poor quality/lighting.
    Uses HSV, LAB color spaces and multiple binary thresholding approaches.
    """
    # ... existing implementation of fourth_pass_processing ...

def process_zip_file(zip_path, border=10, clean=True, open_errors=False):
    # ... existing setup code ...

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        # ... existing setup code ...

        # Try all detection methods until success
        print("\nAttempting basic card detection...")
        success, error_reason = detect_card_basic(str(input_copy_path), output_path, border)
        
        if success:
            print("Basic detection completed successfully")
            is_correct, error_category = display_image(output_path)
            if is_correct:
                print("Basic detection successful!")
                continue
            else:
                print(f"Basic detection result not satisfactory: {error_category}")
        
        # Enhanced detection
        print("\nAttempting enhanced card detection...")
        success, error_reason = detect_card_enhanced(str(input_copy_path), output_path, border)
        
        if success:
            print("Enhanced detection completed successfully")
            is_correct, error_category = display_image(output_path)
            if is_correct:
                print("Enhanced detection successful!")
                continue
            else:
                print(f"Enhanced detection result not satisfactory: {error_category}")
        
        # ML-based detection
        print("\nAttempting ML-based card detection...")
        success, error_reason = detect_card_with_ml(str(input_copy_path), output_path, border)
        
        if success:
            print("ML-based detection completed successfully")
            is_correct, error_category = display_image(output_path)
            if is_correct:
                print("ML-based detection successful!")
                continue
            else:
                print(f"ML-based detection result not satisfactory: {error_category}")
        
        # Aggressive detection
        print("\nAttempting aggressive card detection...")
        success, error_reason = detect_card_aggressive(str(input_copy_path), output_path, border)
        
        if success:
            print("Aggressive detection completed successfully")
            is_correct, error_category = display_image(output_path)
            if is_correct:
                print("Aggressive detection successful!")
                continue
            else:
                print(f"Aggressive detection result not satisfactory: {error_category}")
        
        # If we get here, all detection methods have failed
        print("\nAll detection methods failed. Moving to errors directory...")

def main():
    print("main() function started...")  # Debug print
    parser = argparse.ArgumentParser(description='Process a zip file containing images.')
    parser.add_argument('file_path', help='Path to the zip file or individual image')
    parser.add_argument('--border', type=int, default=5, help='Size of border (in pixels) to add around detected cards. Default is 5.')
    parser.add_argument('--clean', action='store_true', default=True, help='Clean input directory before extraction (default: True)')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Do not clean input directory before extraction')
    parser.add_argument('--open-errors', action='store_true', default=False, help='Open errors directory after processing (default: False)')
    parser.add_argument('--no-open-errors', dest='open_errors', action='store_false', help='Do not open errors directory after processing')
    parser.add_argument('--single-image', action='store_true', help='Process a single image file instead of a zip')
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")  # Debug print
    
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
        
        print(f"Creating directories: {test_output_dir}")  # Debug print
        
        # Create directories
        test_input_dir.mkdir(exist_ok=True, parents=True)
        final_dir.mkdir(exist_ok=True, parents=True)
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy the image to the input directory
        input_copy_path = test_input_dir / file_name
        shutil.copy2(args.file_path, input_copy_path)
        
        # Process the image
        output_path = final_dir / file_name
        success, error_reason = detect_card_basic(str(input_copy_path), output_path, args.border)
        
        if success:
            print("Basic detection completed successfully")
            is_correct, error_category = display_image(output_path)
            if is_correct:
                print("Basic detection successful!")
            else:
                print(f"Basic detection result not satisfactory: {error_category}")
        else:
            print(f"Basic detection failed: {error_reason}")
    else:
        # Process as zip file
        print(f"Processing zip file: {args.file_path}")
        process_zip_file(args.file_path, args.border, args.clean, args.open_errors)

if __name__ == "__main__":
    print("Script is being run as main...")  # Debug print
    main()
    print("Script finished.")  # Debug print 