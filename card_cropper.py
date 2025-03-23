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
    
    # Create input and output directories if they don't exist
    input_dir.mkdir(exist_ok=True, parents=True)
    final_dir.mkdir(exist_ok=True, parents=True)
    debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Create all error directories
    errors_base_dir.mkdir(exist_ok=True, parents=True)
    errors_no_image_dir.mkdir(exist_ok=True, parents=True)
    errors_cut_off_dir.mkdir(exist_ok=True, parents=True)
    errors_skewed_dir.mkdir(exist_ok=True, parents=True)
    errors_other_dir.mkdir(exist_ok=True, parents=True)
    errors_processing_dir.mkdir(exist_ok=True, parents=True)
    
    # Clean input directory if requested
    if clean_input:
        print(f"Cleaning input directory {input_dir}...")
        for item in input_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    # Extract zip file
    print(f"Processing {zip_path}...")
    print(f"Extracting to {input_dir}")
    start_time = time.time()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir)
            
        # Get all image files in the input directory
        image_files = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), input_dir)
                    image_files.append(relative_path)
                    
        total_images = len(image_files)
        print(f"Found {total_images} images in the zip file")
        
        # Track counts and errors
        processed_count = 0
        successful_count = 0
        skipped_count = 0
        
        # Track error images with reasons
        error_images = []  # List to store tuples of (filename, reason)
        user_rejected_images = {
            "no_image": [],
            "cut_off": [],
            "skewed": [],
            "other": []
        }
        
        # Process each image
        for file in image_files:
            # Convert string path to Path object
            file_path = Path(file)
            input_path = input_dir / file_path
            final_path = final_dir / file_path
            
            if input_path.is_file():
                print(f"Processing image {processed_count + 1}/{total_images}: {file}")
                
                # Create parent directories for output files if they don't exist
                final_path.parent.mkdir(exist_ok=True, parents=True)
                
                try:
                    # Check if image is solid color first
                    if is_solid_color_image(input_path):
                        print(f"Automatically categorizing {file} as 'No image' (solid color detected)")
                        error_path = errors_no_image_dir / file_path
                        error_path.parent.mkdir(exist_ok=True, parents=True)
                        user_rejected_images["no_image"].append(str(file))
                        shutil.copy2(input_path, error_path)
                        processed_count += 1
                        continue
                    
                    # Process the image
                    success, error_reason = crop_largest_object(input_path, final_path, border_size)
                    
                    if success:
                        processed_count += 1
                        
                        # Show the cropped image and get user verification
                        is_correct, error_category = display_image(final_path)
                        
                        if is_correct:
                            successful_count += 1
                        else:
                            # Move to appropriate error directory based on the category
                            if error_category == "no_image":
                                error_path = errors_no_image_dir / file_path
                                error_dir_name = "no_image"
                                user_rejected_images["no_image"].append(str(file))
                            elif error_category == "cut_off":
                                error_path = errors_cut_off_dir / file_path
                                error_dir_name = "cut_off"
                                user_rejected_images["cut_off"].append(str(file))
                            elif error_category == "skewed":
                                error_path = errors_skewed_dir / file_path
                                error_dir_name = "skewed"
                                user_rejected_images["skewed"].append(str(file))
                            elif error_category == "other":
                                error_path = errors_other_dir / file_path
                                error_dir_name = "other"
                                user_rejected_images["other"].append(str(file))
                            else:
                                # Fallback if category is not recognized
                                error_path = errors_base_dir / file_path
                                error_dir_name = "unspecified"
                                
                            # Move input image to appropriate error category
                            os.makedirs(os.path.dirname(error_path), exist_ok=True)
                            shutil.copy2(input_path, error_path)
                            print(f"Image rejected as '{error_dir_name}', saved to {error_path}")
                            
                            # Remove the file from final directory
                            if final_path.exists():
                                os.remove(final_path)
                    else:
                        processed_count += 1
                        # If contour detection failed, categorize as "no_image" automatically
                        if "No suitable contours found" in error_reason:
                            print(f"Automatically categorizing {file} as 'No image' (no card detected)")
                            error_path = errors_no_image_dir / file_path
                            error_path.parent.mkdir(exist_ok=True, parents=True)
                            user_rejected_images["no_image"].append(str(file))
                            shutil.copy2(input_path, error_path)
                        else:
                            # Other processing errors
                            error_images.append((file, error_reason))
                            
                            # Move processing errors to their own directory
                            if Path(input_path).is_file():
                                error_path = errors_processing_dir / file_path
                                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                                shutil.copy(str(input_path), str(error_path))
                except Exception as e:
                    processed_count += 1
                    error_reason = f"Error processing image: {e}"
                    error_images.append((file, error_reason))
                    error_path = errors_processing_dir / file_path
                    os.makedirs(os.path.dirname(error_path), exist_ok=True)
                    shutil.copy(str(input_path), str(error_path))
            else:
                print(f"Warning: file not found or not a file: {input_path}")
                skipped_count += 1
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print("\nProcessing complete!")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Successfully cropped {successful_count} images")
        
        # Count errors by category
        no_image_count = len(user_rejected_images["no_image"])
        cut_off_count = len(user_rejected_images["cut_off"])
        skewed_count = len(user_rejected_images["skewed"])
        other_count = len(user_rejected_images["other"])
        processing_error_count = len(error_images)
        
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
    except Exception as e:
        print(f"Error processing zip file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process a zip file containing images.')
    parser.add_argument('zip_path', help='Path to the zip file')
    parser.add_argument('--border', type=int, default=5, help='Size of border (in pixels) to add around detected cards. Default is 5.')
    parser.add_argument('--clean', action='store_true', default=True, help='Clean input directory before extraction (default: True)')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Do not clean input directory before extraction')
    parser.add_argument('--open-errors', action='store_true', default=False, help='Open errors directory after processing (default: False)')
    parser.add_argument('--no-open-errors', dest='open_errors', action='store_false', help='Do not open errors directory after processing')
    
    args = parser.parse_args()
    
    process_zip_file(args.zip_path, args.border, args.clean, args.open_errors)

if __name__ == "__main__":
    main() 