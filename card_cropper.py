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
from term_image.image import from_file

def display_image(image_path):
    """
    Display an image in the terminal using term-image library.
    Returns True if the user confirms the image is correct, False otherwise.
    """
    try:
        # Get image info with OpenCV
        cv_image = cv2.imread(str(image_path))
        if cv_image is None:
            print(f"Error: Could not read image {image_path}")
            return False
            
        # Display image dimensions
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Dimensions: {cv_image.shape[1]}x{cv_image.shape[0]}")
        
        # Use term-image to display the image
        try:
            # Load and render the image
            term_img = from_file(str(image_path))
            
            # Set the size to fit in terminal - fixed for Size issue
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
            
        # Get user verification
        while True:
            response = input("\nIs this image correct? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' for yes or 'n' for no.")
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def crop_largest_object(image_path, output_path, border_size=5):
    """
    Detects and crops the largest object (card/rectangle) in the image,
    correcting for perspective if the rectangle is skewed.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
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
        print(f"No suitable contours found in {image_path}, skipping...")
        cv2.imwrite(str(output_path), orig)
        return False
    
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
        print(f"Invalid dimensions detected in {image_path}, skipping...")
        cv2.imwrite(str(output_path), debug_image)
        return False
    
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
    return True

def process_zip_file(zip_path, border_size=5, clean_input=True, additional_params=None):
    """
    Process a zip file containing images.
    
    Args:
        zip_path: Path to the zip file
        border_size: Size of border to add around detected cards
        clean_input: Whether to clean the input directory before extraction
        additional_params: Dictionary of additional parameters to adjust processing
    """
    # Get the base name of the zip file without extension
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Create input and output directories if they don't exist
    input_dir = Path("input")
    final_dir = Path("output") / "final" / zip_name
    debug_dir = Path("output") / "debug" / zip_name
    errors_dir = Path("output") / "errors" / zip_name
    
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
    errors_dir.mkdir(exist_ok=True, parents=True)
    
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
                if crop_largest_object(input_path, final_path, border_size):
                    processed_count += 1
                    
                    # Display the image and ask for verification
                    print(f"\nVerifying image: {file}")
                    if not display_image(final_path):
                        # Move to errors directory if incorrect
                        error_path = errors_dir / file
                        shutil.move(str(final_path), str(error_path))
                        print(f"Moved incorrect image to: {error_path}")
                        error_count += 1
                    else:
                        print("Image verified as correct.")
                else:
                    skipped_count += 1
                
                # Update progress
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    print(f"Progress: {processed_count}/{total_images} images ({rate:.2f} images/second)")
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Successfully cropped {processed_count} images with {border_size}px border to {final_dir}")
    print(f"Debug images saved to {debug_dir}")
    print(f"Error images moved to {errors_dir}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images")
    if error_count > 0:
        print(f"{error_count} images moved to errors directory")

def main():
    parser = argparse.ArgumentParser(description='Process a zip file containing images.')
    parser.add_argument('zip_path', help='Path to the zip file')
    parser.add_argument('--border', type=int, default=5, help='Size of border (in pixels) to add around detected cards. Default is 5.')
    parser.add_argument('--clean', action='store_true', default=True, help='Clean input directory before extraction (default: True)')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Do not clean input directory before extraction')
    
    args = parser.parse_args()
    
    process_zip_file(args.zip_path, args.border, args.clean)

if __name__ == "__main__":
    main() 