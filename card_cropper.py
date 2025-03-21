import argparse
import zipfile
import os
import shutil
import cv2
import numpy as np
import imutils
from pathlib import Path
import time

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
    debug_path = str(output_path).replace('.jpg', '_debug.jpg').replace('.jpeg', '_debug.jpeg').replace('.png', '_debug.png')
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

def process_zip_file(zip_path, border_size=5):
    # Get the base name of the zip file without extension
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Create input and output directories if they don't exist
    input_dir = Path("input")
    output_dir = Path("output") / zip_name
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
    
    start_time = time.time()
    total_images = 0
    
    # First, count total images
    for root, _, files in os.walk(input_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                total_images += 1
    
    current_image = 0
    
    # Process images
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in image_extensions:
                current_image += 1
                dest_path = os.path.join(output_dir, file)
                print(f"Processing image {current_image}/{total_images}: {file}")
                
                if crop_largest_object(file_path, dest_path, border_size):
                    processed_count += 1
                else:
                    skipped_count += 1
                
                # Print progress update every 10 images
                if current_image % 10 == 0:
                    elapsed_time = time.time() - start_time
                    images_per_second = current_image / elapsed_time if elapsed_time > 0 else 0
                    print(f"Progress: {current_image}/{total_images} images ({images_per_second:.2f} images/second)")
    
    # Print final stats
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Successfully cropped {processed_count} images with {border_size}px border to {output_dir}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (could not detect card)")

def main():
    parser = argparse.ArgumentParser(description='Process a zip file containing images.')
    parser.add_argument('zip_path', help='Path to the zip file')
    parser.add_argument('--border', type=int, default=5, help='Size of border (in pixels) to add around detected cards. Default is 5.')
    args = parser.parse_args()
    
    process_zip_file(args.zip_path, args.border)

if __name__ == "__main__":
    main()