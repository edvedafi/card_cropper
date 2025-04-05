import cv2
import numpy as np
import imutils
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        # Extract borders
        top_border = cropped_img[0:border_check_size, :]
        bottom_border = cropped_img[-border_check_size:, :]
        left_border = cropped_img[:, 0:border_check_size]
        right_border = cropped_img[:, -border_check_size:]
        
        def calc_border_stats(border):
            # Convert to HSV for better color analysis
            hsv_border = cv2.cvtColor(border, cv2.COLOR_BGR2HSV)
            # Calculate standard deviation of each channel
            h_std = np.std(hsv_border[:,:,0])
            s_std = np.std(hsv_border[:,:,1])
            v_std = np.std(hsv_border[:,:,2])
            # Higher std dev means more color variation
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
        
        # Print border statistics for debugging
        print(f"Border std dev - Top: {top_std:.2f}, Bottom: {bottom_std:.2f}, Left: {left_std:.2f}, Right: {right_std:.2f}")
        
        # More lenient threshold for sports cards that may have design elements at edges
        std_threshold = 40.0  # Increased from 25.0
        
        # Count how many borders have high variation
        high_variation_borders = sum([
            top_std > std_threshold,
            bottom_std > std_threshold,
            left_std > std_threshold,
            right_std > std_threshold
        ])
        
        # For sports cards, allow up to 2 borders to have high variation
        # This accounts for design elements like team logos, player names, etc.
        if high_variation_borders <= 2:
            return False
            
        # If 3 or more borders have high variation, check if they're similar to inner pixels
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
        
        # More lenient similarity threshold for sports cards
        similarity_threshold = 30.0  # Increased from 20.0
        
        # Only consider it cut off if 3 or more borders are both high variation and similar to inner content
        cutoff_count = 0
        if top_std > std_threshold and top_similarity < similarity_threshold:
            cutoff_count += 1
        if bottom_std > std_threshold and bottom_similarity < similarity_threshold:
            cutoff_count += 1
        if left_std > std_threshold and left_similarity < similarity_threshold:
            cutoff_count += 1
        if right_std > std_threshold and right_similarity < similarity_threshold:
            cutoff_count += 1
        
        return cutoff_count >= 3
        
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
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Check for horizontal lines (common in sports cards)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # More lenient angle threshold for horizontal lines
                if angle < 30 or angle > 150:
                    horizontal_lines += 1
        
        # 2. Check for text using gradient information
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
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_contours = 0
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # If the contour has 4 points, it might be a rectangle
            if len(approx) == 4:
                rectangular_contours += 1
        
        # 4. Check for Bowman/Topps specific features
        # Look for team logos and player info sections
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Check for team colors (blue for Dodgers, etc)
        team_color_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        team_color_pixels = cv2.countNonZero(team_color_mask)
        team_color_percentage = team_color_pixels / (hsv.shape[0] * hsv.shape[1])
        
        # Adjust criteria for sports cards
        has_horizontal_lines = horizontal_lines >= 2  # Reduced from 3
        has_text = text_percentage > 0.03  # Reduced from 0.05
        has_rectangular_elements = rectangular_contours >= 1  # Reduced from 2
        has_team_colors = team_color_percentage > 0.05
        
        # Print debug information
        print(f"Sports card detection - Horizontal lines: {horizontal_lines}, Text percentage: {text_percentage:.2f}, Rectangular elements: {rectangular_contours}, Team color percentage: {team_color_percentage:.2f}")
        
        # Card passes if it meets at least 3 of the 4 criteria
        criteria_met = sum([has_horizontal_lines, has_text, has_rectangular_elements, has_team_colors])
        return criteria_met >= 3
        
    except Exception as e:
        print(f"Error in sports card detection: {e}")
        return False

def check_crop_quality(source_image_path, test_image_path, interactive=False):
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
    is_solid = is_solid_color_image(test_image_path)
    logging.info(f"Solid color check: {'FAILED' if is_solid else 'PASSED'}")
    if is_solid:
        issues.append("Test image appears to be a solid color")
    
    # 2. Check if card is cut off
    is_cut_off = is_card_cut_off(test_image_path, source_image_path)
    logging.info(f"Cut off check: {'FAILED' if is_cut_off else 'PASSED'}")
    if is_cut_off:
        issues.append("Card appears to be cut off")
    
    # 3. Check if test image is larger than source image
    test_h, test_w = test_img.shape[:2]
    source_h, source_w = source_img.shape[:2]
    is_too_large = test_h > source_h or test_w > source_w
    logging.info(f"Size check: {'FAILED' if is_too_large else 'PASSED'} (test: {test_w}x{test_h}, source: {source_w}x{source_h})")
    if is_too_large:
        issues.append("Test image is larger than source image")
    
    # 4. Check if test image has characteristics of a sports card
    is_sports_card = detect_sports_card(test_img)
    logging.info(f"Sports card check: {'PASSED' if is_sports_card else 'FAILED'}")
    if not is_sports_card:
        issues.append("Test image does not appear to be a valid sports card")
    
    # 5. Check if the complete card exists in test image with proper alignment
    # Convert images to grayscale for processing
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    def find_card_edges(img, gray):
        edges = cv2.Canny(gray, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, None, None, None, None
        card_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(card_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(card_contour)
        return card_contour, box, rect, x, y, w, h
    
    # Find card in source image
    source_result = find_card_edges(source_img, source_gray)
    if source_result[0] is None:
        issues.append("Could not detect card boundaries in source image")
        return False, issues
    source_card, source_box, source_rect, sx, sy, sw, sh = source_result
        
    # Find card in test image
    test_result = find_card_edges(test_img, test_gray)
    if test_result[0] is None:
        issues.append("Could not detect card boundaries in test image")
        return False, issues
    test_card, test_box, test_rect, tx, ty, tw, th = test_result
        
    # Check for skew by comparing angles
    source_angle = source_rect[-1]
    test_angle = test_rect[-1]
    angle_diff = abs(source_angle - test_angle) % 90
    max_angle_diff = 2.0  # Maximum allowed angle difference in degrees
    
    is_properly_aligned = angle_diff <= max_angle_diff
    logging.info(f"Alignment check: {'PASSED' if is_properly_aligned else 'FAILED'} (angle difference: {angle_diff:.1f}°)")
    
    if not is_properly_aligned:
        issues.append(f"Card is skewed by {angle_diff:.1f}°")
        if not interactive:
            return False, issues
    
    # Check margin balance
    test_h, test_w = test_img.shape[:2]
    left_margin = tx
    right_margin = test_w - (tx + tw)
    top_margin = ty
    bottom_margin = test_h - (ty + th)
    
    # Calculate margin ratios (should be close to 1.0 for balanced margins)
    lr_ratio = min(left_margin, right_margin) / max(left_margin, right_margin) if max(left_margin, right_margin) > 0 else 0
    tb_ratio = min(top_margin, bottom_margin) / max(top_margin, bottom_margin) if max(top_margin, bottom_margin) > 0 else 0
    
    min_margin_ratio = 0.7  # Margins should be within 30% of each other
    margins_balanced = lr_ratio >= min_margin_ratio and tb_ratio >= min_margin_ratio
    
    logging.info(f"Margin balance check: {'PASSED' if margins_balanced else 'FAILED'} ")
    logging.info(f"  Left-Right ratio: {lr_ratio:.2f}, Top-Bottom ratio: {tb_ratio:.2f}")
    
    if not margins_balanced:
        issues.append("Card margins are not balanced")
        if not interactive:
            return False, issues
    
    # Check if test image contains the complete card
    # Use the source card contour to create a mask
    test_mask = np.zeros(test_gray.shape, dtype=np.uint8)
    cv2.drawContours(test_mask, [test_card], -1, 255, -1)
    
    # Calculate what percentage of the source card contour is present in the test image
    card_area = cv2.contourArea(source_card)
    test_card_area = cv2.contourArea(test_card)
    
    area_ratio = min(card_area, test_card_area) / max(card_area, test_card_area)
    min_area_ratio = 0.95  # Cards should be very similar in size
    
    is_complete = area_ratio >= min_area_ratio
    logging.info(f"Card completeness check: {'PASSED' if is_complete else 'FAILED'} (area ratio: {area_ratio:.3f})")
    
    if not is_complete:
        issues.append("Test image does not contain the complete card")
        if not interactive:
            return False, issues
    
    # Check for non-background content outside the card area
    # Create a mask for the non-card area
    background_mask = np.ones(test_img.shape[:2], dtype=np.uint8) * 255
    cv2.drawContours(background_mask, [test_card], -1, 0, -1)
    
    # Check the non-card area for non-background content
    non_card_region = cv2.bitwise_and(test_img, test_img, mask=background_mask)
    non_card_gray = cv2.cvtColor(non_card_region, cv2.COLOR_BGR2GRAY)
    
    # Calculate percentage of non-background pixels in the non-card area
    _, outside_thresh = cv2.threshold(non_card_gray, 30, 255, cv2.THRESH_BINARY)
    mask_area = np.count_nonzero(background_mask)
    if mask_area > 0:  # Avoid division by zero
        non_bg_ratio = np.count_nonzero(outside_thresh) / mask_area
    else:
        non_bg_ratio = 0
    
    background_threshold = 0.05  # Allow up to 5% non-background pixels
    has_clean_background = non_bg_ratio <= background_threshold
    
    logging.info(f"Background check: {'PASSED' if has_clean_background else 'FAILED'} (non-background pixels: {non_bg_ratio:.1%})")
    
    if not has_clean_background:
        issues.append("Test image contains non-background elements outside the card")
        if not interactive:
            return False, issues
    
    if interactive:
        # Display quality check results
        print("\nQuality Check Results:")
        print("-" * 50)
        if len(issues) == 0:
            print("✅ All checks passed!")
        else:
            print("❌ Failed checks:")
            for issue in issues:
                print(f"  • {issue}")
        print("-" * 50)
        
        # Display the test image in terminal using iTerm2 image protocol
        try:
            import base64
            import os
            
            def iterm2_img_format(img_data, width='auto', height='auto'):
                b64_data = base64.b64encode(img_data).decode('ascii')
                return f'\033]1337;File=inline=1;size={len(img_data)};width={width};height={height}:{b64_data}\a'
            
            # Convert image to PNG bytes
            img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            success, img_data = cv2.imencode('.png', img_rgb)
            if not success:
                raise Exception("Failed to encode image")
            
            # Calculate dimensions to fit terminal
            term_width = os.get_terminal_size().columns
            img_width = test_img.shape[1]
            scale = min(1.0, term_width / img_width)
            
            print("\nCard preview:")
            if 'TERM_PROGRAM' in os.environ and os.environ['TERM_PROGRAM'] == 'iTerm.app':
                print(iterm2_img_format(img_data.tobytes(), width=f'{scale * 100}%'))
            else:
                print("Image preview requires iTerm2 terminal")
                print(f"Please open {output_path} in your image viewer")
            
            # Ask for user approval
            while True:
                response = input("\nApprove this card? (y/n): ").lower()
                if response in ['y', 'n']:
                    return response == 'y', issues
                print("Please enter 'y' for yes or 'n' for no.")
        except Exception as e:
            print(f"Could not display image preview: {e}")
            print(f"Please open {output_path} in your image viewer")
            return len(issues) == 0, issues
    
    return len(issues) == 0, issues