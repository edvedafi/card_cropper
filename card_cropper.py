import cv2
import numpy as np
import os
from pathlib import Path
import imutils
import zipfile
import tempfile
import shutil

class CardCropper:
    def __init__(self, padding=20, min_card_area_ratio=0.05, max_card_area_ratio=0.95):
        """
        Initialize the CardCropper with configurable parameters.
        
        Args:
            padding (int): Number of pixels to pad around each detected card
            min_card_area_ratio (float): Minimum area ratio relative to image size to be considered a card (5% of image)
            max_card_area_ratio (float): Maximum area ratio relative to image size to be considered a card (95% of image)
        """
        self.padding = padding
        self.min_card_area_ratio = min_card_area_ratio
        self.max_card_area_ratio = max_card_area_ratio
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    def preprocess_image(self, image):
        """
        Preprocess the image using edge detection to find card boundaries.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Preprocessed image ready for contour detection
        """
        # Create debug directory
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)

        # Calculate target size - resize if image is too large
        max_dimension = 1500
        height, width = image.shape[:2]
        scale = min(max_dimension / width, max_dimension / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"Resized image to {new_width}x{new_height}")
        
        cv2.imwrite(str(debug_dir / "0_resized.jpg"), image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(debug_dir / "1_gray.jpg"), gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite(str(debug_dir / "2_blurred.jpg"), blurred)
        
        # Use Canny edge detection with automatic thresholds
        sigma = 0.33
        median = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(blurred, lower, upper)
        cv2.imwrite(str(debug_dir / "3_edges.jpg"), edges)
        
        # Dilate edges to connect gaps
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        cv2.imwrite(str(debug_dir / "4_dilated.jpg"), dilated)
        
        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(str(debug_dir / "5_closed.jpg"), closed)
        
        return closed, scale

    def filter_contours(self, contours, image_shape, image):
        """
        Filter contours to find card-like rectangles.
        
        Args:
            contours: List of contours to filter
            image_shape: Shape of the original image (height, width)
            image: Original image for debugging
        
        Returns:
            List of filtered contours that are likely to be cards
        """
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.min_card_area_ratio
        max_area = image_area * self.max_card_area_ratio
        
        filtered_contours = []
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Create a debug image to show all considered contours
        debug_image = image.copy()
        
        # Create a log file for detailed filtering information
        with open(debug_dir / "contour_filtering.log", "w") as log_file:
            log_file.write(f"Image dimensions: {image_shape[1]}x{image_shape[0]}, Image area: {image_area}\n")
            log_file.write(f"Min area ratio: {self.min_card_area_ratio}, Min area: {min_area}\n")
            log_file.write(f"Max area ratio: {self.max_card_area_ratio}, Max area: {max_area}\n")
            log_file.write(f"Total contours before filtering: {len(contours)}\n\n")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                area_ratio = area / image_area
                
                log_file.write(f"Contour #{i}: Area = {area}, Area ratio = {area_ratio:.4f}\n")
                
                # Draw all contours in red initially
                cv2.drawContours(debug_image, [contour], -1, (0, 0, 255), 2)
                
                # Get centroid for text placement
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    log_file.write("  REJECTED: Invalid moments\n\n")
                    continue
                    
                # Add area ratio text
                cv2.putText(debug_image, f"Area: {area_ratio:.2f}", 
                          (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                if min_area <= area <= max_area:
                    # Get the minimum area rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # Calculate width and height
                    width = min(rect[1])
                    height = max(rect[1])
                    
                    if width == 0 or height == 0:
                        log_file.write("  REJECTED: Zero width or height\n\n")
                        continue
                        
                    aspect_ratio = width / height
                    
                    log_file.write(f"  Passed area check. Width = {width}, Height = {height}, Aspect ratio = {aspect_ratio:.4f}\n")
                    
                    # Draw contours that pass area check in blue
                    cv2.drawContours(debug_image, [box], -1, (255, 0, 0), 2)
                    cv2.putText(debug_image, f"AR: {aspect_ratio:.2f}", 
                              (cx-40, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # More lenient aspect ratio range for baseball cards
                    # Standard baseball card is 2.5" x 3.5" (ratio = 0.714)
                    # Allow a wider range to account for perspective and different card types
                    if 0.5 <= aspect_ratio <= 0.9:
                        filtered_contours.append(box)
                        # Draw accepted contours in green
                        cv2.drawContours(debug_image, [box], -1, (0, 255, 0), 2)
                        log_file.write("  ACCEPTED: Passed all checks\n\n")
                    else:
                        log_file.write(f"  REJECTED: Aspect ratio {aspect_ratio:.4f} outside range [0.5, 0.9]\n\n")
                else:
                    log_file.write(f"  REJECTED: Area {area} outside range [{min_area}, {max_area}]\n\n")
            
            log_file.write(f"Total contours after filtering: {len(filtered_contours)}")
        
        # Save the debug image showing aspect ratios and areas
        cv2.imwrite(str(debug_dir / "contour_analysis.jpg"), debug_image)
        
        # Sort contours by area in descending order
        filtered_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        
        return filtered_contours

    def find_cards(self, image):
        """
        Detect cards in the image using Hough Line Transform and contour detection.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            List of contours that are likely to be cards
        """
        # Create debug directory
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Save original image
        cv2.imwrite(str(debug_dir / "original.jpg"), image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(debug_dir / "gray.jpg"), gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        cv2.imwrite(str(debug_dir / "filtered.jpg"), filtered)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite(str(debug_dir / "thresh.jpg"), thresh)
        
        # Invert the image if it's mostly white (cards on dark background)
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)
        if white_pixels > black_pixels:
            thresh = cv2.bitwise_not(thresh)
            cv2.imwrite(str(debug_dir / "thresh_inverted.jpg"), thresh)
        
        # Use Canny edge detection with lower thresholds
        edges = cv2.Canny(thresh, 30, 150)
        cv2.imwrite(str(debug_dir / "edges.jpg"), edges)
        
        # Dilate edges to connect gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cv2.imwrite(str(debug_dir / "dilated.jpg"), dilated)
        
        # Use Hough Line Transform to detect straight lines
        lines = cv2.HoughLinesP(
            dilated, 1, np.pi/180, threshold=50, 
            minLineLength=max(image.shape) // 10,  # Lines must be at least 1/10 of the image dimension
            maxLineGap=max(image.shape) // 20      # Allow gaps of up to 1/20 of the image dimension
        )
        
        # Create a blank image to draw lines on
        line_image = np.zeros_like(image)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(str(debug_dir / "hough_lines.jpg"), line_image)
        
        # Create a mask from the lines
        line_mask = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        _, line_mask = cv2.threshold(line_mask, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(debug_dir / "line_mask.jpg"), line_mask)
        
        # Dilate the line mask to connect nearby lines
        dilated_mask = cv2.dilate(line_mask, kernel, iterations=3)
        cv2.imwrite(str(debug_dir / "dilated_mask.jpg"), dilated_mask)
        
        # Find contours in the dilated mask
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save debug image with all contours
        debug_all = image.copy()
        cv2.drawContours(debug_all, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "all_contours.jpg"), debug_all)
        
        print(f"Found {len(contours)} initial contours")
        
        # Approximate contours to find rectangles
        approx_contours = []
        for contour in contours:
            # Skip tiny contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Try different epsilon values for better approximation
            for epsilon_factor in [0.01, 0.02, 0.05, 0.1]:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a quadrilateral (4 points) and convex
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    approx_contours.append(approx)
                    break
                # Also accept contours with 5-8 points that might be slightly irregular cards
                elif 4 <= len(approx) <= 8 and cv2.isContourConvex(approx):
                    # Get the minimum area rectangle
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    approx_contours.append(box)
                    break
        
        print(f"Found {len(approx_contours)} rectangular contours")
        
        # Save debug image with approximated contours
        debug_approx = image.copy()
        cv2.drawContours(debug_approx, approx_contours, -1, (255, 0, 0), 2)
        cv2.imwrite(str(debug_dir / "approx_contours.jpg"), debug_approx)
        
        # Filter and sort contours
        card_contours = self.filter_contours(approx_contours, image.shape[:2], image)
        
        # Save debug image with filtered contours
        debug_filtered = image.copy()
        cv2.drawContours(debug_filtered, card_contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "filtered_contours.jpg"), debug_filtered)
        
        # If no cards found, try a more aggressive approach with direct rectangle detection
        if not card_contours:
            print("No cards found with standard approach, trying direct rectangle detection...")
            card_contours = self.detect_rectangles_directly(image)
            
            # Save debug image with directly detected rectangles
            if card_contours:
                debug_direct = image.copy()
                cv2.drawContours(debug_direct, card_contours, -1, (0, 255, 255), 2)
                cv2.imwrite(str(debug_dir / "direct_rectangles.jpg"), debug_direct)
        
        return card_contours
        
    def detect_rectangles_directly(self, image):
        """
        Directly detect rectangles in the image using contour detection on multiple thresholds.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of contours that are likely to be cards
        """
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple threshold values
        all_rectangles = []
        
        # Save a copy of the original image for debugging
        debug_image = image.copy()
        
        # Try different threshold values
        for threshold_value in [100, 150]:  # Focus on the values that work well
            # Apply binary threshold
            _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
            cv2.imwrite(str(debug_dir / f"direct_thresh_{threshold_value}.jpg"), thresh)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw all contours on the debug image
            contour_image = image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / f"direct_contours_{threshold_value}.jpg"), contour_image)
            
            # Process each contour
            for contour in contours:
                # Skip small contours
                area = cv2.contourArea(contour)
                if area < 1000:
                    continue
                
                # Get the minimum area rectangle directly
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Calculate aspect ratio
                width = min(rect[1])
                height = max(rect[1])
                
                if width == 0 or height == 0:
                    continue
                    
                aspect_ratio = width / height
                
                # Calculate area ratio
                image_area = image.shape[0] * image.shape[1]
                area_ratio = area / image_area
                
                # Draw the rectangle on the debug image
                cv2.drawContours(debug_image, [box], -1, (0, 0, 255), 2)
                
                # Add text with aspect ratio and area ratio
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(debug_image, f"AR: {aspect_ratio:.2f}", 
                              (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(debug_image, f"Area: {area_ratio:.2f}", 
                              (cx-40, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Check if aspect ratio and area ratio are reasonable for a card
                # More lenient aspect ratio range
                if 0.5 <= aspect_ratio <= 0.9 and self.min_card_area_ratio <= area_ratio <= self.max_card_area_ratio:
                    all_rectangles.append(box)
                    # Draw accepted rectangles in green
                    cv2.drawContours(debug_image, [box], -1, (0, 255, 0), 2)
        
        # Save the debug image with all detected rectangles
        cv2.imwrite(str(debug_dir / "direct_detection_all.jpg"), debug_image)
        
        # If no rectangles found, try with the original contours
        if not all_rectangles:
            print("No rectangles found with direct detection, trying with original contours...")
            
            # Try with the original image
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Skip small contours
                area = cv2.contourArea(contour)
                if area < 1000:
                    continue
                
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = min(w, h) / max(w, h)
                
                # Calculate area ratio
                image_area = image.shape[0] * image.shape[1]
                area_ratio = area / image_area
                
                # Check if aspect ratio and area ratio are reasonable for a card
                if 0.5 <= aspect_ratio <= 0.9 and self.min_card_area_ratio <= area_ratio <= self.max_card_area_ratio:
                    # Create a box from the bounding rectangle
                    box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                    all_rectangles.append(box)
        
        # Filter out overlapping rectangles
        filtered_rectangles = []
        for rect in all_rectangles:
            # Check if this rectangle significantly overlaps with any already filtered rectangle
            is_duplicate = False
            for filtered_rect in filtered_rectangles:
                # Create masks for both rectangles
                mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
                mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
                
                cv2.drawContours(mask1, [rect], -1, 255, -1)
                cv2.drawContours(mask2, [filtered_rect], -1, 255, -1)
                
                # Calculate intersection area
                intersection = cv2.bitwise_and(mask1, mask2)
                intersection_area = cv2.countNonZero(intersection)
                
                # Calculate areas of both rectangles
                rect_area = cv2.countNonZero(mask1)
                filtered_rect_area = cv2.countNonZero(mask2)
                
                # If intersection is more than 50% of either rectangle, consider it a duplicate
                if intersection_area > 0.5 * min(rect_area, filtered_rect_area):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_rectangles.append(rect)
        
        # Save the debug image with filtered rectangles
        filtered_image = image.copy()
        for rect in filtered_rectangles:
            cv2.drawContours(filtered_image, [rect], -1, (0, 255, 255), 2)
        cv2.imwrite(str(debug_dir / "direct_detection_filtered.jpg"), filtered_image)
        
        print(f"Found {len(filtered_rectangles)} rectangles with direct detection")
        return filtered_rectangles

    def refine_corners(self, points):
        """
        Refine corner points by finding the intersection of lines.
        
        Args:
            points: Approximate corner points
        
        Returns:
            Refined corner points
        """
        # Convert points to float32 and reshape
        pts = points.astype(np.float32).reshape(4, 2)
        
        # Order points clockwise from top-left
        ordered = self.order_points(pts)
        
        # Calculate the width and height of the card
        width = np.sqrt(((ordered[1][0] - ordered[0][0]) ** 2) + ((ordered[1][1] - ordered[0][1]) ** 2))
        height = np.sqrt(((ordered[2][0] - ordered[1][0]) ** 2) + ((ordered[2][1] - ordered[1][1]) ** 2))
        
        # Standard baseball card ratio (2.5" x 3.5")
        card_ratio = 2.5 / 3.5
        
        # Determine if we need to rotate
        current_ratio = width / height
        if abs(current_ratio - card_ratio) > abs(current_ratio - (1/card_ratio)):
            # Card is in portrait orientation, swap width and height
            width, height = height, width
        
        # Calculate the target width and height maintaining aspect ratio
        if width > height * card_ratio:
            width = height * card_ratio
        else:
            height = width / card_ratio
        
        # Create a perfect rectangle with the correct aspect ratio
        rect = np.float32([
            [0, 0],                  # Top-left
            [width - 1, 0],          # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1]           # Bottom-left
        ])
        
        # Get the perspective transform from the ordered points to the perfect rectangle
        matrix = cv2.getPerspectiveTransform(ordered, rect)
        
        # Transform the original points to get perfectly squared corners
        refined = cv2.perspectiveTransform(ordered.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        
        return refined
        
    def order_points(self, pts):
        """
        Order points in clockwise order starting from top-left.
        
        Args:
            pts: Array of 4 points
        
        Returns:
            Ordered points as numpy array
        """
        # Initialize ordered points array
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Calculate sum of coordinates
        s = pts.sum(axis=1)
        
        # Top-left point will have the smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point will have the largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Calculate difference between coordinates
        diff = np.diff(pts, axis=1)
        
        # Top-right point will have the smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point will have the largest difference
        rect[3] = pts[np.argmax(diff)]
        
        # Create a debug image to visualize the ordered points
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Create a blank image
        debug_image = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Scale points to fit in the debug image
        scaled_pts = pts.copy()
        scaled_pts[:, 0] = scaled_pts[:, 0] * 700 / np.max(pts[:, 0]) + 50
        scaled_pts[:, 1] = scaled_pts[:, 1] * 700 / np.max(pts[:, 1]) + 50
        
        scaled_rect = rect.copy()
        scaled_rect[:, 0] = scaled_rect[:, 0] * 700 / np.max(pts[:, 0]) + 50
        scaled_rect[:, 1] = scaled_rect[:, 1] * 700 / np.max(pts[:, 1]) + 50
        
        # Draw the original points in red
        for i, point in enumerate(scaled_pts):
            cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.putText(debug_image, f"Orig {i}", (int(point[0])+10, int(point[1])+10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw the ordered points in green
        for i, point in enumerate(scaled_rect):
            cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.putText(debug_image, f"Ordered {i}", (int(point[0])+10, int(point[1])-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw lines connecting the ordered points
        for i in range(4):
            cv2.line(debug_image, 
                   (int(scaled_rect[i][0]), int(scaled_rect[i][1])), 
                   (int(scaled_rect[(i+1)%4][0]), int(scaled_rect[(i+1)%4][1])), 
                   (0, 255, 255), 2)
        
        cv2.imwrite(str(debug_dir / "ordered_points.jpg"), debug_image)
        
        return rect

    def get_perspective_transform(self, image, points):
        """
        Apply perspective transform to get a top-down view of the card.
        
        Args:
            image: Input image
            points: Four corner points of the card
        
        Returns:
            Transformed image with padding, properly oriented
        """
        # Create debug directory
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Draw the original points on the image
        debug_image = image.copy()
        for i, point in enumerate(points):
            cv2.circle(debug_image, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(debug_image, f"Point {i}", (point[0]+10, point[1]+10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(debug_dir / "original_points.jpg"), debug_image)
        
        # Convert points to float32
        points = np.float32(points)
        
        # Order points in clockwise order starting from top-left
        rect = self.order_points(points)
        
        # Draw the ordered points on the image
        debug_ordered = image.copy()
        for i, point in enumerate(rect):
            cv2.circle(debug_ordered, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.putText(debug_ordered, f"Ordered {i}", (int(point[0])+10, int(point[1])+10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(debug_dir / "ordered_points_on_image.jpg"), debug_ordered)
        
        # Calculate dimensions based on baseball card ratio (2.5" x 3.5")
        card_ratio = 2.5 / 3.5
        
        # Calculate width and height from the ordered points
        width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width = max(width_top, width_bottom)
        
        height_left = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
        height_right = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
        height = max(height_left, height_right)
        
        print(f"Card dimensions: width={width:.1f}, height={height:.1f}, ratio={width/height:.3f}")
        
        # Determine if we need to rotate based on the standard card ratio
        current_ratio = width / height
        if abs(current_ratio - card_ratio) > abs(current_ratio - (1/card_ratio)):
            # Card is in portrait orientation, swap width and height
            width, height = height, width
            print(f"Swapping dimensions: width={width:.1f}, height={height:.1f}, ratio={width/height:.3f}")
        
        # Ensure the card has the correct aspect ratio
        if width / height > card_ratio:
            # Width is too large, adjust it
            width = height * card_ratio
        else:
            # Height is too large, adjust it
            height = width / card_ratio
            
        print(f"Final dimensions: width={width:.1f}, height={height:.1f}, ratio={width/height:.3f}")
        
        # Add padding
        target_width = int(width) + 2 * self.padding
        target_height = int(height) + 2 * self.padding
        
        # Create destination points for a perfectly rectangular card
        dst_pts = np.array([
            [self.padding, self.padding],                          # Top-left
            [target_width - self.padding, self.padding],           # Top-right
            [target_width - self.padding, target_height - self.padding],  # Bottom-right
            [self.padding, target_height - self.padding]           # Bottom-left
        ], dtype=np.float32)
        
        # Get perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(rect, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
        
        # Save intermediate warped image
        cv2.imwrite(str(debug_dir / "warped_intermediate.jpg"), warped)
        
        # Ensure the card is in portrait orientation (height > width)
        if target_width > target_height:
            print("Rotating to portrait orientation")
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            
        # Save final warped image
        cv2.imwrite(str(debug_dir / "warped_final.jpg"), warped)
        
        return warped

    def process_image(self, image_path, output_dir, prefix=""):
        """
        Process a single image file.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save cropped cards
            prefix: Optional prefix for output filenames
            
        Returns:
            Number of cards successfully processed
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return 0
        
        print(f"Image size: {image.shape}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find and process cards
        print("\nStarting card detection...")
        cards = self.find_cards(image)
        
        if not cards:
            print("No cards found in image")
            return 0
        
        # Process each card
        processed_count = 0
        for i, card in enumerate(cards, 1):
            print(f"\nProcessing card {i}:")
            try:
                # Get corner points and calculate aspect ratio
                corners = card.reshape(4, 2)
                print(f"Corner points: {corners}")
                
                rect = cv2.minAreaRect(card)
                width = min(rect[1])
                height = max(rect[1])
                aspect_ratio = width / height
                print(f"Aspect ratio: {aspect_ratio:.3f}")
                
                # Apply perspective transform
                warped = self.get_perspective_transform(image, card)
                if warped is not None:
                    # Create output filename
                    if prefix:
                        output_filename = f"{prefix}_{i}.jpg"
                    else:
                        output_filename = f"{Path(image_path).stem}_{i}.jpg"
                    
                    # Save output image
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), warped)
                    processed_count += 1
                    
                    # Save debug image showing detected contour
                    debug_dir = Path("debug_output")
                    debug_dir.mkdir(exist_ok=True)
                    debug_image = image.copy()
                    cv2.drawContours(debug_image, [card], -1, (0, 255, 0), 2)
                    cv2.imwrite(str(debug_dir / f"card_{i}_detected.jpg"), debug_image)
                    cv2.imwrite(str(debug_dir / f"card_{i}_warped.jpg"), warped)
            except Exception as e:
                print(f"Error processing card {i}: {e}")
        
        print(f"\nSuccessfully processed {processed_count} cards")
        return processed_count

    def process_zip(self, zip_path, output_dir):
        """
        Process all images in a zip file.
        
        Args:
            zip_path: Path to the input zip file
            output_dir: Directory to save all cropped cards
        
        Returns:
            Dictionary with statistics about processed images and cards
        """
        stats = {
            'total_images': 0,
            'processed_images': 0,
            'total_cards': 0,
            'failed_images': []
        }
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create temporary directory for zip extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except zipfile.BadZipFile:
                raise ValueError(f"Invalid zip file: {zip_path}")
            
            # Process all images in the extracted directory
            temp_path = Path(temp_dir)
            for file_path in temp_path.rglob('*'):
                if file_path.suffix.lower() in self.supported_extensions:
                    stats['total_images'] += 1
                    try:
                        # Use the relative path from the zip as the prefix for output files
                        rel_path = file_path.relative_to(temp_path)
                        prefix = str(rel_path.parent / rel_path.stem).replace('/', '_')
                        
                        # Process the image
                        num_cards = self.process_image(
                            file_path,
                            output_dir,
                            prefix=prefix
                        )
                        
                        if num_cards > 0:
                            stats['processed_images'] += 1
                            stats['total_cards'] += num_cards
                        
                    except Exception as e:
                        stats['failed_images'].append({
                            'path': str(file_path.relative_to(temp_path)),
                            'error': str(e)
                        })
        
        return stats

def main():
    """
    Example usage of the CardCropper class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images to detect and crop cards.')
    parser.add_argument('input', help='Input file (zip file or single image)')
    parser.add_argument('--output', '-o', default='output_cards',
                      help='Output directory for cropped cards')
    parser.add_argument('--padding', '-p', type=int, default=10,
                      help='Padding around each card in pixels')
    parser.add_argument('--min-area', type=float, default=0.05,
                      help='Minimum area ratio (0.0-1.0) relative to image size to be considered a card')
    parser.add_argument('--max-area', type=float, default=0.95,
                      help='Maximum area ratio (0.0-1.0) relative to image size to be considered a card')
    args = parser.parse_args()
    
    # Initialize the card cropper
    cropper = CardCropper(
        padding=args.padding,
        min_card_area_ratio=args.min_area,
        max_card_area_ratio=args.max_area
    )
    
    try:
        input_path = Path(args.input)
        if input_path.suffix.lower() == '.zip':
            # Process zip file
            stats = cropper.process_zip(input_path, args.output)
            print(f"Processed {stats['processed_images']}/{stats['total_images']} images")
            print(f"Found {stats['total_cards']} cards total")
            if stats['failed_images']:
                print("\nFailed to process the following images:")
                for failure in stats['failed_images']:
                    print(f"- {failure['path']}: {failure['error']}")
        else:
            # Process single image
            num_cards = cropper.process_image(input_path, args.output)
            print(f"Found {num_cards} cards total")
            
    except Exception as e:
        print(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    main() 