import cv2
import numpy as np
import os
from pathlib import Path
import imutils
import zipfile
import tempfile
import shutil

class CardCropper:
    def __init__(self, padding=10, min_card_area_ratio=0.001, max_card_area_ratio=0.1):
        """
        Initialize the CardCropper with configurable parameters.
        
        Args:
            padding (int): Number of pixels to pad around each detected card
            min_card_area_ratio (float): Minimum area ratio relative to image size to be considered a card (15% of image)
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
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(str(debug_dir / "2_enhanced.jpg"), enhanced)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        cv2.imwrite(str(debug_dir / "3_denoised.jpg"), denoised)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        cv2.imwrite(str(debug_dir / "4_thresh.jpg"), thresh)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(str(debug_dir / "5_morph.jpg"), morph)
        
        # Find edges using Canny with automatic threshold calculation
        sigma = 0.33
        median = np.median(morph)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(morph, lower, upper)
        cv2.imwrite(str(debug_dir / "6_edges.jpg"), edges)
        
        # Dilate edges to connect gaps
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cv2.imwrite(str(debug_dir / "7_dilated.jpg"), dilated)
        
        # Close gaps in the edges
        kernel = np.ones((7,7), np.uint8)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(str(debug_dir / "8_closed.jpg"), closed)
        
        return closed, scale

    def filter_contours(self, contours, image_shape):
        """
        Filter contours to find card-like rectangles.
        
        Args:
            contours: List of contours to filter
            image_shape: Shape of the original image (height, width)
        
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
        debug_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Draw all contours in red initially
            cv2.drawContours(debug_image, [contour], -1, (0, 0, 255), 1)
            
            # Get centroid for text placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
                
            # Add area ratio text
            area_ratio = area / image_area
            cv2.putText(debug_image, f"Area: {area_ratio:.2f}", 
                      (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if min_area <= area <= max_area:
                # Get the minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Calculate aspect ratio using the minimum area rectangle
                width = min(rect[1])
                height = max(rect[1])
                if width == 0 or height == 0:
                    continue
                    
                aspect_ratio = width / height
                
                # Draw contours that pass area check in blue
                cv2.drawContours(debug_image, [box], -1, (255, 0, 0), 2)
                cv2.putText(debug_image, f"AR: {aspect_ratio:.2f}", 
                          (cx-40, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Baseball card aspect ratio is typically around 0.714 (2.5" x 3.5")
                # Allow more tolerance in the range due to perspective distortion
                if 0.5 <= aspect_ratio <= 0.9:
                    filtered_contours.append(box)
                    # Draw accepted contours in green
                    cv2.drawContours(debug_image, [box], -1, (0, 255, 0), 2)
        
        # Save the debug image showing aspect ratios and areas
        cv2.imwrite(str(debug_dir / "contour_analysis.jpg"), debug_image)
        
        # Sort contours by area in descending order
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        return filtered_contours

    def find_cards(self, image):
        """
        Detect cards in the image using contour detection.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            List of contours that are likely to be cards
        """
        # Get preprocessed image and scale factor
        processed, scale = self.preprocess_image(image)
        
        # Save debug image of preprocessed result
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "preprocessed.jpg"), processed)
        
        # Find contours with hierarchy to distinguish outer and inner contours
        contours, hierarchy = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Save debug image with all contours before filtering
        debug_all = image.copy()
        cv2.drawContours(debug_all, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "all_contours.jpg"), debug_all)
        
        print(f"Found {len(contours)} initial contours")
        
        # Approximate contours to find rectangles
        approx_contours = []
        for contour in contours:
            # Calculate perimeter and area
            peri = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 50:
                continue
                
            # Approximate the contour
            epsilon = 0.04 * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If it has 4 points, it might be a card
            if len(approx) == 4:
                # Check if the points form a convex quadrilateral
                if cv2.isContourConvex(approx):
                    approx_contours.append(approx)
        
        print(f"Found {len(approx_contours)} rectangular contours")
        
        # Save debug image with approximated contours
        debug_approx = image.copy()
        cv2.drawContours(debug_approx, approx_contours, -1, (255, 0, 0), 2)
        cv2.imwrite(str(debug_dir / "approx_contours.jpg"), debug_approx)
        
        # Filter and sort contours
        card_contours = self.filter_contours(approx_contours, image.shape[:2])
        
        # Save debug image with filtered contours
        debug_filtered = image.copy()
        cv2.drawContours(debug_filtered, card_contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "filtered_contours.jpg"), debug_filtered)
        
        # If we scaled the image, scale the contours back up
        if scale < 1:
            for i in range(len(card_contours)):
                card_contours[i] = (card_contours[i] / scale).astype(np.int32)
        
        return card_contours

    def order_points(self, pts):
        """
        Order points in clockwise order starting from top-left.
        
        Args:
            pts: Array of 4 points
        
        Returns:
            Ordered points as numpy array
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left point will have the smallest sum
        # Bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point will have the smallest difference
        # Bottom-left point will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def get_perspective_transform(self, image, points):
        """
        Apply perspective transform to get a top-down view of the card.
        
        Args:
            image: Input image
            points: Four corner points of the card
        
        Returns:
            Transformed image with padding
        """
        # Order points in clockwise order
        rect = self.order_points(np.float32(points))
        
        # Calculate width and height of the card
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b)) + (2 * self.padding)
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b)) + (2 * self.padding)
        
        # Define destination points with padding
        dst = np.array([
            [self.padding, self.padding],
            [max_width - self.padding, self.padding],
            [max_width - self.padding, max_height - self.padding],
            [self.padding, max_height - self.padding]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped

    def process_image(self, image_path, output_dir, prefix=''):
        """
        Process an image to detect and crop all cards.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save cropped cards
            prefix: Optional prefix for output filenames
        
        Returns:
            Number of cards detected and processed
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Save original image for reference
        cv2.imwrite(str(output_path / "0_original.jpg"), image)
        print(f"Image size: {image.shape}")
        
        # Find cards
        print("\nStarting card detection...")
        card_contours = self.find_cards(image)
        print(f"Found {len(card_contours)} potential cards")
        
        # Create a debug image showing all detected contours
        debug_all = image.copy()
        cv2.drawContours(debug_all, card_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(output_path / "all_detected_cards.jpg"), debug_all)
        
        # Process each card
        for i, contour in enumerate(card_contours):
            # Get corner points
            corners = contour.reshape(4, 2)
            print(f"\nProcessing card {i+1}:")
            print(f"Corner points: {corners}")
            
            # Calculate aspect ratio
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            height = max(rect[1])
            aspect_ratio = width / height
            print(f"Aspect ratio: {aspect_ratio:.3f}")
            
            # Apply perspective transform
            warped = self.get_perspective_transform(image, corners)
            
            # Create filename with prefix if provided
            filename = f"{prefix}_card_{i+1}.jpg" if prefix else f"card_{i+1}.jpg"
            output_file = output_path / filename
            cv2.imwrite(str(output_file), warped)
            
            # Save debug image showing this card's contour
            debug_image = image.copy()
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            debug_filename = f"{prefix}_debug_{i+1}.jpg" if prefix else f"debug_{i+1}.jpg"
            debug_file = output_path / debug_filename
            cv2.imwrite(str(debug_file), debug_image)
        
        return len(card_contours)

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
                            str(file_path),
                            output_dir,
                            prefix=prefix
                        )
                        
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
    args = parser.parse_args()
    
    # Initialize the card cropper
    cropper = CardCropper(padding=args.padding)
    
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
            print(f"Successfully processed {num_cards} cards")
            
    except Exception as e:
        print(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    main() 