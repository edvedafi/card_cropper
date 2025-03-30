import cv2
import numpy as np
from pathlib import Path
import os
import sys
import zipfile
from typing import List, Tuple, Dict

# Add parent directory to path to import detectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.basic_detector import detect_card as detect_card_basic
from detectors.enhanced_detector import detect_card as detect_card_enhanced
from detectors.ml_detector import detect_card as detect_card_ml
from detectors.aggressive_detector import detect_card as detect_card_aggressive

def process_single_image(image_path: Path, output_path: Path, border_size: int = 5) -> Tuple[bool, str]:
    """
    Process a single image using the best available detector.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) where:
            - success is True if card was successfully detected and cropped
            - error_reason is None on success, or error description on failure
    """
    # Try each detector in order of complexity
    detectors = [
        ('basic', detect_card_basic),
        ('enhanced', detect_card_enhanced),
        ('ml', detect_card_ml),
        ('aggressive', detect_card_aggressive)
    ]
    
    for name, detector in detectors:
        success, error = detector(image_path, output_path, border_size)
        if success:
            print(f"Successfully processed {image_path.name} using {name} detector")
            return True, None
    
    return False, "All detectors failed"

def process_zip_file(zip_path: Path, output_dir: Path, border_size: int = 5) -> Tuple[int, int]:
    """
    Process all images in a zip file.
    
    Args:
        zip_path: Path to the zip file
        output_dir: Directory to save the cropped images
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success_count, total_count) where:
            - success_count is the number of successfully processed images
            - total_count is the total number of images processed
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for extracted images
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files
            zip_ref.extractall(temp_dir)
            
            # Process each image
            for image_path in temp_dir.glob('*'):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    total_count += 1
                    output_path = output_dir / f"{image_path.stem}_cropped{image_path.suffix}"
                    
                    success, error = process_single_image(image_path, output_path, border_size)
                    if success:
                        success_count += 1
                    else:
                        print(f"Failed to process {image_path.name}: {error}")
    
    except Exception as e:
        print(f"Error processing zip file: {str(e)}")
    
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
    
    return success_count, total_count

def main():
    """Main function to process single images or zip files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop cards from images')
    parser.add_argument('input', help='Input image or zip file')
    parser.add_argument('--output', default='output', help='Output directory for cropped images')
    parser.add_argument('--border', type=int, default=5, help='Border size in pixels')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return
    
    if input_path.suffix.lower() == '.zip':
        # Process zip file
        print(f"Processing zip file: {input_path.name}")
        success_count, total_count = process_zip_file(input_path, output_dir, args.border)
        print(f"\nProcessed {success_count} out of {total_count} images successfully")
        
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # Process single image
        print(f"Processing image: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_cropped{input_path.suffix}"
        success, error = process_single_image(input_path, output_path, args.border)
        
        if success:
            print("Successfully processed image")
        else:
            print(f"Failed to process image: {error}")
            
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")

if __name__ == '__main__':
    main() 