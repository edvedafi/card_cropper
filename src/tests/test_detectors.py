import cv2
import numpy as np
from pathlib import Path
import os
import sys
from typing import List, Tuple, Dict

# Add parent directory to path to import detectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.basic_detector import detect_card as detect_card_basic
from detectors.enhanced_detector import detect_card as detect_card_enhanced
from detectors.ml_detector import detect_card as detect_card_ml
from detectors.aggressive_detector import detect_card as detect_card_aggressive

def evaluate_detector(detector_func, image_path: Path, output_path: Path, border_size: int = 5) -> Tuple[bool, str]:
    """
    Evaluate a card detector function on a single image.
    
    Args:
        detector_func: The detector function to evaluate
        image_path: Path to the input image
        output_path: Path to save the cropped image
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        tuple: (success, error_reason) where:
            - success is True if card was successfully detected and cropped
            - error_reason is None on success, or error description on failure
    """
    return detector_func(image_path, output_path, border_size)

def test_detectors(image_path: Path, output_dir: Path, border_size: int = 5) -> Dict[str, Tuple[bool, str]]:
    """
    Test all card detectors on a single image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the cropped images
        border_size: Number of pixels to add as border around the detected card
        
    Returns:
        dict: Dictionary mapping detector names to their (success, error_reason) results
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary of detectors to test
    detectors = {
        'basic': detect_card_basic,
        'enhanced': detect_card_enhanced,
        'ml': detect_card_ml,
        'aggressive': detect_card_aggressive
    }
    
    results = {}
    
    # Test each detector
    for name, detector in detectors.items():
        output_path = output_dir / f"{name}_{image_path.name}"
        success, error = evaluate_detector(detector, image_path, output_path, border_size)
        results[name] = (success, error)
    
    return results

def main():
    """Main function to test detectors on a single image or directory of images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test card detection methods')
    parser.add_argument('input', help='Input image or directory of images')
    parser.add_argument('--output', default='output', help='Output directory for cropped images')
    parser.add_argument('--border', type=int, default=5, help='Border size in pixels')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # Test on single image
        results = test_detectors(input_path, output_dir, args.border)
        
        # Print results
        print(f"\nResults for {input_path.name}:")
        for name, (success, error) in results.items():
            status = "Success" if success else f"Failed: {error}"
            print(f"{name}: {status}")
            
    elif input_path.is_dir():
        # Test on directory of images
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        if not image_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"\nTesting {len(image_files)} images...")
        
        # Create subdirectories for each detector
        for name in ['basic', 'enhanced', 'ml', 'aggressive']:
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for image_path in image_files:
            print(f"\nProcessing {image_path.name}...")
            results = test_detectors(image_path, output_dir, args.border)
            
            # Print results
            for name, (success, error) in results.items():
                status = "Success" if success else f"Failed: {error}"
                print(f"{name}: {status}")
                
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == '__main__':
    main() 