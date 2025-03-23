import os
import sys
import time
import cv2
from pathlib import Path
from card_cropper import crop_largest_object, display_image
import shutil

def test_single_image(image_path, border_size=5):
    """
    Test the card cropping functionality on a single image.
    
    Args:
        image_path: Path to the input image
        border_size: Size of border to add around detected cards
    """
    # Get the base name of the image file
    image_name = os.path.basename(image_path)
    
    # Create output directories
    output_dir = Path("output") / "final" / "test_images"
    debug_dir = Path("output") / "debug" / "test_images"
    errors_dir = Path("output") / "errors" / "test_images"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up output path
    output_path = output_dir / image_name
    
    # Process the image
    start_time = time.time()
    success = crop_largest_object(image_path, output_path, border_size)
    processing_time = time.time() - start_time
    
    if success:
        print(f"Successfully cropped {image_name}")
        print(f"Output dimensions: {cv2.imread(str(output_path)).shape[1]}x{cv2.imread(str(output_path)).shape[0]}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Final image saved to: {output_path}")
        
        # Display the image and ask for verification
        print(f"\nVerifying image: {image_name}")
        if not display_image(output_path):
            # Move to errors directory if incorrect
            error_path = errors_dir / image_name
            shutil.move(str(output_path), str(error_path))
            print(f"Moved incorrect image to: {error_path}")
        else:
            print("Image verified as correct.")
    else:
        print(f"Failed to crop {image_name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_cropper.py <image_path> <border_size>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    border_size = int(sys.argv[2])
    
    test_single_image(image_path, border_size) 