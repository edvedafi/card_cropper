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
from src.tests.detect_quality import check_crop_quality
from src.detectors.basic_detector import detect_card as detect_card_basic
from src.detectors.enhanced_detector import detect_card as detect_card_enhanced
from src.detectors.ml_detector import detect_card as detect_card_ml
from src.detectors.aggressive_detector import detect_card as detect_card_aggressive
from src.detectors.rembg_detector import detect_card as detect_card_rembg

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

def process_image(image_path, output_path, border_size=5):
    """
    Process a single image using all detection methods in sequence.
    Returns True if any method succeeds and passes quality check, False otherwise.
    """
    print(f"\nProcessing image: {os.path.basename(image_path)}")
    print("=" * 50)
    
    # Try each detector in order of complexity
    detectors = [
        ('rembg', detect_card_rembg),  # Try rembg first as it's often reliable
        ('ml', detect_card_ml),
        ('basic', detect_card_basic),
        ('enhanced', detect_card_enhanced),
        ('aggressive', detect_card_aggressive)
    ]
    
    for name, detector in detectors:
        print(f"\nTrying {name} detection...")
        start_time = time.time()
        success, error = detector(image_path, output_path, border_size)
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"✓ {name} detection succeeded in {duration:.2f} seconds")
            print(f"  Output saved to: {output_path}")
            
            # Verify the quality of the detected card
            print("\nVerifying crop quality...")
            is_valid, issues = check_crop_quality(image_path, output_path, interactive=True)
            
            if is_valid:
                print("✓ Quality check passed")
                print("\n" + "=" * 50)
                return True
            else:
                print("✗ Quality check failed")
                print("  Issues found:")
                for issue in issues:
                    print(f"    - {issue}")
                print(f"\nTrying next detection method...")
        else:
            print(f"✗ {name} detection failed in {duration:.2f} seconds")
            print(f"  Error: {error}")
    
    print("\nAll detection methods failed or failed quality check")
    print("=" * 50)
    return False

def process_zip_file(zip_path, output_dir, border_size=5):
    """
    Process all images in a zip file.
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
                    
                    if process_image(image_path, output_path, border_size):
                        success_count += 1
                    else:
                        print(f"Failed to process {image_path.name}")
    
    except Exception as e:
        print(f"Error processing zip file: {str(e)}")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description='Process images or zip files containing images.')
    parser.add_argument('input', help='Input image or zip file')
    parser.add_argument('--border', type=int, default=5, help='Border size in pixels (default: 5)')
    parser.add_argument('--clean', action='store_true', help='Delete contents of debug and output directories before running')
    args = parser.parse_args()
    
    if args.clean:
        # Clean debug and output directories
        for dir_path in ['debug', 'output']:
            if os.path.exists(dir_path):
                print(f"Cleaning {dir_path} directory...")
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
    
    input_path = Path(args.input)
    output_dir = Path('output')
    
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
        
        if process_image(input_path, output_path, args.border):
            print("Successfully processed image")
        else:
            print("Failed to process image")
            
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")

if __name__ == '__main__':
    main() 