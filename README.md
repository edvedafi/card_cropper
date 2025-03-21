# Card Cropper

A Python tool for automatically detecting, cropping, and perspective-correcting multiple cards from a single image or a zip file containing multiple images.

## Features

- Automatic card detection using multiple computer vision techniques
- Multi-strategy approach for handling different backgrounds and lighting conditions
- Flexible grid-based detection for handling any number of cards in a single image
- Perspective correction for skewed or rotated cards
- 10-pixel margin around each cropped card
- Support for multiple cards in a single image (1-9 cards)
- Batch processing of images from zip files
- Detailed debug output for troubleshooting
- Modular and well-documented code

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- imutils

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd cardcropper
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Process a single image:
```bash
python card_cropper.py input_image.jpg
```

2. Process a zip file containing multiple images:
```bash
python card_cropper.py cards.zip
```

3. Customize output directory and parameters:
```bash
python card_cropper.py input_image.jpg --output my_cards --padding 15 --min-area 0.1 --max-area 0.9
```

Available options:
- `--output`, `-o`: Output directory for cropped cards (default: 'output')
- `--padding`: Padding around each card in pixels (default: 10)
- `--min-area`: Minimum card area ratio relative to image size (default: 0.15)
- `--max-area`: Maximum card area ratio relative to image size (default: 0.95)

### Python API

1. Process a single image:
```python
from card_cropper import CardCropper

# Initialize the card cropper
cropper = CardCropper(
    padding=10,
    min_card_area_ratio=0.15,  # Card must be at least 15% of image
    max_card_area_ratio=0.95   # Card can be up to 95% of image
)

# Process an image
output_paths = cropper.process_image("your_image.jpg", "output_dir")
print(f"Processed {len(output_paths)} cards")
```

2. Process a zip file:
```python
from card_cropper import CardCropper

cropper = CardCropper(padding=10)

# Process all images in a zip file
stats = cropper.process_zip(
    "path/to/your/cards.zip",  # Input zip file
    "output_cards"             # Output directory
)
print(f"Processed {stats['processed_images']} images")
print(f"Found {stats['total_cards']} cards total")
```

## Detection Strategies

The tool uses a multi-strategy approach to detect cards:

1. **Specialized Baseball Card Detection**: First attempts to detect cards using multiple thresholding techniques and contour analysis specifically tuned for baseball cards, which have a standard aspect ratio of 2.5" x 3.5" (0.714).

2. **Direct Rectangle Detection**: If specialized detection fails, attempts to detect cards using multiple thresholding techniques optimized for different backgrounds:
   - Simple binary thresholding for light cards on dark backgrounds
   - Otsu's thresholding for automatic threshold selection
   - Adaptive thresholding for handling varying lighting conditions

3. **Grid-Based Detection**: Uses image analysis to detect cards arranged in a grid pattern. This approach does not assume a fixed number of cards but instead:
   - Analyzes image histograms and projections to identify potential card boundaries
   - Detects peaks in vertical and horizontal projections to identify card separations
   - Estimates the actual number of cards based on detected boundaries

4. **Flexible Grid Detection**: For cases where other methods fail, uses an adaptive grid approach:
   - Analyzes the image to determine the likely number of cards (1-9)
   - Detects horizontal and vertical lines that might separate cards
   - Creates a flexible grid based on the detected lines or image aspect ratio
   - Processes each cell in the grid to find and extract cards
   - Maintains the standard baseball card aspect ratio (2.5" x 3.5")
   - Adapts to different layouts including horizontal rows, vertical columns, or grid arrangements

5. **Hough Line Transform**: If other methods fail, uses Hough Line Transform to detect straight lines that might form card edges.

Each strategy includes preprocessing steps like contrast enhancement, noise reduction, and morphological operations to improve detection quality. The system will try each strategy in sequence until cards are successfully detected.

## Nested Contour Filtering

The tool includes intelligent filtering to handle nested contours:
- Detects when smaller contours are contained within larger ones
- Keeps only the largest non-image-sized contours
- Prevents duplicate detection of the same card
- Ensures that contours covering most of the image are ignored

The tool will:
- Filter out contours that are too large (likely the entire image)
- Filter out nested contours that are parts of the same card
- Process all valid card contours regardless of the number of cards in the image

## Debug Output

The tool generates detailed debug images in the `debug_output` directory to help troubleshoot detection issues:

- Original and preprocessed images
- Thresholding results
- Detected contours and edges
- Corner point detection and ordering
- Perspective transformation steps
- Grid-based detection visualization

## Configuration

You can customize the card detection parameters when initializing the `CardCropper`:

```python
cropper = CardCropper(
    padding=10,                # Padding around each card in pixels
    min_card_area_ratio=0.15,  # Card must be at least 15% of image
    max_card_area_ratio=0.95   # Card can be up to 95% of image
)
```

## Supported Image Formats

The following image formats are supported:
- JPEG (.jpg, .jpeg)
- PNG (.png)

## Output

The script will create an output directory containing individual image files for each detected card. Each card will be:
- Perspective-corrected to appear as a top-down view
- Cropped with the specified padding around all edges
- Saved as a separate JPEG file

For single images:
- Files are named `image_name_card_0.jpg`, `image_name_card_1.jpg`, etc.

For zip files:
- Files are named using the original image name
- Example: `image1_card_0.jpg`, `image1_card_1.jpg`

## Error Handling

The script includes robust error handling for:
- Invalid input image paths
- Corrupted zip files
- Image reading errors
- Output directory creation issues
- Card detection and processing failures

When processing zip files, the script will:
- Continue processing even if some images fail
- Provide a summary of successful and failed operations
- Count any errors encountered during processing

## Contributing

Feel free to submit issues and enhancement requests! 