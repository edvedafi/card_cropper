# Card Cropper

A Python tool for automatically detecting, cropping, and perspective-correcting multiple cards from a single image or a zip file containing multiple images.

## Features

- Automatic card detection using computer vision techniques
- Perspective correction for skewed or rotated cards
- 10-pixel margin around each cropped card
- Support for multiple cards in a single image
- Batch processing of images from zip files
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

3. Customize output directory and padding:
```bash
python card_cropper.py input_image.jpg --output my_cards --padding 15
```

Available options:
- `--output`, `-o`: Output directory for cropped cards (default: 'output_cards')
- `--padding`, `-p`: Padding around each card in pixels (default: 10)

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
num_cards = cropper.process_image("your_image.jpg", "output_dir")
print(f"Processed {num_cards} cards")
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
- BMP (.bmp)

## Output

The script will create an output directory containing individual image files for each detected card. Each card will be:
- Perspective-corrected to appear as a top-down view
- Cropped with the specified padding around all edges
- Saved as a separate JPEG file

For single images:
- Files are named `card_1.jpg`, `card_2.jpg`, etc.

For zip files:
- Files are named using the original image name as a prefix
- Example: `folder1_image1_card_1.jpg`, `folder1_image1_card_2.jpg`

## Error Handling

The script includes basic error handling for:
- Invalid input image paths
- Corrupted zip files
- Image reading errors
- Output directory creation issues

When processing zip files, the script will:
- Continue processing even if some images fail
- Provide a summary of successful and failed operations
- List any errors encountered during processing

## Contributing

Feel free to submit issues and enhancement requests! 