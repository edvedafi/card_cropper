# Card Cropper

A Python tool for automatically detecting and cropping cards from images. The tool uses multiple detection methods, from simple to advanced, to handle various image conditions and card types.

## Features

- Multiple detection methods:
  - Basic: Simple contour detection and edge finding
  - Enhanced: Multiple edge detection methods and rotation correction
  - ML-based: YOLO model with fallback to adaptive thresholding
  - Aggressive: Multiple color spaces and aggressive morphological operations
- Support for single images and zip files
- Configurable border size
- Detailed error reporting
- Test mode to evaluate different detection methods

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cardcropper.git
cd cardcropper
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Process a single image:
```bash
python src/card_cropper.py input.jpg --border 10
```

### Process a zip file:
```bash
python src/card_cropper.py input.zip --border 10
```

### Test different detection methods:
```bash
python src/tests/test_detectors.py input.jpg --border 10
```

### Options:
- `--output`: Output directory (default: 'output')
- `--border`: Border size in pixels (default: 5)

## Project Structure

```
cardcropper/
├── src/
│   ├── detectors/
│   │   ├── basic_detector.py
│   │   ├── enhanced_detector.py
│   │   ├── ml_detector.py
│   │   └── aggressive_detector.py
│   ├── tests/
│   │   └── test_detectors.py
│   └── card_cropper.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 