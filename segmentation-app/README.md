# Segmentation Mask Inference Application

A desktop GUI application for binary segmentation inference with mask-based metrics computation.

## Features

- Load PyTorch model weights (.pth) with automatic config detection
- Interactive image segmentation with adjustable threshold and overlay alpha
- Tabbed preview interface (Input/Mask/Overlay)
- Personalized metrics computation from binary masks
- Batch processing of image folders
- Modern web-inspired UI design
- Scrollable interface for smaller screens

## Installation

1. Create a virtual environment:
\`\`\`bash
python -m venv .venv
\`\`\`

2. Activate the virtual environment:
\`\`\`bash
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

Run the application:
\`\`\`bash
python app.py
\`\`\`

### Workflow

1. **Load Model**: Browse and select your trained model weights (.pth file)
2. **Open Image**: Select an image for segmentation (PNG/JPG/TIFF supported)
3. **Adjust Parameters**: Set threshold and overlay alpha using sliders or direct input
4. **Run Segmentation**: Click "Segment" to generate mask and overlay
5. **Save Results**: Export mask and overlay images as needed
6. **Compute Metrics**: Generate detailed metrics from predicted or custom masks
7. **Batch Processing**: Process entire folders of images automatically

### File Structure

- `app.py` - Application entry point
- `main_app.py` - Main application class and UI logic
- `models.py` - Model classes and inference utilities
- `ui_components.py` - Custom UI widgets (ScrollableFrame)
- `theme.py` - UI theme and styling configuration
- `metrics.py` - Metrics computation functions (external dependency)

## Model Requirements

The application expects:
- PyTorch model weights (.pth file)
- Optional: `config.json` in the same directory for automatic architecture detection
- Optional: `dataset_mean.npy` and `dataset_std.npy` for custom normalization

## Batch Processing

Select a folder containing images to process all at once. Results are saved to a `results/` subfolder:
- `*_mask.png` - Binary mask images
- `metrics_summary.csv` - Comprehensive metrics for all processed images

The application automatically skips already processed images for efficient re-runs.
