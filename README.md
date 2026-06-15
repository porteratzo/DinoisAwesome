# DinoisAwesome

Experiments with DINO vision transformer encoders (v2 / v3). Provides feature extraction and patch-level retrieval galleries backed by pandas + memory-mapped NumPy arrays.

## Installation

```bash
pip install -e ".[dev]"
```

## Annotation App

An interactive Flask web app for segmenting and annotating images with SAM 3, located in [`annotationapp/`](annotationapp/).

### Setup

1. Install dependencies (Flask + transformers are included in the base install above):

   ```bash
   pip install -e "."
   ```

2. Set the directory that contains the images you want to annotate:

   ```bash
   export WORKING_DIR=/path/to/your/images
   ```

   Optionally override the SAM model (defaults to `facebook/sam3`):

   ```bash
   export SAM_MODEL_ID=facebook/sam3
   ```

3. Start the server from inside `annotationapp/`:

   ```bash
   cd annotationapp
   python app.py
   ```

   Or from the repo root:

   ```bash
   WORKING_DIR=/path/to/images python annotationapp/app.py
   ```

4. Open `http://localhost:5000` in your browser.

### How it works

- Images under `WORKING_DIR` are listed in the sidebar.
- Draw a bounding box on the image to prompt SAM 3 for a segmentation mask.
- Name the class and save — masks are stored as `.npy` files under `WORKING_DIR/annotations/`.
- Saved annotations are reloaded automatically when you revisit an image.
