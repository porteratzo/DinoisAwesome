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

### Model backends and prompt limitations

The app supports two SAM 3 backends selected by the `USE_TRACKER` environment variable.
Each backend has different supported prompt types — the UI greys out unavailable options
and shows a tooltip on hover explaining the limitation and how to fix it.

| Prompt type | Sam3Model (default) | Sam3TrackerModel (`USE_TRACKER=1`) |
|---|---|---|
| Text | ✅ | ✅ |
| Positive boxes | ✅ | ✅ |
| Negative boxes (right-drag) | ✅ (`input_boxes_labels=0`) | ❌ API has no box-label argument — silently dropped |
| Mixed (text + boxes) | ✅ | ✅ |
| Click points | ❌ No native support — converted to tiny boxes internally, unreliable | ✅ Native `input_points`/`input_labels` |
| Points + Boxes | ❌ Same point limitation | ✅ |

**Choosing a backend:**

- Use the default **Sam3Model** (`USE_TRACKER=0`) for concept/text-driven annotation (e.g. "find all bolts") and when you need negative bounding boxes to exclude regions.
- Use **Sam3TrackerModel** (`USE_TRACKER=1`) for interactive click-based annotation where you click to include/exclude specific pixels.  Negative boxes are not available in this mode — use right-click points (background label=0) instead.

**Why points are unreliable with Sam3Model:**
`Sam3Processor` accepts only `text` and `input_boxes`; there is no `input_points` argument (confirmed in the [official API docs](https://huggingface.co/docs/transformers/en/model_doc/sam3)).  The app works around this by converting each click to a tiny bounding box, but this is a semantically different prompt and the model responds poorly to it.

**Why negative boxes are silently dropped with Sam3TrackerModel:**
`Sam3TrackerProcessor.__call__` accepts only `input_boxes` with no label argument (confirmed in the [official API docs](https://huggingface.co/docs/transformers/en/model_doc/sam3_tracker)).  All boxes are treated as positive.
