"""Flask annotation server for the DinoisAwesome image segmentation tool.

Start with:
    WORKING_DIR=/path/to/images python app.py

Logging is configured before SAM3Service() is instantiated so that our
basicConfig() call takes effect before transformers/torch import their own
handlers (torch may register handlers at import time on some builds).
"""

from __future__ import annotations

import base64
import glob
import io
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image

# sam_service module-level code does NOT import torch; torch is imported lazily
# inside SAM3Service._load() → safe to import the class here before basicConfig.
from sam_service import SAM3Service

# ── Logging ───────────────────────────────────────────────────────────────
# Must be configured before SAM3Service() is instantiated; that call triggers
# _load(), which does `from transformers import ...`, which in turn imports torch.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────
_WORKING_DIR = Path(os.environ.get("WORKING_DIR", ".")).resolve()
_IMAGE_EXTS = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
_log.info("WORKING_DIR: %s", _WORKING_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Model is loaded once at startup; torch/transformers are imported here.
_sam = SAM3Service()


# ── Path helpers ──────────────────────────────────────────────────────────


def _safe_resolve(filepath: str) -> Path:
    """Resolve a relative path within WORKING_DIR; abort(403) on traversal."""
    target = (_WORKING_DIR / filepath).resolve()
    if not str(target).startswith(str(_WORKING_DIR)):
        abort(403)
    return target


def _annotation_subdir(image_path_str: str, class_name: str, instance_id: int) -> Path:
    """Return the per-annotation directory under WORKING_DIR/annotations/."""
    stem = Path(image_path_str.replace("/", "__").replace("\\", "__")).stem
    return _WORKING_DIR / "annotations" / stem / f"{class_name}_{instance_id}"


def _decode_image_b64(image_b64: str) -> Image.Image:
    """Decode a base64 string (plain or data-URI) to a PIL RGB image."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ── Routes ────────────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/images")
def list_images():
    """Return all jpg/png images under WORKING_DIR (excluding annotations/)."""
    found: list[str] = []
    for ext in _IMAGE_EXTS:
        found.extend(glob.glob(str(_WORKING_DIR / "**" / f"*.{ext}"), recursive=True))
    rel_paths = sorted(
        str(Path(p).relative_to(_WORKING_DIR)) for p in found if "annotations" not in Path(p).parts
    )
    return jsonify({"images": rel_paths})


@app.route("/api/image/<path:filepath>")
def serve_image(filepath: str):
    target = _safe_resolve(filepath)
    if not target.is_file():
        abort(404)
    return send_file(target)


@app.route("/api/annotations/<path:filepath>")
def list_annotations(filepath: str):
    """Return all saved annotations for the given image (relative) path."""
    stem = Path(filepath.replace("/", "__").replace("\\", "__")).stem
    ann_dir = _WORKING_DIR / "annotations" / stem
    if not ann_dir.exists():
        return jsonify({"annotations": []})

    annotations = []
    for subdir in sorted(ann_dir.iterdir()):
        if not subdir.is_dir():
            continue
        json_file = subdir / "annotation.json"
        if json_file.exists():
            try:
                annotations.append(json.loads(json_file.read_text()))
            except json.JSONDecodeError:
                _log.warning("Malformed annotation JSON at %s", json_file)
    return jsonify({"annotations": annotations})


@app.route("/api/segment", methods=["POST"])
def segment():
    """Run SAM 3 inference.  Returns HTTP 429 while the model is busy."""
    body = request.get_json(force=True)
    image_b64: str = body.get("image_b64", "")
    prompt_type: str = body.get("prompt_type", "boxes")
    prompt_data: dict = body.get("prompt_data", {})

    if not image_b64:
        return jsonify({"error": "Missing image_b64"}), 400

    if _sam.is_processing:
        return jsonify({"error": "Model busy"}), 429

    try:
        image = _decode_image_b64(image_b64)
    except Exception:
        _log.exception("Failed to decode image")
        return jsonify({"error": "Invalid image data"}), 400

    try:
        if prompt_type == "text":
            text = prompt_data.get("text", "").strip()
            if not text:
                return jsonify({"error": "text required for prompt_type=text"}), 400
            masks = _sam.segment_with_text(image, text)

        elif prompt_type == "boxes":
            boxes: list[list[int]] = prompt_data.get("boxes", [])
            labels: list[int] = prompt_data.get("labels", [1] * len(boxes))
            if not boxes:
                return jsonify({"error": "boxes required for prompt_type=boxes"}), 400
            masks = _sam.segment_with_boxes(image, boxes, labels)

        else:  # mixed
            text = prompt_data.get("text", "").strip()
            boxes = prompt_data.get("boxes", [])
            labels = prompt_data.get("labels", [1] * len(boxes))
            masks = _sam.segment_mixed(image, text, boxes, labels)

    except RuntimeError as exc:
        if "busy" in str(exc).lower():
            return jsonify({"error": "Model busy"}), 429
        _log.exception("SAM 3 inference error")
        return jsonify({"error": "Inference failed"}), 500

    masks_out = [m.tolist() for m in masks]
    return jsonify({"masks": masks_out, "count": len(masks_out)})


@app.route("/api/save", methods=["POST"])
def save_annotation():
    """Persist a boolean mask (.npy) and its metadata (.json)."""
    body = request.get_json(force=True)
    image_path_str: str = body["image_path"]
    class_name: str = body["class_name"]
    instance_id: int = int(body["instance_id"])
    mask_data: list = body["mask_data"]
    prompt_type: str = body.get("prompt_type", "boxes")
    prompt_content: dict = body.get("prompt_content", {})

    subdir = _annotation_subdir(image_path_str, class_name, instance_id)
    subdir.mkdir(parents=True, exist_ok=True)

    np.save(subdir / "mask.npy", np.array(mask_data, dtype=bool))

    annotation = {
        "class": class_name,
        "instance_id": instance_id,
        "prompt_type": prompt_type,
        "prompt_content": prompt_content,
        "image_path": image_path_str,
    }
    (subdir / "annotation.json").write_text(json.dumps(annotation, indent=2))
    _log.info("Saved annotation: %s / %s_%d", image_path_str, class_name, instance_id)
    return jsonify({"status": "ok"})


@app.route("/api/annotation", methods=["DELETE"])
def delete_annotation():
    """Remove a mask/json pair by class + instance ID."""
    body = request.get_json(force=True)
    image_path_str: str = body["image_path"]
    class_name: str = body["class_name"]
    instance_id: int = int(body["instance_id"])

    subdir = _annotation_subdir(image_path_str, class_name, instance_id)
    if not subdir.exists():
        abort(404)
    shutil.rmtree(subdir)
    _log.info("Deleted annotation: %s / %s_%d", image_path_str, class_name, instance_id)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
