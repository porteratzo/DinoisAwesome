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

# ── Load .env from the repo root (parent of this file's directory) ────────
# Done before any HuggingFace/torch import so HF_TOKEN is available for
# from_pretrained() calls inside SAM3Service._load().
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

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
_CLASSES_FILE = _WORKING_DIR / "classes.json"
_log.info("WORKING_DIR: %s", _WORKING_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Model is loaded once at startup; torch/transformers are imported here.
_sam = SAM3Service()


# ── Mask encoding ─────────────────────────────────────────────────────────


def _rle_encode(mask: np.ndarray) -> dict:
    """Encode a (H, W) bool mask as RLE for compact JSON transport.

    Returns {"shape": [H, W], "rle": [count, ...]}.  Run lengths start with
    the False (background) count, alternating False/True.  If the mask starts
    with True a leading 0 is prepended so the even/odd convention holds.
    """
    flat = mask.ravel().astype(np.uint8)
    n = len(flat)
    if n == 0:
        return {"shape": list(mask.shape), "rle": []}
    changes = np.where(np.diff(flat))[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [n]])
    lengths: list[int] = (ends - starts).tolist()
    if flat[0]:
        lengths = [0] + lengths
    return {"shape": list(mask.shape), "rle": lengths}


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


@app.route("/api/thumbnail/<path:filepath>")
def serve_thumbnail(filepath: str):
    """Return a small JPEG thumbnail (max 180×140) for the gallery sidebar."""
    target = _safe_resolve(filepath)
    if not target.is_file():
        abort(404)
    try:
        with Image.open(target) as img:
            img.thumbnail((180, 140))
            thumb = img.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=72)
            buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception:
        _log.exception("Failed to generate thumbnail for %s", filepath)
        abort(500)


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


def _run_inference(image, prompt_type: str, prompt_data: dict, filter_by_prompt: bool):
    """Dispatch to the appropriate SAM method and return a list of masks.

    Raises ValueError with a user-facing message on bad input, or RuntimeError
    ('busy') when the model is already processing.
    """
    fbp = filter_by_prompt
    if prompt_type == "points":
        points: list[list[int]] = prompt_data.get("points", [])
        if not points:
            raise ValueError("points required for prompt_type=points")
        labels: list[int] = prompt_data.get("labels", [1] * len(points))
        return _sam.segment_with_points(image, points, labels, filter_by_prompt=fbp)

    if prompt_type == "text":
        text: str = prompt_data.get("text", "").strip()
        if not text:
            raise ValueError("text required for prompt_type=text")
        return _sam.segment_with_text(image, text)

    if prompt_type == "boxes":
        boxes: list[list[int]] = prompt_data.get("boxes", [])
        if not boxes:
            raise ValueError("boxes required for prompt_type=boxes")
        box_labels: list[int] = prompt_data.get("labels", [1] * len(boxes))
        return _sam.segment_with_boxes(image, boxes, box_labels, filter_by_prompt=fbp)

    if prompt_type == "points_boxes":
        pts: list[list[int]] = prompt_data.get("points", [])
        pt_labels: list[int] = prompt_data.get("point_labels", [1] * len(pts))
        bxs: list[list[int]] = prompt_data.get("boxes", [])
        bx_labels: list[int] = prompt_data.get("box_labels", [1] * len(bxs))
        if not pts and not bxs:
            raise ValueError("points or boxes required for prompt_type=points_boxes")
        return _sam.segment_with_points_and_boxes(image, pts, pt_labels, bxs, bx_labels, filter_by_prompt=fbp)

    # mixed (default)
    text = prompt_data.get("text", "").strip()
    mixed_boxes: list[list[int]] = prompt_data.get("boxes", [])
    mixed_labels: list[int] = prompt_data.get("labels", [1] * len(mixed_boxes))
    return _sam.segment_mixed(image, text, mixed_boxes, mixed_labels, filter_by_prompt=fbp)


@app.route("/api/segment", methods=["POST"])
def segment():
    """Run SAM 3 inference.  Returns HTTP 429 while the model is busy."""
    body = request.get_json(force=True)
    image_b64: str = body.get("image_b64", "")
    prompt_type: str = body.get("prompt_type", "boxes")
    prompt_data: dict = body.get("prompt_data", {})
    filter_by_prompt: bool = bool(body.get("filter_by_prompt", False))

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
        masks = _run_inference(image, prompt_type, prompt_data, filter_by_prompt)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        if "busy" in str(exc).lower():
            return jsonify({"error": "Model busy"}), 429
        _log.exception("SAM 3 inference error")
        return jsonify({"error": "Inference failed"}), 500

    # RLE-encode masks — raw 2D bool arrays are ~8 M values per 4K mask;
    # RLE reduces that to ~20 K integers for a typical segmentation mask.
    masks_out = [_rle_encode(m) for m in masks]
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


@app.route("/api/annotation/mask/<path:filepath>")
def get_annotation_mask(filepath: str):
    """Return the boolean mask for a saved annotation as a JSON 2-D array.

    Query params: class_name, instance_id
    """
    class_name = request.args.get("class_name", "")
    instance_id_str = request.args.get("instance_id", "")
    if not class_name or not instance_id_str:
        return jsonify({"error": "class_name and instance_id are required"}), 400
    try:
        instance_id = int(instance_id_str)
    except ValueError:
        return jsonify({"error": "instance_id must be an integer"}), 400

    subdir = _annotation_subdir(filepath, class_name, instance_id)
    mask_path = subdir / "mask.npy"
    if not mask_path.exists():
        abort(404)

    try:
        mask = np.load(mask_path)
        return jsonify({"mask": _rle_encode(mask)})
    except Exception:
        _log.exception("Failed to load mask from %s", mask_path)
        abort(500)


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


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return the saved class list from WORKING_DIR/classes.json."""
    if _CLASSES_FILE.exists():
        try:
            data = json.loads(_CLASSES_FILE.read_text())
            classes = data.get("classes", ["object"])
        except json.JSONDecodeError:
            _log.warning("Malformed classes.json — using default")
            classes = ["object"]
    else:
        classes = ["object"]
    return jsonify({"classes": classes})


@app.route("/api/classes", methods=["POST"])
def add_class():
    """Append a new class name to WORKING_DIR/classes.json."""
    body = request.get_json(force=True)
    name: str = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Missing name"}), 400

    if _CLASSES_FILE.exists():
        try:
            data = json.loads(_CLASSES_FILE.read_text())
            classes: list[str] = data.get("classes", ["object"])
        except json.JSONDecodeError:
            classes = ["object"]
    else:
        classes = ["object"]

    if name not in classes:
        classes.append(name)
        _CLASSES_FILE.write_text(json.dumps({"classes": classes}, indent=2))
        _log.info("Added class '%s' → %s", name, _CLASSES_FILE)

    return jsonify({"classes": classes})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
