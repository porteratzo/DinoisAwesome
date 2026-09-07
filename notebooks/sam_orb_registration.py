"""Prototype: SAM3-masked registration on abc5 part photos, ORB vs LoFTR.

Simple first pass at image registration for the abc5 part (black plastic
window-regulator assembly with clips/components) photographed on a tiled
floor at different positions:

  1. Segment the part out of each image with SAM3 (text prompt), so the
     background floor never contributes keypoints.
  2. Match a reference image against a handful of other abc5 images of the
     same part, restricted to each image's own foreground mask, estimate a
     homography with RANSAC, and warp each match back onto the reference
     frame.
  3. Two matchers are compared side by side:
       - ORB (cv2): keypoints are detected only inside the foreground mask
         via ``detectAndCompute``'s ``mask=`` argument (not pixel zeroing —
         zeroing would create a hard mask-edge that ORB would happily latch
         corners onto).
       - LoFTR: a detector-free transformer matcher that correlates image
         patches densely instead of relying on sparse corner-like keypoints,
         so it holds up much better on the largely textureless/reflective
         black plastic here — ORB found under 10 RANSAC inliers on 3 of the
         4 test images in the first version of this script.

     LoFTR was picked by surveying github.com/gmberton/vismatch, a wrapper
     around 50+ image-matching models with a unified ``get_matcher()`` API.
     This script does NOT depend on the ``vismatch`` package itself, though:
     ``pip install vismatch`` pulls in an unpinned, very heavy dependency
     tree (lightning, tensorflow-only extras, etc.) and its resolver tried
     to upgrade this env's ``torch`` — shared by every module in this repo
     — to a build without matching CUDA support. Its ``loftr.py`` wrapper is
     a thin ~15-line shim around ``kornia.feature.LoFTR`` plus
     ``cv2.findHomography``, so this script calls kornia directly instead
     (``pip install "kornia>=0.7.3,<0.8.3"`` — 0.8.3 dropped a helper LoFTR
     needs), reproducing the same method without the extra weight. LoFTR has
     no per-keypoint mask concept, so the foreground mask is instead baked
     into the pixels (background zeroed) before matching, and images are
     downscaled first since the transformer backbone is too slow/
     memory-heavy at full 3840x2160.

Outputs go to ``outputs/sam_orb_registration/``:
  ``<stem>_mask.png``              — SAM3 foreground mask used for gating
  ``<stem>_<method>_matches.png``  — inlier matches drawn between ref/image
  ``<stem>_<method>_warped.png``   — image warped into the reference frame
  ``<stem>_<method>_overlay.png``  — reference/warped blend, to eyeball fit

Usage
-----
python notebooks/sam_orb_registration.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Logging must be configured before torch is imported (torch may register
# handlers at import time on some builds).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "annotationapp"))

# isort: off
import torch  # noqa: E402
import cv2  # noqa: E402
from sam_service import SAM3Service  # noqa: E402
# isort: on

DATA_DIR = REPO_ROOT / "data" / "abc5"
OUTPUT_DIR = REPO_ROOT / "outputs" / "sam_orb_registration"

REFERENCE_IMAGE = "LHa_1.jpg"
OTHER_IMAGES = ["LHa_2.jpg", "LHa_3.jpg", "LHa_4.jpg", "LHa_5.jpg"]
SEGMENT_TEXT_PROMPT = "black plastic part"
METHODS = ("orb", "loftr")

ORB_N_FEATURES = 4000
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0

LOFTR_MAX_DIM = 840  # kornia's LoFTR is transformer-based; full-res is too slow/memory-heavy

_EMPTY_RESULT = {
    "num_matches": 0,
    "num_inliers": 0,
    "H": None,
    "inlier_ref": np.empty((0, 2), dtype=np.float32),
    "inlier_other": np.empty((0, 2), dtype=np.float32),
}


def segment_foreground_mask(sam: SAM3Service, image: Image.Image) -> np.ndarray:
    """Return a uint8 (H, W) mask (255=foreground) for the whole part."""
    masks = sam.segment_with_text(image, SEGMENT_TEXT_PROMPT)
    if not masks:
        raise RuntimeError(f"SAM3 found no '{SEGMENT_TEXT_PROMPT}' mask")
    # Multiple instances can come back (e.g. the part plus a stray clip) —
    # keep the largest, since that's the whole-part region we want.
    largest = max(masks, key=lambda m: int(m.sum()))
    return (largest.astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# ORB matcher
# ---------------------------------------------------------------------------


def detect_and_describe(gray: np.ndarray, mask: np.ndarray) -> tuple[Any, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(gray, mask=mask)
    return keypoints, descriptors


def match_orb(kp_ref: Any, desc_ref: np.ndarray, kp_other: Any, desc_other: np.ndarray) -> dict:
    if desc_ref is None or desc_other is None or not kp_ref or not kp_other:
        return dict(_EMPTY_RESULT)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(desc_ref, desc_other, k=2)
    good_matches = [m for m, n in knn_matches if m.distance < LOWE_RATIO * n.distance]
    if len(good_matches) < 4:
        return {**_EMPTY_RESULT, "num_matches": len(good_matches)}

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
    pts_other = np.float32([kp_other[m.trainIdx].pt for m in good_matches])

    H, inlier_mask = cv2.findHomography(pts_other, pts_ref, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return {**_EMPTY_RESULT, "num_matches": len(good_matches)}

    inlier_mask = inlier_mask.ravel().astype(bool)
    return {
        "num_matches": len(good_matches),
        "num_inliers": int(inlier_mask.sum()),
        "H": H,
        "inlier_ref": pts_ref[inlier_mask],
        "inlier_other": pts_other[inlier_mask],
    }


# ---------------------------------------------------------------------------
# LoFTR matcher (method surveyed from github.com/gmberton/vismatch,
# implemented directly against kornia — see module docstring for why)
# ---------------------------------------------------------------------------


def build_loftr_matcher(device: str) -> Any:
    try:
        from kornia.feature import LoFTR
    except ImportError as exc:
        raise ImportError(
            "kornia is required for the 'loftr' registration method.\n"
            'Install it with: pip install "kornia>=0.7.3,<0.8.3"'
        ) from exc
    log.info("Loading LoFTR (kornia, outdoor weights) on device=%s", device)
    model = LoFTR(pretrained="outdoor").to(device)
    model.eval()
    return model


def resize_for_loftr(
    bgr: np.ndarray, mask: np.ndarray, max_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    size = (int(round(w * scale)), int(round(h * scale)))
    bgr_small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return bgr_small, mask_small


def to_loftr_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    """Grayscale uint8 BGR image -> (1, 1, H, W) float tensor in [0, 1]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def match_loftr(
    matcher: Any,
    device: str,
    ref_bgr: np.ndarray,
    ref_mask: np.ndarray,
    other_bgr: np.ndarray,
    other_mask: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run LoFTR on masked, downscaled copies of the images.

    LoFTR is detector-free (dense correlation, no independent keypoint
    stage), so there is no per-keypoint mask to apply as with ORB — the
    foreground mask is instead baked into the pixels (background zeroed)
    before matching. Returns the result dict alongside the resized images
    the keypoints refer to, since they're needed for drawing/warping.
    """
    ref_small, ref_mask_small = resize_for_loftr(ref_bgr, ref_mask, LOFTR_MAX_DIM)
    other_small, other_mask_small = resize_for_loftr(other_bgr, other_mask, LOFTR_MAX_DIM)

    ref_fg = cv2.bitwise_and(ref_small, ref_small, mask=ref_mask_small)
    other_fg = cv2.bitwise_and(other_small, other_small, mask=other_mask_small)

    # image0=other, image1=ref so the returned keypoints0/1 map other -> ref
    # directly, matching what cv2.warpPerspective needs below.
    batch = {
        "image0": to_loftr_tensor(other_fg, device),
        "image1": to_loftr_tensor(ref_fg, device),
    }
    with torch.inference_mode():
        output = matcher(batch)

    pts_other = output["keypoints0"].cpu().numpy()
    pts_ref = output["keypoints1"].cpu().numpy()
    num_matches = len(pts_other)
    if num_matches < 4:
        return {**_EMPTY_RESULT, "num_matches": num_matches}, ref_small, other_small

    # USAC_MAGSAC (as used by vismatch's own RANSAC step) is a more robust
    # estimator than plain RANSAC, worth the (free) upgrade for this path.
    H, inlier_mask = cv2.findHomography(
        pts_other, pts_ref, cv2.USAC_MAGSAC, RANSAC_REPROJ_THRESHOLD
    )
    if H is None:
        return {**_EMPTY_RESULT, "num_matches": num_matches}, ref_small, other_small

    inlier_mask = inlier_mask.ravel().astype(bool)
    result = {
        "num_matches": num_matches,
        "num_inliers": int(inlier_mask.sum()),
        "H": H,
        "inlier_ref": pts_ref[inlier_mask],
        "inlier_other": pts_other[inlier_mask],
    }
    return result, ref_small, other_small


# ---------------------------------------------------------------------------
# Shared visualization / output
# ---------------------------------------------------------------------------


def draw_point_matches(
    img_ref: np.ndarray,
    img_other: np.ndarray,
    pts_ref: np.ndarray,
    pts_other: np.ndarray,
    max_lines: int = 200,
) -> np.ndarray:
    """Draw lines between matched points on a side-by-side canvas."""
    h = max(img_ref.shape[0], img_other.shape[0])
    w_ref = img_ref.shape[1]
    canvas = np.zeros((h, w_ref + img_other.shape[1], 3), dtype=np.uint8)
    canvas[: img_ref.shape[0], :w_ref] = img_ref
    canvas[: img_other.shape[0], w_ref:] = img_other

    rng = np.random.default_rng(0)
    n = len(pts_ref)
    idx = rng.choice(n, size=min(max_lines, n), replace=False) if n else np.empty(0, dtype=int)
    for i in idx:
        p_ref = (int(pts_ref[i, 0]), int(pts_ref[i, 1]))
        p_other = (int(pts_other[i, 0]) + w_ref, int(pts_other[i, 1]))
        color = tuple(int(c) for c in rng.integers(64, 255, size=3))
        cv2.line(canvas, p_ref, p_other, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p_ref, 3, color, -1)
        cv2.circle(canvas, p_other, 3, color, -1)
    return canvas


def save_registration_outputs(
    method: str, stem: str, ref_img: np.ndarray, other_img: np.ndarray, result: dict
) -> None:
    log.info(
        "%s [%s]: %d matches, %d RANSAC inliers",
        stem,
        method,
        result["num_matches"],
        result["num_inliers"],
    )

    match_vis = draw_point_matches(ref_img, other_img, result["inlier_ref"], result["inlier_other"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_{method}_matches.png"), match_vis)

    if result["H"] is None:
        log.warning("%s [%s]: no homography found, skipping warp/overlay", stem, method)
        return

    h, w = ref_img.shape[:2]
    warped = cv2.warpPerspective(other_img, result["H"], (w, h))
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_{method}_warped.png"), warped)

    overlay = cv2.addWeighted(ref_img, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_{method}_overlay.png"), overlay)


def register_image(
    sam: SAM3Service,
    loftr_matcher: Any,
    loftr_device: str,
    ref_bgr: np.ndarray,
    ref_gray: np.ndarray,
    ref_mask: np.ndarray,
    kp_ref: Any,
    desc_ref: np.ndarray,
    other_path: Path,
) -> None:
    stem = other_path.stem
    other_pil = Image.open(other_path).convert("RGB")
    other_bgr = cv2.cvtColor(np.array(other_pil), cv2.COLOR_RGB2BGR)
    other_gray = cv2.cvtColor(other_bgr, cv2.COLOR_BGR2GRAY)

    other_mask = segment_foreground_mask(sam, other_pil)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_mask.png"), other_mask)

    if "orb" in METHODS:
        kp_other, desc_other = detect_and_describe(other_gray, other_mask)
        result = match_orb(kp_ref, desc_ref, kp_other, desc_other)
        save_registration_outputs("orb", stem, ref_bgr, other_bgr, result)

    if "loftr" in METHODS:
        result, ref_small, other_small = match_loftr(
            loftr_matcher, loftr_device, ref_bgr, ref_mask, other_bgr, other_mask
        )
        save_registration_outputs("loftr", stem, ref_small, other_small, result)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading SAM3 (this downloads/loads weights on first run)")
    sam = SAM3Service()

    loftr_matcher = None
    loftr_device = "cuda" if torch.cuda.is_available() else "cpu"
    if "loftr" in METHODS:
        loftr_matcher = build_loftr_matcher(loftr_device)

    ref_path = DATA_DIR / REFERENCE_IMAGE
    ref_pil = Image.open(ref_path).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    log.info("Segmenting reference image %s", REFERENCE_IMAGE)
    ref_mask = segment_foreground_mask(sam, ref_pil)
    cv2.imwrite(str(OUTPUT_DIR / f"{ref_path.stem}_mask.png"), ref_mask)

    kp_ref, desc_ref = detect_and_describe(ref_gray, ref_mask)
    log.info("Reference: %d ORB keypoints inside foreground mask", len(kp_ref))
    if "orb" in METHODS and (desc_ref is None or len(kp_ref) == 0):
        raise RuntimeError("No ORB features found in reference foreground mask")

    for name in OTHER_IMAGES:
        other_path = DATA_DIR / name
        if not other_path.exists():
            log.warning("Skipping missing image %s", other_path)
            continue
        register_image(
            sam,
            loftr_matcher,
            loftr_device,
            ref_bgr,
            ref_gray,
            ref_mask,
            kp_ref,
            desc_ref,
            other_path,
        )

    log.info("Done — outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
