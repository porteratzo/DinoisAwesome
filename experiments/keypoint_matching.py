# %% [markdown]
# # DINO Keypoint Matching
#
# Register named keypoints on a reference image using a Gallery + KeypointHead,
# then locate those same keypoints in a set of query images via nearest-patch
# cosine similarity.
#
# Workflow:
#   1. Build a Gallery from the reference image.
#   2. Register pixel coordinates + labels with KeypointHead.register().
#   3. Call KeypointHead.find() on each query — one encoder pass per query.
#   4. Visualise reference (marked) vs. queries (predicted) side by side.
#   5. Estimate homography (RANSAC) from correspondences and warp reference into query frame.

# %% Logging — must be before torch import
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("keypoint_matching")

import tempfile
from glob import glob
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image

from dinoisawesome import DinoEncoder, Gallery, KeypointHead

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

part_type = "left_under"
ref_number = 1
data_dir = _REPO_ROOT / "data" / "abc"

all_images = glob(str(data_dir / f"{part_type}_*.jpg"))
REF_IMAGE_PATH: str = [i for i in all_images if f"{part_type}_{str(ref_number).zfill(3)}.jpg" in i][
    0
]
TAR_IMAGE_PATH: list[str] = [
    i for i in all_images if f"{part_type}_{str(ref_number).zfill(3)}.jpg" not in i
]

# Keypoints to register — pixel coords in the ORIGINAL image space (before any resize).
# Edit these to match the reference image; or use the cv2.selectROI block below.
KEYPOINTS: list[dict] | None = [
    {"x": 791, "y": 1149, "label": "point1"},
    {"x": 1698, "y": 961, "label": "point2"},
    {"x": 2118, "y": 877, "label": "point3"},
    {"x": 2797, "y": 759, "label": "point4"},
]

# Query transforms applied to target images.
QUERIES: list[tuple[str, object]] = [
    ("identity", lambda img: img),
    ("rotate +15°", lambda img: img.rotate(15, expand=False, fillcolor=(128, 128, 128))),
    ("rotate -30°", lambda img: img.rotate(-30, expand=False, fillcolor=(128, 128, 128))),
    ("flip H", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("flip V", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ("rotate +90°", lambda img: img.rotate(90, expand=True)),
]

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 1024
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DEBIAS = True

MARKER_RADIUS = 18
MARKER_COLORS = [
    (255, 60, 60),
    (60, 180, 60),
    (60, 100, 255),
    (255, 180, 0),
    (180, 0, 255),
    (0, 210, 210),
]

log.info("Reference image: %s", REF_IMAGE_PATH)
log.info("Target images: %d", len(TAR_IMAGE_PATH))

# %% [markdown]
# ## Interactive keypoint selection (optional)
#
# Uncomment and run the block below if you want to pick keypoints interactively
# with cv2.selectROI instead of hard-coding coordinates above.

# %% Interactive keypoint selection (uncomment to use)
# img_cv = cv2.imread(REF_IMAGE_PATH)
# orig_h, orig_w = img_cv.shape[:2]
# resized_cv = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))
# scale_x = orig_w / IMG_SIZE
# scale_y = orig_h / IMG_SIZE
#
# selected = []
# for label in ["point1", "point2", "point3", "point4"]:
#     roi = cv2.selectROI(f"Select {label}", resized_cv)
#     selected.append({
#         "x": int((roi[0] + roi[2] / 2) * scale_x),
#         "y": int((roi[1] + roi[3] / 2) * scale_y),
#         "label": label,
#     })
# cv2.destroyAllWindows()
# KEYPOINTS = selected
# log.info("Selected keypoints: %s", KEYPOINTS)

# %% Load encoder
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
)
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d",
    DINO_VERSION[1],
    DINO_SIZE,
    encoder.patch_size,
    encoder.grid_h,
    encoder.grid_w,
)

# %% Load reference image and draw registered keypoints
ref_img = Image.open(REF_IMAGE_PATH).convert("RGB")
orig_w, orig_h = ref_img.size
log.info("Reference image: %s  (%dx%d px)", Path(REF_IMAGE_PATH).name, orig_w, orig_h)

labels = [kp["label"] for kp in KEYPOINTS]
label_color = {lbl: MARKER_COLORS[i % len(MARKER_COLORS)] for i, lbl in enumerate(labels)}

canvas_ref = np.array(ref_img)
for kp in KEYPOINTS:
    cv2.circle(
        canvas_ref, (kp["x"], kp["y"]), MARKER_RADIUS, label_color[kp["label"]], thickness=-1
    )

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(canvas_ref)
ax.set_title(f"Reference image  ({orig_w}×{orig_h} px)")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% Build gallery from reference image
_gallery_dir = Path(tempfile.mkdtemp(prefix="dino_kp_gallery_"))
log.info("Building gallery in %s …", _gallery_dir)

gallery = Gallery.build(
    encoder=encoder,
    images=[ref_img],
    image_ids=["ref"],
    out_dir=_gallery_dir,
    split="train",
)
log.info(
    "Gallery built: %d patches, blocks=%s",
    len(gallery.patches),
    gallery.config.block_indices,
)

# %% Register keypoints
head = KeypointHead(gallery, encoder)

points = [[kp["x"], kp["y"]] for kp in KEYPOINTS]
head.register(
    image_id="ref",
    points=points,
    labels=labels,
    orig_size=(orig_w, orig_h),
)
head.save()
log.info("Registered keypoints: %s", head.registered_labels)

# %% Draw reference with registered keypoints


def draw_points(
    img: Image.Image,
    pts: list[tuple[int, int]],
    pt_labels: list[str],
    show_labels: bool = True,
) -> np.ndarray:
    canvas = np.array(img.convert("RGB"))
    for (x, y), lbl in zip(pts, pt_labels):
        color = label_color[lbl]
        cv2.circle(canvas, (int(x), int(y)), MARKER_RADIUS, color, thickness=-1)
        cv2.circle(canvas, (int(x), int(y)), MARKER_RADIUS, (255, 255, 255), thickness=2)
        if show_labels:
            cv2.putText(
                canvas,
                lbl,
                (int(x) + MARKER_RADIUS + 4, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )
    return canvas


ref_canvas = draw_points(ref_img, [(kp["x"], kp["y"]) for kp in KEYPOINTS], labels)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ref_canvas)
legend_handles = [
    mpatches.Patch(color=[c / 255 for c in label_color[lbl]], label=lbl) for lbl in labels
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)
ax.set_title("Reference image — registered keypoints")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% Run KeypointHead.find() on all query images
query_results: list[dict] = []
for tar_path in TAR_IMAGE_PATH:
    tar_img = Image.open(tar_path).convert("RGB")
    log.info("Target image: %s  (%dx%d px)", Path(tar_path).name, *tar_img.size)
    for name, transform in QUERIES:
        q_img = transform(tar_img)
        matches = head.find(q_img, labels=labels, debias=DEBIAS)
        query_results.append({"name": name, "path": tar_path, "image": q_img, "matches": matches})
        for m in matches:
            log.info(
                "  [%s] %s → (%d, %d)  sim=%.3f",
                name,
                m["label"],
                m["point"][0],
                m["point"][1],
                m["similarity"],
            )

log.info("Done — %d queries × %d keypoints", len(query_results), len(labels))

# %% Grid visualisation: reference + one column per transform per target
NCOLS = len(QUERIES)
NROWS = len(TAR_IMAGE_PATH) + 1
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 5, NROWS * 5))
if NROWS == 1:
    axes = axes[np.newaxis, :]

axes[0, 0].imshow(ref_canvas)
axes[0, 0].set_title("Reference\n(registered)", fontsize=10)
axes[0, 0].axis("off")
for col in range(1, NCOLS):
    axes[0, col].axis("off")

for row, tar_path in enumerate(TAR_IMAGE_PATH, start=1):
    for col, qr in enumerate(query_results[(row - 1) * len(QUERIES) : row * len(QUERIES)]):
        q_pts = [m["point"] for m in qr["matches"]]
        q_labels = [m["label"] for m in qr["matches"]]
        canvas = draw_points(qr["image"], q_pts, q_labels, show_labels=False)
        axes[row, col].imshow(canvas)
        axes[row, col].set_title(f"{qr['name']} (target {row})", fontsize=10)
        axes[row, col].axis("off")

legend_handles = [
    mpatches.Patch(color=[c / 255 for c in label_color[lbl]], label=lbl) for lbl in labels
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(labels),
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.04),
)
plt.suptitle(
    f"KeypointHead  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  |  "
    f"img_size={IMG_SIZE}  |  debias={DEBIAS}",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.show()

# %% Similarity heatmaps for each keypoint on a chosen query
QUERY_IDX = 0  # index into query_results; adjust to inspect different queries
_DEBIAS_HEAT = False

qr = query_results[QUERY_IDX]
q_img = qr["image"]
q_w, q_h = q_img.size

out = encoder([q_img], layers=[head.block_idx], debias=_DEBIAS_HEAT)
patches = out.patches[0, 0]  # (H, W, D)
H, W, D = patches.shape
flat = F.normalize(patches.reshape(H * W, D), p=2, dim=1)

display_q = np.array(q_img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

n_kp = len(labels)
fig, axes = plt.subplots(2, n_kp, figsize=(n_kp * 4, 8))
if n_kp == 1:
    axes = axes.reshape(2, 1)

for i, lbl in enumerate(labels):
    ref_emb = head._get_label_emb(lbl)  # (D,)
    if ref_emb is None:
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        continue

    sims = (flat @ ref_emb.to(flat.device)).reshape(H, W).cpu().numpy()
    heat_pil = Image.fromarray((sims * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )
    heat_np = np.array(heat_pil)
    heat_color = plt.get_cmap("jet")(heat_np / 255.0)[..., :3]
    blended = np.clip(0.45 * display_q / 255.0 + 0.55 * heat_color, 0, 1)

    im = axes[0, i].imshow(sims, cmap="jet", vmin=sims.min(), vmax=1.0, aspect="auto")
    axes[0, i].set_title(f"{lbl}\n(patch sim map)", fontsize=9)
    axes[0, i].axis("off")
    plt.colorbar(im, ax=axes[0, i], shrink=0.75, pad=0.02)

    axes[1, i].imshow(blended)
    m = next(m for m in qr["matches"] if m["label"] == lbl)
    px_scaled = int(m["point"][0] * IMG_SIZE / q_w)
    py_scaled = int(m["point"][1] * IMG_SIZE / q_h)
    color_f = [c / 255 for c in label_color[lbl]]
    axes[1, i].add_patch(plt.Circle((px_scaled, py_scaled), MARKER_RADIUS, color=color_f, zorder=5))
    axes[1, i].set_title(f"sim={m['similarity']:.3f}", fontsize=9)
    axes[1, i].axis("off")

plt.suptitle(
    f'Similarity heatmaps — query: "{qr["name"]}"  |  '
    f"block={head.block_idx}  debias={_DEBIAS_HEAT}",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Homography estimation (RANSAC) and warp
HOMO_QUERY_IDX = min(len(QUERIES), len(query_results) - 1)

qr = query_results[HOMO_QUERY_IDX]
q_img = qr["image"]
q_w, q_h = q_img.size

match_map = {m["label"]: m["point"] for m in qr["matches"]}
valid_kps = [kp for kp in KEYPOINTS if kp["label"] in match_map]
src_pts = np.array([[kp["x"], kp["y"]] for kp in valid_kps], dtype=np.float32)
dst_pts = np.array([match_map[kp["label"]] for kp in valid_kps], dtype=np.float32)
valid_labels = [kp["label"] for kp in valid_kps]

if len(src_pts) < 4:
    log.warning("Need at least 4 correspondences for homography; got %d.", len(src_pts))
    H_mat, inlier_mask = None, None
else:
    H_mat, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=8.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    log.info(
        "Query: %r | correspondences=%d | RANSAC inliers=%d",
        qr["name"],
        len(src_pts),
        n_inliers,
    )
    log.info("Homography matrix:\n%s", np.array2string(H_mat, precision=4, suppress_small=True))

# %% Warp reference into query frame
if H_mat is not None:
    ref_np = np.array(ref_img)
    q_np = np.array(q_img)
    h_out, w_out = q_np.shape[:2]

    warped = cv2.warpPerspective(ref_np, H_mat, (w_out, h_out))

    WARP_ALPHA = 0.45
    blend_mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    overlay = (
        (
            q_np.astype(np.float32) * (1 - WARP_ALPHA * blend_mask)
            + warped.astype(np.float32) * WARP_ALPHA * blend_mask
        )
        .clip(0, 255)
        .astype(np.uint8)
    )

    canvas_w = orig_w + q_w
    canvas_h = max(orig_h, q_h)
    corresp = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    corresp[:orig_h, :orig_w] = ref_np
    corresp[:q_h, orig_w:] = q_np

    for i, (sp, dp, lbl) in enumerate(zip(src_pts, dst_pts, valid_labels)):
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else True
        color = label_color[lbl] if is_inlier else (120, 120, 120)
        pt1 = (int(sp[0]), int(sp[1]))
        pt2 = (int(dp[0]) + orig_w, int(dp[1]))
        cv2.line(corresp, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(corresp, pt1, MARKER_RADIUS, color, -1)
        cv2.circle(corresp, pt2, MARKER_RADIUS, color, -1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(corresp)
    axes[0].set_title(
        f'Correspondences  (ref → "{qr["name"]}")\n'
        f"inliers {n_inliers}/{len(src_pts)}  (grey = outlier)",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(warped)
    axes[1].set_title("Reference warped into query frame", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay  (warped α={WARP_ALPHA:.0%})", fontsize=10)
    axes[2].axis("off")

    legend_handles = [
        mpatches.Patch(color=[c / 255 for c in label_color[lbl]], label=lbl) for lbl in valid_labels
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(valid_labels),
        fontsize=9,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.suptitle(
        f"Homography  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  |  debias={DEBIAS}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
else:
    log.warning("Skipping warp visualisation (insufficient correspondences).")
