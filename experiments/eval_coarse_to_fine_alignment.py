# %% [markdown]
# # Coarse-to-Fine Image Alignment via DINOv3 + ECC Refinement
#
# Workflow:
#   Exp 1 — Input images
#   Exp 2 — Synthetic view augmentation (what Phase 1 sees)
#   Exp 3 — Master Keypoint Discovery (MNN consensus)
#   Exp 4 — Feature matching + coarse homography (USAC_MAGSAC / RANSAC)
#   Exp 5 — ECC residual refinement
#   Exp 6 — Alignment quality metrics
#
# PLACEHOLDERS — replace before running on real data:
#   KeypointMatcherHead.match() : replace the soft-argmax stub with your trained head

# %% Logging — must be before torch import
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("eval_coarse_to_fine")

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image

from dinoisawesome.encoder import DinoEncoder, ExtractorOutput
from dinoisawesome.keypoint_localization import localize_keypoint, rescale_coords_to_image

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ── Backbone ──────────────────────────────────────────────────────────────────
DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 512  # must be divisible by patch_size (16 for v3)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DEBIAS = True

# ── Phase 1: keypoint discovery ───────────────────────────────────────────────
NUM_VIEWS = 10  # number of synthetic views
MNN_THRESHOLD = 0.85  # min fraction of views with cycle-consistency
SELF_SIM_THRESHOLD = 0.85  # patches above this are generic/background
CYCLE_DIST_PX = 8.0  # max pixel distance for cycle-consistency check

# ── Phase 2: homography ───────────────────────────────────────────────────────
MATCHER_SIGMA = 7.0  # Gaussian suppression radius (patch units)
MATCHER_BETA = 0.04  # softmax temperature for soft-argmax
MIN_MATCH_SIM = 0.2  # reject matches below this cosine similarity
RANSAC_REPROJ_THRESH = 4.0

# ── Phase 3: ECC refinement ───────────────────────────────────────────────────
ECC_WARP_MODE = cv2.MOTION_HOMOGRAPHY
ECC_MAX_ITERS = 200
ECC_EPS = 1e-5
ECC_GAUSS_FILT_SIZE = 5  # Gaussian blur before ECC (0 = skip)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = _REPO_ROOT / "outputs" / "coarse_to_fine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Image paths ───────────────────────────────────────────────────────────────
_SAMPLE_PATH = _REPO_ROOT / "data" / "tiger.jpeg"
IMAGE_A_PATH = Path(os.environ.get("IMAGE_A", str(_SAMPLE_PATH)))
IMAGE_B_PATH = Path(os.environ.get("IMAGE_B", str(_SAMPLE_PATH)))


# %% ── Helper functions ───────────────────────────────────────────────────────


class KeypointMatcherHead:
    """Subpixel keypoint localizer — PLACEHOLDER using cosine-similarity soft-argmax.

    Replace the body of ``match()`` with your trained inference head.
    """

    def __init__(self, sigma: float = 7.0, beta: float = 0.04) -> None:
        self.sigma = sigma
        self.beta = beta

    def match(
        self,
        ref_token: torch.Tensor,      # (D,)  L2-normalised
        target_features: torch.Tensor, # (H_g, W_g, D)  L2-normalised
    ) -> tuple[torch.Tensor, float]:
        H_g, W_g, D = target_features.shape
        flat = F.normalize(target_features.reshape(H_g * W_g, D), p=2, dim=1)
        heatmap = (flat @ ref_token).reshape(H_g, W_g)
        coords = localize_keypoint(heatmap, sigma=self.sigma, beta=self.beta)
        return coords, float(heatmap.max())


def _extract_features(img_rgb: np.ndarray, encoder: DinoEncoder) -> torch.Tensor:
    """Forward pass → (H_g, W_g, D) L2-normalised patch features."""
    out: ExtractorOutput = encoder([Image.fromarray(img_rgb)], debias=DEBIAS)
    return F.normalize(out.patches[0].float(), p=2, dim=-1)


def _sample_feature_map(
    feat: torch.Tensor,      # (H_g, W_g, D)
    px_coords: np.ndarray,   # (N, 2) float32  [x, y] image pixels
    patch_size: int,
) -> torch.Tensor:
    """Bilinearly sample ``feat`` at image-pixel coords → (N, D) L2-normalised."""
    H_g, W_g, D = feat.shape
    patch_x = px_coords[:, 0] / patch_size - 0.5
    patch_y = px_coords[:, 1] / patch_size - 0.5
    norm_x = torch.from_numpy((2.0 * patch_x / (W_g - 1) - 1.0).astype(np.float32))
    norm_y = torch.from_numpy((2.0 * patch_y / (H_g - 1) - 1.0).astype(np.float32))
    grid = torch.stack([norm_x, norm_y], dim=-1).to(feat.device)
    feat_bchw = feat.permute(2, 0, 1).unsqueeze(0).float()
    sampled = F.grid_sample(
        feat_bchw,
        grid.unsqueeze(0).unsqueeze(0),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return F.normalize(sampled.squeeze(0).squeeze(1).T, p=2, dim=1)


def _build_affine_matrix(
    angle_deg: float, scale: float, shear_deg: float,
    tx: float, ty: float, cx: float, cy: float,
) -> np.ndarray:
    """3×3 forward affine homography (source px → destination px)."""
    theta = np.deg2rad(angle_deg)
    shear = np.deg2rad(shear_deg)
    T_c     = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_c_inv = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]], dtype=np.float64)
    R  = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float64)
    S  = np.diag([scale, scale, 1.0])
    Sh = np.array([[1, np.tan(shear), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    T  = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    return (T @ T_c_inv @ Sh @ S @ R @ T_c).astype(np.float32)


def _apply_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness / contrast / blur augmentation."""
    alpha = float(rng.uniform(0.7, 1.3))
    beta  = float(rng.uniform(-30.0, 30.0))
    out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() > 0.5:
        ksize = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    return out


def _alpha_overlay(base: np.ndarray, warp: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask = (warp.sum(axis=2) > 0)[..., None].astype(np.float32)
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha * mask)
        + warp.astype(np.float32) * alpha * mask,
        0, 255,
    ).astype(np.uint8)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64) - a.mean()
    bf = b.astype(np.float64) - b.mean()
    return float(np.sum(af * bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


class _ViewInfo(NamedTuple):
    M_fwd: np.ndarray       # (3, 3) forward affine  (I_A px → view px)
    features: torch.Tensor  # (H_g, W_g, D)
    img_rgb: np.ndarray     # (H, W, 3) uint8 augmented view


@dataclass
class AlignmentResult:
    H_coarse: np.ndarray       # (3, 3) coarse homography  (I_B → I_A frame)
    image_b_coarse: np.ndarray # warped I_B (BGR)
    src_pts: np.ndarray        # (M, 2) master keypoint pixel coords in I_A
    dst_pts: np.ndarray        # (M, 2) matched pixel coords in I_B
    inlier_mask: np.ndarray    # (M,) bool RANSAC inlier mask
    match_sims: np.ndarray     # (M,) cosine similarities


def _generate_views(
    image_a: np.ndarray,
    encoder: DinoEncoder,
    rng: np.random.Generator,
) -> list[_ViewInfo]:
    H_img, W_img = image_a.shape[:2]
    cx, cy = W_img / 2.0, H_img / 2.0
    views: list[_ViewInfo] = []
    for _ in range(NUM_VIEWS):
        M_fwd = _build_affine_matrix(
            angle_deg=float(rng.uniform(-25, 25)),
            scale=float(rng.uniform(0.8, 1.2)),
            shear_deg=float(rng.uniform(-8, 8)),
            tx=float(rng.uniform(-0.1 * W_img, 0.1 * W_img)),
            ty=float(rng.uniform(-0.1 * H_img, 0.1 * H_img)),
            cx=cx, cy=cy,
        )
        view_bgr = cv2.warpPerspective(image_a, M_fwd, (W_img, H_img))
        view_rgb = _apply_photometric(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB), rng)
        feat_k = _extract_features(view_rgb, encoder)
        views.append(_ViewInfo(M_fwd=M_fwd, features=feat_k, img_rgb=view_rgb))
    return views


def discover_master_keypoints(
    image_a: np.ndarray,
    encoder: DinoEncoder,
    views: list[_ViewInfo],
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 1: MNN cycle-consistency across pre-generated synthetic views.

    Returns:
        keypoint_coords: (K, 2) float32 [x, y] pixel coords in I_A.
        mnn_scores:      (K,) float32 per-keypoint consensus fraction.
    """
    img_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    H_img, W_img = img_rgb.shape[:2]

    log.info("Phase 1 | Extracting I_A features (%dx%d)", W_img, H_img)
    feat_a = _extract_features(img_rgb, encoder)
    H_g, W_g, D = feat_a.shape
    N = H_g * W_g
    flat_a = F.normalize(feat_a.reshape(N, D), p=2, dim=1)

    cols = (np.arange(W_g) + 0.5) * encoder.patch_size
    rows = (np.arange(H_g) + 0.5) * encoder.patch_size
    grid_x, grid_y = np.meshgrid(cols, rows)
    patch_px = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    px_hom = np.concatenate([patch_px, np.ones((N, 1), dtype=np.float32)], axis=1).T

    log.info("Phase 1 | Computing MNN scores across %d views", len(views))
    cycle_hits = np.zeros(N, dtype=np.int32)
    for view in views:
        warped_hom = view.M_fwd.astype(np.float64) @ px_hom.astype(np.float64)
        warped_px  = (warped_hom[:2] / warped_hom[2:3]).T.astype(np.float32)
        valid = (
            (warped_px[:, 0] >= 0) & (warped_px[:, 0] < W_img) &
            (warped_px[:, 1] >= 0) & (warped_px[:, 1] < H_img)
        )
        feat_at_warped = _sample_feature_map(view.features, warped_px, encoder.patch_size)
        nn_in_a = (feat_at_warped @ flat_a.T).argmax(dim=1).cpu().numpy()
        nn_px = np.stack([
            (nn_in_a % W_g + 0.5) * encoder.patch_size,
            (nn_in_a // W_g + 0.5) * encoder.patch_size,
        ], axis=1).astype(np.float32)
        dist = np.linalg.norm(patch_px - nn_px, axis=1)
        cycle_hits += (valid & (dist < CYCLE_DIST_PX)).astype(np.int32)

    mnn_score_all = cycle_hits.astype(np.float32) / len(views)
    centroid = flat_a.mean(dim=0)
    self_sim  = (flat_a @ centroid).cpu().numpy()

    keep = (mnn_score_all > MNN_THRESHOLD) & (self_sim < SELF_SIM_THRESHOLD)
    kept_idx = np.where(keep)[0]
    log.info(
        "Phase 1 | Kept %d / %d patches  (mnn>%.2f, self_sim<%.2f)",
        len(kept_idx), N, MNN_THRESHOLD, SELF_SIM_THRESHOLD,
    )
    if len(kept_idx) == 0:
        log.warning("No keypoints survived — falling back to top %d by mnn_score", max(20, N // 10))
        kept_idx = np.argsort(mnn_score_all)[-max(20, N // 10):]

    return patch_px[kept_idx].astype(np.float32), mnn_score_all[kept_idx]


def _mnn_filter(
    src_px: np.ndarray, dst_px: np.ndarray,
    flat_a: torch.Tensor, flat_b: torch.Tensor,
    W_g: int, patch_size: int,
) -> np.ndarray:
    """Boolean mask: keep only mutual-nearest-neighbour correspondences."""
    N = flat_a.shape[0]
    H_g = N // W_g

    def _to_idx(px: np.ndarray) -> np.ndarray:
        col = np.clip((px[:, 0] / patch_size - 0.5).round().astype(int), 0, W_g - 1)
        row = np.clip((px[:, 1] / patch_size - 0.5).round().astype(int), 0, H_g - 1)
        return row * W_g + col

    src_idx = _to_idx(src_px)
    dst_idx = _to_idx(dst_px)
    nn_of_dst_in_a = (flat_b[dst_idx] @ flat_a.T).argmax(dim=1).cpu().numpy()
    nn_of_src_in_b = (flat_a[src_idx] @ flat_b.T).argmax(dim=1).cpu().numpy()
    return (nn_of_dst_in_a == src_idx) & (nn_of_src_in_b == dst_idx)


def align_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    master_kp_px: np.ndarray,
    encoder: DinoEncoder,
    matcher: KeypointMatcherHead,
) -> AlignmentResult:
    """Phase 2: match master keypoints into I_B → coarse homography."""
    H_img, W_img = image_a.shape[:2]
    K = len(master_kp_px)
    log.info("Phase 2 | Matching %d master keypoints into I_B", K)

    feat_a = _extract_features(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB), encoder)
    feat_b = _extract_features(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB), encoder)
    H_g, W_g, D = feat_a.shape
    feat_b_norm = F.normalize(feat_b.reshape(H_g * W_g, D), p=2, dim=1).reshape(H_g, W_g, D)
    ref_tokens = _sample_feature_map(feat_a, master_kp_px, encoder.patch_size)

    src_list: list[np.ndarray] = []
    dst_patch_list: list[torch.Tensor] = []
    sim_list: list[float] = []
    for i in range(K):
        coords_patch, sim = matcher.match(ref_tokens[i], feat_b_norm)
        if sim >= MIN_MATCH_SIM:
            src_list.append(master_kp_px[i])
            dst_patch_list.append(coords_patch)
            sim_list.append(sim)

    if len(src_list) < 4:
        raise RuntimeError(
            f"Only {len(src_list)} matches above sim_threshold={MIN_MATCH_SIM}."
        )

    src_np = np.array(src_list, dtype=np.float32)
    dst_patch = torch.stack(dst_patch_list, dim=0)
    dst_px = (
        rescale_coords_to_image(dst_patch, (H_g, W_g), (H_img, W_img))
        .cpu().numpy().astype(np.float32)
    )

    flat_a = F.normalize(feat_a.reshape(H_g * W_g, D), p=2, dim=1)
    flat_b = F.normalize(feat_b.reshape(H_g * W_g, D), p=2, dim=1)
    mnn_mask = _mnn_filter(src_np, dst_px, flat_a, flat_b, W_g, encoder.patch_size)
    log.info(
        "Phase 2 | MNN filter: %d / %d correspondences retained",
        int(mnn_mask.sum()), len(src_np),
    )
    src_np  = src_np[mnn_mask]
    dst_px  = dst_px[mnn_mask]
    sims_np = np.array(sim_list, dtype=np.float32)[mnn_mask]

    if len(src_np) < 4:
        raise RuntimeError("Too few MNN-verified matches to estimate homography.")

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H_coarse, inlier_mask = cv2.findHomography(src_np, dst_px, method, RANSAC_REPROJ_THRESH)
    inlier_mask = inlier_mask.ravel().astype(bool)
    log.info(
        "Phase 2 | Homography inliers: %d / %d  (method=%s)",
        int(inlier_mask.sum()), len(src_np),
        "USAC_MAGSAC" if method == cv2.USAC_MAGSAC else "RANSAC",
    )

    h_out, w_out = image_b.shape[:2]
    image_b_coarse = cv2.warpPerspective(image_b, H_coarse, (w_out, h_out))
    return AlignmentResult(
        H_coarse=H_coarse,
        image_b_coarse=image_b_coarse,
        src_pts=src_np,
        dst_pts=dst_px,
        inlier_mask=inlier_mask,
        match_sims=sims_np,
    )


def refine_alignment(
    image_a: np.ndarray,
    image_b_coarse: np.ndarray,
    H_coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3: ECC residual refinement → H_final = H_fine @ H_coarse."""
    log.info("Phase 3 | Running ECC refinement")
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b_coarse, cv2.COLOR_BGR2GRAY)
    if ECC_GAUSS_FILT_SIZE > 0:
        ks = ECC_GAUSS_FILT_SIZE | 1
        gray_a = cv2.GaussianBlur(gray_a, (ks, ks), 0)
        gray_b = cv2.GaussianBlur(gray_b, (ks, ks), 0)

    H_fine: np.ndarray = np.eye(3, dtype=np.float32) if ECC_WARP_MODE == cv2.MOTION_HOMOGRAPHY \
        else np.eye(2, 3, dtype=np.float32)
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERS, ECC_EPS)
    try:
        _, H_fine = cv2.findTransformECC(
            gray_a.astype(np.float32), gray_b.astype(np.float32),
            H_fine, ECC_WARP_MODE, term,
        )
    except cv2.error as exc:
        log.warning("ECC did not converge (%s) — using identity residual", exc)
        H_fine = np.eye(3, dtype=np.float32)

    if ECC_WARP_MODE != cv2.MOTION_HOMOGRAPHY:
        tmp = np.eye(3, dtype=np.float32)
        tmp[:2, :] = H_fine
        H_fine = tmp

    residual_norm = float(np.linalg.norm(H_fine - np.eye(3, dtype=np.float32)))
    log.info("Phase 3 | ECC residual |H_fine - I| = %.4f", residual_norm)
    return H_fine, H_fine @ H_coarse


# %% Load backbone and images
if not IMAGE_A_PATH.exists():
    raise FileNotFoundError(
        f"Sample image not found at {IMAGE_A_PATH}. "
        "Set IMAGE_A / IMAGE_B env vars to point at your images."
    )

encoder = DinoEncoder(
    version=DINO_VERSION,   # type: ignore[arg-type]
    size=DINO_SIZE,         # type: ignore[arg-type]
    img_size=IMG_SIZE,
    layers=1,
    weights_dir=DINO_WEIGHTS_DIR,
)
log.info(
    "Backbone: DINOv%s-%s | patch_size=%d | grid=%dx%d | device=%s",
    DINO_VERSION[1], DINO_SIZE,
    encoder.patch_size, encoder.grid_h, encoder.grid_w, encoder.device,
)

matcher = KeypointMatcherHead(sigma=MATCHER_SIGMA, beta=MATCHER_BETA)

image_a = cv2.imread(str(IMAGE_A_PATH))
image_b_raw = cv2.imread(str(IMAGE_B_PATH))
if image_a is None or image_b_raw is None:
    raise FileNotFoundError("Could not load I_A or I_B — check IMAGE_A / IMAGE_B.")
image_a = cv2.resize(image_a, (IMG_SIZE, IMG_SIZE))

# Synthetic I_B with a known ground-truth warp (replace with real I_B for actual data)
rng_demo = np.random.default_rng(0)
M_gt = _build_affine_matrix(
    angle_deg=12.0, scale=0.92, shear_deg=3.0,
    tx=15.0, ty=-10.0,
    cx=IMG_SIZE / 2.0, cy=IMG_SIZE / 2.0,
)
image_b = cv2.warpPerspective(image_a, M_gt, (IMG_SIZE, IMG_SIZE))
log.info("Demo: I_B generated from I_A with a known synthetic warp")

# %% Experiment 1 — Input images
log.info("Experiment 1: Input images …")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"$I_A$  (source, {IMG_SIZE}×{IMG_SIZE})", fontsize=12)
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
axes[1].set_title(
    f"$I_B$  (target — synthetic warp: rot=12°, scale=0.92, shear=3°, tx=15, ty=−10)",
    fontsize=12,
)
axes[1].axis("off")
plt.suptitle("Exp 1: Inputs — I_A and I_B to be aligned", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp1_input_images.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 2 — Synthetic view augmentation (what Phase 1 sees)
log.info("Experiment 2: Synthetic view augmentation …")

views = _generate_views(image_a, encoder, rng_demo)

n_show = min(6, NUM_VIEWS)
fig, axes = plt.subplots(2, n_show // 2, figsize=(16, 7))
axes = axes.ravel()
for i, ax in enumerate(axes):
    ax.imshow(views[i].img_rgb)
    ax.set_title(f"View {i}", fontsize=9)
    ax.axis("off")
plt.suptitle(
    f"Exp 2: Synthetic views generated from I_A  "
    f"(random affine + photometric augmentation, {NUM_VIEWS} total)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp2_synthetic_views.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 3 — Master Keypoint Discovery
log.info("Experiment 3: Master Keypoint Discovery …")

master_kp_px, mnn_scores = discover_master_keypoints(image_a, encoder, views)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: keypoints on I_A coloured by MNN score
axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
sc = axes[0].scatter(
    master_kp_px[:, 0], master_kp_px[:, 1],
    c=mnn_scores, cmap="plasma",
    vmin=MNN_THRESHOLD, vmax=1.0,
    s=18, linewidths=0.3, edgecolors="white", zorder=5,
)
plt.colorbar(sc, ax=axes[0], label="MNN consensus score", fraction=0.03, pad=0.02)
axes[0].set_title(
    f"Master keypoints on $I_A$  (K={len(master_kp_px)}, threshold={MNN_THRESHOLD:.2f})",
    fontsize=11,
)
axes[0].axis("off")

# Right: histogram of MNN scores across all patches
img_rgb_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
feat_a_all = _extract_features(img_rgb_a, encoder)
H_g, W_g, D = feat_a_all.shape
flat_a_all = F.normalize(feat_a_all.reshape(H_g * W_g, D), p=2, dim=1)
# Re-use the cycle_hits we'd need to recompute — approximate with the kept scores distribution
axes[1].hist(mnn_scores, bins=20, color="#5b9bd5", edgecolor="white", linewidth=0.5)
axes[1].axvline(MNN_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                label=f"threshold={MNN_THRESHOLD:.2f}")
axes[1].set_xlabel("MNN consensus score")
axes[1].set_ylabel("count (kept keypoints)")
axes[1].set_title("MNN score distribution (kept patches only)", fontsize=11)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Exp 3: Master Keypoint Discovery  |  {NUM_VIEWS} synthetic views  "
    f"|  DINOv{DINO_VERSION[1]}-{DINO_SIZE}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp3_master_keypoints.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 4 — Feature matching and coarse homography
log.info("Experiment 4: Feature matching + coarse homography …")

alignment = align_images(image_a, image_b, master_kp_px, encoder, matcher)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: correspondence lines between I_A and I_B (inliers green, outliers red)
canvas_w = IMG_SIZE * 2
canvas = np.concatenate(
    [cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB),
     cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)],
    axis=1,
)
axes[0].imshow(canvas)
for i, (src, dst) in enumerate(zip(alignment.src_pts, alignment.dst_pts)):
    color = "#2ecc71" if alignment.inlier_mask[i] else "#e74c3c"
    lw = 0.8 if alignment.inlier_mask[i] else 0.4
    axes[0].plot(
        [src[0], dst[0] + IMG_SIZE], [src[1], dst[1]],
        color=color, linewidth=lw, alpha=0.7,
    )
axes[0].scatter(alignment.src_pts[:, 0], alignment.src_pts[:, 1],
                s=6, c="#2ecc71", zorder=5)
axes[0].scatter(alignment.dst_pts[:, 0] + IMG_SIZE, alignment.dst_pts[:, 1],
                s=6, c="#3498db", zorder=5)
axes[0].set_title(
    f"Correspondences  |  inliers {alignment.inlier_mask.sum()}/{len(alignment.inlier_mask)}\n"
    f"(green=inlier, red=outlier  |  left=I_A, right=I_B)",
    fontsize=10,
)
axes[0].axis("off")

# Right: coarse alignment overlay
axes[1].imshow(cv2.cvtColor(_alpha_overlay(image_a, alignment.image_b_coarse), cv2.COLOR_BGR2RGB))
axes[1].set_title("Coarse alignment overlay  (I_B warped onto I_A)", fontsize=11)
axes[1].axis("off")

plt.suptitle(
    f"Exp 4: Feature matching + coarse homography  "
    f"|  sim≥{MIN_MATCH_SIM}  |  RANSAC reproj={RANSAC_REPROJ_THRESH}px",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp4_matching_and_coarse.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 5 — ECC residual refinement
log.info("Experiment 5: ECC refinement …")

H_fine, H_final = refine_alignment(image_a, alignment.image_b_coarse, alignment.H_coarse)
image_b_final = cv2.warpPerspective(image_b, H_final, (IMG_SIZE, IMG_SIZE))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
axes[0].set_title("$I_A$  (reference)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(_alpha_overlay(image_a, alignment.image_b_coarse), cv2.COLOR_BGR2RGB))
axes[1].set_title("Coarse alignment  ($H_{\\rm coarse}$)", fontsize=11)
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(_alpha_overlay(image_a, image_b_final), cv2.COLOR_BGR2RGB))
axes[2].set_title(
    f"After ECC  ($H_{{\\rm fine}} \\cdot H_{{\\rm coarse}}$)\n"
    f"|H_fine − I| = {float(np.linalg.norm(H_fine - np.eye(3))):.4f}",
    fontsize=11,
)
axes[2].axis("off")

plt.suptitle(
    f"Exp 5: ECC residual refinement  |  warp_mode=HOMOGRAPHY  "
    f"|  max_iters={ECC_MAX_ITERS}  eps={ECC_EPS}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp5_ecc_refinement.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 6 — Alignment quality metrics
log.info("Experiment 6: Alignment quality metrics …")

gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_c = cv2.cvtColor(alignment.image_b_coarse, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_f = cv2.cvtColor(image_b_final, cv2.COLOR_BGR2GRAY).astype(np.float32)
ov_c = (gray_a > 0) & (gray_c > 0)
ov_f = (gray_a > 0) & (gray_f > 0)

mse_c = _mse(gray_a[ov_c], gray_c[ov_c])
mse_f = _mse(gray_a[ov_f], gray_f[ov_f])
ncc_c = _ncc(gray_a[ov_c], gray_c[ov_c])
ncc_f = _ncc(gray_a[ov_f], gray_f[ov_f])
log.info(
    "Metrics | coarse  MSE=%.2f  NCC=%.4f | final  MSE=%.2f  NCC=%.4f",
    mse_c, ncc_c, mse_f, ncc_f,
)

labels_bar = ["Coarse", "Final (ECC)"]
colors = ["#5b9bd5", "#70ad47"]

fig, (ax_mse, ax_ncc) = plt.subplots(1, 2, figsize=(9, 4))

ax_mse.bar(labels_bar, [mse_c, mse_f], color=colors)
ax_mse.set_ylabel("MSE (lower = better)")
ax_mse.set_title("Mean Squared Error")
for rect, v in zip(ax_mse.patches, [mse_c, mse_f]):
    ax_mse.text(
        rect.get_x() + rect.get_width() / 2.0,
        rect.get_height() * 1.01,
        f"{v:.2f}", ha="center", va="bottom", fontsize=10,
    )

ax_ncc.bar(labels_bar, [ncc_c, ncc_f], color=colors)
ax_ncc.set_ylabel("NCC (higher = better)")
ax_ncc.set_title("Normalised Cross-Correlation")
ax_ncc.set_ylim(min(ncc_c, ncc_f) * 0.95, 1.02)
for rect, v in zip(ax_ncc.patches, [ncc_c, ncc_f]):
    ax_ncc.text(
        rect.get_x() + rect.get_width() / 2.0,
        rect.get_height() * 1.001,
        f"{v:.4f}", ha="center", va="bottom", fontsize=10,
    )

plt.suptitle("Exp 6: Alignment quality — coarse vs. ECC-refined", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp6_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

log.info("Done. Outputs written to %s", OUTPUT_DIR)

# %%
