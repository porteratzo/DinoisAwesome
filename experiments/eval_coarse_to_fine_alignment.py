# %% [markdown]
# # Coarse-to-Fine Image Alignment via DINOv3 + ECC Refinement
#
# Workflow:
#   Phase 1 — Synthetic view generation + automated Master Keypoint discovery (MNN consensus)
#   Phase 2 — DINOv3 feature matching + coarse homography (USAC_MAGSAC / RANSAC)
#   Phase 3 — ECC residual refinement (cv2.findTransformECC)
#   Phase 4 — Evaluation metrics + visualisation
#
# PLACEHOLDERS — replace before running on real data:
#   _load_backbone()       : swap the mock DinoEncoder config for your real weights
#   KeypointMatcherHead    : replace the soft-argmax stub with your trained head

# %% Logging — must be before torch import
from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
_log = logging.getLogger("eval_coarse_to_fine")

# %% Imports
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinoisawesome.encoder import DinoEncoder, ExtractorOutput
from dinoisawesome.keypoint_localization import localize_keypoint, rescale_coords_to_image

# %%  ── Configuration ─────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent


@dataclass
class PipelineConfig:
    # ── Backbone ──────────────────────────────────────────────────────────────
    dino_version: str = "v3"
    dino_size: str = "base"
    img_size: int = 512  # must be divisible by patch_size (16 for v3)
    dino_weights_dir: str | None = field(default_factory=lambda: os.environ.get("DINO_WEIGHTS_DIR"))
    debias: bool = True

    # ── Phase 1: keypoint discovery ───────────────────────────────────────────
    num_views: int = 10  # number of synthetic views
    mnn_threshold: float = 0.85  # min fraction of views with cycle-consistency
    # Patches whose mean-direction cosine sim to the I_A centroid exceeds this
    # are treated as background/low-texture and discarded.
    self_sim_threshold: float = 0.85
    cycle_dist_px: float = 8.0  # max pixel distance for cycle-consistency check

    # ── Phase 2: homography ───────────────────────────────────────────────────
    matcher_sigma: float = 7.0  # Gaussian suppression radius (patch units)
    matcher_beta: float = 0.04  # softmax temperature for soft-argmax
    min_match_sim: float = 0.2  # reject matches below this cosine similarity
    ransac_reproj_thresh: float = 4.0

    # ── Phase 3: ECC refinement ───────────────────────────────────────────────
    ecc_warp_mode: int = cv2.MOTION_HOMOGRAPHY
    ecc_max_iters: int = 200
    ecc_eps: float = 1e-5
    ecc_gauss_filt_size: int = 5  # Gaussian blur before ECC (0 = skip)

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir: Path = _REPO_ROOT / "outputs" / "coarse_to_fine"
    save_figures: bool = True
    show_figures: bool = False


# %%  ── Backbone loader (PLACEHOLDER) ────────────────────────────────────────


def _load_backbone(cfg: PipelineConfig) -> DinoEncoder:
    """Load the DINOv3 backbone.

    Extend this function to load from a custom checkpoint when the default
    torch-hub weights differ from what your experiments used.
    """
    encoder = DinoEncoder(
        version=cfg.dino_version,  # type: ignore[arg-type]
        size=cfg.dino_size,  # type: ignore[arg-type]
        img_size=cfg.img_size,
        layers=1,
        weights_dir=cfg.dino_weights_dir,
    )
    _log.info(
        "Backbone: DINOv%s-%s | patch_size=%d | grid=%dx%d | device=%s",
        cfg.dino_version[1],
        cfg.dino_size,
        encoder.patch_size,
        encoder.grid_h,
        encoder.grid_w,
        encoder.device,
    )
    return encoder


# %%  ── KeypointMatcherHead stub (PLACEHOLDER) ───────────────────────────────


class KeypointMatcherHead:
    """Subpixel keypoint localizer for a single reference–target pair.

    PLACEHOLDER — replace the body of ``match()`` with your trained inference head.
    The current implementation is a pure-cosine-similarity baseline using
    Gaussian-suppressed soft-argmax from ``dinoisawesome.keypoint_localization``.

    Args:
        sigma: Gaussian suppression radius in patch units.
        beta:  Softmax temperature (smaller = sharper peak).
    """

    def __init__(self, sigma: float = 7.0, beta: float = 0.04) -> None:
        self.sigma = sigma
        self.beta = beta

    def match(
        self,
        ref_token: torch.Tensor,  # (D,)  L2-normalised reference embedding
        target_features: torch.Tensor,  # (H_g, W_g, D)  L2-normalised target map
    ) -> tuple[torch.Tensor, float]:
        """Localise ``ref_token`` in ``target_features`` with subpixel precision.

        Returns:
            coords: ``(2,)`` tensor — ``(x, y)`` in **patch-grid units**.
            sim:    Peak cosine similarity as a scalar float.
        """
        # ── REPLACE THIS BLOCK with your trained head's forward pass ──────────
        H_g, W_g, D = target_features.shape
        flat = F.normalize(target_features.reshape(H_g * W_g, D), p=2, dim=1)
        heatmap = (flat @ ref_token).reshape(H_g, W_g)  # (H_g, W_g)
        coords = localize_keypoint(heatmap, sigma=self.sigma, beta=self.beta)  # (2,)
        return coords, float(heatmap.max())
        # ── END PLACEHOLDER ───────────────────────────────────────────────────


# %%  ── Shared feature utility ─────────────────────────────────────────────────


def _extract_features(
    img_rgb: np.ndarray,
    encoder: DinoEncoder,
    cfg: PipelineConfig,
) -> torch.Tensor:
    """Run one encoder forward pass; return ``(H_g, W_g, D)`` L2-normalised patches."""
    out: ExtractorOutput = encoder([Image.fromarray(img_rgb)], debias=cfg.debias)
    return F.normalize(out.patches[0].float(), p=2, dim=-1)


def _sample_feature_map(
    feat: torch.Tensor,  # (H_g, W_g, D)
    px_coords: np.ndarray,  # (N, 2) float32  [x, y] in image-pixel space
    patch_size: int,
) -> torch.Tensor:
    """Bilinearly sample ``feat`` at arbitrary image-pixel coordinates.

    Returns ``(N, D)`` L2-normalised features.
    """
    H_g, W_g, D = feat.shape
    # Convert pixel coords → patch-grid indices → normalised [-1, 1] for grid_sample
    patch_x = px_coords[:, 0] / patch_size - 0.5
    patch_y = px_coords[:, 1] / patch_size - 0.5
    norm_x = torch.from_numpy((2.0 * patch_x / (W_g - 1) - 1.0).astype(np.float32))
    norm_y = torch.from_numpy((2.0 * patch_y / (H_g - 1) - 1.0).astype(np.float32))

    grid = torch.stack([norm_x, norm_y], dim=-1).to(feat.device)  # (N, 2)
    # grid_sample expects (B, C, H, W) input and (B, 1, N, 2) grid
    feat_bchw = feat.permute(2, 0, 1).unsqueeze(0).float()  # (1, D, H_g, W_g)
    sampled = F.grid_sample(
        feat_bchw,
        grid.unsqueeze(0).unsqueeze(0),  # (1, 1, N, 2)
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )  # (1, D, 1, N)
    return F.normalize(sampled.squeeze(0).squeeze(1).T, p=2, dim=1)  # (N, D)


# %%  ── Helpers ────────────────────────────────────────────────────────────────


def _build_affine_matrix(
    angle_deg: float,
    scale: float,
    shear_deg: float,
    tx: float,
    ty: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Return a 3x3 forward affine homography (source-pixel → destination-pixel)."""
    theta = np.deg2rad(angle_deg)
    shear = np.deg2rad(shear_deg)
    T_c = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_c_inv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    S = np.diag([scale, scale, 1.0])
    Sh = np.array([[1, np.tan(shear), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    return (T @ T_c_inv @ Sh @ S @ R @ T_c).astype(np.float32)


def _apply_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness, contrast, and optional Gaussian blur augmentation."""
    alpha = float(rng.uniform(0.7, 1.3))
    beta = float(rng.uniform(-30.0, 30.0))
    out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() > 0.5:
        ksize = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    return out


class _ViewInfo(NamedTuple):
    M_fwd: np.ndarray  # (3, 3) forward affine matrix (I_A px → view px)
    features: torch.Tensor  # (H_g, W_g, D)


# %%  ── Phase 1: Master Keypoint Discovery ────────────────────────────────────


def discover_master_keypoints(
    image_a: np.ndarray,
    encoder: DinoEncoder,
    cfg: PipelineConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Discover robust Master Keypoints on ``image_a`` via synthetic-view MNN consensus.

    For each synthetic view generated from I_A:
      1. Map every patch centre in I_A forward via the affine matrix M_k.
      2. Bilinearly sample the view's feature at that warped location.
      3. Find the nearest-neighbour back in I_A's feature map.
      4. A patch is cycle-consistent for view k if the NN lands within
         ``cfg.cycle_dist_px`` of the original patch centre.

    Patches with ``mnn_score > cfg.mnn_threshold`` and low global self-similarity
    (i.e. distinctive, not background) are kept as Master Keypoints.

    Args:
        image_a: Source image ``(H, W, 3)`` uint8 BGR.
        encoder: Loaded DINOv3 backbone.
        cfg:     Pipeline configuration.
        rng:     Optional random state.

    Returns:
        keypoint_coords: ``(K, 2)`` float32 ``[x, y]`` pixel coords in I_A.
        mnn_scores:      ``(K,)`` float32 per-keypoint consensus fraction.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    img_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    H_img, W_img = img_rgb.shape[:2]
    cx, cy = W_img / 2.0, H_img / 2.0

    _log.info("Phase 1 | Extracting I_A features (%dx%d)", W_img, H_img)
    feat_a = _extract_features(img_rgb, encoder, cfg)  # (H_g, W_g, D)
    H_g, W_g, D = feat_a.shape
    N = H_g * W_g

    flat_a = F.normalize(feat_a.reshape(N, D), p=2, dim=1)  # (N, D)

    # Patch-centre pixel coordinates for every patch in I_A
    cols = (np.arange(W_g) + 0.5) * encoder.patch_size
    rows = (np.arange(H_g) + 0.5) * encoder.patch_size
    grid_x, grid_y = np.meshgrid(cols, rows)
    patch_px = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)  # (N, 2)

    # ── Generate synthetic views ──────────────────────────────────────────────
    _log.info("Phase 1 | Generating %d synthetic views", cfg.num_views)
    views: list[_ViewInfo] = []
    for k in range(cfg.num_views):
        M_fwd = _build_affine_matrix(
            angle_deg=float(rng.uniform(-25, 25)),
            scale=float(rng.uniform(0.8, 1.2)),
            shear_deg=float(rng.uniform(-8, 8)),
            tx=float(rng.uniform(-0.1 * W_img, 0.1 * W_img)),
            ty=float(rng.uniform(-0.1 * H_img, 0.1 * H_img)),
            cx=cx,
            cy=cy,
        )
        view_bgr = cv2.warpPerspective(image_a, M_fwd, (W_img, H_img))
        view_rgb = _apply_photometric(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB), rng)
        feat_k = _extract_features(view_rgb, encoder, cfg)
        views.append(_ViewInfo(M_fwd=M_fwd, features=feat_k))
        _log.debug("  View %02d synthesised", k)

    # ── MNN cycle-consistency ─────────────────────────────────────────────────
    _log.info("Phase 1 | Computing MNN scores across %d views", cfg.num_views)
    cycle_hits = np.zeros(N, dtype=np.int32)

    # Homogeneous patch centres for batch matrix-multiply
    px_hom = np.concatenate([patch_px, np.ones((N, 1), dtype=np.float32)], axis=1).T  # (3, N)

    for view in views:
        # Warp patch centres into the synthetic view's pixel space
        warped_hom = view.M_fwd.astype(np.float64) @ px_hom.astype(np.float64)  # (3, N)
        warped_px = (warped_hom[:2] / warped_hom[2:3]).T.astype(np.float32)  # (N, 2)

        valid = (
            (warped_px[:, 0] >= 0)
            & (warped_px[:, 0] < W_img)
            & (warped_px[:, 1] >= 0)
            & (warped_px[:, 1] < H_img)
        )

        # Bilinearly sample view features at warped patch centres
        feat_at_warped = _sample_feature_map(view.features, warped_px, encoder.patch_size)  # (N, D)

        # NN forward: sampled view feature → best match back in I_A
        nn_in_a = (feat_at_warped @ flat_a.T).argmax(dim=1).cpu().numpy()  # (N,)

        nn_col = nn_in_a % W_g
        nn_row = nn_in_a // W_g
        nn_px_x = (nn_col + 0.5) * encoder.patch_size
        nn_px_y = (nn_row + 0.5) * encoder.patch_size
        nn_px = np.stack([nn_px_x, nn_px_y], axis=1).astype(np.float32)  # (N, 2)

        dist = np.linalg.norm(patch_px - nn_px, axis=1)
        cycle_hits += (valid & (dist < cfg.cycle_dist_px)).astype(np.int32)

    mnn_score_all = cycle_hits.astype(np.float32) / cfg.num_views  # (N,)

    # ── Self-similarity / background filter ──────────────────────────────────
    # A patch whose feature aligns closely with the I_A centroid is generic/
    # background and offers poor discriminability for matching.
    centroid = flat_a.mean(dim=0)  # (D,) unnormalised
    self_sim = (flat_a @ centroid).cpu().numpy()  # (N,)

    # ── Threshold and return ──────────────────────────────────────────────────
    keep = (mnn_score_all > cfg.mnn_threshold) & (self_sim < cfg.self_sim_threshold)
    kept_idx = np.where(keep)[0]

    _log.info(
        "Phase 1 | Kept %d / %d patches  (mnn_score>%.2f, self_sim<%.2f)",
        len(kept_idx),
        N,
        cfg.mnn_threshold,
        cfg.self_sim_threshold,
    )

    if len(kept_idx) == 0:
        _log.warning(
            "No master keypoints survived both filters — falling back to top %d by mnn_score",
            max(20, N // 10),
        )
        kept_idx = np.argsort(mnn_score_all)[-max(20, N // 10) :]

    return patch_px[kept_idx].astype(np.float32), mnn_score_all[kept_idx]


# %%  ── Phase 2: Coarse Alignment ─────────────────────────────────────────────


@dataclass
class AlignmentResult:
    H_coarse: np.ndarray  # (3, 3) coarse homography  (I_B → I_A frame)
    image_b_coarse: np.ndarray  # warped I_B (BGR)
    src_pts: np.ndarray  # (M, 2) source keypoint pixel coords in I_A
    dst_pts: np.ndarray  # (M, 2) matched pixel coords in I_B
    inlier_mask: np.ndarray  # (M,) bool RANSAC inlier mask
    match_sims: np.ndarray  # (M,) cosine similarities


def _mnn_filter(
    src_px: np.ndarray,  # (M, 2) pixel coords in I_A
    dst_px: np.ndarray,  # (M, 2) pixel coords in I_B
    flat_a: torch.Tensor,  # (N, D)
    flat_b: torch.Tensor,  # (N, D)
    W_g: int,
    patch_size: int,
) -> np.ndarray:
    """Boolean mask keeping only mutual-nearest-neighbour correspondences."""
    N = flat_a.shape[0]
    H_g = N // W_g

    def _to_idx(px: np.ndarray) -> np.ndarray:
        col = np.clip((px[:, 0] / patch_size - 0.5).round().astype(int), 0, W_g - 1)
        row = np.clip((px[:, 1] / patch_size - 0.5).round().astype(int), 0, H_g - 1)
        return row * W_g + col

    src_idx = _to_idx(src_px)
    dst_idx = _to_idx(dst_px)

    # For each dst match, check its NN in I_A equals the source patch
    nn_of_dst_in_a = (flat_b[dst_idx] @ flat_a.T).argmax(dim=1).cpu().numpy()
    # For each src match, check its NN in I_B equals the dst patch
    nn_of_src_in_b = (flat_a[src_idx] @ flat_b.T).argmax(dim=1).cpu().numpy()

    return (nn_of_dst_in_a == src_idx) & (nn_of_src_in_b == dst_idx)


def align_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    master_kp_px: np.ndarray,
    encoder: DinoEncoder,
    matcher: KeypointMatcherHead,
    cfg: PipelineConfig,
) -> AlignmentResult:
    """Phase 2: match master keypoints into I_B and estimate a coarse homography.

    For each master keypoint:
      1. Retrieve its L2-normalised DINOv3 token from I_A's feature map.
      2. Call ``matcher.match()`` to obtain a subpixel coordinate in I_B.
      3. Apply an MNN constraint at patch level to prune outlier matches.
      4. Estimate H_coarse via cv2.findHomography (USAC_MAGSAC or RANSAC).
      5. Warp I_B with H_coarse to produce a coarsely aligned image.

    Args:
        image_a:      Source image BGR uint8.
        image_b:      Target image BGR uint8.
        master_kp_px: ``(K, 2)`` master keypoint pixel coords in I_A.
        encoder:      Loaded backbone.
        matcher:      KeypointMatcherHead instance.
        cfg:          Pipeline configuration.

    Returns:
        AlignmentResult with H_coarse and the warped I_B.
    """
    H_img, W_img = image_a.shape[:2]
    K = len(master_kp_px)
    _log.info("Phase 2 | Matching %d master keypoints into I_B", K)

    feat_a = _extract_features(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB), encoder, cfg)
    feat_b = _extract_features(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB), encoder, cfg)
    H_g, W_g, D = feat_a.shape

    feat_b_norm = F.normalize(feat_b.reshape(H_g * W_g, D), p=2, dim=1).reshape(H_g, W_g, D)

    # Reference token for each master keypoint (bilinear sample from I_A feat map)
    ref_tokens = _sample_feature_map(feat_a, master_kp_px, encoder.patch_size)  # (K, D)

    src_list: list[np.ndarray] = []
    dst_patch_list: list[torch.Tensor] = []
    sim_list: list[float] = []

    for i in range(K):
        coords_patch, sim = matcher.match(ref_tokens[i], feat_b_norm)
        if sim >= cfg.min_match_sim:
            src_list.append(master_kp_px[i])
            dst_patch_list.append(coords_patch)
            sim_list.append(sim)

    if len(src_list) < 4:
        raise RuntimeError(
            f"Only {len(src_list)} matches above sim_threshold={cfg.min_match_sim} "
            "— cannot estimate homography."
        )

    src_np = np.array(src_list, dtype=np.float32)
    dst_patch = torch.stack(dst_patch_list, dim=0)  # (M, 2) patch-grid units
    dst_px = (
        rescale_coords_to_image(dst_patch, (H_g, W_g), (H_img, W_img))
        .cpu()
        .numpy()
        .astype(np.float32)
    )  # (M, 2) pixel units

    # ── MNN constraint ────────────────────────────────────────────────────────
    flat_a = F.normalize(feat_a.reshape(H_g * W_g, D), p=2, dim=1)
    flat_b = F.normalize(feat_b.reshape(H_g * W_g, D), p=2, dim=1)
    mnn_mask = _mnn_filter(src_np, dst_px, flat_a, flat_b, W_g, encoder.patch_size)
    _log.info(
        "Phase 2 | MNN filter retained %d / %d correspondences",
        int(mnn_mask.sum()),
        len(src_np),
    )
    src_np = src_np[mnn_mask]
    dst_px = dst_px[mnn_mask]
    sims_np = np.array(sim_list, dtype=np.float32)[mnn_mask]

    if len(src_np) < 4:
        raise RuntimeError("Too few MNN-verified matches to estimate homography.")

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H_coarse, inlier_mask = cv2.findHomography(src_np, dst_px, method, cfg.ransac_reproj_thresh)
    inlier_mask = inlier_mask.ravel().astype(bool)
    _log.info(
        "Phase 2 | Homography inliers: %d / %d  (method=%s)",
        int(inlier_mask.sum()),
        len(src_np),
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


# %%  ── Phase 3: ECC Residual Refinement ──────────────────────────────────────


def refine_alignment(
    image_a: np.ndarray,
    image_b_coarse: np.ndarray,
    H_coarse: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3: estimate a residual warp via ECC and compose with H_coarse.

    ``cv2.findTransformECC`` is initialised at the identity so it searches
    only for a small residual correction on top of the coarse alignment.
    The final transform is H_final = H_fine @ H_coarse.

    Args:
        image_a:        Source image BGR uint8.
        image_b_coarse: Coarsely aligned target BGR uint8 (output of Phase 2).
        H_coarse:       3x3 coarse homography from Phase 2.
        cfg:            Pipeline configuration.

    Returns:
        H_fine:  (3, 3) residual homography estimated by ECC.
        H_final: (3, 3) composed total homography.
    """
    _log.info("Phase 3 | Running ECC refinement")

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b_coarse, cv2.COLOR_BGR2GRAY)

    if cfg.ecc_gauss_filt_size > 0:
        ks = cfg.ecc_gauss_filt_size | 1  # ensure odd
        gray_a = cv2.GaussianBlur(gray_a, (ks, ks), 0)
        gray_b = cv2.GaussianBlur(gray_b, (ks, ks), 0)

    if cfg.ecc_warp_mode == cv2.MOTION_HOMOGRAPHY:
        H_fine: np.ndarray = np.eye(3, dtype=np.float32)
    else:
        H_fine = np.eye(2, 3, dtype=np.float32)

    term = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg.ecc_max_iters,
        cfg.ecc_eps,
    )
    try:
        _, H_fine = cv2.findTransformECC(
            gray_a.astype(np.float32),
            gray_b.astype(np.float32),
            H_fine,
            cfg.ecc_warp_mode,
            term,
        )
    except cv2.error as exc:
        _log.warning("ECC did not converge (%s) — using identity residual", exc)
        H_fine = np.eye(3, dtype=np.float32)

    if cfg.ecc_warp_mode != cv2.MOTION_HOMOGRAPHY:
        tmp = np.eye(3, dtype=np.float32)
        tmp[:2, :] = H_fine
        H_fine = tmp

    residual_norm = float(np.linalg.norm(H_fine - np.eye(3, dtype=np.float32)))
    _log.info("Phase 3 | ECC residual |H_fine - I| = %.4f", residual_norm)

    H_final = H_fine @ H_coarse
    return H_fine, H_final


# %%  ── Phase 4: Evaluation & Visualisation ────────────────────────────────────


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64) - a.mean()
    bf = b.astype(np.float64) - b.mean()
    return float(np.sum(af * bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _alpha_overlay(base: np.ndarray, warp: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask = (warp.sum(axis=2) > 0)[..., None].astype(np.float32)
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha * mask) + warp.astype(np.float32) * alpha * mask,
        0,
        255,
    ).astype(np.uint8)


def _savefig(fig: plt.Figure, path: Path, cfg: PipelineConfig) -> None:
    if cfg.save_figures:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        _log.info("Saved → %s", path)
    if cfg.show_figures:
        plt.show()
    plt.close(fig)


def evaluate_and_visualize(
    image_a: np.ndarray,
    image_b: np.ndarray,
    image_b_coarse: np.ndarray,
    H_final: np.ndarray,
    master_kp_px: np.ndarray,
    mnn_scores: np.ndarray,
    alignment: AlignmentResult,
    cfg: PipelineConfig,
) -> None:
    """Phase 4: log quality metrics and write three diagnostic figures.

    Outputs written to ``cfg.out_dir``:

    1. ``keypoints.png``  — Master keypoints on I_A, colour-coded by MNN score.
    2. ``alignment.png``  — Coarse vs. final (ECC-refined) overlay comparison.
    3. ``metrics.png``    — MSE and NCC bar chart, coarse vs. final.

    Metrics are computed on the valid (non-black-border) overlap region only.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    H_img, W_img = image_a.shape[:2]

    image_b_final = cv2.warpPerspective(image_b, H_final, (W_img, H_img))

    # ── Overlap-aware quality metrics ─────────────────────────────────────────
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c = cv2.cvtColor(image_b_coarse, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_f = cv2.cvtColor(image_b_final, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ov_c = (gray_a > 0) & (gray_c > 0)
    ov_f = (gray_a > 0) & (gray_f > 0)

    mse_c = _mse(gray_a[ov_c], gray_c[ov_c])
    mse_f = _mse(gray_a[ov_f], gray_f[ov_f])
    ncc_c = _ncc(gray_a[ov_c], gray_c[ov_c])
    ncc_f = _ncc(gray_a[ov_f], gray_f[ov_f])

    _log.info(
        "Metrics | coarse  MSE=%.2f  NCC=%.4f | final  MSE=%.2f  NCC=%.4f",
        mse_c,
        ncc_c,
        mse_f,
        ncc_f,
    )

    # ── Figure 1: Master Keypoints on I_A ────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 9))
    ax1.imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
    sc = ax1.scatter(
        master_kp_px[:, 0],
        master_kp_px[:, 1],
        c=mnn_scores,
        cmap="plasma",
        vmin=cfg.mnn_threshold,
        vmax=1.0,
        s=18,
        linewidths=0.3,
        edgecolors="white",
        zorder=5,
    )
    plt.colorbar(sc, ax=ax1, label="MNN consensus score", fraction=0.03, pad=0.02)
    ax1.set_title(
        f"Master Keypoints on $I_A$  (N={len(master_kp_px)}, threshold={cfg.mnn_threshold:.2f})",
        fontsize=12,
    )
    ax1.axis("off")
    plt.tight_layout()
    _savefig(fig1, cfg.out_dir / "keypoints.png", cfg)

    # ── Figure 2: Alignment overlay before / after ECC ───────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    axes2[0].imshow(cv2.cvtColor(_alpha_overlay(image_a, image_b_coarse), cv2.COLOR_BGR2RGB))
    axes2[0].set_title(
        f"Coarse alignment ($H_{{\\rm coarse}}$)\nMSE={mse_c:.1f}  NCC={ncc_c:.4f}",
        fontsize=11,
    )
    axes2[0].axis("off")
    axes2[1].imshow(cv2.cvtColor(_alpha_overlay(image_a, image_b_final), cv2.COLOR_BGR2RGB))
    axes2[1].set_title(
        f"After ECC refinement ($H_{{\\rm final}}$)\nMSE={mse_f:.1f}  NCC={ncc_f:.4f}",
        fontsize=11,
    )
    axes2[1].axis("off")
    plt.suptitle(
        f"$I_B$ warped onto $I_A$  |  "
        f"inliers {alignment.inlier_mask.sum()}/{len(alignment.inlier_mask)}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    _savefig(fig2, cfg.out_dir / "alignment.png", cfg)

    # ── Figure 3: Metrics bar chart ───────────────────────────────────────────
    fig3, (ax_mse, ax_ncc) = plt.subplots(1, 2, figsize=(8, 4))
    labels_bar = ["Coarse", "Final (ECC)"]
    colors = ["#5b9bd5", "#70ad47"]

    ax_mse.bar(labels_bar, [mse_c, mse_f], color=colors)
    ax_mse.set_ylabel("MSE (lower = better)")
    ax_mse.set_title("Mean Squared Error")
    for rect, v in zip(ax_mse.patches, [mse_c, mse_f]):
        ax_mse.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() * 1.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_ncc.bar(labels_bar, [ncc_c, ncc_f], color=colors)
    ax_ncc.set_ylabel("NCC (higher = better)")
    ax_ncc.set_title("Normalised Cross-Correlation")
    ax_ncc.set_ylim(min(ncc_c, ncc_f) * 0.95, 1.02)
    for rect, v in zip(ax_ncc.patches, [ncc_c, ncc_f]):
        ax_ncc.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() * 1.001,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle("Alignment quality: coarse vs. ECC-refined", fontsize=12)
    plt.tight_layout()
    _savefig(fig3, cfg.out_dir / "metrics.png", cfg)


# %%  ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    cfg = PipelineConfig()

    # ── Image loading ─────────────────────────────────────────────────────────
    # Defaults to the bundled tiger sample for a self-contained demo.
    # Override via IMAGE_A / IMAGE_B environment variables.
    sample_path = _REPO_ROOT / "data" / "tiger.jpeg"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample image not found at {sample_path}. "
            "Point IMAGE_A / IMAGE_B env vars to your own images."
        )

    image_a_path = Path(os.environ.get("IMAGE_A", str(sample_path)))
    image_b_path = Path(os.environ.get("IMAGE_B", str(sample_path)))

    image_a = cv2.imread(str(image_a_path))
    image_b_raw = cv2.imread(str(image_b_path))
    if image_a is None or image_b_raw is None:
        raise FileNotFoundError("Could not load I_A or I_B — check IMAGE_A / IMAGE_B.")

    image_a = cv2.resize(image_a, (cfg.img_size, cfg.img_size))

    # Synthetic I_B — apply a known perturbation so alignment quality is measurable.
    # Replace this block with a real I_B when running on actual data.
    rng_demo = np.random.default_rng(0)
    M_gt = _build_affine_matrix(
        angle_deg=12.0,
        scale=0.92,
        shear_deg=3.0,
        tx=15.0,
        ty=-10.0,
        cx=cfg.img_size / 2.0,
        cy=cfg.img_size / 2.0,
    )
    image_b = cv2.warpPerspective(image_a, M_gt, (cfg.img_size, cfg.img_size))
    _log.info("Demo: I_B generated from I_A with a known synthetic warp")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    encoder = _load_backbone(cfg)
    matcher = KeypointMatcherHead(sigma=cfg.matcher_sigma, beta=cfg.matcher_beta)

    master_kp_px, mnn_scores = discover_master_keypoints(image_a, encoder, cfg, rng=rng_demo)
    alignment = align_images(image_a, image_b, master_kp_px, encoder, matcher, cfg)
    H_fine, H_final = refine_alignment(image_a, alignment.image_b_coarse, alignment.H_coarse, cfg)
    evaluate_and_visualize(
        image_a,
        image_b,
        alignment.image_b_coarse,
        H_final,
        master_kp_px,
        mnn_scores,
        alignment,
        cfg,
    )
    _log.info("Done. Outputs written to %s", cfg.out_dir)


if __name__ == "__main__":
    main()
