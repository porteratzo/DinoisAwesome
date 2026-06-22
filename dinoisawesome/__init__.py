"""DinoisAwesome: DINO ViT feature extraction and patch-level retrieval gallery."""

from .anomaly_head import AnomalyHead
from .encoder import DinoEncoder, ExtractorOutput
from .foreground_head import ForegroundHead
from .gallery import Gallery, GalleryConfig
from .instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    detect_instances,
    extract_patch_tokens,
    extract_peaks,
    visualize,
)
from .keypoint_head import KeypointHead
from .keypoint_localization import (
    apply_gaussian_suppression,
    localize_keypoint,
    make_coordinate_grid,
    rescale_coords_to_image,
    temperature_softmax,
)

__all__ = [
    "DinoEncoder",
    "ExtractorOutput",
    "Gallery",
    "GalleryConfig",
    "AnomalyHead",
    "ForegroundHead",
    "KeypointHead",
    "apply_gaussian_suppression",
    "localize_keypoint",
    "make_coordinate_grid",
    "rescale_coords_to_image",
    "temperature_softmax",
    "compute_density_map",
    "compute_exemplar_features",
    "detect_instances",
    "extract_patch_tokens",
    "extract_peaks",
    "visualize",
]
