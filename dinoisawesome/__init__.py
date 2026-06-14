"""DinoisAwesome: DINO ViT feature extraction and patch-level retrieval gallery."""

from .anomaly_head import AnomalyHead
from .encoder import DinoEncoder, ExtractorOutput
from .foreground_head import ForegroundHead
from .gallery import Gallery, GalleryConfig
from .keypoint_head import KeypointHead
from .instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    detect_instances,
    extract_patch_tokens,
    extract_peaks,
    visualize,
)

__all__ = [
    "DinoEncoder",
    "ExtractorOutput",
    "Gallery",
    "GalleryConfig",
    "AnomalyHead",
    "ForegroundHead",
    "KeypointHead",
    "compute_density_map",
    "compute_exemplar_features",
    "detect_instances",
    "extract_patch_tokens",
    "extract_peaks",
    "visualize",
]
