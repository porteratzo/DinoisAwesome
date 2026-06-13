"""DinoisAwesome: DINO ViT feature extraction and patch-level retrieval gallery."""

from .encoder import DinoEncoder, ExtractorOutput
from .gallery import Gallery, GalleryConfig
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
    "compute_density_map",
    "compute_exemplar_features",
    "detect_instances",
    "extract_patch_tokens",
    "extract_peaks",
    "visualize",
]
