"""DinoisAwesome: DINO ViT feature extraction and patch-level retrieval gallery."""

from .annotation_utils import load_annotations
from .anomaly_head import AnomalyHead
from .background_mask import compute_foreground_mask
from .encoder import DinoEncoder, ExtractorOutput
from .encoder_cache import EncoderFingerprint, EncoderWithCache, inspect_cache
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
from .prototype_head import PrototypeAnomalyHead

__all__ = [
    "load_annotations",
    "DinoEncoder",
    "ExtractorOutput",
    "EncoderFingerprint",
    "EncoderWithCache",
    "inspect_cache",
    "Gallery",
    "GalleryConfig",
    "AnomalyHead",
    "PrototypeAnomalyHead",
    "ForegroundHead",
    "compute_foreground_mask",
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
