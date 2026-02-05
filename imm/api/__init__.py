from imm.estimators import (
    create_homography_estimator,
    create_pnp_estimator,
    create_relative_pose_estimator,
)
from imm.extractors._helper import EXTRACTORS_REGISTRY, create_extractor
from imm.matchers._helper import MATCHERS_REGISTRY, create_matcher
from imm.registry.factory import download_model_weights
from imm.utils.io import read_image


def list_extractors():
    """List all available feature extractors."""
    return EXTRACTORS_REGISTRY.list_models


def list_matchers():
    """List all available feature matchers."""
    return MATCHERS_REGISTRY.list_models


__all__ = [
    "create_extractor",
    "create_matcher",
    "create_homography_estimator",
    "create_pnp_estimator",
    "create_relative_pose_estimator",
    "read_image",
    "list_extractors",
    "list_matchers",
    "download_model_weights",
    "EXTRACTORS_REGISTRY",
    "MATCHERS_REGISTRY",
]
