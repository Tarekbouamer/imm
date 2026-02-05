__version__ = "0.1.0"

from imm.utils.logger import setup_logger

try:
    from loguru import logger

    setup_logger(app_name="imm")

except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")

from imm.api import (
    EXTRACTORS_REGISTRY,
    MATCHERS_REGISTRY,
    create_extractor,
    create_matcher,
    download_model_weights,
)

__all__ = [
    "create_extractor",
    "create_matcher",
    "download_model_weights",
    "EXTRACTORS_REGISTRY",
    "MATCHERS_REGISTRY",
]
