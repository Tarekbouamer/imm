__version__ = "0.1"


try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")


from .feature_matcher import ExtractorManager, FeaturesWriter

__all__ = ["ExtractorManager", "FeaturesWriter"]
