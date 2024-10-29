__version__ = "0.1"

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")
