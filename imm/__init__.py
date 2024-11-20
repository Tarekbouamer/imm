__version__ = "0.1"

from imm.utils.logger import setup_logger

try:
    from loguru import logger

    setup_logger(app_name="imm")

except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")
