from loguru import logger

try:
    import poselib
except ImportError:
    poselib = None
    logger.warning("PoseLib not found. PoseLib estimators will not work.")

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not found. OpenCV estimators will not work.")

try:
    import pycolmap
except ImportError:
    pycolmap = None
    logger.warning("Pycolmap not found. Pycolmap estimators will not work.")


def get_backend():
    """Get the default backend for estimation.

    Returns:
        Default backend for estimation based on available libraries.
    """
    if cv2 is not None:
        return "opencv"
    elif poselib is not None:
        return "poselib"
    elif pycolmap is not None:
        return "pycolmap"
    else:
        raise ValueError(
            "No backend found for estimation. Install opencv-python, poselib, or pycolmap.")
