<<<<<<< HEAD
from loguru import logger

try:
    import poselib
except ImportError:
    poselib = None
    logger.warning("PoseLib not found. PoseLibHomographyEstimator will not work.")

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not found. CvHomographyEstimator will not work.")

try:
    import pycolmap
except ImportError:
    pycolmap = None
    logger.warning("Pycolmap not found. PycolmapHomographyEstimator will not work.")


def get_backend():
    """
    Get the default backend for homography estimation.

    Returns:
        Default backend for homography estimation.
    """
    if cv2 is not None:
        return "opencv"
    elif poselib is not None:
        return "poselib"
    elif pycolmap is not None:
        return "pycolmap"
    else:
        raise ValueError("No backend found for homography estimation.")
=======
from imm.geometry import Camera
>>>>>>> 5e32819 (feat: Add utility functions for configuration merging and key extension)
