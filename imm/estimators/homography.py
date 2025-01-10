from typing import Any, Dict, Union

import numpy as np
from loguru import logger

from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from .estimator import Estimator

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


CV_H_SOLVERS = {
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
    "rho": cv2.RHO,
    "usac": cv2.USAC_DEFAULT,
    "usac_parallel": cv2.USAC_PARALLEL,
    "usac_accurate": cv2.USAC_ACCURATE,
    "usac_fast": cv2.USAC_FAST,
    "usac_prosac": cv2.USAC_PROSAC,
    "usac_magsac": cv2.USAC_MAGSAC,
}


class OpenCVHomographyEstimator(Estimator):
    """
    Homography estimator using OpenCV.

    Args:
        solver: Solver method to use (default: "ransac").
        inlier_threshold: RANSAC reprojection threshold (default: 0.5).
        max_iters: Maximum number of iterations (default: 1000).
        confidence: Confidence level for the estimation (default: 0.998).
    """

    def __init__(
        self,
        solver: str = "ransac",
        inlier_threshold: float = 0.5,
        max_iters: int = 1000,
        confidence: float = 0.998,
        **kwargs,
    ):
        super().__init__()

        if solver not in CV_H_SOLVERS:
            raise ValueError(f"Invalid solver: {solver}. Valid options are: {list(CV_H_SOLVERS.keys())}")

        self.solver = solver
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, and number of inliers
        """

        # Check if OpenCV is available
        if cv2 is None:
            logger.error("OpenCV not found. CvHomographyEstimator will not work.")
            return {
                "H": None,
                "success": False,
                "inliers": 0,
            }

        try:
            # Validate type
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError("At least 4 points are required to estimate homography.")

            # Compute homography
            H, mask = cv2.findHomography(
                pts0,
                pts1,
                method=CV_H_SOLVERS[self.solver],
                ransacReprojThreshold=self.inlier_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )

            if H is None:
                return {
                    "H": H,
                    "success": False,
                    "inliers": 0,
                }

            # Count inliers
            inliers = int(mask.sum()) if mask is not None else 0

            return {
                "H": H,
                "success": True,
                "inliers": inliers,
            }

        except Exception as e:
            logger.error(f"Error in OpenCVHomographyEstimator: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "H": None,
                "success": False,
                "inliers": 0,
            }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"solver='{self.solver}', "
            f"inlier_threshold={self.inlier_threshold}, "
            f"max_iters={self.max_iters}, "
            f"confidence={self.confidence})"
        )


class PoseLibHomographyEstimator(Estimator):
    """Homography estimator using PoseLib.

    Args:
        inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        max_iters: The maximum number of RANSAC iterations (default: 1000).
        min_iters: The minimum number of RANSAC iterations (default: 50).
        confidence: The confidence level for the estimation (default: 0.99999).
        progressive_sampling: Whether to use progressive sampling (default: False
    """

    def __init__(
        self,
        inlier_threshold: float = 2.0,
        max_iters: int = 1000,
        min_iters: int = 50,
        confidence: float = 0.99999,
        progressive_sampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.confidence = confidence
        self.progressive_sampling = progressive_sampling

        assert (
            self.min_iters < self.max_iters
        ), f"min_iters={self.min_iters} should be less than max_iters={self.max_iters}"

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, and number of inliers
        """

        # Check if PoseLib is available
        if poselib is None:
            logger.error("PoseLib not found. PoseLibHomographyEstimator will not work.")
            return {
                "H": None,
                "success": False,
                "inliers": 0,
            }

        try:
            # Validate type
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError("At least 4 points are required to estimate homography.")

            # Ransac options
            ransac_options = {
                "max_iterations": self.max_iters,
                "min_iterations": self.min_iters,
                "success_prob": self.confidence,
                "max_reproj_error": self.inlier_threshold,
                "progressive_sampling": self.progressive_sampling,
            }

            # Estimate homography
            H, status = poselib.estimate_homography(
                pts0,
                pts1,
                ransac_opt=ransac_options,
            )

            if H is None:
                return {
                    "H": H,
                    "success": False,
                    "inliers": 0,
                }

            return {
                "H": H,
                "success": True,
                **status,
            }

        except Exception as e:
            logger.error(f"Error in PoseLibHomographyEstimator: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "H": None,
                "success": False,
                "inliers": 0,
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, max_iters={self.max_iters}, min_iters={self.min_iters}, confidence={self.confidence}, progressive_sampling={self.progressive_sampling})"


class PycolmapHomographyEstimator(Estimator):
    """
    Homography estimator using Pycolmap.

    Args:
        inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        min_inlier_ratio: The minimum inlier ratio (default: 0.1).
        confidence: The confidence level for the estimation (default: 0.9999).
        max_iters: The maximum number of RANSAC iterations (default: 100000).
        min_iters: The minimum number of RANSAC iterations (default: 1000).
    """

    def __init__(
        self,
        inlier_threshold: float = 2.0,
        min_inlier_ratio: float = 0.1,
        confidence: float = 0.9999,
        max_iters: int = 100000,
        min_iters: int = 1000,
        **kwargs,
    ):
        super().__init__()

        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.confidence = confidence
        self.max_iters = max_iters
        self.min_iters = min_iters

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, and number of inliers
        """

        # Check if Pycolmap is available
        if pycolmap is None:
            logger.error("Pycolmap not found. PycolmapHomographyEstimator will not work.")
            return {"H": None, "success": False, "inliers": 0}

        try:
            # Validate type and shape
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError("At least 4 points are required to estimate homography.")

            # RANSAC options
            options = pycolmap.RANSACOptions()
            options.max_error = self.inlier_threshold
            options.min_inlier_ratio = self.min_inlier_ratio
            options.confidence = self.confidence
            options.max_num_trials = self.max_iters
            options.min_num_trials = self.min_iters

            # Estimate homography
            res = pycolmap.homography_matrix_estimation(pts0, pts1, options)

            if res is None:
                return {"H": None, "success": False, "inliers": 0}

            return {"H": res["H"], "success": True if res is not None else False, "inliers": res["inliers"]}
        except Exception as e:
            logger.error(
                f"Error in PycolmapHomographyEstimator: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}"
            )
            return {"H": None, "success": False, "inliers": 0}

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, min_inlier_ratio={self.min_inlier_ratio}, confidence={self.confidence}, max_iters={self.max_iters}, min_iters={self.min_iters})"


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


class HomographyEstimator(Estimator):
    """
    Unified Homography Estimator.

    Args:
        method: Estimation method to use (default: "cv2").
        solver: Solver method to use (default: "ransac").
        inlier_threshold: RANSAC reprojection threshold (default: 0.5).
        max_iters: Maximum number of iterations (default: 1000).
        confidence: Confidence level for the estimation (default: 0.998).
    """

    def __init__(
        self,
        method: str = None,
        solver: str = "ransac",
        inlier_threshold: float = 0.5,
        max_iters: int = 1000,
        confidence: float = 0.998,
        **kwargs,
    ):
        super().__init__()

        method = method if not None else get_backend()

        if method == "opencv":
            self.estimator = OpenCVHomographyEstimator(solver, inlier_threshold, max_iters, confidence)
        elif method == "poselib":
            self.estimator = PoseLibHomographyEstimator(inlier_threshold, max_iters, confidence, **kwargs)
        elif method == "pycolmap":
            self.estimator = PycolmapHomographyEstimator(inlier_threshold, 0.1, confidence, max_iters, 1000)
        else:
            raise ValueError(f"Invalid method: {method}. Valid options are: cv2, poselib, pycolmap")

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, and number of inliers
        """
        return self.estimator.estimate(pts0, pts1)

    def __repr__(self):
        return self.estimator.__repr__()
