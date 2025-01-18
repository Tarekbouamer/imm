from typing import Any, Dict, Union

import numpy as np
from loguru import logger

from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from ._helper import get_backend
from .estimator import Estimator

try:
    import poselib
except ImportError:
    poselib = None
    logger.warning("PoseLib not found. PoseLibFundamentalEstimator will not work.")

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not found. CvFundamentalEstimator will not work.")

try:
    import pycolmap
except ImportError:
    pycolmap = None
    logger.warning("Pycolmap not found. PycolmapFundamentalEstimator will not work.")


CV_F_SOLVERS = {
    "ransac": cv2.FM_RANSAC,
    "lmeds": cv2.FM_LMEDS,
    "8pt": cv2.FM_8POINT,
}


class OpenCVFundamentalEstimator(Estimator):
    """
    Fundamental matrix estimator using OpenCV.

    Args:
        solver: Solver method to use (default: "ransac").
        inlier_threshold: RANSAC reprojection threshold (default: 0.5).
        max_iters: Maximum number of iterations (default: 1000).
        confidence: Confidence level for the estimation (default: 0.998).
    """

    def __init__(
        self,
        solver: str = "ransac",
        inlier_threshold: float = 1.0,
        max_iters: int = 1000,
        confidence: float = 0.998,
        **kwargs,
    ):
        super().__init__()

        if solver not in CV_F_SOLVERS:
            raise ValueError(f"Invalid solver: {solver}. Valid options are: {list(CV_F_SOLVERS.keys())}")

        self.solver = solver
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate fundamental matrix.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated fundamental matrix, success status, inliers, and inliers count.
        """

        # Check if OpenCV is available
        if cv2 is None:
            logger.error("OpenCV not found. CvFundamentalEstimator will not work.")
            return {
                "F": None,
                "success": False,
                "inliers": None,
                "num_inliers": 0,
            }

        try:
            # Validate type
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 8 or len(pts1) < 8:
                raise ValueError("At least 8 points are required to estimate the fundamental matrix.")

            # Compute fundamental matrix
            F, mask = cv2.findFundamentalMat(
                pts0,
                pts1,
                method=CV_F_SOLVERS[self.solver],
                ransacReprojThreshold=self.inlier_threshold,
                confidence=self.confidence,
                maxIters=self.max_iters,
            )

            if F is None:
                return {
                    "F": F,
                    "success": False,
                    "inliers": 0,
                    "num_inliers": 0,
                }

            # Count inliers
            num_inliers = int(mask.sum()) if mask is not None else 0

            return {
                "F": F,
                "success": True,
                "inliers": mask.reshape(-1).astype(bool),
                "num_inliers": num_inliers,
            }

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "F": None,
                "success": False,
                "inliers": None,
                "num_inliers": 0,
            }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"solver='{self.solver}', "
            f"inlier_threshold={self.inlier_threshold}, "
            f"max_iters={self.max_iters}, "
            f"confidence={self.confidence})"
        )


class PoseLibFundamentalEstimator(Estimator):
    """Fundamental matrix estimator using PoseLib.

    Args:
        inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        max_iters: The maximum number of RANSAC iterations (default: 1000).
        confidence: The confidence level for the estimation (default: 0.99999).
        progressive_sampling: Whether to use progressive sampling (default: False).
    """

    def __init__(
        self,
        inlier_threshold: float = 2.0,
        max_iters: int = 1000,
        confidence: float = 0.99999,
        progressive_sampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.progressive_sampling = progressive_sampling

        # RANSAC options
        self.ransac_options = {
            "max_epipolar_error": inlier_threshold,
            "max_iterations": max_iters,
            "success_prob": confidence,
            "progressive_sampling": progressive_sampling,
        }

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate fundamental matrix.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated fundamental matrix, success status, inliers, and inliers count.
        """

        # Check if PoseLib is available
        if poselib is None:
            logger.error("PoseLib not found. PoseLibFundamentalEstimator will not work.")
            return {
                "F": None,
                "success": False,
                "inliers": None,
                "num_inliers": 0,
            }

        try:
            # Validate type
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 8 or len(pts1) < 8:
                raise ValueError("At least 8 points are required to estimate the fundamental matrix.")

            # Estimate fundamental matrix
            F, status = poselib.estimate_fundamental(
                pts0,
                pts1,
                ransac_opt=self.ransac_options,
            )

            if F is None:
                return {"F": F, "success": False, "inliers": None}

            return {
                "F": F,
                "success": True,
                **status,
            }

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "F": None,
                "success": False,
                "inliers": None,
                "num_inliers": 0,
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, max_iters={self.max_iters}, min_iters={self.min_iters}, confidence={self.confidence}, progressive_sampling={self.progressive_sampling})"


class PycolmapFundamentalEstimator(Estimator):
    """
    Fundamental matrix estimator using Pycolmap.

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

        # RANSAC options
        self.options = pycolmap.RANSACOptions()
        self.options.max_error = inlier_threshold
        self.options.min_inlier_ratio = min_inlier_ratio
        self.options.confidence = confidence
        self.options.max_num_trials = max_iters
        self.options.min_num_trials = min_iters

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate fundamental matrix.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated fundamental matrix, success status, inliers, and inliers count.
        """

        # Check if Pycolmap is available
        if pycolmap is None:
            logger.error("Pycolmap not found. PycolmapFundamentalEstimator will not work.")
            return {"F": None, "success": False, "inliers": 0}

        try:
            # Validate type and shape
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 8 or len(pts1) < 8:
                raise ValueError("At least 8 points are required to estimate the fundamental matrix.")

            # Estimate fundamental matrix
            res = pycolmap.fundamental_matrix_estimation(pts0, pts1, self.options)

            if res is None:
                return {"F": None, "success": False, "inliers": None, "num_inliers": 0}

            return {
                "F": res["F"],
                "success": True if res is not None else False,
                "inliers": res["inliers"],
                "num_inliers": res["num_inliers"],
            }
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {"F": None, "success": False, "inliers": None, "num_inliers": 0}

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, min_inlier_ratio={self.min_inlier_ratio}, confidence={self.confidence}, max_iters={self.max_iters}, min_iters={self.min_iters})"


class FundamentalEstimator(Estimator):
    """
    Unified Fundamental Matrix Estimator.

    Args:
        backend (str): Backend to use. Default is the available backend by priority.
        solver (str): Solver method to use (default: "ransac").
        inlier_threshold (float): RANSAC reprojection threshold (default: 0.5).
        max_iters (int): Maximum number of iterations (default: 1000).
        confidence (float): Confidence level for the estimation (default: 0.998).
    """

    def __init__(
        self,
        backend: str = None,
        solver: str = "ransac",
        inlier_threshold: float = 0.5,
        max_iters: int = 1000,
        confidence: float = 0.998,
        **kwargs,
    ):
        super().__init__()

        # Choose the backend
        backend = backend if backend is not None else get_backend()

        if backend == "opencv":
            self.estimator = OpenCVFundamentalEstimator(solver, inlier_threshold, max_iters, confidence)
        elif backend == "poselib":
            self.estimator = PoseLibFundamentalEstimator(inlier_threshold, max_iters, confidence, **kwargs)
        elif backend == "pycolmap":
            self.estimator = PycolmapFundamentalEstimator(inlier_threshold, 0.1, confidence, max_iters, 1000)
        else:
            raise ValueError(backend)

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[Any]]:
        """
        Estimate fundamental matrix.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated fundamental matrix, success status, inliers, and inliers count.
        """
        return self.estimator.estimate(pts0, pts1)

    def __repr__(self):
        return f"{self.__class__.__name__}(estimator={self.estimator})"
