from typing import Any, Dict, TypedDict, Union

import numpy as np
from loguru import logger

from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from ._helper import get_backend
from .estimator import Estimator

try:
    import poselib
except ImportError:
    poselib = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pycolmap
except ImportError:
    pycolmap = None


class HomographyResult(TypedDict):
    H: np.ndarray | None
    success: bool
    inliers: np.ndarray | None
    num_inliers: int


def to_inlier_mask(inliers, num_points: int) -> np.ndarray | None:
    if inliers is None:
        return None

    arr = np.asarray(inliers).ravel()
    if arr.size == 0:
        return np.zeros(num_points, dtype=bool)

    if arr.dtype == bool:
        if arr.size != num_points:
            raise ValueError(
                "Inlier mask size does not match number of points")
        return arr

    if arr.size == num_points and np.all((arr == 0) | (arr == 1)):
        return arr.astype(bool)

    if arr.max() < num_points and arr.min() >= 0:
        mask = np.zeros(num_points, dtype=bool)
        mask[arr.astype(int)] = True
        return mask

    raise ValueError("Unsupported inliers format")


CV_H_SOLVERS = {}
if cv2 is not None:
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
    """Homography estimator using OpenCV.

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

        if cv2 is None:
            raise ImportError(
                "OpenCVHomographyEstimator requires `opencv-python`. "
                "Install it with: pip install opencv-python"
            )

        if solver not in CV_H_SOLVERS:
            raise ValueError(
                f"Invalid solver: {solver}. Valid options are: {list(CV_H_SOLVERS.keys())}")

        self.solver = solver
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> HomographyResult:
        """Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, inliers and inliers count.
        """

        try:
            pts0 = np.asarray(pts0, dtype=np.float32)
            pts1 = np.asarray(pts1, dtype=np.float32)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError(
                    "At least 4 points are required to estimate homography.")

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
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Count inliers
            inliers = to_inlier_mask(
                mask, len(pts0)) if mask is not None else None
            num_inliers = int(inliers.sum()) if inliers is not None else 0

            return {
                "H": H,
                "success": True,
                "inliers": inliers,
                "num_inliers": num_inliers,
            }

        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "H": None,
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


class PoseLibHomographyEstimator(Estimator):
    """Homography estimator using PoseLib.

    Args:
        inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        max_iters: The maximum number of RANSAC iterations (default: 1000).
        confidence: The confidence level for the estimation (default: 0.99999).
        progressive_sampling: Whether to use progressive sampling (default: False
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

        if poselib is None:
            raise ImportError(
                "PoseLibHomographyEstimator requires `poselib`. "
                "Install it with: pip install poselib"
            )

        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.progressive_sampling = progressive_sampling

        # RANSAC options
        self.ransac_options = {
            "max_error": inlier_threshold,
            "max_iterations": max_iters,
            "success_prob": confidence,
            "progressive_sampling": progressive_sampling,
        }

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> HomographyResult:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, inliers and inliers count.
        """

        try:
            # Validate type
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)

            # Validate shape
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError(
                    "At least 4 points are required to estimate homography.")

            # Estimate homography
            H, status = poselib.estimate_homography(
                pts0, pts1, self.ransac_options, {})

            if H is None:
                return {"H": H, "success": False, "inliers": None, "num_inliers": 0}

            inliers = to_inlier_mask(status.get(
                "inliers"), len(pts0)) if status else None
            num_inliers = int(inliers.sum()) if inliers is not None else 0

            return {
                "H": H,
                "success": True,
                "inliers": inliers,
                "num_inliers": num_inliers,
            }

        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {
                "H": None,
                "success": False,
                "inliers": None,
                "num_inliers": 0,
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, max_iters={self.max_iters}, confidence={self.confidence}, progressive_sampling={self.progressive_sampling})"


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

        if pycolmap is None:
            raise ImportError(
                "PycolmapHomographyEstimator requires `pycolmap`. "
                "Install it with: pip install pycolmap"
            )

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

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> HomographyResult:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, inliers and inliers count.
        """

        try:
            # Validate type and shape
            CHECK_TYPE(pts0, np.ndarray)
            CHECK_TYPE(pts1, np.ndarray)
            CHECK_SHAPE(pts0, (-1, 2))
            CHECK_SHAPE(pts1, (-1, 2))

            # Check sufficient points
            if len(pts0) < 4 or len(pts1) < 4:
                raise ValueError(
                    "At least 4 points are required to estimate homography.")

            # Estimate homography
            res = pycolmap.homography_matrix_estimation(
                pts0, pts1, self.options)

            if res is None:
                return {"H": None, "success": False, "inliers": None, "num_inliers": 0}

            inliers = to_inlier_mask(res.get("inliers"), len(pts0))
            num_inliers = int(inliers.sum()) if inliers is not None else 0

            return {
                "H": res["H"],
                "success": True,
                "inliers": inliers,
                "num_inliers": num_inliers,
            }
        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shape: pts0={pts0.shape}, pts1={pts1.shape}")
            return {"H": None, "success": False, "inliers": None, "num_inliers": 0}

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, min_inlier_ratio={self.min_inlier_ratio}, confidence={self.confidence}, max_iters={self.max_iters}, min_iters={self.min_iters})"


class HomographyEstimator(Estimator):
    """Unified Homography Estimator.

    Args:
        backend (str): Backend to use. Default is the available backend by priority.
        solver (str): Solver method to use (default: "ransac").
        inlier_threshold (float): RANSAC reprojection threshold (default: 0.5).
        max_iters (int): Maximum number of iterations (default: 1000).
        confidence (float): Confidence level for the estimation (default: 0.998).
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

        # Choose the backend
        method = method if method is not None else get_backend()

        if method == "opencv":
            self.estimator = OpenCVHomographyEstimator(
                solver=solver,
                inlier_threshold=inlier_threshold,
                max_iters=max_iters,
                confidence=confidence,
                **kwargs,
            )
        elif method == "poselib":
            self.estimator = PoseLibHomographyEstimator(
                inlier_threshold=inlier_threshold,
                max_iters=max_iters,
                confidence=confidence,
                **kwargs,
            )
        elif method == "pycolmap":
            self.estimator = PycolmapHomographyEstimator(
                inlier_threshold=inlier_threshold,
                confidence=confidence,
                max_iters=max_iters,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid method: {method}. Valid options are: opencv, poselib, pycolmap")

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> HomographyResult:
        """Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).

        Returns:
            Dictionary containing the estimated homography matrix, success status, inliers and inliers count.
        """
        return self.estimator.estimate(pts0, pts1)

    def __repr__(self):
        return f"{self.__class__.__name__}(estimator={self.estimator})"
