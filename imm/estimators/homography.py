from typing import Dict, Union

import numpy as np
from loguru import logger

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
    def __init__(
        self, solver: str = "ransac", inlier_threshold: float = 0.5, max_iters: int = 1000, confidence: float = 0.998
    ):
        """
        Homography estimator using OpenCV.

        Args:
            solver: Solver method to use (default: "ransac").
            inlier_threshold: RANSAC reprojection threshold (default: 0.5).
            max_iters: Maximum number of iterations (default: 1000).
            confidence: Confidence level for the estimation (default: 0.998).
        """
        super().__init__()
        self.solver = solver
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).
        """
        pts0 = pts0.reshape(-1, 2)
        pts1 = pts1.reshape(-1, 2)
        
        print(pts0.shape)
        print(pts1.shape)

        # Compute homography
        H, mask = cv2.findHomography(
            pts0,
            pts1,
            method=CV_H_SOLVERS[self.solver],
            ransacReprojThreshold=self.inlier_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        result = {
            "H": H,
            "success": H is not None,
            "inliers": int(mask.sum()) if mask is not None else 0,
        }

        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"solver='{self.solver}', "
            f"inlier_threshold={self.inlier_threshold}, "
            f"max_iters={self.max_iters}, "
            f"confidence={self.confidence})"
        )


class PoseLibHomographyEstimator(Estimator):
    def __init__(
        self,
        inlier_threshold: float = 2.0,
        max_iters: int = 1000,
        min_iters: int = 50,
        confidence: float = 0.99999,
        progressive_sampling: bool = False,
    ):
        """
        Homography estimator using PoseLib.

        Args:
            inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        """
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.confidence = confidence
        self.progressive_sampling = progressive_sampling

        assert (
            self.min_iters < self.max_iters
        ), f"min_iters={self.min_iters} should be less than max_iters={self.max_iters}"

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[np.ndarray, bool, Dict]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).
        """
        pts0 = pts0.reshape(-1, 2)
        pts1 = pts1.reshape(-1, 2)

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

        result = {
            "H": H,
            "success": H is not None,
            **status,
        }

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}), max_iters={self.max_iters}, min_iters={self.min_iters}, confidence={self.confidence}, progressive_sampling={self.progressive_sampling}"


class PycolmapHomographyEstimator(Estimator):
    def __init__(
        self,
        inlier_threshold: float = 2.0,
        min_inlier_ratio: float = 0.1,
        confidence: float = 0.9999,
        max_iters: int = 100000,
        min_iters: int = 1000,
    ):
        """
        Homography estimator using Pycolmap.

        Args:
            inlier_threshold: The threshold for the maximum reprojection error (default: 2.0).
        """
        super().__init__()

        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.confidence = confidence
        self.max_iters = max_iters
        self.min_iters = min_iters

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict:
        pts0 = pts0.reshape(-1, 2)
        pts1 = pts1.reshape(-1, 2)

        # RANSAC options
        options = pycolmap.RANSACOptions()
        options.max_error = self.inlier_threshold
        options.min_inlier_ratio = self.min_inlier_ratio
        options.confidence = self.confidence
        options.max_num_trials = self.max_iters
        options.min_num_trials = self.min_iters

        # Estimate homography
        res = pycolmap.homography_matrix_estimation(pts0, pts1, options)

        return {"H": res["H"], "success": True if res is not None else False, "inliers": res["inliers"]}

    def __repr__(self):
        return f"{self.__class__.__name__}(inlier_threshold={self.inlier_threshold}, min_inlier_ratio={self.min_inlier_ratio}, confidence={self.confidence}, max_iters={self.max_iters}, min_iters={self.min_iters})"
