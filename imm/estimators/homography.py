from typing import Dict, Union

import cv2
import numpy as np
import poselib

from .estimator import Estimator


def CHECK_DIM(pts: np.ndarray) -> bool:
    """Check if the keypoints have the correct shape."""
    if not isinstance(pts, np.ndarray):
        raise ValueError("Keypoints must be a numpy array.")
    if len(pts.shape) != 2 or pts.shape[1] != 2:
        raise ValueError("Keypoints must have the shape (N, 2).")
    return True


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


class CvHomographyEstimator(Estimator):
    def __init__(
        self, solver: str = "ransac", ransac_th: float = 0.5, max_iters: int = 1000, confidence: float = 0.998
    ):
        """
        Homography estimator using OpenCV.

        Args:
            solver: Solver method to use (default: "ransac").
            ransac_th: RANSAC reprojection threshold (default: 0.5).
            max_iters: Maximum number of iterations (default: 1000).
            confidence: Confidence level for the estimation (default: 0.998).
        """
        super().__init__()
        self.solver = solver
        self.ransac_th = ransac_th
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[np.ndarray, bool, int]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).
        """
        CHECK_DIM(pts0)
        CHECK_DIM(pts1)

        # Compute homography
        H, mask = cv2.findHomography(
            pts0,
            pts1,
            method=CV_H_SOLVERS[self.solver],
            ransacReprojThreshold=self.ransac_th,
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
            f"ransac_th={self.ransac_th}, "
            f"max_iters={self.max_iters}, "
            f"confidence={self.confidence})"
        )


class PoseLibHomographyEstimator(Estimator):
    def __init__(self, ransac_th: float = 2.0):
        """
        Homography estimator using PoseLib.

        Args:
            ransac_th: The threshold for the maximum reprojection error (default: 2.0).
        """
        super().__init__()
        self.ransac_th = ransac_th

    def estimate(self, pts0: np.ndarray, pts1: np.ndarray) -> Dict[str, Union[np.ndarray, bool, Dict]]:
        """
        Estimate homography.

        Args:
            pts0: First set of points (Nx2 array).
            pts1: Second set of points (Nx2 array).
        """
        CHECK_DIM(pts0)
        CHECK_DIM(pts1)

        # Estimate homography
        H, status = poselib.estimate_homography(
            pts0,
            pts1,
            {
                "max_reproj_error": self.ransac_th,
            },
        )

        result = {
            "H": H,
            "success": H is not None,
            **status,
        }

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(ransac_th={self.ransac_th})"
