from typing import Dict, Union

import cv2
import numpy as np
from loguru import logger

from imm.estimators._helper import Camera

from ._conversions import convert_points_to_homogeneous

CV_SOLVERS = {
    "ransac": cv2.RANSAC,
    "usac_magsac": cv2.USAC_MAGSAC,
}


class OpenCVRelativePoseEstimator:
    def __init__(
        self,
        solver: str = "ransac",
        threshold: float = 1.0,
        confidence: float = 0.999,
        max_iters: int = 1000,
    ):
        if solver not in CV_SOLVERS:
            raise ValueError(f"Invalid solver: {solver}. Valid options are: {list(CV_SOLVERS.keys())}")

        self.solver = solver
        self.threshold = threshold
        self.confidence = confidence
        self.max_iters = max_iters

    def estimate(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        camera0: Camera = None,
        camera1: Camera = None,
    ) -> Dict[str, Union[np.ndarray, int, bool]]:
        if not isinstance(pts0, np.ndarray) or not isinstance(pts1, np.ndarray):
            raise TypeError("pts0 and pts1 must be numpy arrays.")

        if pts0.shape != pts1.shape or pts0.shape[1] != 2:
            raise ValueError("pts0 and pts1 must have the same shape (N, 2).")

        try:
            # Five correspondences
            if pts0.shape[0] < 5:
                logger.warning(f"Number of correspondences is less than 5: {pts0.shape[0]}.")
                return {"success": False, "R": None, "t": None, "inliers": None, "num_inliers": 0}

            # Undistort points if camera matrices are provided
            pts0 = cv2.undistortPoints(pts0, camera0, distCoeffs=distCoeffs).reshape(-1, 2)
            pts1 = cv2.undistortPoints(pts1, camera1, distCoeffs=distCoeffs).reshape(-1, 2)

            f_mean = np.cat([camera0.fx, camera1.fx]).mean().item()
            norm_threshold = self.threshold / f_mean

            # Convert to homogeneous coordinates
            # pts0 = convert_points_to_homogeneous(pts0)
            # pts1 = convert_points_to_homogeneous(pts1)

            # Compute the essential matrix
            E, mask = cv2.findEssentialMat(
                pts0,
                pts1,
                cameraMatrix=np.eye(3),
                method=CV_SOLVERS[self.solver],
                threshold=norm_threshold,
                prob=self.confidence,
                maxIters=self.max_iters,
            )

            if E is None:
                logger.warning("Essential matrix computation failed.")
                return {"success": False, "R": None, "t": None, "inliers": None, "num_inliers": 0}

            # Recover the relative pose (R and t)
            _, R, t, mask_recover = cv2.recoverPose(E, pts0, pts1, cameraMatrix=np.eye(3), mask=mask)

            return {
                "success": True,
                "R": R,
                "t": t,
                "inliers": mask.reshape(-1).astype(bool),
                "num_inliers": int(mask_recover.sum()),
            }

        except Exception as e:
            logger.error(
                f"Error in OpenCVRelativePoseEstimator: {e}, Input shapes pts0: {pts0.shape}, pts1: {pts1.shape}"
            )
            return {"success": False, "R": None, "t": None, "inliers": None, "num_inliers": 0}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"solver='{self.solver}', "
            f"threshold={self.threshold}, "
            f"confidence={self.confidence}, "
            f"max_iters={self.max_iters})"
        )
