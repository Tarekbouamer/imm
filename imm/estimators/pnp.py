from typing import Dict

import numpy as np
from loguru import logger

from .estimator import Estimator

try:
    import pycolmap
except ImportError:
    pycolmap = None
    logger.warning("Pycolmap not found. PycolmapPnPEstimator will not work.")

try:
    import poselib
except ImportError:
    poselib = None
    logger.warning("PoseLib not found. PoseLibPnPEstimator will not work.")


try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not found. OpenCVPnPEstimator will not work.")


class PycolmapPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0):
        self.max_reproj_error = max_reproj_error

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Dict, **kwargs):
        """Estimate the pose using Pycolmap.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)

        Returns:
            Dict: qvec (rotation quaternion) and tvec (translation) and success (bool) and inliers (int)

        """
        pts2d = pts2d.reshape(-1, 2)
        pts3d = pts3d.reshape(-1, 3)

        # Options
        estimation_options = {
            "ransac": {
                "max_error": self.max_reproj_error,
            },
        }
        #
        refinement_options = {"refine_focal_length": True}

        # compute pose
        ret = pycolmap.absolute_pose_estimation(
            pts2d,
            pts3d,
            camera,
            # estimation_options, refinement_options
        )

        return {
            "qvec": ret["cam_from_world"].rotation.quat,
            "tvec": ret["cam_from_world"].translation,
            "success": True,
            "inliers": ret["inliers"],
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"max_reproj_error={self.max_reproj_error})"


class PoseLibPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0, max_epipolar_error=1.0, max_iterations=100):
        self.max_reproj_error = max_reproj_error
        self.max_epipolar_error = max_epipolar_error
        self.max_iterations = max_iterations

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Dict, **kwargs) -> Dict:
        """Estimate the pose using PoseLib.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)

        """
        pts2d = pts2d.reshape(-1, 2)
        pts3d = pts3d.reshape(-1, 3)

        ransac_options = {
            "max_reproj_error": self.max_reproj_error,
            "max_epipolar_error": self.max_epipolar_error,
        }

        bundle_options = {
            "max_iterations": self.max_iterations,
        }

        pose, info = poselib.estimate_absolute_pose(
            pts2d,
            pts3d,
            camera,
            ransac_options,
            bundle_options,
        )

        result = {
            "qvec": pose.q,
            "tvec": pose.t,
            "success": pose is not None,
            "inliers": info["inliers"],
        }

        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_reproj_error={self.max_reproj_error}, "
            f"max_epipolar_error={self.max_epipolar_error}, "
            f"max_iterations={self.max_iterations})"
        )


class OpenCVPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=8.0):
        self.max_reproj_error = max_reproj_error

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Dict, dist: np.ndarray = None, **kwargs) -> Dict:
        """
        Estimate the pose using OpenCV PnP.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)
            dist (np.ndarray): Distortion coefficients (5, 1)

        Returns:
            Dict: qvec (rotation quaternion) and tvec (translation) and success (bool) and inliers (int)

        """

        #
        pts2d = pts2d.reshape(-1, 2)
        pts3d = pts3d.reshape(-1, 3)

        # Convert camera dictionary to camera matrix
        cam_matrix = np.array(
            [[camera["params"][0], 0, camera["params"][2]], [0, camera["params"][1], camera["params"][3]], [0, 0, 1]]
        )

        # Solve PnP
        success, rotation, translation = cv2.solvePnP(
            pts3d,
            pts2d,
            cam_matrix,
            dist,
        )

        # Convert rotation vector to quaternion
        rvec = rotation.ravel()
        R = cv2.Rodrigues(rvec)[0]

        # Convert rotation matrix to quaternion
        w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        w4 = 4.0 * w
        x = (R[2, 1] - R[1, 2]) / w4
        y = (R[0, 2] - R[2, 0]) / w4
        z = (R[1, 0] - R[0, 1]) / w4
        qvec = np.array([w, x, y, z])

        result = {
            "qvec": qvec,
            "tvec": translation.ravel(),
            "success": success,
        }

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(max_reproj_error={self.max_reproj_error})"
