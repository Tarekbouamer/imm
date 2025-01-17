from typing import Dict, Optional

import numpy as np
from loguru import logger

from imm.estimators._camera import Camera
from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from ._helper import get_backend
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


class OpenCVPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=8.0, **kwargs):
        self.max_reproj_error = max_reproj_error

    def estimate(
        self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Camera, dist: Optional[np.ndarray] = None, **kwargs
    ) -> Dict:
        """
        Estimate the pose using OpenCV PnP.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)
            dist (np.ndarray): Distortion coefficients (5, 1)

        Returns:
            Dict: A dictionary containing:
                - qvec (Optional[np.ndarray]): Rotation quaternion (4,)
                - tvec (Optional[np.ndarray]): Translation vector (3,)
                - success (bool): Whether the pose estimation was successful
        """

        # Validate inputs
        CHECK_TYPE(pts2d, np.ndarray)
        CHECK_TYPE(pts3d, np.ndarray)
        CHECK_SHAPE(pts2d, (-1, 2))
        CHECK_SHAPE(pts3d, (-1, 3))

        try:
            #
            if len(pts2d) < 4:
                logger.warning(f"Not enough points to estimate pose less than 4: {len(pts2d)}")
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                }

            # Solve PnP
            success, rotation, translation = cv2.solvePnP(
                pts3d,
                pts2d,
                camera.K,
                dist,
            )

            #
            if not success:
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                }

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

            return {
                "qvec": qvec,
                "tvec": translation.ravel(),
                "success": success,
            }

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shapes: {pts2d.shape}, {pts3d.shape}")
            return {
                "qvec": None,
                "tvec": None,
                "success": False,
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(max_reproj_error={self.max_reproj_error})"


class PycolmapPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0, **kwargs):
        self.max_reproj_error = max_reproj_error

        self.estimation_options = {
            "ransac": {
                "max_error": self.max_reproj_error,
            },
        }

        self.refinement_options = {"refine_focal_length": True}

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Camera, **kwargs) -> Dict:
        """Estimate the pose using Pycolmap.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)

        Returns:
            Dict: A dictionary containing:
                - qvec (Optional[np.ndarray]): Rotation quaternion (4,)
                - tvec (Optional[np.ndarray]): Translation vector (3,)
                - success (bool): Whether the pose estimation was successful
                - inliers (Optional[int]): Number of inliers (not used in this method)
        """
        # Validate inputs
        CHECK_TYPE(pts2d, np.ndarray)
        CHECK_TYPE(pts3d, np.ndarray)
        CHECK_SHAPE(pts2d, (-1, 2))
        CHECK_SHAPE(pts3d, (-1, 3))

        try:
            #
            if len(pts2d) < 4:
                logger.warning(f"Not enough points to estimate pose less than 4: {len(pts2d)}")
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                    "inliers": None,
                }

            # Compute pose
            result = pycolmap.absolute_pose_estimation(
                pts2d, pts3d, camera.todict(), self.estimation_options, self.refinement_options
            )

            if result is None:
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                    "inliers": None,
                }

            return {
                "qvec": result["qvec"],
                "tvec": result["tvec"],
                "success": True,
                "inliers": result["inliers"],
            }

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shapes: {pts2d.shape}, {pts3d.shape}")
            return {
                "qvec": None,
                "tvec": None,
                "success": False,
                "inliers": None,
            }

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"max_reproj_error={self.max_reproj_error})"


class PoseLibPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0, max_epipolar_error=1.0, max_iterations=100, **kwargs):
        self.max_reproj_error = max_reproj_error
        self.max_epipolar_error = max_epipolar_error
        self.max_iterations = max_iterations

        self.ransac_options = {
            "max_reproj_error": max_reproj_error,
            "max_epipolar_error": max_epipolar_error,
        }

        self.bundle_options = {
            "max_iterations": max_iterations,
        }

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Camera, **kwargs) -> Dict:
        """Estimate the pose using PoseLib.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)

        Returns:
            Dict: A dictionary containing:
                - qvec (Optional[np.ndarray]): Rotation quaternion (4,)
                - tvec (Optional[np.ndarray]): Translation vector (3,)
                - success (bool): Whether the pose estimation was successful
                - inliers (Optional[int]): Number of inliers
        """
        # Validate inputs
        CHECK_TYPE(pts2d, np.ndarray)
        CHECK_TYPE(pts3d, np.ndarray)
        CHECK_SHAPE(pts2d, (-1, 2))
        CHECK_SHAPE(pts3d, (-1, 3))

        try:
            if len(pts2d) < 4:
                logger.warning(f"Not enough points to estimate pose less than 4: {len(pts2d)}")
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                    "inliers": None,
                }

            # Estimate pose
            pose, info = poselib.estimate_absolute_pose(
                pts2d,
                pts3d,
                camera.todict(),
                self.ransac_options,
                self.bundle_options,
            )

            if pose is None:
                return {
                    "qvec": None,
                    "tvec": None,
                    "success": False,
                    "inliers": None,
                }

            return {
                "qvec": pose.q,
                "tvec": pose.t,
                "success": pose is not None,
                "inliers": info["inliers"],
            }
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}, Input shapes: {pts2d.shape}, {pts3d.shape}")
            return {
                "qvec": None,
                "tvec": None,
                "success": False,
                "inliers": None,
            }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_reproj_error={self.max_reproj_error}, "
            f"max_epipolar_error={self.max_epipolar_error}, "
            f"max_iterations={self.max_iterations})"
        )


class PnPEstimator(Estimator):
    """Unified PnP Estimator.

    Args:
        backend (str): Backend to use (opencv, pycolmap, poselib)

    """

    def __init__(
        self,
        backend: str,
        max_reproj_error: float = 12.0,
        max_epipolar_error: float = 1.0,
        max_iterations: int = 100,
        **kwargs,
    ):
        super().__init__()

        # Choose the backend
        backend = backend if not None else get_backend()

        if self.backend == "opencv":
            self.estimator = OpenCVPnPEstimator(max_reproj_error=max_reproj_error, **kwargs)
        elif self.backend == "pycolmap":
            self.estimator = PycolmapPnPEstimator(max_reproj_error=max_reproj_error, **kwargs)
        elif self.backend == "poselib":
            self.estimator = PoseLibPnPEstimator(
                max_reproj_error=max_reproj_error,
                max_epipolar_error=max_epipolar_error,
                max_iterations=max_iterations,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid backend: {backend}. Valid options are: ['opencv', 'pycolmap', 'poselib']")

    def estimate(self, pts2d: np.ndarray, pts3d: np.ndarray, camera: Camera, **kwargs) -> Dict:
        """Estimate the pose.

        Args:
            pts2d (np.ndarray): 2D points (N, 2)
            pts3d (np.ndarray): 3D points (N, 3)
            camera (Dict): Camera parameters (model, width, height, params)

        Returns:
            Dict: A dictionary containing:
                - qvec (Optional[np.ndarray]): Rotation quaternion (4,)
                - tvec (Optional[np.ndarray]): Translation vector (3,)
                - success (bool): Whether the pose estimation was successful
                - inliers (Optional[int]): Number of inliers

        """
        return self.estimator.estimate(pts2d, pts3d, camera, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(backend={self.backend})"
