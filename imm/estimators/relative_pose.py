from typing import Dict, Optional, Union

import cv2
import numpy as np
from loguru import logger

from imm.estimators._camera import Camera
from imm.estimators.estimator import Estimator
from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from ._conversions import convert_points_from_homogeneous, essential_from_Rt
from ._helper import get_backend

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

CV_SOLVERS = {
    "ransac": cv2.RANSAC,
    "usac_magsac": cv2.USAC_MAGSAC,
}


class OpenCVRelativePoseEstimator(Estimator):
    """Estimate the relative pose between two images using OpenCV.

    Args:
        solver (str): Solver method to use. Default is "ransac".
        threshold (float): Maximum reprojection error threshold. Default is 1.0.
        confidence (float): Confidence level. Default is 0.999.
        max_iters (int): Maximum number of iterations. Default is 1000.
    """

    def __init__(
        self,
        solver: str = "ransac",
        threshold: float = 1.0,
        confidence: float = 0.999,
        max_iters: int = 1000,
    ):
        super().__init__()
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
        camera0: Optional[Camera] = None,
        camera1: Optional[Camera] = None,
    ) -> Dict[str, Union[np.ndarray, int, bool]]:
        """Estimate the relative pose between two sets of 2D points.

        Args:
            pts0 (np.ndarray): 2D points in the first image, shape (N, 2).
            pts1 (np.ndarray): 2D points in the second image, shape (N, 2).
            camera0 (Optional[Camera]): Camera model of the first image.
            camera1 (Optional[Camera]): Camera model of the second image.

        Returns:
            Dict[str, Union[np.ndarray, int, bool]]: Dictionary containing the following
                - success (bool): True if the estimation was successful.
                - E (np.ndarray): Essential matrix.
                - R (np.ndarray): Rotation matrix.
                - t (np.ndarray): Translation vector.
                - inliers (np.ndarray): Inliers mask.
                - num_inliers (int): Number of inliers.
        """

        # validate inputs
        CHECK_TYPE(pts0, np.ndarray)
        CHECK_TYPE(pts1, np.ndarray)
        CHECK_SHAPE(pts0, (-1, 2))
        CHECK_SHAPE(pts1, (-1, 2))

        try:
            # Five correspondences
            if pts0.shape[0] < 5:
                logger.warning(f"Number of correspondences is less than 5: {pts0.shape[0]}.")
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Normalize the points
            pts0 = camera0.image2camera(pts0)
            pts1 = camera1.image2camera(pts1)

            # Normalize the threshold
            f_mean = np.array([camera0.fx, camera1.fx]).mean().item()
            norm_threshold = self.threshold / f_mean

            # Convert to homogeneous coordinates
            pts0 = convert_points_from_homogeneous(pts0)
            pts1 = convert_points_from_homogeneous(pts1)

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
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Recover the relative pose (R and t)
            _, R, t, mask_recover = cv2.recoverPose(E, pts0, pts1, cameraMatrix=np.eye(3), mask=mask)

            return {
                "success": True,
                "E": E,
                "R": R,
                "t": t,
                "inliers": mask.reshape(-1).astype(bool),
                "num_inliers": int(mask_recover.sum()),
            }

        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shapes pts0: {pts0.shape}, pts1: {pts1.shape}"
            )
            return {
                "success": False,
                "E": None,
                "R": None,
                "t": None,
                "inliers": None,
                "num_inliers": 0,
            }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"solver='{self.solver}', "
            f"threshold={self.threshold}, "
            f"confidence={self.confidence}, "
            f"max_iters={self.max_iters})"
        )


class PycolmapRelativePoseEstimator(Estimator):
    """Estimate the relative pose between two images using Pycolmap.

    Args:
        threshold (float): Maximum reprojection error threshold. Default is 4.0.
        confidence (float): Confidence level. Default is 0.999.
        max_iters (int): Maximum number of iterations. Default is 1000.
    """

    def __init__(self, threshold: float = 4.0, confidence: float = 0.999, max_iters: int = 1000, **kwargs):
        super().__init__()

        # Options
        self.options = pycolmap.TwoViewGeometryOptions()
        self.options.ransac.max_error = threshold
        self.options.ransac.confidence = confidence
        self.options.ransac.max_num_trials = max_iters

    def estimate(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        camera0: Optional[Camera] = None,
        camera1: Optional[Camera] = None,
    ) -> Dict[str, Union[np.ndarray, int, bool]]:
        """Estimate the relative pose between two sets of 2D points.

        Args:
            pts0 (np.ndarray): 2D points in the first image, shape (N, 2).
            pts1 (np.ndarray): 2D points in the second image, shape (N, 2).
            camera0 (Optional[Camera]): Camera model of the first image.
            camera1 (Optional[Camera]): Camera model of the second image.

        Returns:
            Dict[str, Union[np.ndarray, int, bool]]: Dictionary containing the following
                - success (bool): True if the estimation was successful.
                - E (np.ndarray): Essential matrix.
                - R (np.ndarray): Rotation matrix.
                - t (np.ndarray): Translation vector.
                - inliers (np.ndarray): Inliers mask.
                - num_inliers (int): Number of inliers.
        """
        # validate inputs
        CHECK_TYPE(pts0, np.ndarray)
        CHECK_TYPE(pts1, np.ndarray)
        CHECK_SHAPE(pts0, (-1, 2))
        CHECK_SHAPE(pts1, (-1, 2))

        try:
            if pts0.shape[0] < 5:
                logger.warning(f"Number of correspondences is less than 5: {pts0.shape[0]}.")
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Estimate two-view geometry
            ret = pycolmap.two_view_geometry_estimation(pts0, pts1, camera0.todict(), camera1.todict(), self.options)

            if ret["success"] is False:
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Convert the quaternion to rotation matrix
            R = pycolmap.qvec_to_rotmat(ret["qvec"])
            t = ret["tvec"]

            inliers = ret["inliers"]
            num_inliers = len(inliers)

            # Essential matrix
            E = essential_from_Rt(R, t)

            return {
                "success": True,
                "E": E,
                "R": R,
                "t": t,
                "inliers": inliers,
                "num_inliers": num_inliers,
            }

        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shapes pts0: {pts0.shape}, pts1: {pts1.shape}"
            )
            return {
                "success": False,
                "E": None,
                "R": None,
                "t": None,
                "inliers": None,
                "num_inliers": 0,
            }


class PoseLibRelativePoseEstimator(Estimator):
    """Estimate the relative pose between two images using PoseLib.

    Args:
        threshold (float): Maximum epipolar error threshold. Default is 1.0.
        confidence (float): Confidence level. Default is 0.999.
        max_iters (int): Maximum number of iterations. Default is 1000.
    """

    def __init__(self, threshold: float = 1.0, confidence: float = 0.999, max_iters: int = 1000, **kwargs):
        super().__init__()

        self.threshold = threshold
        self.confidence = confidence
        self.max_iters = max_iters

        self.ransac_opt = {
            "max_iterations": max_iters,
            "success_prob": confidence,
            "max_epipolar_error": threshold,
        }

        self.bundle_opt = {}

    def estimate(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        camera0: Optional[Camera] = None,
        camera1: Optional[Camera] = None,
    ) -> Dict[str, Union[np.ndarray, int, bool]]:
        """Estimate the relative pose between two sets of 2D points.

        Args:
            pts0 (np.ndarray): 2D points in the first image, shape (N, 2).
            pts1 (np.ndarray): 2D points in the second image, shape (N, 2).
            camera0 (Optional[Camera]): Camera model of the first image.
            camera1 (Optional[Camera]): Camera model of the second image.

        Returns:
            Dict[str, Union[np.ndarray, int, bool]]: Dictionary containing the following
                - success (bool): True if the estimation was successful.
                - E (np.ndarray): Essential matrix.
                - R (np.ndarray): Rotation matrix.
                - t (np.ndarray): Translation vector.
                - inliers (np.ndarray): Inliers mask.
                - num_inliers (int): Number of inliers.
        """
        # validate inputs
        CHECK_TYPE(pts0, np.ndarray)
        CHECK_TYPE(pts1, np.ndarray)
        CHECK_SHAPE(pts0, (-1, 2))
        CHECK_SHAPE(pts1, (-1, 2))

        try:
            if pts0.shape[0] < 5:
                logger.warning(f"Number of correspondences is less than 5: {pts0.shape[0]}.")
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Estimate the relative pose
            ret, info = poselib.estimate_relative_pose(
                pts0, pts1, camera0.todict(), camera1.todict(), self.ransac_opt, self.bundle_opt
            )

            if ret is None:
                return {
                    "success": False,
                    "E": None,
                    "R": None,
                    "t": None,
                    "inliers": None,
                    "num_inliers": 0,
                }

            # Rt
            R = ret.R
            t = ret.t

            # Essential matrix
            E = essential_from_Rt(R, t)

            return {
                "success": True,
                "E": E,
                "R": R,
                "t": t,
                "inliers": info["inliers"],
                "num_inliers": info["num_inliers"],
            }

        except Exception as e:
            logger.error(
                f"Error in {self.__class__.__name__}: {e}, Input shapes pts0: {pts0.shape}, pts1: {pts1.shape}"
            )
            return {
                "success": False,
                "E": None,
                "R": None,
                "t": None,
                "inliers": None,
                "num_inliers": 0,
            }


class RelativePoseEstimator(Estimator):
    """Unified Relative Pose Estimator.

    Args:
        backend (str): Backend to use. Default is the available backend by priority.
        threshold (float): Maximum reprojection error threshold. Default is 1.0.
        confidence (float): Confidence level. Default is 0.999.
        max_iters (int): Maximum number of iterations. Default is 1000.
    """

    def __init__(
        self,
        backend: str = None,
        solver: str = "ransac",
        threshold: float = 1.0,
        confidence: float = 0.999,
        max_iters: int = 1000,
        **kwargs,
    ):
        super().__init__()

        # Choose the backend
        backend = backend if not None else get_backend()

        if backend == "opencv":
            self.estimator = OpenCVRelativePoseEstimator(
                solver="ransac", threshold=threshold, confidence=confidence, max_iters=max_iters
            )
        elif backend == "pycolmap":
            self.estimator = PycolmapRelativePoseEstimator(
                threshold=threshold, confidence=confidence, max_iters=max_iters
            )
        elif backend == "poselib":
            self.estimator = PoseLibRelativePoseEstimator(
                threshold=threshold, confidence=confidence, max_iters=max_iters
            )
        else:
            raise ValueError(f"Invalid backend: {backend}. Valid options are: ['opencv', 'pycolmap', 'poselib']")

    def estimate(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        camera0: Optional[Camera] = None,
        camera1: Optional[Camera] = None,
    ) -> Dict[str, Union[np.ndarray, int, bool]]:
        """Estimate the relative pose between two sets of 2D points.

        Args:
            pts0 (np.ndarray): 2D points in the first image, shape (N, 2).
            pts1 (np.ndarray): 2D points in the second image, shape (N, 2).
            camera0 (Optional[Camera]): Camera model of the first image.
            camera1 (Optional[Camera]): Camera model of the second image.

        Returns:
            Dict[str, Union[np.ndarray, int, bool]]: Dictionary containing the following
                - success (bool): True if the estimation was successful.
                - E (np.ndarray): Essential matrix.
                - R (np.ndarray): Rotation matrix.
                - t (np.ndarray): Translation vector.
                - inliers (np.ndarray): Inliers mask.
                - num_inliers (int): Number of inliers.

        """
        return self.estimator.estimate(pts0, pts1, camera0, camera1)

    def __repr__(self):
        return f"{self.__class__.__name__}(backend='{self.estimator.__class__.__name__}')"
