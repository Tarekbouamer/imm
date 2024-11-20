import numpy as np
import poselib
from loguru import logger

from imm.utils.check import CHECK_SHAPE, CHECK_TYPE

from .estimator import Estimator

try:
    import pycolmap
except ImportError:
    pycolmap = None


class PycolmapPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0):
        self.max_reproj_error = max_reproj_error

    def estimate(self, pts2d, pts3d, cam, **kwargs):
        try:
            CHECK_TYPE(pts2d, np.ndarray)
            CHECK_TYPE(pts3d, np.ndarray)

            # Shape
            CHECK_SHAPE(pts2d, (-1, 2))
            CHECK_SHAPE(pts3d, (-1, 3))

            ret = pycolmap.absolute_pose_estimation(pts2d, pts3d, cam)

        except Exception as e:
            logger.error(f"Error in PycolmapPnPEstimator: {e}")
            ret = None

        return ret

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"max_reproj_error={self.max_reproj_error})"


class PoseLibPnPEstimator(Estimator):
    def __init__(self, max_reproj_error=12.0, max_epipolar_error=1.0, max_iterations=100):
        self.max_reproj_error = max_reproj_error
        self.max_epipolar_error = max_epipolar_error
        self.max_iterations = max_iterations

    def estimate(self, pts2d, pts3d, cam, **kwargs):
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
            cam,
            **ransac_options,
            **bundle_options,
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
