from .estimator import Estimator
import poselib

from imm.utils.device import to_numpy
import pycolmap


class PycolmapPnPEstimator(Estimator):
    default_cfg = {
        "ransac": {
            "max_error": 12.0,
        },
        "refinement": {},
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def estimate(self, pts2d, pts3d, cam, **kwargs):
        # convert to numpy array
        pts2d = to_numpy(pts2d)
        pts3d = to_numpy(pts3d)

        pts2d = pts2d.reshape(-1, 2)
        pts3d = pts3d.reshape(-1, 3)

        ret = pycolmap.absolute_pose_estimation(pts2d, pts3d, cam)

        return ret


class PoseLibPnPEstimator(Estimator):
    default_cfg = {
        "ransac": {
            "max_reproj_error": 12.0,
            "max_epipolar_error": 1.0,
        },
        "bundle": {"max_iterations": 100},
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def estimate(self, pts2d, pts3d, cam, **kwargs):
        pts2d = pts2d.reshape(-1, 2)
        pts3d = pts3d.reshape(-1, 3)

        pose, info = poselib.estimate_absolute_pose(
            pts2d,
            pts3d,
            cam,
            {
                "max_reproj_error": self.cfg.ransac["max_reproj_error"],
                "max_epipolar_error": self.cfg.ransac["max_epipolar_error"],
            },
            {
                "max_iterations": self.cfg.bundle["max_iterations"],
            },
        )

        result = {
            "qvec": pose.q,
            "tvec": pose.t,
            "success": pose is not None,
            "inliers": info["inliers"],
        }

        return result
