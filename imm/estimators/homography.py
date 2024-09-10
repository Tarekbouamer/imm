from .estimator import Estimator
import poselib
import cv2

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
    default_cfg = {
        "ransac_th": 0.5,
        "options": {"solver": "ransac", "max_iters": 1, "confidence": 0.998},
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def estimate(self, pts0, pts1, **kwargs):
        H, mask = cv2.findHomography(
            pts0,
            pts1,
            method=CV_H_SOLVERS[self.cfg.options.solver],
            ransacReprojThreshold=self.cfg.ransac_th,
            maxIters=self.cfg.options.max_iters,
            confidence=self.cfg.options.confidence,
        )

        result = {
            "H": H,
            "success": H is not None,
            "inliers": int(mask.sum()) if mask is not None else 0,
        }

        return result


class PoseLibHomographyEstimator(Estimator):
    default_cfg = {"ransac_th": 2.0, "options": {}}

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def estimate(self, pts0, pts1, **kwargs):
        H, status = poselib.estimate_homography(
            pts0,
            pts1,
            {
                "max_reproj_error": self.cfg.ransac_th,
            },
        )

        result = {
            "H": H,
            "success": H is not None,
            **status,
        }

        return result
