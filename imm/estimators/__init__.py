from .homography import OpenCVHomographyEstimator, PoseLibHomographyEstimator, PycolmapHomographyEstimator, CV_H_SOLVERS  # noqa F401
from .pnp import OpenCVPnPEstimator, PoseLibPnPEstimator, PycolmapPnPEstimator

ESTIMATORS_2D = ["homography"]
ESTIMATORS_3D = ["pnp"]


def create_homography_estimator(
    backend: str,
    solver: str = "ransac",
    inlier_threshold: float = 4.0,
    max_iters: int = 10000,
    confidence: float = 0.9998,
):
    """Create a homography estimator."""

    if backend == "opencv":
        return OpenCVHomographyEstimator(
            solver=solver, inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence
        )
    elif backend == "poselib":
        return PoseLibHomographyEstimator(inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence)
    elif backend == "pycolmap":
        return PycolmapHomographyEstimator(
            inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence
        )
    else:
        raise ValueError(f"Unknown homography estimator: {backend}", available=["opencv", "poselib", "pycolmap"])


def create_pnp_estimator(
    backend: str, max_reproj_error: float = 12.0, max_epipolar_error: float = 1.0, max_iterations: int = 100
):
    """Create a PnP estimator."""
    if backend == "poselib":
        return PoseLibPnPEstimator(
            max_reproj_error=max_reproj_error, max_epipolar_error=max_epipolar_error, max_iterations=max_iterations
        )
    elif backend == "pycolmap":
        return PycolmapPnPEstimator(max_reproj_error=max_reproj_error)

    elif backend == "opencv":
        return OpenCVPnPEstimator(max_reproj_error=max_reproj_error)
    else:
        raise ValueError(f"Unknown PnP estimator: {backend}", available=["poselib", "pycolmap", "opencv"])
