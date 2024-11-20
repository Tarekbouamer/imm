from .homography import CvHomographyEstimator, PoseLibHomographyEstimator, CV_H_SOLVERS  # noqa F401
from .pnp import PoseLibPnPEstimator, PycolmapPnPEstimator

ESTIMATORS_2D = ["homography"]
ESTIMATORS_3D = ["pnp"]


def create_homography_estimator(backend: str, solver: str, thd: float, max_iters: int, confidence: float):
    """Create a homography estimator."""

    if backend == "cv":
        return CvHomographyEstimator(solver, thd, max_iters, confidence)
    elif backend == "poselib":
        return PoseLibHomographyEstimator(solver, thd, max_iters, confidence)
    else:
        raise ValueError(f"Unknown homography estimator: {backend}", available=["cv", "poselib"])


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
    else:
        raise ValueError(f"Unknown PnP estimator: {backend}", available=["poselib", "pycolmap"])
