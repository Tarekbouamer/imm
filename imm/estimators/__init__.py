from loguru import logger

from .homography import (  # noqa F401
    CV_H_SOLVERS,
    OpenCVHomographyEstimator,
    PoseLibHomographyEstimator,
    PycolmapHomographyEstimator,
)
from .pnp import OpenCVPnPEstimator, PoseLibPnPEstimator, PycolmapPnPEstimator

ESTIMATORS_2D = ["homography"]
ESTIMATORS_3D = ["pnp"]


def create_homography_estimator(
    backend: str,
    solver: str = "ransac",
    inlier_threshold: float = 4.0,
    max_iters: int = 10000,
    confidence: float = 0.9998,
    **kwargs,
):
    "Create a homography estimator with the specified backend." ""

    if backend == "opencv":
        estimator = OpenCVHomographyEstimator(
            solver=solver, inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence, **kwargs
        )
    elif backend == "poselib":
        estimator = PoseLibHomographyEstimator(
            inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence, **kwargs
        )
    elif backend == "pycolmap":
        estimator = PycolmapHomographyEstimator(
            inlier_threshold=inlier_threshold, max_iters=max_iters, confidence=confidence, **kwargs
        )
    else:
        raise ValueError(f"Unknown homography estimator: {backend}", available=["opencv", "poselib", "pycolmap"])

    logger.info(f"Created homography estimator: {estimator}")
    return estimator


def create_pnp_estimator(
    backend: str, max_reproj_error: float = 12.0, max_epipolar_error: float = 1.0, max_iterations: int = 100, **kwargs
):
    """Create a PnP estimator with the specified backend."""

    # TODO : Verify All the available backends and typing of the parameters

    if backend == "poselib":
        estimator = PoseLibPnPEstimator(
            max_reproj_error=max_reproj_error,
            max_epipolar_error=max_epipolar_error,
            max_iterations=max_iterations,
            **kwargs,
        )
    elif backend == "pycolmap":
        estimator = PycolmapPnPEstimator(max_reproj_error=max_reproj_error, **kwargs)

    elif backend == "opencv":
        estimator = OpenCVPnPEstimator(max_reproj_error=max_reproj_error, **kwargs)
    else:
        raise ValueError(f"Unknown PnP estimator: {backend}", available=["poselib", "pycolmap", "opencv"])

    logger.info(f"Created PnP estimator: {estimator}")
    return estimator
