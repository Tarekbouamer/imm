from loguru import logger

from imm.estimators.relative_pose import (
    OpenCVRelativePoseEstimator,
    PoseLibRelativePoseEstimator,
    PycolmapRelativePoseEstimator,
)

from .homography import (
    CV_H_SOLVERS,  # noqa
    OpenCVHomographyEstimator,
    PoseLibHomographyEstimator,
    PycolmapHomographyEstimator,
)
from .pnp import OpenCVPnPEstimator, PoseLibPnPEstimator, PycolmapPnPEstimator


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


def create_relative_pose_estimator(
    backend: str,
    solver: str = "ransac",
    threshold: float = 1.0,
    confidence: float = 0.999,
    max_iters: int = 1000,
    **kwargs,
):
    """Create a relative pose estimator with the specified backend."""

    if backend == "opencv":
        estimator = OpenCVRelativePoseEstimator(
            solver=solver, threshold=threshold, confidence=confidence, max_iters=max_iters, **kwargs
        )
    elif backend == "poselib":
        estimator = PoseLibRelativePoseEstimator(
            threshold=threshold, confidence=confidence, max_iters=max_iters, **kwargs
        )
    elif backend == "pycolmap":
        estimator = PycolmapRelativePoseEstimator(
            threshold=threshold, confidence=confidence, max_iters=max_iters, **kwargs
        )
    else:
        raise ValueError(f"Unknown relative pose estimator: {backend}", available=["opencv", "poselib", "pycolmap"])

    logger.info(f"Created relative pose estimator: {estimator}")
    return estimator
