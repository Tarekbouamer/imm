from typing import Tuple

import numpy as np
import pytest

from imm.estimators._camera import Camera
from imm.estimators.relative_pose import (
    CV_RP_SOLVERS,
    OpenCVRelativePoseEstimator,
    PoseLibRelativePoseEstimator,
    PycolmapRelativePoseEstimator,
    RelativePoseEstimator,
)


def generate_synthetic_data(
    num_points: int = 8, noise_std: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D point correspondences, ground truth rotation, and translation.
    """
    # Ground truth rotation and translation
    R_gt = np.random.rand(3, 3)
    U, _, Vt = np.linalg.svd(R_gt)
    R_gt = U @ Vt  # Ensure R is a valid rotation matrix
    t_gt = np.random.rand(3, 1)

    # Generate random 3D points
    points_3d = np.random.rand(num_points, 3) * 1

    # Create Cameras
    camera0_dict = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [640, 480, 320, 240],
    }
    camera1_dict = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [640, 480, 320, 240],
    }

    camera0 = Camera.from_dict(camera0_dict)
    camera1 = Camera.from_dict(camera1_dict)

    # Project points
    pts0, msk0 = camera0.project(points_3d)
    pts1, msk1 = camera1.project(points_3d @ R_gt.T + t_gt.T)

    pts0 = pts0[msk1]

    # Add Gaussian noise to the points
    pts0 += np.random.normal(0, noise_std, pts0.shape)
    pts1 += np.random.normal(0, noise_std, pts1.shape)

    return pts0, pts1, R_gt, t_gt, camera0, camera1


@pytest.mark.parametrize("solver", CV_RP_SOLVERS.keys())
def test_opencv_solvers(solver: str) -> None:
    """
    Test all OpenCV solvers for relative pose estimation.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = OpenCVRelativePoseEstimator(solver=solver, threshold=1.0, confidence=0.999, max_iters=1000)
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], f"Estimation failed for solver: {solver}"
    assert result["E"] is not None, f"Essential matrix is None for solver: {solver}"
    assert result["R"] is not None, f"Rotation matrix is None for solver: {solver}"
    assert result["t"] is not None, f"Translation vector is None for solver: {solver}"
    assert result["num_inliers"] > 0, f"No inliers found for solver: {solver}"


def test_poselib_relative_pose_estimator() -> None:
    """
    Test the PoseLib backend for relative pose estimation.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PoseLibRelativePoseEstimator(threshold=1.0, confidence=0.999, max_iters=1000)
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], "Estimation failed"
    assert result["E"] is not None, "Essential matrix is None"
    assert result["R"] is not None, "Rotation matrix is None"
    assert result["t"] is not None, "Translation vector is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_pycolmap_relative_pose_estimator() -> None:
    """
    Test the Pycolmap backend for relative pose estimation.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PycolmapRelativePoseEstimator(threshold=1.0, confidence=0.999, max_iters=1000)
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], "Estimation failed"
    assert result["E"] is not None, "Essential matrix is None"
    assert result["R"] is not None, "Rotation matrix is None"
    assert result["t"] is not None, "Translation vector is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_opencv() -> None:
    """
    Test the unified estimator with the OpenCV backend.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = RelativePoseEstimator(
        backend="opencv", solver="ransac", threshold=1.0, confidence=0.999, max_iters=1000
    )
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], "Estimation failed"
    assert result["E"] is not None, "Essential matrix is None"
    assert result["R"] is not None, "Rotation matrix is None"
    assert result["t"] is not None, "Translation vector is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_poselib() -> None:
    """
    Test the unified estimator with the PoseLib backend.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = RelativePoseEstimator(backend="poselib", threshold=1.0, confidence=0.999, max_iters=1000)
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], "Estimation failed"
    assert result["E"] is not None, "Essential matrix is None"
    assert result["R"] is not None, "Rotation matrix is None"
    assert result["t"] is not None, "Translation vector is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_pycolmap() -> None:
    """
    Test the unified estimator with the Pycolmap backend.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = RelativePoseEstimator(backend="pycolmap", threshold=1.0, confidence=0.999, max_iters=1000)
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert result["success"], "Estimation failed"
    assert result["E"] is not None, "Essential matrix is None"
    assert result["R"] is not None, "Rotation matrix is None"
    assert result["t"] is not None, "Translation vector is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_insufficient_points() -> None:
    """
    Test the case where insufficient points are provided.
    """
    pts0, pts1, R_gt, t_gt, camera0, camera1 = generate_synthetic_data(num_points=4, noise_std=0.1)

    estimator = RelativePoseEstimator(
        backend="opencv", solver="ransac", threshold=1.0, confidence=0.999, max_iters=1000
    )
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert not result["success"], "Estimation should fail with insufficient points"
    assert result["E"] is None, "Essential matrix should be None"
    assert result["R"] is None, "Rotation matrix should be None"
    assert result["t"] is None, "Translation vector should be None"


def test_degenerate_case() -> None:
    """
    Test the case where all points are collinear (degenerate case).
    """
    pts0 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    pts1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    camera0 = {
        "model": "SIMPLE_PINHOLE",
        "width": 640,
        "height": 480,
        "params": [800, 320, 240],
    }
    camera1 = {
        "model": "SIMPLE_PINHOLE",
        "width": 640,
        "height": 480,
        "params": [800, 320, 240],
    }

    estimator = RelativePoseEstimator(
        backend="opencv", solver="ransac", threshold=1.0, confidence=0.999, max_iters=1000
    )
    result = estimator.estimate(pts0, pts1, camera0, camera1)

    assert not result["success"], "Estimation should fail for degenerate case"
    assert result["E"] is None, "Essential matrix should be None"
    assert result["R"] is None, "Rotation matrix should be None"
    assert result["t"] is None, "Translation vector should be None"


def test_invalid_backend() -> None:
    """
    Test the case where an invalid backend is provided.
    """
    with pytest.raises(ValueError) as exc_info:
        RelativePoseEstimator(backend="invalid_backend")
    assert (
        str(exc_info.value) == "Invalid backend: invalid_backend. Valid options are: ['opencv', 'pycolmap', 'poselib']"
    ), "Error message does not match"
