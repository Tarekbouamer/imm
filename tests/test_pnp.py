from typing import Tuple

import numpy as np
import pytest

from imm.estimators._camera import Camera
from imm.estimators.pnp import (
    OpenCVPnPEstimator,
    PnPEstimator,
    PoseLibPnPEstimator,
    PycolmapPnPEstimator,
)


def generate_synthetic_data(
    num_points: int = 8, noise_std: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Camera]:
    """
    Generate synthetic 2D-3D point correspondences and a ground truth pose.
    """
    # Ground truth rotation and translation
    R_gt = np.random.rand(3, 3)
    U, _, Vt = np.linalg.svd(R_gt)
    R_gt = U @ Vt  # Ensure R is a valid rotation matrix
    t_gt = np.random.rand(3, 1)

    # Generate random 3D points
    points_3d = np.random.rand(num_points, 3) * 1

    # Create a camera
    camera_dict = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [800, 800, 320, 240],  # [fx, fy, cx, cy]
    }
    camera = Camera.from_dict(camera_dict)

    # Project 3D points into 2D using the ground truth pose
    points_3d_homogeneous = np.hstack((points_3d, np.ones((num_points, 1))))
    transformation_matrix = np.hstack((R_gt, t_gt))
    points_3d_transformed = (transformation_matrix @ points_3d_homogeneous.T).T
    pts2d, _ = camera.project(points_3d_transformed[:, :3])

    # Add Gaussian noise to the 2D points
    pts2d += np.random.normal(0, noise_std, pts2d.shape)

    return pts2d, points_3d, R_gt, t_gt, camera


def test_opencv_pnp_estimator() -> None:
    """
    Test the OpenCV PnP estimator.
    """
    pts2d, pts3d, R_gt, t_gt, camera = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = OpenCVPnPEstimator(max_reproj_error=12.0)
    result = estimator.estimate(pts2d, pts3d, camera)

    assert result["success"], "Estimation failed"
    assert result["qvec"] is not None, "Rotation quaternion is None"
    assert result["tvec"] is not None, "Translation vector is None"


def test_pycolmap_pnp_estimator() -> None:
    """
    Test the Pycolmap PnP estimator.
    """
    pts2d, pts3d, R_gt, t_gt, camera = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PycolmapPnPEstimator(max_reproj_error=12.0)
    result = estimator.estimate(pts2d, pts3d, camera)

    assert result["success"], "Estimation failed"
    assert result["qvec"] is not None, "Rotation quaternion is None"
    assert result["tvec"] is not None, "Translation vector is None"


def test_poselib_pnp_estimator() -> None:
    """
    Test the PoseLib PnP estimator.
    """
    pts2d, pts3d, R_gt, t_gt, camera = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PoseLibPnPEstimator(max_reproj_error=12.0, max_epipolar_error=1.0, max_iterations=100)
    result = estimator.estimate(pts2d, pts3d, camera)

    assert result["success"], "Estimation failed"
    assert result["qvec"] is not None, "Rotation quaternion is None"
    assert result["tvec"] is not None, "Translation vector is None"


def test_insufficient_points() -> None:
    """
    Test the case where insufficient points are provided.
    """
    pts2d, pts3d, R_gt, t_gt, camera = generate_synthetic_data(num_points=3, noise_std=0.1)

    estimator = PnPEstimator(backend="opencv", max_reproj_error=12.0)
    result = estimator.estimate(pts2d, pts3d, camera)

    assert not result["success"], "Estimation should fail with insufficient points"
    assert result["qvec"] is None, "Rotation quaternion should be None"
    assert result["tvec"] is None, "Translation vector should be None"


def test_degenerate_case() -> None:
    """
    Test the case where all points are collinear (degenerate case).
    """
    pts2d = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    pts3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])

    camera_dict = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [800, 800, 320, 240],  # [fx, fy, cx, cy]
    }
    camera = Camera.from_dict(camera_dict)

    estimator = PnPEstimator(backend="opencv", max_reproj_error=12.0)
    result = estimator.estimate(pts2d, pts3d, camera)

    assert not result["success"], "Estimation should fail for degenerate case"
    assert result["qvec"] is None, "Rotation quaternion should be None"
    assert result["tvec"] is None, "Translation vector should be None"


def test_invalid_backend() -> None:
    """
    Test the case where an invalid backend is provided.
    """
    with pytest.raises(ValueError) as exc_info:
        PnPEstimator(backend="invalid_backend")
    assert (
        str(exc_info.value) == "Invalid backend: invalid_backend. Valid options are: ['opencv', 'pycolmap', 'poselib']"
    ), "Error message does not match"
