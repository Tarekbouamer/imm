from typing import Tuple

import numpy as np
import pytest

from imm.estimators.homography import (
    CV_H_SOLVERS,
    HomographyEstimator,
    OpenCVHomographyEstimator,
    PoseLibHomographyEstimator,
    PycolmapHomographyEstimator,
)


def generate_synthetic_data(num_points: int = 8, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D point correspondences and a ground truth homography matrix.
    """
    H_gt = np.random.rand(3, 3)
    H_gt = H_gt / np.linalg.norm(H_gt)

    pts0 = np.random.rand(num_points, 2) * 100
    pts0_h = np.hstack((pts0, np.ones((num_points, 1))))
    pts1_h = H_gt @ pts0_h.T
    pts1_h = pts1_h.T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:]

    pts0 += np.random.normal(0, noise_std, pts0.shape)
    pts1 += np.random.normal(0, noise_std, pts1.shape)

    return pts0, pts1, H_gt


@pytest.mark.parametrize(
    "solver",
    CV_H_SOLVERS.keys(),
)
def test_opencv_solvers(solver: str):
    """
    Test all OpenCV solvers for homography estimation.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = OpenCVHomographyEstimator(solver=solver, inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], f"Estimation failed for solver: {solver}"
    assert result["H"] is not None, f"Homography matrix is None for solver: {solver}"
    assert result["num_inliers"] > 0, f"No inliers found for solver: {solver}"


def test_poselib_homography_estimator():
    """
    Test the PoseLib backend for homography estimation.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PoseLibHomographyEstimator(inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["H"] is not None, "Homography matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_pycolmap_homography_estimator():
    """
    Test the Pycolmap backend for homography estimation.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PycolmapHomographyEstimator(inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["H"] is not None, "Homography matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_opencv():
    """
    Test the unified estimator with the OpenCV backend.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = HomographyEstimator(method="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["H"] is not None, "Homography matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_poselib():
    """
    Test the unified estimator with the PoseLib backend.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = HomographyEstimator(method="poselib", inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["H"] is not None, "Homography matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_pycolmap():
    """
    Test the unified estimator with the Pycolmap backend.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = HomographyEstimator(method="pycolmap", inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["H"] is not None, "Homography matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_insufficient_points():
    """
    Test the case where insufficient points are provided.
    """
    pts0, pts1, H_gt = generate_synthetic_data(num_points=3, noise_std=0.1)

    estimator = HomographyEstimator(method="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert not result["success"], "Estimation should fail with insufficient points"
    assert result["H"] is None, "Homography matrix should be None"


def test_degenerate_case():
    """
    Test the case where all points are collinear (degenerate case).
    """
    pts0 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    pts1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

    estimator = HomographyEstimator(method="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert not result["success"], "Estimation should fail for degenerate case"
    assert result["H"] is None, "Homography matrix should be None"
