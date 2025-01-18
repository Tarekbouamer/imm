from typing import Tuple

import numpy as np
import pytest

from imm.estimators.fundamental import (
    CV_F_SOLVERS,
    FundamentalEstimator,
    OpenCVFundamentalEstimator,
    PoseLibFundamentalEstimator,
    PycolmapFundamentalEstimator,
)


def generate_synthetic_data(num_points: int = 8, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D point correspondences and a ground truth fundamental matrix.
    """
    F_gt = np.random.rand(3, 3)
    F_gt = F_gt / np.linalg.norm(F_gt)

    pts0 = np.random.rand(num_points, 2) * 100
    pts0_h = np.hstack((pts0, np.ones((num_points, 1))))
    pts1_h = F_gt @ pts0_h.T
    pts1_h = pts1_h.T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:]

    pts0 += np.random.normal(0, noise_std, pts0.shape)
    pts1 += np.random.normal(0, noise_std, pts1.shape)

    return pts0, pts1, F_gt


@pytest.mark.parametrize("solver", CV_F_SOLVERS)
def test_opencv_solvers(solver: str):
    """
    Test all OpenCV solvers for fundamental matrix estimation.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = OpenCVFundamentalEstimator(solver=solver, inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], f"Estimation failed for solver: {solver}"
    assert result["F"] is not None, f"Fundamental matrix is None for solver: {solver}"
    assert result["num_inliers"] > 0, f"No inliers found for solver: {solver}"


def test_poselib_fundamental_estimator():
    """
    Test the PoseLib backend for fundamental matrix estimation.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PoseLibFundamentalEstimator(inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["F"] is not None, "Fundamental matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_pycolmap_fundamental_estimator():
    """
    Test the Pycolmap backend for fundamental matrix estimation.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = PycolmapFundamentalEstimator(inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["F"] is not None, "Fundamental matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_opencv():
    """
    Test the unified estimator with the OpenCV backend.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = FundamentalEstimator(backend="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["F"] is not None, "Fundamental matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_poselib():
    """
    Test the unified estimator with the PoseLib backend.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = FundamentalEstimator(backend="poselib", inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["F"] is not None, "Fundamental matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_unified_estimator_pycolmap():
    """
    Test the unified estimator with the Pycolmap backend.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=100, noise_std=0.1)

    estimator = FundamentalEstimator(backend="pycolmap", inlier_threshold=2.0, max_iters=1000)
    result = estimator.estimate(pts0, pts1)

    assert result["success"], "Estimation failed"
    assert result["F"] is not None, "Fundamental matrix is None"
    assert result["num_inliers"] > 0, "No inliers found"


def test_insufficient_points():
    """
    Test the case where insufficient points are provided.
    """
    pts0, pts1, F_gt = generate_synthetic_data(num_points=4, noise_std=0.1)

    estimator = FundamentalEstimator(backend="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert not result["success"], "Estimation should fail with insufficient points"
    assert result["F"] is None, "Fundamental matrix should be None"


def test_degenerate_case():
    """
    Test the case where all points are collinear (degenerate case).
    """
    pts0 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    pts1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

    estimator = FundamentalEstimator(backend="opencv", solver="ransac", inlier_threshold=0.5)
    result = estimator.estimate(pts0, pts1)

    assert not result["success"], "Estimation should fail for degenerate case"
    assert result["F"] is None, "Fundamental matrix should be None"
