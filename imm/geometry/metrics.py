
from typing import Dict, Optional, Tuple

import numpy as np


def pose_error(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Tuple[float, float]:
    """Compute rotation and translation errors in degrees.

    Returns: (rotation_error_deg, translation_error_deg)
    """
    # Validate shapes
    assert R_est.shape == (3, 3), f"R_est must be (3, 3), got {R_est.shape}"
    assert R_gt.shape == (3, 3), f"R_gt must be (3, 3), got {R_gt.shape}"
    t_est = np.asarray(t_est).reshape(-1)
    t_gt = np.asarray(t_gt).reshape(-1)
    assert len(t_est) == 3, f"t_est must have 3 elements, got {len(t_est)}"
    assert len(t_gt) == 3, f"t_gt must have 3 elements, got {len(t_gt)}"

    # Rotation error
    R_err = R_gt.T @ R_est
    trace = np.trace(R_err)
    # Clamp to handle numerical errors
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error_rad = np.arccos(cos_angle)
    rot_error_deg = np.degrees(rot_error_rad)

    # Translation direction error (angular error)
    t_est_normalized = t_est.ravel() / (np.linalg.norm(t_est) + 1e-8)
    t_gt_normalized = t_gt.ravel() / (np.linalg.norm(t_gt) + 1e-8)

    cos_t_angle = np.clip(np.dot(t_est_normalized, t_gt_normalized), -1.0, 1.0)
    # abs to handle sign ambiguity
    trans_error_rad = np.arccos(np.abs(cos_t_angle))
    trans_error_deg = np.degrees(trans_error_rad)

    return rot_error_deg, trans_error_deg


def epipolar_error(
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    method: str = "sampson",
) -> np.ndarray:
    """Compute epipolar distance. method: 'sampson' or 'symmetric'."""
    # Validate shapes
    assert F.shape == (3, 3), f"F must be (3, 3), got {F.shape}"
    assert pts1.ndim == 2 and pts1.shape[
        1] == 2, f"pts1 must be (N, 2), got {pts1.shape}"
    assert pts2.ndim == 2 and pts2.shape[
        1] == 2, f"pts2 must be (N, 2), got {pts2.shape}"
    assert pts1.shape == pts2.shape, f"pts1 and pts2 must have same shape, got {pts1.shape} vs {pts2.shape}"

    # Convert to homogeneous coordinates
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

    if method == "sampson":
        # Sampson distance (first-order geometric approximation)
        Fx1 = (F @ pts1_h.T).T
        FTx2 = (F.T @ pts2_h.T).T

        numerator = np.sum(pts2_h * Fx1, axis=1) ** 2
        denominator = (
            Fx1[:, 0]**2 + Fx1[:, 1]**2 +
            FTx2[:, 0]**2 + FTx2[:, 1]**2 + 1e-8
        )
        return numerator / denominator

    elif method == "symmetric":
        # Symmetric epipolar distance
        Fx1 = (F @ pts1_h.T).T
        FTx2 = (F.T @ pts2_h.T).T

        algebraic_dist = np.sum(pts2_h * Fx1, axis=1)
        dist1 = np.abs(algebraic_dist) / \
            (np.linalg.norm(Fx1[:, :2], axis=1) + 1e-8)

        algebraic_dist2 = np.sum(pts1_h * FTx2, axis=1)
        dist2 = np.abs(algebraic_dist2) / \
            (np.linalg.norm(FTx2[:, :2], axis=1) + 1e-8)

        return np.sqrt(dist1**2 + dist2**2)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'sampson' or 'symmetric'.")


def reprojection_error(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compute reprojection error for 3D-2D correspondences."""
    # Validate shapes
    assert pts_2d.ndim == 2 and pts_2d.shape[
        1] == 2, f"pts_2d must be (N, 2), got {pts_2d.shape}"
    assert pts_3d.ndim == 2 and pts_3d.shape[
        1] == 3, f"pts_3d must be (N, 3), got {pts_3d.shape}"
    assert pts_2d.shape[0] == pts_3d.shape[
        0], f"pts_2d and pts_3d must have same N, got {pts_2d.shape[0]} vs {pts_3d.shape[0]}"
    assert K.shape == (3, 3), f"K must be (3, 3), got {K.shape}"
    assert R.shape == (3, 3), f"R must be (3, 3), got {R.shape}"
    t = np.asarray(t).reshape(-1)
    assert len(t) == 3, f"t must have 3 elements, got {len(t)}"

    t = t.reshape(3, 1)

    # Project 3D points
    pts_3d_cam = (R @ pts_3d.T + t).T  # (N, 3)
    pts_2d_proj = (K @ pts_3d_cam.T).T  # (N, 3)

    # Convert to Euclidean coordinates
    pts_2d_proj = pts_2d_proj[:, :2] / (pts_2d_proj[:, 2:3] + 1e-8)

    # Compute error
    errors = np.linalg.norm(pts_2d - pts_2d_proj, axis=1)

    return errors


def homography_error(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Compute reprojection error under homography."""
    # Validate shapes
    assert H.shape == (3, 3), f"H must be (3, 3), got {H.shape}"
    assert pts1.ndim == 2 and pts1.shape[
        1] == 2, f"pts1 must be (N, 2), got {pts1.shape}"
    assert pts2.ndim == 2 and pts2.shape[
        1] == 2, f"pts2 must be (N, 2), got {pts2.shape}"
    assert pts1.shape == pts2.shape, f"pts1 and pts2 must have same shape, got {pts1.shape} vs {pts2.shape}"

    # Convert to homogeneous
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])

    # Transform
    pts2_pred = (H @ pts1_h.T).T
    pts2_pred = pts2_pred[:, :2] / (pts2_pred[:, 2:3] + 1e-8)

    # Error
    return np.linalg.norm(pts2 - pts2_pred, axis=1)


def compute_auc(
    errors: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    max_threshold: float = 10.0,
    num: int = 100,
) -> float:
    """Compute AUC of inlier rate vs threshold.
    """
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)

    if thresholds is None:
        thresholds = np.linspace(0.0, float(
            max_threshold), int(num), dtype=np.float64)
        norm = float(max_threshold)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if thresholds.size < 2:
            raise ValueError("thresholds must contain at least 2 values")
        thresholds = np.sort(thresholds)
        norm = float(thresholds[-1] - thresholds[0])
        if norm <= 0:
            raise ValueError("thresholds must span a positive range")

    # Vectorized inlier-rate curve
    accuracies = (errors[:, None] <= thresholds[None, :]).mean(axis=0)

    auc = float(np.trapz(accuracies, thresholds) / norm)
    return auc


def pose_auc(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    max_rot_error: float = 10.0,
    max_trans_error: float = 10.0,
) -> Dict[str, float]:
    """Compute AUC for pose errors."""
    # Validate shapes
    assert R_est.shape == (3, 3), f"R_est must be (3, 3), got {R_est.shape}"
    assert R_gt.shape == (3, 3), f"R_gt must be (3, 3), got {R_gt.shape}"
    t_est_flat = np.asarray(t_est).reshape(-1)
    t_gt_flat = np.asarray(t_gt).reshape(-1)
    assert len(
        t_est_flat) == 3, f"t_est must have 3 elements, got {len(t_est_flat)}"
    assert len(
        t_gt_flat) == 3, f"t_gt must have 3 elements, got {len(t_gt_flat)}"

    rot_err, trans_err = pose_error(R_est, t_est, R_gt, t_gt)

    # For single pose, create threshold arrays
    rot_thresholds = np.linspace(0, max_rot_error, 100)
    trans_thresholds = np.linspace(0, max_trans_error, 100)

    # Compute "AUC" as binary accuracy integrated over thresholds
    rot_accuracies = (rot_err <= rot_thresholds).astype(float)
    trans_accuracies = (trans_err <= trans_thresholds).astype(float)

    rot_auc = np.trapz(rot_accuracies, rot_thresholds) / max_rot_error
    trans_auc = np.trapz(trans_accuracies, trans_thresholds) / max_trans_error

    # Combined: both rotation and translation below threshold
    combined_accuracies = rot_accuracies * trans_accuracies
    combined_auc = np.trapz(combined_accuracies,
                            rot_thresholds) / max_rot_error

    return {
        "rotation_auc": rot_auc,
        "translation_auc": trans_auc,
        "combined_auc": combined_auc,
        "rotation_error": rot_err,
        "translation_error": trans_err,
    }
