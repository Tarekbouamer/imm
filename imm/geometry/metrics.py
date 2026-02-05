
from typing import Dict, Optional, Tuple

import numpy as np


def pose_error(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute pose error between estimated and ground truth pose.

    Args:
        R_est: Estimated rotation matrix (3, 3).
        t_est: Estimated translation vector (3,) or (3, 1).
        R_gt: Ground truth rotation matrix (3, 3).
        t_gt: Ground truth translation vector (3,) or (3, 1).

    Returns:
        Tuple of (rotation_error_degrees, translation_error_degrees).
            - rotation_error: Angular error in degrees
            - translation_error: Angular error of translation direction in degrees
    """
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
    """
    Compute epipolar distance/error for point correspondences.

    Args:
        F: Fundamental matrix (3, 3).
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).
        method: "sampson" or "symmetric" epipolar distance.

    Returns:
        Epipolar errors (N,).
    """
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
    """
    Compute reprojection error for 3D-2D correspondences.

    Args:
        pts_2d: 2D points (N, 2).
        pts_3d: 3D points (N, 3).
        K: Camera intrinsic matrix (3, 3).
        R: Rotation matrix (3, 3).
        t: Translation vector (3,) or (3, 1).

    Returns:
        Reprojection errors (N,).
    """
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
    """
    Compute reprojection error under homography.

    Args:
        H: Homography matrix (3, 3).
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).

    Returns:
        Reprojection errors (N,).
    """
    # Convert to homogeneous
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])

    # Transform
    pts2_pred = (H @ pts1_h.T).T
    pts2_pred = pts2_pred[:, :2] / (pts2_pred[:, 2:3] + 1e-8)

    # Error
    return np.linalg.norm(pts2 - pts2_pred, axis=1)


def compute_auc(errors: np.ndarray, thresholds: Optional[np.ndarray] = None, max_threshold: float = 10.0) -> float:
    """
    Compute Area Under Curve (AUC) for error distribution.

    AUC measures the fraction of samples with error below threshold, integrated over thresholds.

    Args:
        errors: Error values (N,).
        thresholds: Threshold values for computing curve. If None, uses linspace from 0 to max_threshold.
        max_threshold: Maximum threshold value if thresholds is None.

    Returns:
        AUC value (normalized to [0, 1]).
    """
    if thresholds is None:
        thresholds = np.linspace(0, max_threshold, 100)

    # Compute fraction of errors below each threshold
    accuracies = np.array([np.mean(errors <= t) for t in thresholds])

    # Compute AUC using trapezoidal rule, normalized by threshold range
    auc = np.trapz(accuracies, thresholds) / max_threshold

    return auc



def compute_precision_recall_curve(
    errors: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve for errors.

    Args:
        errors: Error values (N,).
        thresholds: Threshold values. If None, uses unique sorted error values.

    Returns:
        Tuple of (precisions, recalls) arrays.
    """
    if thresholds is None:
        thresholds = np.sort(np.unique(errors))

    precisions = []
    recalls = []

    for threshold in thresholds:
        inliers = errors <= threshold
        recall = np.mean(inliers)
        # For geometric errors, precision = recall (all predictions are considered)
        precision = recall

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def pose_auc(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    max_rot_error: float = 10.0,
    max_trans_error: float = 10.0,
) -> Dict[str, float]:
    """
    Compute AUC for pose errors.

    Args:
        R_est: Estimated rotation matrix (3, 3).
        t_est: Estimated translation vector (3,) or (3, 1).
        R_gt: Ground truth rotation matrix (3, 3).
        t_gt: Ground truth translation vector (3,) or (3, 1).
        max_rot_error: Maximum rotation error threshold in degrees.
        max_trans_error: Maximum translation error threshold in degrees.

    Returns:
        Dictionary with rotation_auc, translation_auc, and combined_auc.
    """
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

