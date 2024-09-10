from collections import OrderedDict

import cv2
import numpy as np
import torch
from imm.utils.device import to_numpy
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric
from loguru import logger


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance."""

    # normalize keypoints
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    #  convert to homogeneous
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    # compute Sampson distance
    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    # compute distance
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))  # N
    return d


def compute_symmetrical_epipolar_errors(mkpts0, mkpts1, K0, K1, T_0to1):
    # essential matrix
    Tx = numeric.cross_product_matrix(T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]

    # epipolar error
    epi_errs = symmetric_epipolar_distance(mkpts0, mkpts1, E_mat, K0, K1)
    epi_errs = epi_errs.numpy()

    return {"epi_errs": epi_errs}


def estimate_pose(kpts0, kpts1, K0, K1, ransac_thd, ransac_conf=0.99999):
    # if tensor, convert to numpy
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.numpy()
        kpts1 = kpts1.numpy()

    if isinstance(K0, torch.Tensor):
        K0 = K0.numpy()
        K1 = K1.numpy()

    if len(kpts0) < 5:
        logger.warning(f"Cannot estimate, few points {len(kpts0)}")
        return None

    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = ransac_thd / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        np.eye(3),
        threshold=ransac_thr,
        prob=ransac_conf,
        method=cv2.RANSAC,
    )

    if E is None:
        logger.warning("E is None while trying to recover pose.")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity

    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def compute_pose_errors(mkpts0, mkpts1, K0, K1, T_0to1, ransac_thd=1.0, ransac_conf=0.99999):
    out = {}

    # estimate pose
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, ransac_thd=ransac_thd, ransac_conf=ransac_conf)

    if ret is None:
        out["R_errs"] = np.inf
        out["t_errs"] = np.inf
        out["inliers"] = np.array([]).astype(np.bool_)
    else:
        R, t, inliers = ret

        # pose error
        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)

        out["R_errs"] = R_err
        out["t_errs"] = t_err
        out["inliers"] = inliers

    return out


def error_auc(errors, thresholds, coef=100.0):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    # percentage
    return {f"auc@{t}": coef * auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, coef=100.0):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)

    # percentage
    return {f"prec@{t:.0e}": coef * prec for t, prec in zip(thresholds, precs)}


def aggregate_metrics(args, metrics):
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics["pair_names"]))
    unq_ids = list(unq_ids.values())
    logger.info(f"Aggregating metrics over {len(unq_ids)} unique items...")

    # pose auc thresholds
    logger.info(f"compute AUC for {args.auc_thresholds}")
    pose_errors = np.max(np.stack([metrics["R_errs"], metrics["t_errs"]]), axis=0)[unq_ids]
    pose_aucs = error_auc(pose_errors, args.auc_thresholds)

    # matching precision
    epip_thresholds = [args.epip_thd]
    epip_precs = epidist_prec(np.array(metrics["epi_errs"], dtype=object)[unq_ids], epip_thresholds)

    return {**pose_aucs, **epip_precs}


def compute_metrics(mkpts0, mkpts1, K0, K1, T_0to1, ransac_thd, ransac_conf):
    # compute epipolar error
    epip_metric = compute_symmetrical_epipolar_errors(mkpts0, mkpts1, K0, K1, T_0to1)

    # compute pose error
    pose_metric = compute_pose_errors(
        mkpts0,
        mkpts1,
        K0,
        K1,
        T_0to1,
        ransac_thd=ransac_thd,
        ransac_conf=ransac_conf,
    )
    # aggregate
    metrics = {**epip_metric, **pose_metric}
    to_numpy(metrics)

    return {**epip_metric, **pose_metric}
