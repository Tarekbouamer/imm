import cv2
import numpy as np
import pydegensac
from loguru import logger
from tabulate import tabulate

from imm.utils.logging import create_small_table

np.set_printoptions(precision=5)


def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))

    N = len(errors)
    errors = np.append([0.0], np.sort(errors))
    recalls = np.arange(N + 1) / N

    aucs = []
    for thresholds in thresholds:
        last_index = np.searchsorted(errors, thresholds)
        rcs_ = np.append(recalls[:last_index], recalls[last_index - 1])
        errs_ = np.append(errors[:last_index], thresholds)

        aucs.append(np.trapz(rcs_, x=errs_) / thresholds)

    return np.array(aucs)


def print_table(title, data, header):
    tb = tabulate(
        data,
        header,
        tablefmt="pipe",
        floatfmt=".2f",
        stralign="center",
        numalign="center",
    )
    logger.info(f"{title} \n" + tb)


def eval_homography(dists_sa, dists_si, dists_sv, thresholds):
    #
    results = {}

    #
    logger.info("Homography Estimation scores")

    # correct
    correct_sa = np.mean([[float(dist <= t) for t in thresholds] for dist in dists_sa], axis=0)
    correct_si = np.mean([[float(dist <= t) for t in thresholds] for dist in dists_si], axis=0)
    correct_sv = np.mean([[float(dist <= t) for t in thresholds] for dist in dists_sv], axis=0)

    results["correct"] = {"i": correct_si, "v": correct_sv, "a": correct_sa}
    logger.info("Correct metrics: \n" + create_small_table(results["correct"]))

    # aucs
    auc_sa = cal_error_auc(dists_sa, thresholds=thresholds)
    auc_si = cal_error_auc(dists_si, thresholds=thresholds)
    auc_sv = cal_error_auc(dists_sv, thresholds=thresholds)

    results["auc"] = {"i": auc_si, "v": auc_sv, "a": auc_sa}

    logger.info("AUC metrics: \n" + create_small_table(results["auc"]))

    #
    return results


def eval_matching(
    i_err,
    v_err,
    seq_types,
    num_features,
    num_matches,
    thresholds=[1, 3, 5, 10],
    save_path=None,
):
    logger.info("Matching scores")

    results = {}

    n_i = 52
    n_v = 56

    # features
    mean_features = np.mean(num_features)
    min_features = np.min(num_features)
    max_features = np.max(num_features)

    results["features"] = {
        "mean": mean_features,
        "min": min_features,
        "max": max_features,
    }

    logger.info("Features metrics: \n" + create_small_table(results["features"]))

    # matching
    mean_matches = np.mean(num_matches)
    i_matches = np.mean(num_matches[seq_types == "i"])
    v_matches = np.mean(num_matches[seq_types == "v"])

    results["matches"] = {"mean": mean_matches, "i": i_matches, "v": v_matches}

    logger.info("Matches metrics: \n" + create_small_table(results["matches"]))

    # MMA
    thresholds = np.array(thresholds)
    results["thresholds"] = thresholds
    #
    ierr = np.array([i_err[th] / (n_i * 5) for th in thresholds])
    verr = np.array([v_err[th] / (n_v * 5) for th in thresholds])
    aerr = np.array([(i_err[th] + v_err[th]) / ((n_i + n_v) * 5) for th in thresholds])

    results["mma"] = {"i": ierr, "v": verr, "a": aerr}
    logger.info("MMA metrics: \n" + create_small_table(results["mma"]))

    return results


def scale_homography(sw, sh):
    return np.array([[sw, 0, 0], [0, sh, 0], [0, 0, 1]])


def reproj_distance_error(p1s, p2s, homography):
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))

    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))

    return dist


def homography_error(kps0, kps1, H_gt, corners, ransac_thd, h_solver="magsac"):
    try:
        if h_solver == "cv":
            H_est, inliers = cv2.findHomography(kps0, kps1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thd)
        elif h_solver == "magsac":
            H_est, inliers = cv2.findHomography(kps0, kps1, cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_thd)
        elif h_solver == "pydegensac":
            H_est, inliers = pydegensac.findHomography(kps0, kps1, ransac_thd)

        success = True
    except Exception:
        H_est = None
        success = False

    # compute distances
    if H_est is not None:
        real_warped_corners = np.dot(corners, np.transpose(H_gt))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]

        warped_corners = np.dot(corners, np.transpose(H_est))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

        corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        irat = np.mean(inliers)
    else:
        corner_dist = np.nan
        irat = 0
        inliers = []

    return success, corner_dist, irat
