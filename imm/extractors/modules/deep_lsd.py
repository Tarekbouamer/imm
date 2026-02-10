import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components


class VGGUNet(torch.nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if tiny:
            sizes = [32, 64, 128, 256]
        else:
            sizes = [64, 128, 256, 512]

        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
            nn.Conv2d(sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
        )

        # Decoder blocks
        self.deblock4 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.deblock3 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.deblock2 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.deblock1 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )

    def forward(self, inputs):
        # Encoding
        features = [self.block1(inputs)]
        for block in [self.block2, self.block3, self.block4]:
            features.append(block(self.pool(features[-1])))

        # Decoding
        out = self.deblock4(features[-1])
        for deblock, feat in zip([self.deblock3, self.deblock2, self.deblock1], features[:-1][::-1]):
            out = deblock(torch.cat([F.interpolate(out, feat.shape[2:4], mode="bilinear"), feat], dim=1))

        return out  # dim = 32 if tiny else 64


###########################################################
###
###########################################################


def get_segment_overlap(seg_coord1d):
    """Given a list of segments parameterized by the 1D coordinate
    of the endpoints, compute the overlap with the segment [0, 1]."""
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = (
        (seg_coord1d[..., 1] > 0)
        * (seg_coord1d[..., 0] < 1)
        * (np.minimum(seg_coord1d[..., 1], 1) - np.maximum(seg_coord1d[..., 0], 0))
    )
    return overlap


def get_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5, return_overlap=False, mode="min"):
    """Compute the symmetrical orthogonal line distance between two sets
    of lines and the average overlapping ratio of both lines.
    Enforce a high line distance for small overlaps.
    This is compatible for nD objects (e.g. both lines in 2D or 3D)."""
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2
    min_overlaps = np.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == "mean":
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


def project_point_to_line(line_segs, points):
    """Given a list of line segments and a list of points (2D or 3D coordinates),
    compute the orthogonal projection of all points on all lines.
    This returns the 1D coordinates of the projection on the line,
    as well as the list of orthogonal distances."""
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = ((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2) / np.linalg.norm(dir_vec, axis=2) ** 2
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


def merge_line_cluster(lines):
    """Merge a cluster of line segments.
    First compute the principal direction of the lines, compute the
    endpoints barycenter, project the endpoints onto the middle line,
    keep the two extreme projections.
    Args:
        lines: a (n, 2, 2) np array containing n lines.
    Returns:
        The merged (2, 2) np array line segment.
    """
    # Get the principal direction of the endpoints
    points = lines.reshape(-1, 2)
    weights = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    weights = np.repeat(weights, 2)[:, None]
    weights /= weights.sum()  # More weights for longer lines
    avg_points = (points * weights).sum(axis=0)
    points_bar = points - avg_points[None]
    cov = 1 / 3 * np.einsum("ij,ik->ijk", points_bar, points_bar * weights).sum(axis=0)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = np.array([1, 0]) if a >= c else np.array([0, 1])
    else:
        m = (c - a + np.sqrt((a - c) ** 2 + 4 * b**2)) / (2 * b)
        u = np.array([1, m]) / np.sqrt(1 + m**2)

    # Get the center of gravity of all endpoints
    cross = np.mean(points, axis=0)

    # Project the endpoints on the line defined by cross and u
    avg_line_seg = np.stack([cross, cross + u], axis=0)
    proj = project_point_to_line(avg_line_seg[None], points)[0]

    # Take the two extremal projected endpoints
    new_line = np.stack([cross + np.amin(proj) * u, cross + np.amax(proj) * u], axis=0)
    return new_line


def merge_lines(lines, thresh=5.0, overlap_thresh=0.0):
    """Given a set of lines, merge close-by lines together.
    Two lines are merged when their orthogonal distance is smaller
    than a threshold and they have a positive overlap.
    Args:
        lines: a (N, 2, 2) np array.
        thresh: maximum orthogonal distance between two lines to be merged.
        overlap_thresh: maximum distance between 2 endpoints to merge
                        two aligned lines.
    Returns:
        The new lines after merging.
    """
    if len(lines) == 0:
        return lines

    # Compute the pairwise orthogonal distances and overlap
    orth_dist, overlaps = get_orth_line_dist(lines, lines, return_overlap=True)

    # Define clusters of close-by lines to merge
    if overlap_thresh == 0:
        adjacency_mat = (overlaps > 0) * (orth_dist < thresh)
    else:
        # Filter using the distance between the two closest endpoints
        n = len(lines)
        endpoints = lines.reshape(n * 2, 2)
        close_endpoint = np.linalg.norm(endpoints[:, None] - endpoints[None], axis=2)
        close_endpoint = close_endpoint.reshape(n, 2, n, 2).transpose(0, 2, 1, 3).reshape(n, n, 4)
        close_endpoint = np.amin(close_endpoint, axis=2)
        adjacency_mat = ((overlaps > 0) | (close_endpoint < overlap_thresh)) * (orth_dist < thresh)
    n_comp, components = connected_components(adjacency_mat, directed=False)

    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = lines[components == i]
        new_lines.append(merge_line_cluster(cluster))

    return np.stack(new_lines, axis=0)


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


def compute_image_grad(img, ksize=7):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), 1).astype(np.float32)
    dx = np.zeros_like(blur_img)
    dy = np.zeros_like(blur_img)
    dx[:, 1:] = (blur_img[:, 1:] - blur_img[:, :-1]) / 2
    dx[1:, 1:] = dx[:-1, 1:] + dx[1:, 1:]
    dy[1:] = (blur_img[1:] - blur_img[:-1]) / 2
    dy[1:, 1:] = dy[1:, :-1] + dy[1:, 1:]
    gradnorm = np.sqrt(dx**2 + dy**2)
    gradangle = np.arctan2(dy, dx)
    return dx, dy, gradnorm, gradangle


def align_with_grad_angle(angle, img):
    """Starting from an angle in [0, pi], find the sign of the angle based on
    the image gradient of the corresponding pixel."""
    # Image gradient
    img_grad_angle = compute_image_grad(img)[3]

    # Compute the distance of the image gradient to the angle
    # and angle - pi
    pred_grad = np.mod(angle, np.pi)  # in [0, pi]
    pos_dist = np.minimum(np.abs(img_grad_angle - pred_grad), 2 * np.pi - np.abs(img_grad_angle - pred_grad))
    neg_dist = np.minimum(
        np.abs(img_grad_angle - pred_grad + np.pi), 2 * np.pi - np.abs(img_grad_angle - pred_grad + np.pi)
    )

    # Assign the new grad angle to the closest of the two
    is_pos_closest = np.argmin(np.stack([neg_dist, pos_dist], axis=-1), axis=-1).astype(bool)
    new_grad_angle = np.where(is_pos_closest, pred_grad, pred_grad - np.pi)
    return new_grad_angle, img_grad_angle


def preprocess_angle(angle, img, mask=False):
    """Convert a grad angle field into a line level angle, using
    the image gradient to get the right orientation."""
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = np.mod(oriented_grad_angle - np.pi / 2, 2 * np.pi)
    if mask:
        oriented_grad_angle[0] = -1024
        oriented_grad_angle[:, 0] = -1024
    return oriented_grad_angle.astype(np.float64), img_grad_angle


def sample_along_line(lines, img, n_samples=10, mode="mean"):
    """Sample a fixed number of points along each line and interpolate
    an img at these points, and finally aggregate the values."""
    # Get the sample positions
    t = np.linspace(0, 1, 10)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None] - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the img at the samples and aggregate the values
    if mode == "mean":
        # Average
        val = bilinear_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.mean(val.reshape(-1, n_samples), axis=-1)
    elif mode == "angle":
        # Average of angles
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)
        val = np.arctan2(np.sin(val).sum(axis=-1), np.cos(val).sum(axis=-1))
    elif mode == "median":
        # Median
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.median(val.reshape(-1, n_samples), axis=-1)
    else:
        # No aggregation
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)

    return val


def get_line_orientation(lines, angle):
    """Get the orientation in [-pi, pi] of a line, based on the gradient."""
    grad_val = sample_along_line(lines, angle, mode="angle")
    line_ori = np.mod(np.arctan2(lines[:, 1, 0] - lines[:, 0, 0], lines[:, 1, 1] - lines[:, 0, 1]), np.pi)

    pos_dist = np.minimum(np.abs(grad_val - line_ori), 2 * np.pi - np.abs(grad_val - line_ori))
    neg_dist = np.minimum(np.abs(grad_val - line_ori + np.pi), 2 * np.pi - np.abs(grad_val - line_ori + np.pi))
    line_ori = np.where(pos_dist <= neg_dist, line_ori, line_ori - np.pi)
    return line_ori


def nn_interpolate_numpy(img, x, y):
    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
    return img[yi, xi]


def filter_outlier_lines(
    img,
    lines,
    df,
    angle,
    mode="inlier_thresh",
    use_grad=False,
    inlier_thresh=0.5,
    df_thresh=1.5,
    ang_thresh=np.pi / 6,
    n_samples=50,
):
    """Filter out outlier lines either by comparing the average DF and
        line level values to a threshold or by counting the number of inliers
        across the line. It can also optionally use the image gradient.
    Args:
        img: the original image.
        lines: a (N, 2, 2) np array.
        df: np array with the distance field.
        angle: np array with the grad angle field.
        mode: 'mean' or 'inlier_thresh'.
        use_grad: True to use the image gradient instead of line_level.
        inlier_thresh: ratio of inliers to get accepted.
        df_thresh, ang_thresh: thresholds to determine a valid value.
        n_samples: number of points sampled along each line.
    Returns:
        A tuple with the filtered lines and a mask of valid lines.
    """
    # Get the right orientation of the line_level and the lines orientation
    oriented_line_level, img_grad_angle = preprocess_angle(angle, img)
    orientations = get_line_orientation(lines, oriented_line_level)

    # Get the sample positions
    t = np.linspace(0, 1, n_samples)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None] - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the DF and angle map
    df_samples = bilinear_interpolate_numpy(df, samples[:, 1], samples[:, 0])
    df_samples = df_samples.reshape(-1, n_samples)
    if use_grad:
        oriented_line_level = np.mod(img_grad_angle - np.pi / 2, 2 * np.pi)
    ang_samples = nn_interpolate_numpy(oriented_line_level, samples[:, 1], samples[:, 0]).reshape(-1, n_samples)

    # Check the average value or number of inliers
    if mode == "mean":
        df_check = np.mean(df_samples, axis=1) < df_thresh
        ang_avg = np.arctan2(np.sin(ang_samples).sum(axis=1), np.cos(ang_samples).sum(axis=1))
        ang_diff = np.minimum(np.abs(ang_avg - orientations), 2 * np.pi - np.abs(ang_avg - orientations))
        ang_check = ang_diff < ang_thresh
        valid = df_check & ang_check
    elif mode == "inlier_thresh":
        df_check = df_samples < df_thresh
        ang_diff = np.minimum(
            np.abs(ang_samples - orientations[:, None]), 2 * np.pi - np.abs(ang_samples - orientations[:, None])
        )
        ang_check = ang_diff < ang_thresh
        valid = (df_check & ang_check).mean(axis=1) > inlier_thresh
    else:
        raise ValueError("Unknown filtering mode: " + mode)

    return lines[valid], valid
