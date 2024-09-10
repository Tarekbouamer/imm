import torch


def max_pool(x, nms_radius: int):
    return torch.nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    scores = scores[None]
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores, nms_radius)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    return torch.where(max_mask, scores, zeros)[0]


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    # Still dynamic despite trace warning
    kpts_len = torch.tensor(keypoints.shape[0])
    max_keypoints = torch.minimum(torch.tensor(k), kpts_len)
    scores, indices = torch.topk(scores, max_keypoints, dim=0)  # type: ignore
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5

    keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
    keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=-1)

    # normalize to (-1, 1)
    keypoints = keypoints * 2 - 1
    descriptors = torch.nn.functional.grid_sample(
        descriptors,
        keypoints.view(b, 1, -1, 2),
        mode="bilinear",
        align_corners=True,
    )

    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors
