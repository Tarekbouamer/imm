import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor([[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]]).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor([[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]]).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor([[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]]).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = batch == depth_wise_max
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = batch == local_max
        del local_max

        dii = F.conv2d(batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1).view(b, c, h, w)
        dij = F.conv2d(batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1).view(b, c, h, w)
        djj = F.conv2d(batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(is_depth_wise_max, torch.min(is_local_max, is_not_edge))
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor([[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]]).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor([[1.0, 0, -1.0], [0, 0.0, 0], [-1.0, 0, 1.0]]).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor([[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]]).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1).view(b, c, h, w)
        dij = F.conv2d(batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1).view(b, c, h, w)
        djj = F.conv2d(batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1).view(b, c, h, w)
        dj = F.conv2d(batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)


class EmptyTensorError(Exception):
    pass


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right),
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left]
        + w_top_right * dense_features[:, i_top_right, j_top_right]
        + w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left]
        + w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack(
            [
                torch.stack([i_top_left, j_top_left], dim=0),
                torch.stack([i_top_right, j_top_right], dim=0),
                torch.stack([i_bottom_left, j_bottom_left], dim=0),
                torch.stack([i_bottom_right, j_bottom_right], dim=0),
            ],
            dim=0,
        )
        return [descriptors, pos, ids, corners]
