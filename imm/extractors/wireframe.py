from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch import Tensor

from imm.base import FeatureModel
from imm.extractors._helper import EXTRACTORS_REGISTRY, create_extractor
from imm.misc import _cfg


def sample_descriptors_corner_conv(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def lines_to_wireframe(lines, line_scores, dense_desc, s_desc, nms_radius=4, force_num_lines=None, max_num_lines=None):
    """Given a set of lines, their score and dense descriptors,
        merge close-by endpoints and compute a wireframe defined by
        its junctions and connectivity.
    Returns:
        junctions: list of [num_junc, 2] tensors listing all wireframe junctions
        junc_scores: list of [num_junc] tensors with the junction score
        junc_descs: list of [dim, num_junc] tensors with the junction descriptors
        connectivity: list of [num_junc, num_junc] bool arrays with True when 2
        junctions are connected
        new_lines: the new set of [b_size, num_lines, 2, 2] lines
        lines_junc_idx: a [b_size, num_lines, 2] tensor with the indices of the
        junctions of each endpoint
        num_true_junctions: a list of the number of valid junctions for each image
        in the batch, i.e. before filling with random ones
    """
    b_size, _, h, w = dense_desc.shape
    device = lines.device
    h, w = h * s_desc, w * s_desc
    lines_end_points = lines.reshape(b_size, -1, 2)

    (
        junctions,
        junc_scores,
        connectivity,
        new_lines,
        lines_junc_idx,
        num_true_junctions,
    ) = [], [], [], [], [], []
    for bs in range(b_size):
        # Cluster the junctions that are close-by
        db = DBSCAN(eps=nms_radius, min_samples=1).fit(lines_end_points[bs].cpu().numpy())
        clusters = db.labels_
        n_clusters = len(set(clusters))

        num_true_junctions.append(n_clusters)

        # Compute the average junction and score for each cluster
        clusters = torch.tensor(clusters, dtype=torch.long, device=device)
        new_junc = torch.zeros(n_clusters, 2, dtype=torch.float, device=device)
        new_junc.scatter_reduce_(
            0,
            clusters[:, None].repeat(1, 2),
            lines_end_points[bs],
            reduce="mean",
            include_self=False,
        )
        junctions.append(new_junc)
        new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
        new_scores.scatter_reduce_(
            0,
            clusters,
            torch.repeat_interleave(line_scores[bs], 2),
            reduce="mean",
            include_self=False,
        )
        junc_scores.append(new_scores)

        # Compute the new lines
        new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))
        lines_junc_idx.append(clusters.reshape(-1, 2))

        if force_num_lines:
            # Add random junctions (with no connectivity)
            missing = max_num_lines * 2 - len(junctions[-1])
            junctions[-1] = torch.cat(
                [
                    junctions[-1],
                    torch.rand(missing, 2).to(lines) * lines.new_tensor([[w - 1, h - 1]]),
                ],
                dim=0,
            )
            junc_scores[-1] = torch.cat([junc_scores[-1], torch.zeros(missing).to(lines)], dim=0)

            junc_connect = torch.eye(max_num_lines * 2, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)
        else:
            # Compute the junction connectivity
            junc_connect = torch.eye(n_clusters, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)

    junctions = torch.stack(junctions, dim=0)
    new_lines = torch.stack(new_lines, dim=0)
    lines_junc_idx = torch.stack(lines_junc_idx, dim=0)

    # Interpolate the new junction descriptors
    junc_descs = sample_descriptors_corner_conv(junctions, dense_desc, s_desc).mT

    return junctions, junc_scores, junc_descs, connectivity, new_lines, lines_junc_idx, num_true_junctions


def parse_input(data: Dict[str, Any]) -> Dict[str, Tensor]:
    for k, v in data.items():
        #
        if isinstance(v, list):
            v = torch.cat(v, dim=0)

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if v.dim() == 4:
            data[k] = v
        else:
            data[k] = v.unsqueeze(0)

    return data


class Wireframe(FeatureModel):
    required_data_keys = ["image"]

    def __init__(self, point_extractor, line_extractor, cfg) -> None:
        super().__init__(cfg)

        # Point extractor
        self.point_extractor = point_extractor

        # Line extractor
        self.line_extractor = line_extractor

    def transform_inputs(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # to 4D
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        return data

    def get_predictions(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        #
        b_size, _, h, w = data["image"].shape

        # 1. Line detection
        l_pred = self.line_extractor.extract(data)
        l_pred = parse_input(l_pred)

        lines = l_pred["lines"]
        line_scores = l_pred["scores"]

        # Normalize the line scores
        line_scores = line_scores / line_scores.max(dim=1)[0][:, None] + 1e-8

        # From [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        lines = lines.reshape(b_size, -1, 2, 2)

        # 2. Keypoint prediction
        p_pred = self.point_extractor.extract(data)
        p_pred = parse_input(p_pred)

        kpts = p_pred["kpts"]
        desc = p_pred["desc"]
        dense_desc = p_pred["dense_desc"]
        keypoint_scores = p_pred["scores"]

        # Permute the descriptors to [B, D, N]
        desc = desc.permute(0, 2, 1)

        return kpts, lines, desc, dense_desc, keypoint_scores, line_scores

    def forward(self, data):
        #
        b_size, _, h, w = data["image"].shape
        device = data["image"].device

        # 1. Get the predictions
        kpts, lines, desc, dense_desc, keypoint_scores, line_scores = self.get_predictions(data)

        s_desc = data["image"].shape[2] // dense_desc.shape[2]

        # 2. Remove keypoints that are too close to line endpoints
        if self.cfg.merge_points:
            # Compute the distance between each keypoint and each line endpoint
            line_end_points = lines.reshape(b_size, -1, 2)
            dist_pt_lines = torch.cdist(kpts, line_end_points)

            # Remove keypoints that are too close to any line endpoint
            pts_to_remove = torch.any(dist_pt_lines < self.cfg.nms_radius, dim=2)
            kpts = kpts[0][~pts_to_remove[0]][None]
            keypoint_scores = keypoint_scores[0][~pts_to_remove[0]][None]
            desc = desc[0][~pts_to_remove[0]][None]

        # 3. Connect the lines together to form a wireframe
        orig_lines = lines.clone()
        if self.cfg.merge_line_endpoints and len(lines[0]) > 0:
            # Merge first close-by endpoints to connect lines

            line_points, line_pts_scores, line_descs, line_association, lines, lines_junc_idx, n_true_junctions = (
                lines_to_wireframe(
                    lines,
                    line_scores,
                    dense_desc,
                    s_desc=s_desc,
                    nms_radius=self.cfg.nms_radius,
                    force_num_lines=self.cfg.force_num_lines,
                    max_num_lines=self.cfg.max_lines,
                )
            )

            # Add the keypoints to the junctions and fill the rest with random keypoints
            (all_points, all_scores, all_descs, pl_associativity) = [], [], [], []
            for bs in range(b_size):
                all_points.append(torch.cat([line_points[bs], kpts[bs]], dim=0))
                all_scores.append(torch.cat([line_pts_scores[bs], keypoint_scores[bs]], dim=0))
                all_descs.append(torch.cat([line_descs[bs], desc[bs]], dim=0))

                associativity = torch.eye(len(all_points[-1]), dtype=torch.bool, device=device)
                associativity[: n_true_junctions[bs], : n_true_junctions[bs]] = line_association[bs][
                    : n_true_junctions[bs], : n_true_junctions[bs]
                ]
                pl_associativity.append(associativity)

            all_points = torch.stack(all_points, dim=0)
            all_scores = torch.stack(all_scores, dim=0)
            all_descs = torch.stack(all_descs, dim=0)
            pl_associativity = torch.stack(pl_associativity, dim=0)
        else:
            # Lines are independent
            all_points = torch.cat([lines.reshape(b_size, -1, 2), kpts], dim=1)
            n_pts = all_points.shape[1]
            num_lines = lines.shape[1]
            n_true_junctions = [num_lines * 2] * b_size
            all_scores = torch.cat(
                [
                    torch.repeat_interleave(line_scores, 2, dim=1),
                    keypoint_scores,
                ],
                dim=1,
            )
            line_descs = sample_descriptors_corner_conv(
                lines.reshape(b_size, -1, 2), dense_desc, s_desc
            ).mT  # [B, n_lines * 2, desc_dim]
            all_descs = torch.cat([line_descs, desc], dim=1)
            pl_associativity = torch.eye(n_pts, dtype=torch.bool, device=device)[None].repeat(b_size, 1, 1)
            lines_junc_idx = torch.arange(num_lines * 2, device=device).reshape(1, -1, 2).repeat(b_size, 1, 1)

        del dense_desc  # Remove dense descriptors to save memory
        torch.cuda.empty_cache()

        ret = {}
        ret["lines"] = lines
        ret["line_scores"] = line_scores
        ret["kpts"] = all_points
        ret["keypoint_scores"] = all_scores
        ret["desc"] = all_descs
        # ret["pl_associativity"] = pl_associativity
        # ret["num_junctions"] = torch.tensor(n_true_junctions)
        # ret["orig_lines"] = orig_lines
        ret["lines_junc_idx"] = lines_junc_idx
        return ret


# default configurations
default_cfgs = {
    "sp_lsd_wireframe": _cfg(
        force_num_keypoints=False,
        max_keypoints=None,
        force_num_lines=False,
        max_lines=None,
        min_length=15,
        merge_points=True,
        merge_line_endpoints=True,
        nms_radius=3,
    ),
    "sp_deeplsd_wireframe": _cfg(
        force_num_keypoints=False,
        max_keypoints=None,
        force_num_lines=False,
        max_lines=None,
        min_length=15,
        merge_points=True,
        merge_line_endpoints=True,
        nms_radius=3,
    ),
}


def _make_model(
    point_extractor: str,
    line_extractor: str,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create point extractor
    point_extractor = create_extractor(name=point_extractor, cfg=cfg, pretrained=pretrained)

    # create line extractor
    line_extractor = create_extractor(name=line_extractor, cfg=cfg, pretrained=pretrained)

    # create model
    model = Wireframe(point_extractor, line_extractor, cfg=cfg)

    return model


@EXTRACTORS_REGISTRY.register(name="sp_lsd_wireframe", default_cfg=default_cfgs["sp_lsd_wireframe"])
def sp_lsd_wireframe(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(point_extractor="superpoint", line_extractor="lsd", cfg=cfg, **kwargs)
