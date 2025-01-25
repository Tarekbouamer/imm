import warnings
from typing import Any, Dict

import torch
import torch.utils.checkpoint
from torch import nn

from imm.base.matcher import MatcherModel
from imm.matchers._helper import MATCHERS_REGISTRY
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from .modules.gluestick import (
    AttentionalGNN,
    EndPtEncoder,
    KeypointEncoder,
    arange_like,
    log_double_softmax,
    normalize_keypoints,
)

warnings.filterwarnings("ignore", category=UserWarning)

ETH_EPS = 1e-8


class GlueStick(MatcherModel):
    required_data_keys = [
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
        "keypoint_scores0",
        "keypoint_scores1",
    ]

    DEFAULT_LOSS_CONF = {"nll_weight": 1.0, "nll_balancing": 0.5, "reward_weight": 0.0, "bottleneck_l2_weight": 0.0}

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        if self.cfg.bottleneck_dim is not None:
            self.bottleneck_down = nn.Conv1d(self.cfg.input_dim, self.cfg.bottleneck_dim, kernel_size=1)
            self.bottleneck_up = nn.Conv1d(self.cfg.bottleneck_dim, self.cfg.input_dim, kernel_size=1)
            nn.init.constant_(self.bottleneck_down.bias, 0.0)
            nn.init.constant_(self.bottleneck_up.bias, 0.0)

        if self.cfg.input_dim != self.cfg.descriptor_dim:
            self.input_proj = nn.Conv1d(self.cfg.input_dim, self.cfg.descriptor_dim, kernel_size=1)
            nn.init.constant_(self.input_proj.bias, 0.0)

        self.kenc = KeypointEncoder(self.cfg.descriptor_dim, self.cfg.keypoint_encoder)
        self.lenc = EndPtEncoder(self.cfg.descriptor_dim, self.cfg.keypoint_encoder)
        self.gnn = AttentionalGNN(
            self.cfg.descriptor_dim,
            self.cfg.GNN_layers,
            checkpointed=self.cfg.checkpointed,
            inter_supervision=self.cfg.inter_supervision,
            num_line_iterations=self.cfg.num_line_iterations,
            line_attention=self.cfg.line_attention,
        )
        self.final_proj = nn.Conv1d(self.cfg.descriptor_dim, self.cfg.descriptor_dim, kernel_size=1)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)
        self.final_line_proj = nn.Conv1d(self.cfg.descriptor_dim, self.cfg.descriptor_dim, kernel_size=1)
        nn.init.constant_(self.final_line_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_line_proj.weight, gain=1)
        if self.cfg.inter_supervision is not None:
            self.inter_line_proj = nn.ModuleList(
                [
                    nn.Conv1d(self.cfg.descriptor_dim, self.cfg.descriptor_dim, kernel_size=1)
                    for _ in self.cfg.inter_supervision
                ]
            )
            self.layer2idx = {}
            for i, l in enumerate(self.cfg.inter_supervision):
                nn.init.constant_(self.inter_line_proj[i].bias, 0.0)
                nn.init.orthogonal_(self.inter_line_proj[i].weight, gain=1)
                self.layer2idx[l] = i

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)
        line_bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("line_bin_score", line_bin_score)

    def _get_matches(self, scores_mat):
        max0 = scores_mat[:, :-1, :-1].max(2)
        max1 = scores_mat[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores_mat.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.cfg.match_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return m0, m1, mscores0, mscores1

    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0, lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)

        line_scores = torch.einsum("bdn,bdm->bnm", mldesc0, mldesc1)
        line_scores = line_scores / self.cfg.descriptor_dim**0.5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]
        line_scores = torch.gather(
            line_scores, dim=2, index=lines_junc_idx1[:, None, :].repeat(1, line_scores.shape[1], 1)
        )
        line_scores = torch.gather(line_scores, dim=1, index=lines_junc_idx0[:, :, None].repeat(1, 1, n2_lines1))
        line_scores = line_scores.reshape((-1, n2_lines0 // 2, 2, n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * torch.maximum(
            line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1],
            line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0],
        )
        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(line_scores)
        return (line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores)

    def transform_inputs(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                if isinstance(data[k][0], torch.Tensor):
                    data[k] = torch.stack(data[k])
        return data

    def process_matches(self, data, preds):
        # keypoints
        kpts0 = data["kpts0"][0]
        kpts1 = data["kpts1"][0]

        matches0 = preds["matches0"][0]
        mscores0 = preds["match_scores0"][0]

        valid = torch.where(matches0 != -1)[0]

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]

        # lines
        lines0 = data["lines0"][0]
        lines1 = data["lines1"][0]

        line_matches0 = preds["line_matches0"][0]
        line_scores0 = preds["line_match_scores0"][0]

        valid = torch.where(line_matches0 != -1)[0]

        mlines0 = lines0[valid]
        mlines1 = lines1[line_matches0[valid]]

        ret = {
            "kpts0": kpts0,
            "kpts1": kpts1,
            "matches": matches0,
            "mscores": mscores0,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            # lines
            "lines0": lines0,
            "lines1": lines1,
            "lines_matches": line_matches0,
            "lines_scores": line_scores0,
            "mlines0": mlines0,
            "mlines1": mlines1,
        }

        return ret

    def forward(self, data):
        device = data["kpts0"].device
        b_size = len(data["kpts0"])
        image_size0 = data["size0"] if "size0" in data else data["image0"].shape
        image_size1 = data["size1"] if "size1" in data else data["image1"].shape

        pred = {}
        desc0, desc1 = data["desc0"], data["desc1"]
        kpts0, kpts1 = data["kpts0"], data["kpts1"]

        # permute desc
        desc0 = desc0.permute(0, 2, 1)
        desc1 = desc1.permute(0, 2, 1)

        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = data["lines0"].shape[1], data["lines1"].shape[1]
        if n_kpts0 == 0 or n_kpts1 == 0:
            # No detected keypoints nor lines
            pred["log_assignment"] = torch.zeros(b_size, n_kpts0, n_kpts1, dtype=torch.float, device=device)
            pred["matches0"] = torch.full((b_size, n_kpts0), -1, device=device, dtype=torch.int64)
            pred["matches1"] = torch.full((b_size, n_kpts1), -1, device=device, dtype=torch.int64)
            pred["match_scores0"] = torch.zeros((b_size, n_kpts0), device=device, dtype=torch.float32)
            pred["match_scores1"] = torch.zeros((b_size, n_kpts1), device=device, dtype=torch.float32)
            pred["line_log_assignment"] = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)
            pred["line_matches0"] = torch.full((b_size, n_lines0), -1, device=device, dtype=torch.int64)
            pred["line_matches1"] = torch.full((b_size, n_lines1), -1, device=device, dtype=torch.int64)
            pred["line_match_scores0"] = torch.zeros((b_size, n_lines0), device=device, dtype=torch.float32)
            pred["line_match_scores1"] = torch.zeros((b_size, n_kpts1), device=device, dtype=torch.float32)
            return pred

        lines0 = data["lines0"].flatten(1, 2)
        lines1 = data["lines1"].flatten(1, 2)
        lines_junc_idx0 = data["lines_junc_idx0"].flatten(1, 2)  # [b_size, num_lines * 2]
        lines_junc_idx1 = data["lines_junc_idx1"].flatten(1, 2)

        if self.cfg.bottleneck_dim is not None:
            pred["down_descriptors0"] = desc0 = self.bottleneck_down(desc0)
            pred["down_descriptors1"] = desc1 = self.bottleneck_down(desc1)
            desc0 = self.bottleneck_up(desc0)
            desc1 = self.bottleneck_up(desc1)
            desc0 = nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = nn.functional.normalize(desc1, p=2, dim=1)
            pred["bottleneck_descriptors0"] = desc0
            pred["bottleneck_descriptors1"] = desc1
            if self.cfg.loss.nll_weight == 0:
                desc0 = desc0.detach()
                desc1 = desc1.detach()

        if self.cfg.input_dim != self.cfg.descriptor_dim:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, image_size0)
        kpts1 = normalize_keypoints(kpts1, image_size1)

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)
        desc0 = desc0 + self.kenc(kpts0, data["keypoint_scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["keypoint_scores1"])

        if n_lines0 != 0 and n_lines1 != 0:
            # Pre-compute the line encodings
            lines0 = normalize_keypoints(lines0, image_size0).reshape(b_size, n_lines0, 2, 2)
            lines1 = normalize_keypoints(lines1, image_size1).reshape(b_size, n_lines1, 2, 2)
            line_enc0 = self.lenc(lines0, data["line_scores0"])
            line_enc1 = self.lenc(lines1, data["line_scores1"])
        else:
            line_enc0 = torch.zeros(b_size, self.cfg.descriptor_dim, n_lines0 * 2, dtype=torch.float, device=device)
            line_enc1 = torch.zeros(b_size, self.cfg.descriptor_dim, n_lines1 * 2, dtype=torch.float, device=device)

        desc0, desc1 = self.gnn(desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)

        # Match all points (KP and line junctions)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        kp_scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        kp_scores = kp_scores / self.cfg.descriptor_dim**0.5
        kp_scores = log_double_softmax(kp_scores, self.bin_score)
        m0, m1, mscores0, mscores1 = self._get_matches(kp_scores)
        pred["log_assignment"] = kp_scores
        pred["matches0"] = m0
        pred["matches1"] = m1
        pred["match_scores0"] = mscores0
        pred["match_scores1"] = mscores1

        # Match the lines
        if n_lines0 > 0 and n_lines1 > 0:
            (line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores) = self._get_line_matches(
                desc0[:, :, : 2 * n_lines0],
                desc1[:, :, : 2 * n_lines1],
                lines_junc_idx0,
                lines_junc_idx1,
                self.final_line_proj,
            )
            if self.cfg.inter_supervision:
                for l in self.cfg.inter_supervision:
                    (line_scores_i, m0_lines_i, m1_lines_i, mscores0_lines_i, mscores1_lines_i) = (
                        self._get_line_matches(
                            self.gnn.inter_layers[l][0][:, :, : 2 * n_lines0],
                            self.gnn.inter_layers[l][1][:, :, : 2 * n_lines1],
                            lines_junc_idx0,
                            lines_junc_idx1,
                            self.inter_line_proj[self.layer2idx[l]],
                        )
                    )
                    pred[f"line_{l}_log_assignment"] = line_scores_i
                    pred[f"line_{l}_matches0"] = m0_lines_i
                    pred[f"line_{l}_matches1"] = m1_lines_i
                    pred[f"line_{l}_match_scores0"] = mscores0_lines_i
                    pred[f"line_{l}_match_scores1"] = mscores1_lines_i
        else:
            line_scores = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)
            m0_lines = torch.full((b_size, n_lines0), -1, device=device, dtype=torch.int64)
            m1_lines = torch.full((b_size, n_lines1), -1, device=device, dtype=torch.int64)
            mscores0_lines = torch.zeros((b_size, n_lines0), device=device, dtype=torch.float32)
            mscores1_lines = torch.zeros((b_size, n_lines1), device=device, dtype=torch.float32)
            raw_line_scores = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)
        pred["line_log_assignment"] = line_scores
        pred["line_matches0"] = m0_lines
        pred["line_matches1"] = m1_lines
        pred["line_match_scores0"] = mscores0_lines
        pred["line_match_scores1"] = mscores1_lines
        pred["raw_line_scores"] = raw_line_scores

        return pred


default_cfgs = {
    "gluestick": _cfg(
        drive="https://drive.google.com/uc?id=1HycIjTLV9iNc0th_g4-LlzSH94LhadzM",
        input_dim=256,
        descriptor_dim=256,
        keypoint_encoder=[32, 64, 128, 256],
        GNN_layers=["self", "cross"] * 9,
        sinkhorn_iterations=20,
        match_threshold=0.2,
        #
        bottleneck_dim=None,
        checkpointed=False,
        inter_supervision=None,
        num_line_iterations=1,
        line_attention=False,
        # weights=None,
    )
}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create model
    model = GlueStick(cfg=cfg)

    # load pretrained
    if pretrained:
        load_model_weights(model, name, cfg)

    return model


@MATCHERS_REGISTRY.register(name="gluestick", default_cfg=default_cfgs["gluestick"])
def gluestick(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="gluestick", cfg=cfg, **kwargs)
