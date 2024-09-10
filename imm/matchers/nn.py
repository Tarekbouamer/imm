# logger
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from imm.base import MatcherModel
from imm.misc import _cfg

from ._helper import MATCHERS_REGISTRY


class MutualNearestNeighbor(MatcherModel):
    """mutual nearest neighboor matcher"""

    required_inputs = ["desc0", "desc1"]

    def __init__(self, cfg: Dict = {}):
        super().__init__(cfg)
        self.dummy = torch.nn.Parameter()

    def transform_inputs(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                if isinstance(data[k][0], torch.Tensor):
                    data[k] = torch.stack(data[k])
        return data

    def process_matches(self, data: Dict[str, Any], preds: torch.Tensor) -> Dict[str, Any]:
        # mutuals
        matches = preds["matches"][0]
        mscores = preds["mscores"][0]

        kpts0 = data["kpts0"][0]
        kpts1 = data["kpts1"][0]

        valid = torch.where(matches != -1)[0]

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        scores = mscores[valid]

        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mscores": scores,
            "kpts0": kpts0,
            "kpts1": kpts1,
        }

    def forward(self, data: Dict[str, torch.Tensor]):
        # unpack
        desc0 = data["desc0"][0]
        desc1 = data["desc1"][0]

        assert desc0.shape[1] != 0 and desc1.shape[1] != 0, f"desc0: {desc0.shape}, desc1: {desc1.shape}"
        assert desc0.shape[0] == desc1.shape[0], f"desc0: {desc0.shape}, desc1: {desc1.shape}"

        # normalize
        desc0 = functional.normalize(desc0, dim=0)
        desc1 = functional.normalize(desc1, dim=0)

        # similarity
        nn_sim = torch.einsum("dn, dm->nm", desc0, desc1)

        # mutual nearest neighbor
        nn_dist_01, nn_idx_01 = torch.max(nn_sim, dim=1)
        nn_dist_10, nn_idx_10 = torch.max(nn_sim, dim=0)

        nn_dist = 2 * (1 - nn_dist_01)

        #
        ids1 = torch.arange(0, nn_sim.shape[0], device=desc0.device)

        # cross check
        mask = ids1 == nn_idx_10[nn_idx_01]

        # thd
        if self.cfg.match_threshold > 0.0:
            mask = mask & (nn_dist <= self.cfg.match_threshold)

        matches = torch.where(mask, nn_idx_01, nn_idx_01.new_tensor(-1))
        scores = torch.where(mask, (nn_dist + 1) / 2.0, nn_dist.new_tensor(0.0))

        #
        out = {"matches": matches.unsqueeze(0), "mscores": scores.unsqueeze(0)}

        return out


default_cfgs = {"nn": _cfg(match_threshold=0.4)}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    return MutualNearestNeighbor(cfg=cfg)


@MATCHERS_REGISTRY.register(name="nn", default_cfg=default_cfgs["nn"])
def nn(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="nn", cfg=cfg, **kwargs)
