from typing import Any, Dict

from .lightglue import LightGlue
from torch import nn
import torch

from imm.base.matcher import MatcherModel
from imm.matchers._helper import MATCHERS_REGISTRY
from imm.misc import _cfg
from imm.registry.factory import load_model_weights


class LighterGlue(MatcherModel):
    """
    Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
        "name": "xfeat",  # just for interfacing
        "input_dim": 64,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": 0.95,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    def __init__(self, cfg: Dict[str, Any] = {}) -> None:
        super().__init__(cfg)
        LightGlue.default_conf = self.default_conf_xfeat

        # load model
        self.net = LightGlue(None)
        self.net.eval()

    def transform_inputs(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                if isinstance(data[k][0], torch.Tensor):
                    data[k] = torch.stack(data[k])
        return data

    def process_matches(self, data: Dict[str, Any], preds: torch.Tensor) -> Dict[str, Any]:
        #
        matches = preds["matches"][0]
        mscores = preds["scores"][0]

        kpts0 = data["kpts0"][0]
        kpts1 = data["kpts1"][0]

        # valid matches
        mkpts0 = kpts0[matches[..., 0]]
        mkpts1 = kpts1[matches[..., 1]]

        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mscores": mscores,
            "matches": matches,
            "kpts0": kpts0,
            "kpts1": kpts1,
        }

    @torch.inference_mode()
    def forward(self, data):
        result = self.net(data)
        return result


default_cfgs = {
    "lighterglue": _cfg(
        drive="https://drive.google.com/uc?id=1UpnI8sbG_wP8gNr4vZafU3fAwYhoaaXV",
        features="superpoint",
        match_threshold=0.1,
    ),
}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create model
    model = LighterGlue(cfg=cfg, **kwargs)

    # load pretrained
    if pretrained:
        load_model_weights(model.net, name, cfg)

    return model


@MATCHERS_REGISTRY.register(name="lighterglue", default_cfg=default_cfgs["lighterglue"])
def lighterglue(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="lighterglue", cfg=cfg, **kwargs)
