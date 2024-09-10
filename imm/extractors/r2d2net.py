import torch
import torch.nn.functional as functional

from imm.base import FeatureModel, tfn_image_net
from imm.extractors.modules.r2d2net_modules import (
    Fast_Quad_L2Net_ConfCFS,
    NonMaxSuppression,
    Quad_L2Net_ConfCFS,
)
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import EXTRACTORS_REGISTRY


class R2d2Net(FeatureModel):
    required_inputs = ["image"]

    def __init__(self, net, cfg):
        super().__init__(cfg)

        #
        self.net = net

        self.detector = NonMaxSuppression(
            rel_thr=cfg["reliability_threshold"],
            rep_thr=cfg["repetability_threshold"],
        )

    def transform_inputs(self, data):
        """transform model inputs"""

        # to 4D
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        image = tfn_image_net(data["image"])
        return image

    def forward(self, image):
        # # transform inputs
        # image = self.transform_inputs(data)

        # extract features
        features = self.net.extract_features(image)

        # ureliability &&  urepeatability
        ureliability = self.net.clf(features**2)
        urepeatability = self.net.sal(features**2)

        # reliability &&  repeatability
        repeatability = self.net.softmax(urepeatability)
        reliability = self.net.softmax(ureliability)

        # nms
        y, x = self.detector(reliability, repeatability)

        # scores
        C = reliability[0, 0, y, x]
        Q = repeatability[0, 0, y, x]
        scores = C * Q

        # keypoints
        X = x.float()
        Y = y.float()

        keypoints = torch.stack([X, Y], dim=-1)

        if self.cfg["max_keypoints"] > 0:
            idxs = (-scores).argsort()[: self.cfg["max_keypoints"]]
            keypoints = keypoints[idxs]
            scores = scores[idxs]

        # compute
        x, y = keypoints[:, 0], keypoints[:, 1]
        x = x.long()
        y = y.long()

        descriptors = functional.normalize(features, p=2, dim=1)
        descriptors = descriptors[0, :, y, x]

        return {
            "kpts": [keypoints],
            "scores": [scores],
            "desc": [descriptors],
        }


default_cfgs = {
    "r2d2_WASF_N16": _cfg(
        drive="https://drive.google.com/uc?id=1yHiLse1yopT7Ylsx6iVZ3M-_WRaKssp9",
        max_keypoints=5000,
        reliability_threshold=0.7,
        repetability_threshold=0.7,
        descriptor_dim=128,
    ),
    "r2d2_WASF_N8_big": _cfg(
        drive="https://drive.google.com/uc?id=1qUtQMZPU8x4Kv0jwbm22bEK6tNvO7qPi",
        max_keypoints=5000,
        reliability_threshold=0.7,
        repetability_threshold=0.7,
        descriptor_dim=128,
    ),
    "r2d2_WAF_N16": _cfg(
        drive="https://drive.google.com/uc?id=1SPPnagMOXv0aFEBUAhlY42WFZ2C6ArFg",
        max_keypoints=5000,
        reliability_threshold=0.7,
        repetability_threshold=0.7,
        descriptor_dim=128,
    ),
    "faster2d2_WASF_N16": _cfg(
        drive="https://drive.google.com/uc?id=1glXoORF9-7N6zR4-fFengt_J1lMyQaZV",
        max_keypoints=5000,
        reliability_threshold=0.7,
        repetability_threshold=0.7,
        descriptor_dim=128,
    ),
    "faster2d2_WASF_N8_big": _cfg(
        drive="https://drive.google.com/uc?id=1gvRap5g0ORnk9s4YCR7md-qs2JMeMGqn",
        max_keypoints=5000,
        reliability_threshold=0.7,
        repetability_threshold=0.7,
        descriptor_dim=128,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    if name == "r2d2_WASF_N16":
        net = Quad_L2Net_ConfCFS()

    if name == "r2d2_WASF_N8_big":
        net = Quad_L2Net_ConfCFS(mchan=6)

    if name == "r2d2_WAF_N16":
        net = Quad_L2Net_ConfCFS()

    if name == "faster2d2_WASF_N16":
        net = Fast_Quad_L2Net_ConfCFS()

    if name == "faster2d2_WASF_N8_big":
        net = Fast_Quad_L2Net_ConfCFS(mchan=6)

    if pretrained:
        load_model_weights(net, name, cfg, state_key="state_dict", replace=("module.", ""))

    return R2d2Net(net, cfg)


@EXTRACTORS_REGISTRY.register(name="r2d2_WASF_N16", default_cfg=default_cfgs["r2d2_WASF_N16"])
def r2d2_WASF_N16(cfg=None, **kwargs):
    return _make_model(name="r2d2_WASF_N16", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(name="r2d2_WASF_N8_big", default_cfg=default_cfgs["r2d2_WASF_N8_big"])
def r2d2_WASF_N8_big(cfg=None, **kwargs):
    return _make_model(name="r2d2_WASF_N8_big", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(name="r2d2_WAF_N16", default_cfg=default_cfgs["r2d2_WAF_N16"])
def r2d2_WAF_N16(cfg=None, **kwargs):
    return _make_model(name="r2d2_WAF_N16", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(name="faster2d2_WASF_N16", default_cfg=default_cfgs["faster2d2_WASF_N16"])
def faster2d2_WASF_N16(cfg=None, **kwargs):
    return _make_model(name="faster2d2_WASF_N16", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(
    name="faster2d2_WASF_N8_big",
    default_cfg=default_cfgs["faster2d2_WASF_N8_big"],
)
def faster2d2_WASF_N8_big(cfg=None, **kwargs):
    return _make_model(name="faster2d2_WASF_N8_big", cfg=cfg, **kwargs)
