from typing import Any, Dict

import torch
import torch.nn.functional as F

from imm.base import FeatureModel
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import EXTRACTORS_REGISTRY
from .modules.disk_modules import Detector, thin_setup
from .modules.unet import Unet

DEFAULT_SETUP = {**thin_setup, "bias": True, "padding": True}


class DISK(FeatureModel):
    required_inputs = ["image"]

    def __init__(self, cfg, setup=DEFAULT_SETUP) -> None:
        super().__init__(cfg)

        #
        self.cfg = cfg

        #
        self.desc_dim = cfg["descriptor_dim"]
        #
        self.unet = Unet(
            in_features=3,
            size=cfg["kernel_size"],
            down=[16, 32, 64, 64, 64],
            up=[64, 64, 64, cfg["descriptor_dim"] + 1],
            setup=setup,
        )

        #
        self.detector = Detector(window=cfg["window"])

    def transform_inputs(self, data):
        """transform model inputs"""

        image = data["image"]
        if image.dim() == 3:
            image = image.unsqueeze(0)

        orig_h, orig_w = image.shape[-2:]
        new_h = round(orig_h / 16) * 16
        new_w = round(orig_w / 16) * 16

        image = F.pad(image, (0, new_w - orig_w, 0, new_h - orig_h))
        self.ori_size = (orig_w, orig_h)

        return {"image": image}

    def extract_features(self, data):
        #
        data = self.transform_inputs(data)

        return self.unet(data["image"])

    def detect(self, data, features=None):
        if features is None:
            features = self.extract_features(data)

        assert features.shape[1] == self.desc_dim + 1
        heatmap = features[:, self.desc_dim :]

        _keypoints = self.detector.nms(heatmap)

        # keypoints and scores
        keypoints = _keypoints[0].xys
        scores = _keypoints[0].logp

        # valid
        orig_w, orig_h = self.ori_size
        valid = torch.all(keypoints <= keypoints.new_tensor([orig_w, orig_h]) - 1, 1)
        keypoints = keypoints[valid]
        scores = scores[valid]

        # max and sort
        if self.cfg["max_keypoints"] > 0:
            idxs = (-scores).argsort()[: self.cfg["max_keypoints"]]
            keypoints = keypoints[idxs]
            scores = scores[idxs]

        out = {"kpts": keypoints, "scores": scores}

        return out

    def compute(self, data, features):
        keypoints = data["kpts"]
        scores = data["scores"]

        # keypoints
        x, y = keypoints.T

        # descriptors
        descriptors = features[:, : self.desc_dim][0]

        descriptors = descriptors[:, y, x].T
        descriptors = F.normalize(descriptors, dim=-1)

        data = {"kpts": keypoints, "scores": scores, "desc": descriptors}

        return data

    def forward(self, data):
        features = self.extract_features(data)
        data = self.detect(data, features)
        data = self.compute(data, features)

        kpts = data["kpts"]
        desc = data["desc"]
        scores = data["scores"]

        # transpose
        desc = desc.T

        return {
            "kpts": [kpts],
            "desc": [desc],
            "scores": [scores],
        }


default_cfgs = {
    "disk_depth": _cfg(
        drive="https://drive.google.com/uc?id=1SMNY0swehee2I9TNkvp2VMCJm0BTsRH5",
        kernel_size=5,
        window=8,
        max_keypoints=-1,
        descriptor_dim=128,
    ),
    "disk_epipolar": _cfg(
        drive="https://drive.google.com/uc?id=1hldj_irmF2BXI_AzUktKM3Qg57TjIdyv",
        kernel_size=5,
        window=8,
        max_keypoints=-1,
        descriptor_dim=128,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    # net
    model = DISK(cfg=cfg)

    if pretrained:
        load_model_weights(model, name, cfg, state_key="extractor")

    return model


@EXTRACTORS_REGISTRY.register(name="disk_depth", default_cfg=default_cfgs["disk_depth"])
def disk_depth(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="disk_depth", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(name="disk_epipolar", default_cfg=default_cfgs["disk_epipolar"])
def disk_epipolar(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="disk_epipolar", cfg=cfg, **kwargs)
