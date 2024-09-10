from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from imm.base import FeatureModel, tfn_image_net
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import EXTRACTORS_REGISTRY, create_extractor
from .modules.caps_modules import ResUNet


class CAPSnet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        # cfg
        self.cfg = cfg

        # description
        self.net = ResUNet(
            pretrained=False,
            encoder=cfg["backbone"],
            coarse_out_ch=cfg["coarse_feat_dim"],
            fine_out_ch=cfg["coarse_feat_dim"],
        )

    def forward(self, image):
        return self.net(image)

    @staticmethod
    def normalize(coord, h, w):
        c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).to(coord)
        coord_norm = (coord - c) / c
        return coord_norm

    def sample_feat_by_coord(self, x, coord_n, norm=False):
        feat = F.grid_sample(x, coord_n.unsqueeze(2), align_corners=True).squeeze(-1)
        if norm:
            feat = F.normalize(feat)
        feat = feat.transpose(1, 2)
        return feat


class CAPS(FeatureModel):
    required_inputs = ["image"]

    def __init__(self, detector, descriptor, cfg) -> None:
        super().__init__(cfg)
        self.detector = detector
        self.descriptor = descriptor

    def transform_inputs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)
        data["image"] = tfn_image_net(data["image"])
        return data

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        # Detection
        preds = self.detector.extract({"image": data["image"].clone()})

        # Keypoints
        kpts = preds["kpts"][0]
        scores = preds["scores"][0]

        # Transform inputs for descriptor
        data = self.transform_inputs(data)

        # Description
        xc, xf = self.descriptor(data["image"])
        hi, wi = data["image"].size()[2:]

        coord_n = self.descriptor.normalize(kpts.unsqueeze(0), hi, wi)
        feat_c = self.descriptor.sample_feat_by_coord(xc, coord_n)
        feat_f = self.descriptor.sample_feat_by_coord(xf, coord_n)

        descriptors = torch.cat((feat_c, feat_f), -1).squeeze(0)
        descriptors = descriptors.T
        return {"kpts": [kpts], "scores": [scores], "desc": [descriptors]}


default_cfgs = {
    "caps_sp": _cfg(
        drive="https://drive.google.com/uc?id=1vj3q7_phDcHiwIdDYppfSfoQiPe3lnOD",
        backbone="resnet50",
        descriptor_dim=256,
        coarse_feat_dim=128,
        fine_feat_dim=128,
        max_keypoints=-1,
    )
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    # detector
    detector = create_extractor("superpoint", pretrained=True)

    # descriptor
    descriptor = CAPSnet(cfg)

    if pretrained:
        load_model_weights(descriptor, name, cfg, state_key="state_dict")

    model = CAPS(detector, descriptor=descriptor, cfg=cfg)

    return model


@EXTRACTORS_REGISTRY.register(name="caps_sp", default_cfg=default_cfgs["caps_sp"])
def caps_sp(cfg=None, **kwargs):
    return _make_model(name="caps_sp", cfg=cfg, **kwargs)
