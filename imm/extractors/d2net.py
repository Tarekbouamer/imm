from typing import Any, Dict

import torch
import torch.nn.functional as F

from imm.base import FeatureModel
from imm.extractors.modules.d2net_modules import (
    DenseFeatureExtractionModule,
    HandcraftedLocalizationModule,
    HardDetectionModule,
    downscale_positions,
    interpolate_dense_features,
    upscale_positions,
)
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import EXTRACTORS_REGISTRY


class D2Net(FeatureModel):
    required_inputs = ["image"]

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)

        self.dense_feature_extraction = DenseFeatureExtractionModule(use_relu=True, use_cuda=False)
        self.detection = HardDetectionModule()
        self.localization = HandcraftedLocalizationModule()

    def transform_inputs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transform model inputs"""
        image = data["image"].flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = image * 255 - norm.view(1, 3, 1, 1)  # Caffe normalization
        return {"image": image}

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.dense_feature_extraction(data["image"])

        # Detect keypoints
        detections = self.detection(features)[0]
        fmap_pos = torch.nonzero(detections).t()

        displacements = self.localization(features)[0]
        displacements_i = displacements[0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]
        displacements_j = displacements[1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]

        mask = torch.min(torch.abs(displacements_i) < 0.5, torch.abs(displacements_j) < 0.5)
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([displacements_i[mask], displacements_j[mask]], dim=0)

        fmap_keypoints = fmap_pos[1:, :].float() + valid_displacements
        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        scores = features[0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]

        keypoints = keypoints.transpose(1, 0)[:, [1, 0]]  # N, 2 and swap x, y

        if self.cfg["max_keypoints"] > 0:
            idxs = (-scores).argsort()[: self.cfg["max_keypoints"]]
            keypoints = keypoints[idxs]
            scores = scores[idxs]

        # Compute descriptors
        keypoints_t = keypoints[:, [1, 0]].transpose(1, 0)  # swap x, y and transpose
        fmap_keypoints = downscale_positions(keypoints_t, scaling_steps=2)

        raw_descriptors, _, _ = interpolate_dense_features(fmap_keypoints, features[0])
        descriptors = F.normalize(raw_descriptors, dim=0)

        return {"kpts": [keypoints], "scores": [scores], "desc": [descriptors]}


default_cfgs = {
    "d2net_ots": _cfg(
        url="https://dusmanu.com/files/d2-net/d2_ots.pth",
        multiscale=False,
        max_keypoints=-1,
        descriptor_dim=512,
    ),
    "d2net_tf": _cfg(
        url="https://dusmanu.com/files/d2-net/d2_tf.pth",
        multiscale=False,
        max_keypoints=-1,
        descriptor_dim=512,
    ),
    "d2_tf_no_phototourism": _cfg(
        url="https://dusmanu.com/files/d2-net/d2_tf_no_phototourism.pth",
        multiscale=False,
        max_keypoints=-1,
        descriptor_dim=512,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    model = D2Net(cfg=cfg)

    if pretrained:
        load_model_weights(model, name, cfg, state_key="model")

    return model


@EXTRACTORS_REGISTRY.register(name="d2net_ots", default_cfg=default_cfgs["d2net_ots"])
def d2net_ots(cfg=None, **kwargs):
    return _make_model(name="d2net_ots", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(name="d2net_tf", default_cfg=default_cfgs["d2net_tf"])
def d2net_tf(cfg=None, **kwargs):
    return _make_model(name="d2net_tf", cfg=cfg, **kwargs)


@EXTRACTORS_REGISTRY.register(
    name="d2_tf_no_phototourism",
    default_cfg=default_cfgs["d2_tf_no_phototourism"],
)
def d2_tf_no_phototourism(cfg=None, **kwargs):
    return _make_model(name="d2_tf_no_phototourism", cfg=cfg, **kwargs)
