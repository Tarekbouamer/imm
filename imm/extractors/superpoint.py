from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from imm.base import FeatureModel, tfn_grayscale
from imm.extractors.modules.superpoints_modules import (
    remove_borders,
    sample_descriptors,
    simple_nms,
    top_k_keypoints,
)
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import EXTRACTORS_REGISTRY


class SuperPoint(FeatureModel):
    required_inputs = ["image"]

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # model
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.cfg["descriptor_dim"], kernel_size=1, stride=1, padding=0)

    def transform_inputs(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # to 4D
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # grayscale
        data["image"] = tfn_grayscale(data["image"])

        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        # Shared Encoder
        x = self.relu(self.conv1a(data["image"]))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.cfg["nms_radius"])

        # Extract keypoints
        keypoints = [torch.nonzero(s > self.cfg["keypoint_threshold"]) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[remove_borders(k, s, self.cfg["remove_borders"], h * 8, w * 8) for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.cfg["max_keypoints"] >= 0:
            keypoints, scores = list(zip(*[top_k_keypoints(k, s, self.cfg["max_keypoints"]) for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)]

        #
        # descriptors = [d.permute(1, 0) for d in descriptors]

        return {"kpts": keypoints, "scores": list(scores), "desc": descriptors}


# default configurations
default_cfgs = {
    "superpoint": _cfg(
        drive="https://drive.google.com/uc?id=1JjRJ5RLa3yx4VOSZ17mryJoXiWGbeCa1",
        descriptor_dim=256,
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        remove_borders=4,
    )
}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create model
    model = SuperPoint(cfg=cfg)

    # load pretrained
    if pretrained:
        load_model_weights(model, name, cfg)
    return model


@EXTRACTORS_REGISTRY.register(name="superpoint", default_cfg=default_cfgs["superpoint"])
def superpoint(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="superpoint", cfg=cfg, **kwargs)
