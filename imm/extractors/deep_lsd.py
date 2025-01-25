from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from pytlsd import lsd as lsd_py
from torch import Tensor, nn

from imm.base import FeatureModel, tfn_grayscale
from imm.extractors._helper import EXTRACTORS_REGISTRY
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from .modules.deep_lsd import (
    VGGUNet,
    filter_outlier_lines,
    merge_lines,
    preprocess_angle,
)


class DeepLSD(FeatureModel):
    required_inputs = ["image"]

    default_conf = {
        "line_neighborhood": 5,
        "multiscale": False,
        "scale_factors": [1.0, 1.5],
        "detect_lines": True,
        "line_detection_params": {
            "merge": False,
            "grad_nfa": True,
            "filtering": "normal",
            "grad_thresh": 3,
        },
    }

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Base network
        self.backbone = VGGUNet(tiny=False)
        dim = 64

        # Predict the distance field and angle to the nearest line
        # DF head
        self.df_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )

        # Closest line direction head
        self.angle_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def normalize_df(self, df):
        return -torch.log(df / self.cfg["line_neighborhood"] + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.cfg["line_neighborhood"]

    def ms_forward(self, data):
        """Do several forward passes at multiple image resolutions
        and aggregate the results before extracting the lines."""
        img_size = data["image"].shape[2:]

        # Forward pass for each scale
        pred_df, pred_angle = [], []
        for s in self.cfg["scale_factors"]:
            img = F.interpolate(data["image"], scale_factor=s, mode="bilinear")
            with torch.no_grad():
                base = self.backbone(img)
                pred_df.append(self.denormalize_df(self.df_head(base)))
                pred_angle.append(self.angle_head(base) * np.pi)

        # Fuse the outputs together
        for i in range(len(self.cfg["scale_factors"])):
            pred_df[i] = F.interpolate(pred_df[i], img_size, mode="bilinear").squeeze(1)
            pred_angle[i] = F.interpolate(pred_angle[i], img_size, mode="nearest").squeeze(1)
        fused_df = torch.stack(pred_df, dim=0).mean(dim=0)
        fused_angle = torch.median(torch.stack(pred_angle, dim=0), dim=0)[0]

        out = {"df": fused_df, "line_level": fused_angle}
        return out

    def detect_afm_lines(self, img, df, line_level, filtering="normal", merge=False, grad_thresh=3, grad_nfa=True):
        """Detect lines from the line distance and angle field.
        Offer the possibility to ignore line in high DF values,
        and to merge close-by lines."""
        gradnorm = np.maximum(5 - df, 0).astype(np.float64)
        angle = line_level.astype(np.float64) - np.pi / 2
        angle = preprocess_angle(angle, img, mask=True)[0]
        angle[gradnorm < grad_thresh] = -1024

        # Detect lines
        lines = lsd_py(img.astype(np.float64), scale=1.0, gradnorm=gradnorm, gradangle=angle, grad_nfa=grad_nfa)[
            :, :4
        ].reshape(-1, 2, 2)

        # Optionally filter out lines based on the DF and line_level
        if filtering:
            if filtering == "strict":
                df_thresh, ang_thresh = 1.0, np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            angle = line_level - np.pi / 2
            lines = filter_outlier_lines(
                img,
                lines[:, :, [1, 0]],
                df,
                angle,
                mode="inlier_thresh",
                use_grad=False,
                inlier_thresh=0.5,
                df_thresh=df_thresh,
                ang_thresh=ang_thresh,
            )[0][:, :, [1, 0]]

        # Merge close-by lines together
        if merge:
            lines = merge_lines(lines, thresh=4, overlap_thresh=0).astype(np.float32)

        return lines

    def transform_inputs(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # to 4D
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # grayscale
        data["image"] = tfn_grayscale(data["image"])
        B, C, H, W = data["image"].shape
        data["size"] = torch.tensor([W, H])

        return data

    def forward(self, data):
        outputs = {}

        if self.cfg["multiscale"]:
            outputs = self.ms_forward(data)
        else:
            base = self.backbone(data["image"])

            # DF prediction
            outputs["df_norm"] = self.df_head(base).squeeze(1)
            outputs["df"] = self.denormalize_df(outputs["df_norm"])

            # Closest line direction prediction
            outputs["line_level"] = self.angle_head(base).squeeze(1) * np.pi

        # Detect line segments
        if self.cfg["detect_lines"]:
            lines = []
            np_img = (data["image"].cpu().numpy()[:, 0] * 255).astype(np.uint8)
            np_df = outputs["df"].cpu().numpy()
            np_ll = outputs["line_level"].cpu().numpy()
            for img, df, ll in zip(np_img, np_df, np_ll):
                line = self.detect_afm_lines(img, df, ll, **self.cfg["line_detection_params"])
                lines.append(line)

            # filter out lines that are too short
            for line in lines:
                lengths = np.linalg.norm(line[:, 0] - line[:, 1], axis=1)
                lines = line[lengths > self.cfg["min_length"]]

                scores = np.sqrt(lengths[lengths >= self.cfg["min_length"]])

                # keep the best scoring
                indices = np.argsort(-scores)

                if self.cfg["max_lines"] is not None and self.cfg["max_lines"] > 0:
                    indices = indices[: self.cfg["max_lines"]]
                    lines = lines[indices]
                    scores = scores[indices]

            outputs["lines"] = lines.reshape(-1, 4)
            outputs["scores"] = scores

        return outputs


# default configurations
default_cfgs = {
    "deep_lsd": _cfg(
        url="https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar ",
        line_neighborhood=5,
        multiscale=False,
        scale_factors=[1.0, 1.5],
        min_length=30,
        max_lines=-1,
        detect_lines=True,
        line_detection_params={
            "merge": False,
            "grad_nfa": True,
            "filtering": "normal",
            "grad_thresh": 3,
        },
    )
}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create model
    model = DeepLSD(cfg=cfg)

    # load pretrained
    if pretrained:
        load_model_weights(model, name, cfg, state_key="model")
    return model


@EXTRACTORS_REGISTRY.register(name="deep_lsd", default_cfg=default_cfgs["deep_lsd"])
def deep_lsd(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="deep_lsd", cfg=cfg, **kwargs)
