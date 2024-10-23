from typing import Dict

import torch
from einops.einops import rearrange
from loguru import logger
from pyparsing import Any
import torchvision.transforms as tfm

from imm.base.matcher import MatcherModel
from imm.matchers._helper import MATCHERS_REGISTRY
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from .modules.efficient_loftr_modules import CoarseMatching, FineMatching, FinePreprocess, LocalFeatureTransformer, build_backbone


def detect_NaN(feat_0, feat_1):
    logger.info("NaN detected in feature")
    logger.info(f"#NaN in feat_0: {torch.isnan(feat_0).int().sum()}, #NaN in feat_1: {torch.isnan(feat_1).int().sum()}")
    feat_0[torch.isnan(feat_0)] = 0
    feat_1[torch.isnan(feat_1)] = 0


def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    return matcher


_config = {
    "backbone_type": "RepVGG",
    "align_corner": False,
    "resolution": (8, 1),
    "fine_window_size": 8,
    "mp": False,
    "replace_nan": True,
    "half": False,
    "backbone": {"block_dims": [64, 128, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross"] * 4,
        "agg_size0": 4,
        "agg_size1": 4,
        "no_flash": False,
        "rope": True,
        "npe": [832, 832, 832, 832],
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 2,
        "dsmax_temperature": 0.1,
        "skip_softmax": False,
        "fp16matmul": False,
        "train_coarse_percent": 0.2,
        "train_pad_num_gt_min": 200,
    },
    "match_fine": {"local_regress_temperature": 10.0, "local_regress_slicedim": 8},
}


def resize_to_divisible(img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
    """Resize to be divisible by a factor. Useful for ViT based models.

    Args:
        img (torch.Tensor): img as tensor, in (*, H, W) order
        divisible_by (int, optional): factor to make sure img is divisible by. Defaults to 14.

    Returns:
        torch.Tensor: img tensor with divisible shape
    """
    h, w = img.shape[-2:]

    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    return img


class EfficinetLoFTR(MatcherModel):
    required_inputs = [
        "image0",
        "image1",
    ]

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # Modules
        self.backbone = build_backbone(self.cfg)
        self.loftr_coarse = LocalFeatureTransformer(self.cfg)
        self.coarse_matching = CoarseMatching(self.cfg["match_coarse"])
        self.fine_preprocess = FinePreprocess(self.cfg)
        self.fine_matching = FineMatching(self.cfg)

    def transform_inputs(self, data: Dict) -> Dict:
        """transform model inputs"""

        image0 = resize_to_divisible(data["image0"], 32)
        image1 = resize_to_divisible(data["image1"], 32)

        image0 = tfm.Grayscale()(image0)
        image1 = tfm.Grayscale()(image1)

        if image0.dim() == 3:
            image0 = image0.unsqueeze(0)
            image1 = image1.unsqueeze(0)

        return {"image0": image0, "image1": image1}

    def process_matches(self, data: dict, preds: dict) -> dict:
        # mutuals
        matches = preds["matches"][0]
        mscores = preds["mscores"][0]

        kpts0 = mkpts0 = preds["kpts0"][0]
        kpts1 = mkpts1 = preds["kpts1"][0]

        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mscores": mscores,
            "kpts0": torch.empty(0, 2),
            "kpts1": torch.empty(0, 2),
        }

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        #
        image0 = data["image0"]
        image1 = data["image1"]

        # 1. Local Feature CNN
        out_data = {}
        out_data.update(
            {
                "bs": image0.size(0),
                "hw0_i": image0.shape[2:],
                "hw1_i": image1.shape[2:],
            }
        )

        if out_data["hw0_i"] == out_data["hw1_i"]:  # faster & better BN convergence
            ret_dict = self.backbone(torch.cat([image0, image1], dim=0))
            feats_c = ret_dict["feats_c"]
            out_data.update(
                {
                    "feats_x2": ret_dict["feats_x2"],
                    "feats_x1": ret_dict["feats_x1"],
                }
            )
            (feat_c0, feat_c1) = feats_c.split(out_data["bs"])
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(image0), self.backbone(image1)
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            out_data.update(
                {
                    "feats_x2_0": ret_dict0["feats_x2"],
                    "feats_x1_0": ret_dict0["feats_x1"],
                    "feats_x2_1": ret_dict1["feats_x2"],
                    "feats_x1_1": ret_dict1["feats_x1"],
                }
            )

        mul = self.cfg["resolution"][0] // self.cfg["resolution"][1]
        out_data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            }
        )

        # 2. coarse-level loftr module
        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        # detect NaN during mixed precision training
        if self.cfg["replace_nan"] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            detect_NaN(feat_c0, feat_c1)

        # 3. match coarse-level
        self.coarse_matching(
            feat_c0,
            feat_c1,
            out_data,
            mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0,
            mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1,
        )

        # prevent fp16 overflow during mixed precision training
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, out_data)

        # detect NaN during mixed precision training
        if self.cfg["replace_nan"] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
            detect_NaN(feat_f0_unfold, feat_f1_unfold)

        del feat_c0, feat_c1, mask_c0, mask_c1

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, out_data)

        #
        keypoints0 = []
        keypoints1 = []
        scores = []
        matches = []

        for bs in range(out_data["bs"]):
            mask = out_data["m_bids"] == bs
            #
            keypoints0.append(out_data["mkpts0_f"][mask])
            keypoints1.append(out_data["mkpts1_f"][mask])
            scores.append(out_data["mconf"][mask])
            matches.append(torch.arange(len(out_data["mconf"][mask])))

        return {
            "kpts0": keypoints0,
            "kpts1": keypoints1,
            "mscores": scores,
            "matches": matches,
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


default_cfgs = {
    "efficient_loftr": _cfg(
        drive="https://drive.google.com/uc?id=1jFy2JbMKlIp82541TakhQPaoyB5qDeic",
        gray=True,
        match_threshold=0.2,
        temp_bug_fix=True,
        **_config,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    model = EfficinetLoFTR(cfg=cfg)

    # load
    if pretrained:
        load_model_weights(model, name, cfg, state_key="state_dict", replace=("matcher.", ""))

    return model


@MATCHERS_REGISTRY.register(name="efficient_loftr", default_cfg=default_cfgs["efficient_loftr"])
def efficient_loftr(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="efficient_loftr", cfg=cfg, **kwargs)
