from typing import Any, Dict

import torch
import torchvision.transforms as tfm
from kornia.feature.loftr.backbone import build_backbone
from kornia.feature.loftr.loftr_module import (
    FinePreprocess,
    LocalFeatureTransformer,
)
from kornia.feature.loftr.utils.coarse_matching import CoarseMatching
from kornia.feature.loftr.utils.fine_matching import FineMatching
from kornia.feature.loftr.utils.position_encoding import PositionEncodingSine
from kornia.geometry import resize

from imm.base.matcher import MatcherModel
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import MATCHERS_REGISTRY

_config = {
    "backbone_type": "ResNetFPN",
    "resolution": (8, 2),
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": [
            "self",
            "cross",
            "self",
            "cross",
            "self",
            "cross",
            "self",
            "cross",
        ],
        "attention": "linear",
        "temp_bug_fix": False,
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 0,  # 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": False,
        "train_coarse_percent": 0.2,  # 0.3, #
        "train_pad_num_gt_min": 200,
    },
    "fine": {
        "d_model": 128,
        "d_ffn": 128,
        "nhead": 8,
        "layer_names": ["self", "cross"],
        "attention": "linear",
    },
}


class LoFTR(MatcherModel):
    required_inputs = [
        "image0",
        "image1",
    ]

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # override match_coarse
        self.cfg.match_coarse.thr = self.cfg.match_threshold
        self.cfg.coarse.temp_bug_fix = self.cfg.temp_bug_fix

        # Modules
        self.backbone = build_backbone(self.cfg)

        self.pos_encoding = PositionEncodingSine(
            self.cfg["coarse"]["d_model"],
            temp_bug_fix=self.cfg["coarse"]["temp_bug_fix"],
        )
        self.loftr_coarse = LocalFeatureTransformer(self.cfg["coarse"])
        self.coarse_matching = CoarseMatching(self.cfg["match_coarse"])
        self.fine_preprocess = FinePreprocess(self.cfg)
        self.loftr_fine = LocalFeatureTransformer(self.cfg["fine"])
        self.fine_matching = FineMatching()

    def transform_inputs(self, data: Dict) -> Dict:
        """transform model inputs"""
        image0 = tfm.Grayscale()(data["image0"])
        image1 = tfm.Grayscale()(data["image1"])

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

    def forward(self, data: dict) -> dict:
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
            feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = (
                feats_c.split(out_data["bs"]),
                feats_f.split(out_data["bs"]),
            )
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = (
                self.backbone(image0),
                self.backbone(image1),
            )

        out_data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        feat_c0 = self.pos_encoding(feat_c0).permute(0, 2, 3, 1)
        n, h, w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        n1, h1, w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        mask_c0 = mask_c1 = None  # mask is useful in training

        if "mask0" in data:
            mask_c0 = resize(data["mask0"], out_data["hw0_c"], interpolation="nearest").flatten(-2)
            # mask_c0 = data['mask0'].flatten(-2)

        if "mask1" in data:
            mask_c1 = resize(data["mask1"], out_data["hw1_c"], interpolation="nearest").flatten(-2)
            # mask_c1 = data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, out_data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, out_data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

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


default_cfgs = {
    "loftr_indoor_ds_new": _cfg(
        drive="https://drive.google.com/uc?id=1UqYrFzAQO7pgEA2n1S_IpNyX39a5Sw4C",
        gray=True,
        match_threshold=0.2,
        temp_bug_fix=True,
        **_config,
    ),
    "loftr_indoor_ds": _cfg(
        drive="https://drive.google.com/uc?id=14daC7NRRqmdB6T3KjIp8AERn2x9IhqZ0",
        gray=True,
        match_threshold=0.2,
        temp_bug_fix=False,
        **_config,
    ),
    "loftr_outdoor_ds": _cfg(
        drive="https://drive.google.com/uc?id=1kGV9QHuqSKVPucr5QNTdErsGSymkFIdy",
        gray=True,
        match_threshold=0.2,
        temp_bug_fix=False,
        **_config,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    model = LoFTR(cfg=cfg)

    # load
    if pretrained:
        load_model_weights(model, name, cfg, state_key="state_dict", replace=("matcher.", ""))

    return model


@MATCHERS_REGISTRY.register(name="loftr_indoor_ds_new", default_cfg=default_cfgs["loftr_indoor_ds_new"])
def loftr_indoor_ds_new(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="loftr_indoor_ds_new", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="loftr_indoor_ds", default_cfg=default_cfgs["loftr_indoor_ds"])
def loftr_indoor_ds(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="loftr_indoor_ds", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="loftr_outdoor_ds", default_cfg=default_cfgs["loftr_outdoor_ds"])
def loftr_outdoor_ds(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="loftr_outdoor_ds", cfg=cfg, **kwargs)
