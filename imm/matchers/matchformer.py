from typing import Any, Dict
import torch

from einops.einops import rearrange
import torchvision.transforms as tfn

from imm.base.matcher import MatcherModel
from imm.matchers._helper import MATCHERS_REGISTRY
from imm.misc import _cfg
from imm.registry.factory import load_model_weights
from .efficient_loftr import resize_to_divisible


from .modules.matchformer import FinePreprocess, CoarseMatching, FineMatching, build_backbone

_config = {
    "backbone_type": "largela",  # litela, largela, litesea, largesea
    "scens": "indoor",  # indoor, outdoor
    "resolution": (8, 2),  # (8,2), (8,4)
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 0,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": False,
        "train_coarse_percent": 0.2,
        "train_pad_num_gt_min": 200,
        "sparse_spvs": True,
    },
    "fine": {
        "d_model": 128,
        "d_ffn": 128,
    },
}


class MatchFormer(MatcherModel):
    required_inputs = [
        "image0",
        "image1",
    ]

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # Misc
        self.backbone = build_backbone(self.cfg)
        self.coarse_matching = CoarseMatching(self.cfg["match_coarse"])
        self.fine_preprocess = FinePreprocess(self.cfg)
        self.fine_matching = FineMatching()

    def transform_inputs(self, data: Dict) -> Dict:
        """transform model inputs"""

        image0 = resize_to_divisible(data["image0"], 32)
        image1 = resize_to_divisible(data["image1"], 32)

        # Grayscale
        image0 = tfn.Grayscale()(image0)
        image1 = tfn.Grayscale()(image1)

        # Get shapes
        w0, h0 = image0.shape[2], image0.shape[1]
        w1, h1 = image1.shape[2], image1.shape[1]

        Ws = max(w0, h0, w1, h1)
        Hs = min(w0, h0, w1, h1)

        # resize
        image0 = tfn.Resize([Hs, Ws], antialias=True)(image0)
        image1 = tfn.Resize([Hs, Ws], antialias=True)(image1)

        scale00 = torch.tensor([data["image0"].shape[2] / Ws, data["image0"].shape[1] / Hs]).to(image0.device)
        scale11 = torch.tensor([data["image1"].shape[2] / Ws, data["image1"].shape[1] / Hs]).to(image1.device)

        # add batch dim
        if image0.dim() == 3:
            image0 = image0.unsqueeze(0)
            image1 = image1.unsqueeze(0)

        return {
            "image0": image0,
            "image1": image1,
            "scale00": scale00,
            "scale11": scale11,
        }

    def process_matches(self, data: dict, preds: dict) -> dict:
        # mutuals
        matches = preds["matches"][0]
        mscores = preds["mscores"][0]

        # keypoints
        mkpts0 = preds["kpts0"][0] * data["scale00"]
        mkpts1 = preds["kpts1"][0] * data["scale11"]

        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mscores": mscores,
            "kpts0": torch.empty(0, 2),
            "kpts1": torch.empty(0, 2),
        }

    def forward(self, data):
        #
        image0 = data["image0"]
        image1 = data["image1"]

        #
        out_data = {}
        out_data.update({"bs": image0.size(0), "hw0_i": image0.shape[2:], "hw1_i": image1.shape[2:]})

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)

        if out_data["hw0_i"] == out_data["hw1_i"]:
            feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(out_data["bs"]), feats_f.split(out_data["bs"])
        else:
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(image0), self.backbone(image1)

        out_data.update({"hw0_c": feat_c0.shape[2:], "hw1_c": feat_c1.shape[2:], "hw0_f": feat_f0.shape[2:], "hw1_f": feat_f1.shape[2:]})

        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        # match coarse-level
        self.coarse_matching(feat_c0, feat_c1, out_data, mask_c0=mask_c0, mask_c1=mask_c1)

        # fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, out_data)

        # match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, out_data)

        # return
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
    "matchformer_largesea": _cfg(
        cfg_base=_config,
        drive="https://drive.google.com/uc?id=1EjeSvU3ARZg5mn2PlqNDWMu9iwS7Zf_m",
        backbone_type="largesea",
        scens="indoor",
        resolution=(8, 2),
        coarse={
            "d_model": 256,
            "d_ffn": 256,
        },
    ),
    "matchformer_litela": _cfg(
        cfg_base=_config,
        drive="https://drive.google.com/uc?id=11ClOQ_VrlsT7PxK6jQr5AW1Fd0YMbB3R",
        backbone_type="litela",
        scens="indoor",
        resolution=(8, 4),
        coarse={
            "d_model": 192,
            "d_ffn": 192,
        },
    ),
    "matchformer_largela": _cfg(
        cfg_base=_config,
        drive="https://drive.google.com/uc?id=1Ii-z3dwNwGaxoeFVSE44DqHdMhubYbQf",
        backbone_type="largela",
        scens="outdoor",
        resolution=(8, 2),
        coarse={
            "d_model": 256,
            "d_ffn": 256,
        },
    ),
    "matchformer_litesea": _cfg(
        cfg_base=_config,
        drive="https://drive.google.com/uc?id=1etaU9mM8bGT2AKT56ph6iqUdpV1daFBz",
        backbone_type="litesea",
        scens="outdoor",
        resolution=(8, 4),
        coarse={
            "d_model": 192,
            "d_ffn": 192,
        },
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    model = MatchFormer(cfg=cfg)

    # load
    if pretrained:
        load_model_weights(model, name, cfg, replace=("matcher.", ""))

    return model


@MATCHERS_REGISTRY.register(name="matchformer_largela", default_cfg=default_cfgs["matchformer_largela"])
def matchformer_largela(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="matchformer_largela", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="matchformer_largesea", default_cfg=default_cfgs["matchformer_largesea"])
def matchformer_largesea(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="matchformer_largesea", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="matchformer_litela", default_cfg=default_cfgs["matchformer_litela"])
def matchformer_litela(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="matchformer_litela", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="matchformer_litesea", default_cfg=default_cfgs["matchformer_litesea"])
def matchformer_litesea(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="matchformer_litesea", cfg=cfg, **kwargs)
