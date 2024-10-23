import torch
from einops.einops import rearrange
import torchvision.transforms as tfn

from imm.base import MatcherModel
from imm.matchers.modules.aspanformer_modules import (
    CoarseMatching,
    FineMatching,
    FinePreprocess,
    LocalFeatureTransformer,
    LocalFeatureTransformer_Flow,
    PositionEncodingSine,
    build_backbone,
)
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
        "d_flow": 128,
        "nhead": 8,
        "nlevel": 3,
        "ini_layer_num": 2,
        "layer_num": 4,
        "nsample": [2, 8],
        "radius_scale": 5,
        "coarsest_level": [36, 36],  # [15, 20]
        "train_res": [832, 832],  # [480, 640]
        "test_res": [1152, 1152],  # None
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 2,  # 0
        "match_type": "dual_softmax",
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": False,
        "train_coarse_percent": 0.3,  # 0.2
        "train_pad_num_gt_min": 200,
        "sparse_spvs": True,
        "learnable_ds_temp": True,
    },
    "fine": {
        "d_model": 128,
        "d_ffn": 128,
        "nhead": 8,
        "layer_names": ["self", "cross"],
        "attention": "linear",
    },
    "loss": {
        "coarse_type": "focal",
        "coarse_weight": 1.0,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "pos_weight": 1.0,
        "neg_weight": 1.0,
        "fine_type": "l2_with_std",
        "fine_weight": 1.0,
        "fine_correct_thr": 1.0,
        "flow_weight": 0.1,
    },
}


class ASpanFormer(MatcherModel):
    required_inputs = [
        "image0",
        "image1",
    ]

    def __init__(self, cfg):
        super().__init__(cfg)

        # override match_coarse
        self.cfg.match_coarse.thr = self.cfg.match_threshold
        self.cfg.match_coarse.border_rm = self.cfg.border_rm

        # override coarse
        self.cfg.coarse.coarsest_level = self.cfg.coarsest_level
        self.cfg.coarse.test_res = self.cfg.test_res

        # Modules
        self.backbone = build_backbone(self.cfg)

        #
        self.pos_encoding = PositionEncodingSine(
            self.cfg["coarse"]["d_model"],
            pre_scaling=[
                self.cfg["coarse"]["train_res"],
                self.cfg["coarse"]["test_res"],
            ],
        )
        self.loftr_coarse = LocalFeatureTransformer_Flow(self.cfg["coarse"])
        self.coarse_matching = CoarseMatching(self.cfg["match_coarse"])
        self.fine_preprocess = FinePreprocess(self.cfg)
        self.loftr_fine = LocalFeatureTransformer(self.cfg["fine"])
        self.fine_matching = FineMatching()
        self.coarsest_level = self.cfg["coarse"]["coarsest_level"]

    def transform_inputs(self, data):
        image0 = tfn.Grayscale()(data["image0"])
        image1 = tfn.Grayscale()(data["image1"])

        # resize to 640 x 480
        image0 = tfn.Resize([480, 640], antialias=True)(image0)
        image1 = tfn.Resize([480, 640], antialias=True)(image1)

        scale00 = torch.tensor([data["image0"].shape[2] / 640, data["image0"].shape[1] / 480]).to(image0.device)
        scale11 = torch.tensor([data["image1"].shape[2] / 640, data["image1"].shape[1] / 480]).to(image1.device)

        if image0.dim() == 3:
            image0 = image0.unsqueeze(0)
            image1 = image1.unsqueeze(0)

        return {
            "image0": image0,
            "image1": image1,
            "scale00": scale00,
            "scale11": scale11,
        }

    def process_matches(self, data: torch.Dict[str, torch.Any], preds: torch.Tensor) -> torch.Dict[str, torch.Any]:
        # mutuals
        matches = preds["matches"][0]
        mscores = preds["mscores"][0]

        # keypoints
        mkpts0 = preds["kpts0"][0] * data["scale00"]
        mkpts1 = preds["kpts1"][0] * data["scale11"]

        return {
            "matches": matches,
            "mscores": mscores,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "kpts0": torch.empty(0, 2),
            "kpts1": torch.empty(0, 2),
        }

    def forward(self, data, online_resize=False):
        data = data.copy()

        data["pos_scale0"], data["pos_scale1"] = None, None

        # 1. Local Feature CNN
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data["image0"], data["image1"]], dim=0))

            (feat_c0, feat_c1), (feat_f0, feat_f1) = (
                feats_c.split(data["bs"]),
                feats_f.split(data["bs"]),
            )
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = (
                self.backbone(data["image0"]),
                self.backbone(data["image1"]),
            )

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module [N, HW, C]
        [feat_c0, pos_encoding0] = self.pos_encoding(feat_c0, data["pos_scale0"])
        [feat_c1, pos_encoding1] = self.pos_encoding(feat_c1, data["pos_scale1"])

        feat_c0 = rearrange(feat_c0, "n c h w -> n c h w ")
        feat_c1 = rearrange(feat_c1, "n c h w -> n c h w ")

        # adjust ds
        ds0 = [
            int(data["hw0_c"][0] / self.coarsest_level[0]),
            int(data["hw0_c"][1] / self.coarsest_level[1]),
        ]

        ds1 = [
            int(data["hw1_c"][0] / self.coarsest_level[0]),
            int(data["hw1_c"][1] / self.coarsest_level[1]),
        ]

        # if online_resize:
        #     ds0, ds1 = [4, 4], [4, 4]

        # mask training
        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0 = data["mask0"].flatten(-2)
            mask_c1 = data["mask1"].flatten(-2)

        # coarse-level loftr module
        feat_c0, feat_c1, flow_list = self.loftr_coarse(
            feat_c0,
            feat_c1,
            pos_encoding0,
            pos_encoding1,
            mask_c0,
            mask_c1,
            ds0,
            ds1,
        )
        # 3. match coarse-level and register predicted offset
        self.coarse_matching(feat_c0, feat_c1, flow_list, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # # 6. resize match coordinates back to input resolution
        # if online_resize:
        #     data['mkpts0_f'] *= data['online_resize_scale0']
        #     data['mkpts1_f'] *= data['online_resize_scale1']

        keypoints0 = []
        keypoints1 = []
        scores = []
        matches = []

        for bs in range(data["bs"]):
            mask = data["m_bids"] == bs
            #
            keypoints0.append(data["mkpts0_f"][mask])
            keypoints1.append(data["mkpts1_f"][mask])
            scores.append(data["mconf"][mask])
            matches.append(torch.arange(len(data["mconf"][mask])))

        return {
            "kpts0": keypoints0,
            "kpts1": keypoints1,
            "mscores": scores,
            "matches": matches,
        }

    def resize_input(self, data, train_res, df=32):
        h0, w0, h1, w1 = (
            data["image0"].shape[2],
            data["image0"].shape[3],
            data["image1"].shape[2],
            data["image1"].shape[3],
        )
        data["image0"], data["image1"] = (
            self.resize_df(data["image0"], df),
            self.resize_df(data["image1"], df),
        )

        if len(train_res) == 1:
            train_res_h = train_res_w = train_res
        else:
            train_res_h, train_res_w = train_res[0], train_res[1]
        data["pos_scale0"], data["pos_scale1"] = (
            [
                train_res_h / data["image0"].shape[2],
                train_res_w / data["image0"].shape[3],
            ],
            [
                train_res_h / data["image1"].shape[2],
                train_res_w / data["image1"].shape[3],
            ],
        )
        data["online_resize_scale0"], data["online_resize_scale1"] = (
            torch.tensor([w0 / data["image0"].shape[3], h0 / data["image0"].shape[2]])[None].cuda(),
            torch.tensor([w1 / data["image1"].shape[3], h1 / data["image1"].shape[2]])[None].cuda(),
        )

    def resize_df(self, image, df=32):
        h, w = image.shape[2], image.shape[3]
        h_new, w_new = h // df * df, w // df * df
        if h != h_new or w != w_new:
            img_new = tfn.Resize([h_new, w_new]).forward(image)
        else:
            img_new = image
        return img_new


default_cfgs = {
    "aspanformer_indoor": _cfg(
        drive="https://drive.google.com/uc?id=1p-Wbx26qsw3zSy1Cv7NTCMOYdlxy5CMs",
        gray=True,
        match_threshold=0.2,
        coarsest_level=[15, 20],
        border_rm=0,
        test_res=[480, 640],
        **_config,
    ),
    "aspanformer_outdoor": _cfg(
        drive="https://drive.google.com/uc?id=1XBLixHDw9HasBkABTMIdAP5LbILM1OdE",
        gray=True,
        match_threshold=0.2,
        coarsest_level=[15, 15],
        test_res=[480, 480],
        border_rm=2,
        **_config,
    ),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):
    #
    model = ASpanFormer(cfg=cfg)

    # load
    if pretrained:
        load_model_weights(model, name, cfg, state_key="state_dict", replace=("matcher.", ""))

    return model


@MATCHERS_REGISTRY.register(name="aspanformer_indoor", default_cfg=default_cfgs["aspanformer_indoor"])
def aspanformer_indoor(cfg=None, **kwargs):
    return _make_model(name="aspanformer_indoor", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="aspanformer_outdoor", default_cfg=default_cfgs["aspanformer_outdoor"])
def aspanformer_outdoor(cfg=None, **kwargs):
    return _make_model(name="aspanformer_outdoor", cfg=cfg, **kwargs)


# if __name__ == "__main__":
#     from core.visualization import plot_matches

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     path0 = "assets/phototourism_sample_images/st_pauls_cathedral_30776973_2635313996.jpg"
#     path1 = "assets/phototourism_sample_images/st_pauls_cathedral_37347628_10902811376.jpg"

#     matcher = create_matcher("aspanformer_outdoor")
#     matcher = matcher.to(device)
#     matcher = matcher.eval()

#     with torch.no_grad():
#         preds, image0, image1 = matcher.match_pairs(path0, path1, device=device, max_size=480)
#         preds = to_numpy(preds)

#     # show
#     plot_matches(image0, image1, preds)
