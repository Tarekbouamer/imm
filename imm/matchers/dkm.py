# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from imm.utils.device import to_numpy
# from imm.registry.factory import load_model_weights
# from core.registry.register import get_pretrained_cfg

# from imm.models.matchers._base import MatcherModel
# from imm.models.matchers.misc import create_matcher, register_matcher
# from imm.models.matchers.modules.dkm_modules import (
#     DFN,
#     GP,
#     ConvRefiner,
#     CosKernel,
#     Decoder,
#     ResNet50,
# )
# from imm.misc import _cfg

# default_dkm_cfg = {
#     "encoder": {
#         "pretrained": False,
#         "high_res": False,
#         "freeze_bn": False
#     },
#     "sample_mode": "threshold_balanced",
#     "match_threshold": 0.05,
#     "max_keypoints": 5000
# }


# class DKM(MatcherModel):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         #
#         self.cfg = {**default_dkm_cfg, **cfg}

#         # encoder
#         self.encoder = ResNet50(
#             pretrained=False, high_res=False, freeze_bn=False)

#         # decoder
#         coordinate_decoder = DFN()
#         #
#         gp32 = GP(CosKernel)
#         gp16 = GP(CosKernel)
#         gps = nn.ModuleDict({"32": gp32, "16": gp16})
#         #
#         proj = nn.ModuleDict({"16": nn.Conv2d(1024, 512, 1, 1),
#                               "32": nn.Conv2d(2048, 512, 1, 1)})
#         #
#         conv_refiner = nn.ModuleDict({
#             "16": ConvRefiner(2 * 512+128+(2*7+1)**2,
#                               2 * 512+128+(2*7+1)**2,
#                               3,
#                               displacement_emb_dim=128,
#                               local_corr_radius=7,
#                               corr_in_other=True),
#             "8": ConvRefiner(2 * 512+64+(2*3+1)**2,
#                              2 * 512+64+(2*3+1)**2,
#                              3,
#                              displacement_emb_dim=64,
#                              local_corr_radius=3,
#                              corr_in_other=True),
#             "4": ConvRefiner(2 * 256+32+(2*2+1)**2,
#                              2 * 256+32+(2*2+1)**2,
#                              3,
#                              displacement_emb_dim=32,
#                              local_corr_radius=2,
#                              corr_in_other=True),
#             "2": ConvRefiner(2 * 64+16,
#                              128+16,
#                              3,
#                              displacement_emb_dim=16),
#             "1": ConvRefiner(2 * 3+6,
#                              24,
#                              3,
#                              displacement_emb_dim=6
#                              )})

#         self.decoder = Decoder(coordinate_decoder, gps,
#                                proj, conv_refiner, detach=True)

#     def __extract_backbone_features(self, batch):
#         x_q = batch["image0"]
#         x_s = batch["image1"]

#         feature_pyramid = self.encoder(x_q), self.encoder(x_s)

#         return feature_pyramid

#     def __sample(self, dense_matches, dense_certainty, max_keypoints=10000):

#         if "threshold" in self.cfg["sample_mode"]:
#             upper_thresh = self.cfg["match_threshold"]
#             dense_certainty = dense_certainty.clone()
#             dense_certainty[dense_certainty > upper_thresh] = 1

#         elif "pow" in self.cfg["sample_mode"]:
#             dense_certainty = dense_certainty**(1/3)

#         elif "naive" in self.cfg["sample_mode"]:
#             dense_certainty = torch.ones_like(dense_certainty)

#         matches, certainty = (
#             dense_matches.reshape(-1, 4),
#             dense_certainty.reshape(-1),
#         )

#         expansion_factor = 4 if "balanced" in self.cfg["sample_mode"] else 1
#         good_samples = torch.multinomial(certainty,
#                                          num_samples=min(
#                                              expansion_factor*max_keypoints, len(certainty)),
#                                          replacement=False)
#         good_matches, good_certainty = matches[good_samples], certainty[good_samples]

#         if "balanced" not in self.cfg["sample_mode"]:
#             return good_matches, good_certainty

#         density = kde(good_matches, std=0.1)
#         p = 1 / (density+1)
#         # Basically should have at least 10 perfect neighbours, or around 100 ok ones
#         p[density < 10] = 1e-7
#         balanced_samples = torch.multinomial(p,
#                                              num_samples=min(
#                                                  max_keypoints, len(good_certainty)),
#                                              replacement=False)
#         return good_matches[balanced_samples], good_certainty[balanced_samples]

#     def __to_pixel_coordinates(self, matches, H_A, W_A, H_B, W_B):
#         kpts_A, kpts_B = matches[..., :2], matches[..., 2:]

#         kpts_A = torch.stack(
#             (W_A/2 * (kpts_A[..., 0]+1), H_A/2 * (kpts_A[..., 1]+1)), axis=-1)
#         kpts_B = torch.stack(
#             (W_B/2 * (kpts_B[..., 0]+1), H_B/2 * (kpts_B[..., 1]+1)), axis=-1)

#         return kpts_A, kpts_B

#     def _forward(self, batch):

#         # extract features
#         feature_pyramid = self.__extract_backbone_features(batch)
#         f_q_pyramid, f_s_pyramid = feature_pyramid

#         # decoder
#         dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)

#         return dense_corresps

#     def transform_inputs(self, _data):

#         data = {}
#         data["image0"] = self.transforms(_data["image0"])
#         data["image1"] = self.transforms(_data["image1"])
#         return data

#     def forward(self, data):

#         #
#         b, _, h0, w0 = data["image0"].shape
#         b, _, h1, w1 = data["image1"].shape

#         # extract dense correspondances [32, 16, 8, 4, 2, 1]
#         dense_corresps = self._forward(data)

#         # low res certainty
#         d16_certainty = dense_corresps[16]["dense_certainty"]
#         low_res_certainty = F.interpolate(d16_certainty, size=(
#             h0, w0), align_corners=False, mode="bilinear")
#         low_res_certainty = 0.5 * low_res_certainty * (low_res_certainty < 0.)

#         # high res
#         d1_certainty = dense_corresps[1]["dense_certainty"]
#         query_to_support = dense_corresps[1]["dense_flow"]

#         #
#         dense_certainty = d1_certainty - low_res_certainty
#         query_to_support = query_to_support.permute(0, 2, 3, 1)

#         # logits -> probs
#         dense_certainty = dense_certainty.sigmoid()

#         # filter out bounds
#         if (query_to_support.abs() > 1).any() and True:
#             wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
#             dense_certainty[wrong[:, None]] = 0

#         query_to_support = torch.clamp(query_to_support, -1, 1)

#         # im1 meshgrid coords
#         query_coords = torch.meshgrid((torch.linspace(-1 + 1 / h0, 1 - 1 / h0, h0, device=query_to_support.device),
#                                        torch.linspace(-1 + 1 / w0, 1 - 1 / w0, w0, device=query_to_support.device)))

#         query_coords = torch.stack((query_coords[1], query_coords[0]))
#         query_coords = query_coords[None].expand(b, 2, h0, w0)
#         query_coords = query_coords.permute(0, 2, 3, 1)

#         # wrap
#         warp = torch.cat((query_coords, query_to_support), dim=-1)

#         keypoints0, keypoints1, matches_scores, matches = [], [], [], []
#         for it in range(b):
#             # sample
#             it_matches, it_certainty = self.__sample(
#                 warp[it], dense_certainty[it], max_keypoints=self.cfg["max_keypoints"])

#             # coordinates
#             it_kpts0, it_kpts1 = self.__to_pixel_coordinates(
#                 it_matches, h0, w0, h1, w1)

#             #
#             keypoints0.append(it_kpts0)
#             keypoints1.append(it_kpts1)
#             matches_scores.append(it_certainty)
#             matches.append(torch.arange(len(it_certainty)))

#         # output
#         out = {}
#         out['keypoints0'] = keypoints0
#         out['keypoints1'] = keypoints1
#         out['matches_scores'] = matches_scores
#         out['matches'] = matches

#         return out


# def _make_model(name, cfg=None, pretrained=True, **kwargs):

#     # cfg
#     default_cfg = get_pretrained_cfg(name)
#     cfg = {**default_cfg, **cfg}

#     #
#     model = DKM(cfg)

#     # load
#     if pretrained:
#         load_model_weights(model, name, cfg)

#     return model


# default_cfgs = {
#     'dkm_outdoor':
#         _cfg(drive='https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth',
#              normalize="imagenet"),

#     'dkm_indoor':
#         _cfg(drive='https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth',
#              normalize="imagenet"),
# }


# @MATCHERS_REGISTRY
# def dkm_outdoor(cfg=None, **kwargs):
#     return _make_model(name="dkm_outdoor", cfg=cfg, **kwargs)


# @MATCHERS_REGISTRY
# def dkm_indoor(cfg=None, **kwargs):
#     return _make_model(name="dkm_indoor", cfg=cfg, **kwargs)


# if __name__ == '__main__':

#     from core.visualization import plot_matches

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     path0 = "assets/phototourism_sample_images/st_pauls_cathedral_30776973_2635313996.jpg"
#     path1 = "assets/phototourism_sample_images/st_pauls_cathedral_37347628_10902811376.jpg"

#     matcher = create_matcher("dkm_indoor")
#     matcher = matcher.to(device)
#     matcher = matcher.eval()


#     with torch.no_grad():
#         preds, image0, image1 = matcher.match_pairs(
#             path0, path1, device=device, max_size=(640, 480))
#         preds = to_numpy(preds)

#     # show
#     plot_matches(image0, image1, preds)
