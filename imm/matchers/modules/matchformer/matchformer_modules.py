# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.einops import rearrange
# import math

# from kornia.geometry.subpix import dsnt
# from kornia.utils.grid import create_meshgrid
# from functools import partial
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from einops.einops import repeat


# INF = 1e9


# #############################################################
# ### Coarse Matching
# #############################################################


# def mask_border(m, b: int, v):
#     """Mask borders with value
#     Args:
#         m (torch.Tensor): [N, H0, W0, H1, W1]
#         b (int)
#         v (m.dtype)
#     """
#     if b <= 0:
#         return

#     m[:, :b] = v
#     m[:, :, :b] = v
#     m[:, :, :, :b] = v
#     m[:, :, :, :, :b] = v
#     m[:, -b:] = v
#     m[:, :, -b:] = v
#     m[:, :, :, -b:] = v
#     m[:, :, :, :, -b:] = v


# def mask_border_with_padding(m, bd, v, p_m0, p_m1):
#     if bd <= 0:
#         return

#     m[:, :bd] = v
#     m[:, :, :bd] = v
#     m[:, :, :, :bd] = v
#     m[:, :, :, :, :bd] = v

#     h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
#     h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
#     for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
#         m[b_idx, h0 - bd :] = v
#         m[b_idx, :, w0 - bd :] = v
#         m[b_idx, :, :, h1 - bd :] = v
#         m[b_idx, :, :, :, w1 - bd :] = v


# def compute_max_candidates(p_m0, p_m1):
#     """Compute the max candidates of all pairs within a batch

#     Args:
#         p_m0, p_m1 (torch.Tensor): padded masks
#     """
#     h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
#     h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
#     max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
#     return max_cand


# class CoarseMatching(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         # general config
#         self.thr = config["thr"]
#         self.border_rm = config["border_rm"]
#         # -- # for trainig fine-level LoFTR
#         self.train_coarse_percent = config["train_coarse_percent"]
#         self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

#         # we provide 2 options for differentiable matching
#         self.match_type = config["match_type"]
#         if self.match_type == "dual_softmax":
#             self.temperature = config["dsmax_temperature"]
#         else:
#             raise NotImplementedError()

#     def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
#         """
#         Args:
#             feat0 (torch.Tensor): [N, L, C]
#             feat1 (torch.Tensor): [N, S, C]
#             data (dict)
#             mask_c0 (torch.Tensor): [N, L] (optional)
#             mask_c1 (torch.Tensor): [N, S] (optional)
#         Update:
#             data (dict): {
#                 'b_ids' (torch.Tensor): [M'],
#                 'i_ids' (torch.Tensor): [M'],
#                 'j_ids' (torch.Tensor): [M'],
#                 'gt_mask' (torch.Tensor): [M'],
#                 'mkpts0_c' (torch.Tensor): [M, 2],
#                 'mkpts1_c' (torch.Tensor): [M, 2],
#                 'mconf' (torch.Tensor): [M]}
#             NOTE: M' != M during training.
#         """
#         N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

#         # normalize
#         feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

#         if self.match_type == "dual_softmax":
#             sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
#             if mask_c0 is not None:
#                 sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
#             conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

#         data.update({"conf_matrix": conf_matrix})

#         # predict coarse matches from conf_matrix
#         data.update(**self.get_coarse_match(conf_matrix, data))

#     @torch.no_grad()
#     def get_coarse_match(self, conf_matrix, data):
#         """
#         Args:
#             conf_matrix (torch.Tensor): [N, L, S]
#             data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
#         Returns:
#             coarse_matches (dict): {
#                 'b_ids' (torch.Tensor): [M'],
#                 'i_ids' (torch.Tensor): [M'],
#                 'j_ids' (torch.Tensor): [M'],
#                 'gt_mask' (torch.Tensor): [M'],
#                 'm_bids' (torch.Tensor): [M],
#                 'mkpts0_c' (torch.Tensor): [M, 2],
#                 'mkpts1_c' (torch.Tensor): [M, 2],
#                 'mconf' (torch.Tensor): [M]}
#         """
#         axes_lengths = {"h0c": data["hw0_c"][0], "w0c": data["hw0_c"][1], "h1c": data["hw1_c"][0], "w1c": data["hw1_c"][1]}
#         _device = conf_matrix.device
#         # 1. confidence thresholding
#         mask = conf_matrix > self.thr
#         mask = rearrange(mask, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", **axes_lengths)
#         if "mask0" not in data:
#             mask_border(mask, self.border_rm, False)
#         else:
#             mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
#         mask = rearrange(mask, "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)", **axes_lengths)

#         # 2. mutual nearest
#         mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

#         # 3. find all valid coarse matches
#         # this only works when at most one `True` in each row
#         mask_v, all_j_ids = mask.max(dim=2)
#         b_ids, i_ids = torch.where(mask_v)
#         j_ids = all_j_ids[b_ids, i_ids]
#         mconf = conf_matrix[b_ids, i_ids, j_ids]

#         # 4. Random sampling of training samples for fine-level LoFTR
#         # (optional) pad samples with gt coarse-level matches
#         if self.training:
#             # NOTE:
#             # The sampling is performed across all pairs in a batch without manually balancing
#             # #samples for fine-level increases w.r.t. batch_size
#             if "mask0" not in data:
#                 num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
#             else:
#                 num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
#             num_matches_train = int(num_candidates_max * self.train_coarse_percent)
#             num_matches_pred = len(b_ids)
#             assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

#             # pred_indices is to select from prediction
#             if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
#                 pred_indices = torch.arange(num_matches_pred, device=_device)
#             else:
#                 pred_indices = torch.randint(num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=_device)

#             # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
#             gt_pad_indices = torch.randint(
#                 len(data["spv_b_ids"]), (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),), device=_device
#             )
#             mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=_device)  # set conf of gt paddings to all zero

#             b_ids, i_ids, j_ids, mconf = map(
#                 lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
#                 *zip([b_ids, data["spv_b_ids"]], [i_ids, data["spv_i_ids"]], [j_ids, data["spv_j_ids"]], [mconf, mconf_gt]),
#             )

#         # These matches select patches that feed into fine-level network
#         coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

#         # 4. Update with matches in original image resolution
#         scale = data["hw0_i"][0] / data["hw0_c"][0]
#         scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
#         scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
#         mkpts0_c = torch.stack([i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]], dim=1) * scale0
#         mkpts1_c = torch.stack([j_ids % data["hw1_c"][1], j_ids // data["hw1_c"][1]], dim=1) * scale1

#         # These matches is the current prediction (for visualization)
#         coarse_matches.update(
#             {
#                 "gt_mask": mconf == 0,
#                 "m_bids": b_ids[mconf != 0],  # mconf == 0 => gt matches
#                 "mkpts0_c": mkpts0_c[mconf != 0],
#                 "mkpts1_c": mkpts1_c[mconf != 0],
#                 "mconf": mconf[mconf != 0],
#             }
#         )

#         return coarse_matches


# #############################################################
# ### Fine Matching
# #############################################################


# class FineMatching(nn.Module):
#     """FineMatching with s2d paradigm"""

#     def __init__(self):
#         super().__init__()

#     def forward(self, feat_f0, feat_f1, data):
#         """
#         Args:
#             feat0 (torch.Tensor): [M, WW, C]
#             feat1 (torch.Tensor): [M, WW, C]
#             data (dict)
#         Update:
#             data (dict):{
#                 'expec_f' (torch.Tensor): [M, 3],
#                 'mkpts0_f' (torch.Tensor): [M, 2],
#                 'mkpts1_f' (torch.Tensor): [M, 2]}
#         """
#         M, WW, C = feat_f0.shape
#         W = int(math.sqrt(WW))
#         scale = data["hw0_i"][0] / data["hw0_f"][0]
#         self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

#         # corner case: if no coarse matches found
#         if M == 0:
#             assert self.training == False, "M is always >0, when training, see coarse_matching.py"
#             # logger.warning('No matches found in coarse-level.')
#             data.update(
#                 {
#                     "expec_f": torch.empty(0, 3, device=feat_f0.device),
#                     "mkpts0_f": data["mkpts0_c"],
#                     "mkpts1_f": data["mkpts1_c"],
#                 }
#             )
#             return

#         feat_f0_picked = feat_f0_picked = feat_f0[:, WW // 2, :]
#         sim_matrix = torch.einsum("mc,mrc->mr", feat_f0_picked, feat_f1)
#         softmax_temp = 1.0 / C**0.5
#         heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

#         # compute coordinates from heatmap
#         coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
#         grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

#         # compute std over <x, y>
#         var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
#         std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

#         # for fine-level supervision
#         data.update({"expec_f": torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

#         # compute absolute kpt coords
#         self.get_fine_match(coords_normalized, data)

#     @torch.no_grad()
#     def get_fine_match(self, coords_normed, data):
#         W, WW, C, scale = self.W, self.WW, self.C, self.scale

#         # mkpts0_f and mkpts1_f
#         mkpts0_f = data["mkpts0_c"]
#         scale1 = scale * data["scale1"][data["b_ids"]] if "scale0" in data else scale
#         mkpts1_f = data["mkpts1_c"] + (coords_normed * (W // 2) * scale1)[: len(data["mconf"])]

#         data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})


# #############################################################
# ### Fine Preprocess
# #############################################################


# class FinePreprocess(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.config = config
#         self.cat_c_feat = config["fine_concat_coarse_feat"]
#         self.W = self.config["fine_window_size"]

#         d_model_c = self.config["coarse"]["d_model"]
#         d_model_f = self.config["fine"]["d_model"]
#         self.d_model_f = d_model_f
#         if self.cat_c_feat:
#             self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
#             self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

#     def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
#         W = self.W
#         stride = data["hw0_f"][0] // data["hw0_c"][0]

#         data.update({"W": W})
#         if data["b_ids"].shape[0] == 0:
#             feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
#             feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
#             return feat0, feat1

#         # 1. unfold(crop) all local windows
#         feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
#         feat_f0_unfold = rearrange(feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2)
#         feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
#         feat_f1_unfold = rearrange(feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2)

#         # 2. select only the predicted matches
#         feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]  # [n, ww, cf]
#         feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]

#         # option: use coarse-level loftr feature as context: concat and linear
#         if self.cat_c_feat:
#             feat_c_win = self.down_proj(
#                 torch.cat([feat_c0[data["b_ids"], data["i_ids"]], feat_c1[data["b_ids"], data["j_ids"]]], 0)
#             )  # [2n, c]
#             feat_cf_win = self.merge_feat(
#                 torch.cat(
#                     [
#                         torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
#                         repeat(feat_c_win, "n c -> n ww c", ww=W**2),  # [2n, ww, cf]
#                     ],
#                     -1,
#                 )
#             )
#             feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

#         return feat_f0_unfold, feat_f1_unfold


# #############################################################
# ### Backbones
# #############################################################


# def build_backbone(config):
#     if config["backbone_type"] == "litela":
#         return Matchformer_LA_lite()
#     elif config["backbone_type"] == "largela":
#         return Matchformer_LA_large()
#     elif config["backbone_type"] == "litesea":
#         return Matchformer_SEA_lite()
#     elif config["backbone_type"] == "largesea":
#         return Matchformer_SEA_large()
#     else:
#         raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")


# #############################################################
# ### Matchformer_LA_large
# #############################################################


# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).contiguous().view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# def elu_feature_map(x):
#     return torch.nn.functional.elu(x) + 1


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, eps=1e-6, cross=False):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.cross = cross
#         self.feature_map = elu_feature_map
#         self.eps = eps
#         self.dim = dim
#         self.num_heads = num_heads
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

#     def forward(self, x):
#         x_q, x_kv = x, x
#         B, N, C = x_q.shape
#         MiniB = B // 2
#         query = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 1, 2, 3)
#         kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)

#         if self.cross:
#             k1, k2 = kv[0].split(MiniB)
#             v1, v2 = kv[1].split(MiniB)
#             key = torch.cat([k2, k1], dim=0)
#             value = torch.cat([v2, v1], dim=0)
#         else:
#             key, value = kv[0], kv[1]

#         Q = self.feature_map(query)
#         K = self.feature_map(key)
#         v_length = value.size(1)
#         value = value / v_length
#         KV = torch.einsum("nshd,nshv->nhdv", K, value)
#         Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
#         x = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
#         x = x.contiguous().view(B, -1, C)

#         return x


# class Block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         cross=False,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, cross=cross)
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, H, W):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm(x), H, W))

#         return x


# class Positional(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         return x * self.sigmoid(self.pa_conv(x))


# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, with_pos=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
#         self.num_patches = self.H * self.W
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))

#         self.with_pos = with_pos
#         if self.with_pos:
#             self.pos = Positional(embed_dim)

#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         x = self.proj(x)
#         if self.with_pos:
#             x = self.pos(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)

#         return x, H, W


# class AttentionBlock(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         in_chans=1,
#         embed_dims=128,
#         patch_size=7,
#         num_heads=1,
#         mlp_ratios=4,
#         qkv_bias=True,
#         drop_rate=0.0,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         stride=2,
#         depths=1,
#         cross=[False, False, True],
#     ):
#         super().__init__()
#         self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dims)
#         self.block = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims,
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     drop=drop_rate,
#                     drop_path=0,
#                     norm_layer=norm_layer,
#                     cross=cross[i],
#                 )
#                 for i in range(depths)
#             ]
#         )
#         self.norm = norm_layer(embed_dims)

#     def forward(self, x):
#         B = x.shape[0]
#         x, H, W = self.patch_embed(x)
#         for i, blk in enumerate(self.block):
#             x = blk(x, H, W)
#         x = self.norm(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         return x


# class Matchformer_LA_large(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         in_chans=1,
#         embed_dims=[128, 192, 256, 512],
#         num_heads=[8, 8, 8, 8],
#         stage1_cross=[False, False, True],
#         stage2_cross=[False, False, True],
#         stage3_cross=[False, True, True],
#         stage4_cross=[False, True, True],
#     ):
#         super().__init__()
#         # Attention
#         self.AttentionBlock1 = AttentionBlock(
#             img_size=img_size // 2,
#             patch_size=7,
#             num_heads=num_heads[0],
#             mlp_ratios=4,
#             in_chans=in_chans,
#             embed_dims=embed_dims[0],
#             stride=2,
#             depths=3,
#             cross=stage1_cross,
#         )
#         self.AttentionBlock2 = AttentionBlock(
#             img_size=img_size // 4,
#             patch_size=3,
#             num_heads=num_heads[1],
#             mlp_ratios=4,
#             in_chans=embed_dims[0],
#             embed_dims=embed_dims[1],
#             stride=2,
#             depths=3,
#             cross=stage2_cross,
#         )
#         self.AttentionBlock3 = AttentionBlock(
#             img_size=img_size // 16,
#             patch_size=3,
#             num_heads=num_heads[2],
#             mlp_ratios=4,
#             in_chans=embed_dims[1],
#             embed_dims=embed_dims[2],
#             stride=2,
#             depths=3,
#             cross=stage3_cross,
#         )
#         self.AttentionBlock4 = AttentionBlock(
#             img_size=img_size // 32,
#             patch_size=3,
#             num_heads=num_heads[3],
#             mlp_ratios=4,
#             in_chans=embed_dims[2],
#             embed_dims=embed_dims[3],
#             stride=2,
#             depths=3,
#             cross=stage4_cross,
#         )

#         # FPN
#         self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
#         self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[3], embed_dims[3]),
#             nn.BatchNorm2d(embed_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[3], embed_dims[2]),
#         )

#         self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[2], embed_dims[2]),
#             nn.BatchNorm2d(embed_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[2], embed_dims[1]),
#         )
#         self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[1], embed_dims[1]),
#             nn.BatchNorm2d(embed_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[1], embed_dims[0]),
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # stage 1 # 1/2
#         x = self.AttentionBlock1(x)
#         out1 = x
#         # stage 2 # 1/4
#         x = self.AttentionBlock2(x)
#         out2 = x
#         # stage 3 # 1/8
#         x = self.AttentionBlock3(x)
#         out3 = x
#         # stage 3 # 1/16
#         x = self.AttentionBlock4(x)
#         out4 = x

#         # FPN
#         c4_out = self.layer4_outconv(out4)
#         _, _, H, W = out3.shape
#         c4_out_2x = F.interpolate(c4_out, size=(H, W), mode="bilinear", align_corners=True)
#         c3_out = self.layer3_outconv(out3)
#         _, _, H, W = out2.shape
#         c3_out = self.layer3_outconv2(c3_out + c4_out_2x)
#         c3_out_2x = F.interpolate(c3_out, size=(H, W), mode="bilinear", align_corners=True)
#         c2_out = self.layer2_outconv(out2)
#         _, _, H, W = out1.shape
#         c2_out = self.layer2_outconv2(c2_out + c3_out_2x)
#         c2_out_2x = F.interpolate(c2_out, size=(H, W), mode="bilinear", align_corners=True)
#         c1_out = self.layer1_outconv(out1)
#         c1_out = self.layer1_outconv2(c1_out + c2_out_2x)

#         return c3_out, c1_out


# #############################################################
# ### Matchformer_LA_lite
# #############################################################


# class Matchformer_LA_lite(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         in_chans=1,
#         embed_dims=[128, 192, 256, 512],
#         num_heads=[8, 8, 8, 8],
#         stage1_cross=[False, False, True],
#         stage2_cross=[False, False, True],
#         stage3_cross=[False, True, True],
#         stage4_cross=[False, True, True],
#     ):
#         super().__init__()
#         # Attention
#         self.AttentionBlock1 = AttentionBlock(
#             img_size=img_size // 2,
#             patch_size=7,
#             num_heads=num_heads[0],
#             mlp_ratios=4,
#             in_chans=in_chans,
#             embed_dims=embed_dims[0],
#             stride=4,
#             depths=3,
#             cross=stage1_cross,
#         )
#         self.AttentionBlock2 = AttentionBlock(
#             img_size=img_size // 4,
#             patch_size=3,
#             num_heads=num_heads[1],
#             mlp_ratios=4,
#             in_chans=embed_dims[0],
#             embed_dims=embed_dims[1],
#             stride=2,
#             depths=3,
#             cross=stage2_cross,
#         )
#         self.AttentionBlock3 = AttentionBlock(
#             img_size=img_size // 16,
#             patch_size=3,
#             num_heads=num_heads[2],
#             mlp_ratios=4,
#             in_chans=embed_dims[1],
#             embed_dims=embed_dims[2],
#             stride=2,
#             depths=3,
#             cross=stage3_cross,
#         )
#         self.AttentionBlock4 = AttentionBlock(
#             img_size=img_size // 32,
#             patch_size=3,
#             num_heads=num_heads[3],
#             mlp_ratios=4,
#             in_chans=embed_dims[2],
#             embed_dims=embed_dims[3],
#             stride=2,
#             depths=3,
#             cross=stage4_cross,
#         )

#         # FPN
#         self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
#         self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[3], embed_dims[3]),
#             nn.BatchNorm2d(embed_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[3], embed_dims[2]),
#         )

#         self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[2], embed_dims[2]),
#             nn.BatchNorm2d(embed_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[2], embed_dims[1]),
#         )
#         self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[1], embed_dims[1]),
#             nn.BatchNorm2d(embed_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[1], embed_dims[0]),
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # stage 1 # 1/4
#         x = self.AttentionBlock1(x)
#         out1 = x
#         # stage 2 # 1/8
#         x = self.AttentionBlock2(x)
#         out2 = x
#         # stage 3 # 1/16
#         x = self.AttentionBlock3(x)
#         out3 = x
#         # stage 3 # 1/32
#         x = self.AttentionBlock4(x)
#         out4 = x

#         # FPN
#         c4_out = self.layer4_outconv(out4)
#         _, _, H, W = out3.shape
#         c4_out_2x = F.interpolate(c4_out, size=(H, W), mode="bilinear", align_corners=True)
#         c3_out = self.layer3_outconv(out3)
#         _, _, H, W = out2.shape
#         c3_out = self.layer3_outconv2(c3_out + c4_out_2x)
#         c3_out_2x = F.interpolate(c3_out, size=(H, W), mode="bilinear", align_corners=True)
#         c2_out = self.layer2_outconv(out2)
#         _, _, H, W = out1.shape
#         c2_out = self.layer2_outconv2(c2_out + c3_out_2x)
#         c2_out_2x = F.interpolate(c2_out, size=(H, W), mode="bilinear", align_corners=True)
#         c1_out = self.layer1_outconv(out1)
#         c1_out = self.layer1_outconv2(c1_out + c2_out_2x)

#         return c2_out, c1_out


# #############################################################
# ###  Matchformer_SEA_lite
# #############################################################


# class Matchformer_SEA_large(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         in_chans=1,
#         embed_dims=[128, 192, 256, 512],
#         num_heads=[1, 2, 4, 8],
#         sr_ratios=[4, 2, 2, 1],
#         stage1_cross=[False, False, True],
#         stage2_cross=[False, False, True],
#         stage3_cross=[False, True, True],
#         stage4_cross=[False, True, True],
#     ):
#         super().__init__()
#         # Attention
#         self.AttentionBlock1 = AttentionBlock(
#             img_size=img_size // 2,
#             patch_size=7,
#             num_heads=num_heads[0],
#             mlp_ratios=4,
#             in_chans=in_chans,
#             embed_dims=embed_dims[0],
#             stride=2,
#             sr_ratios=sr_ratios[0],
#             depths=3,
#             cross=stage1_cross,
#         )
#         self.AttentionBlock2 = AttentionBlock(
#             img_size=img_size // 4,
#             patch_size=3,
#             num_heads=num_heads[1],
#             mlp_ratios=4,
#             in_chans=embed_dims[0],
#             embed_dims=embed_dims[1],
#             stride=2,
#             sr_ratios=sr_ratios[1],
#             depths=3,
#             cross=stage2_cross,
#         )
#         self.AttentionBlock3 = AttentionBlock(
#             img_size=img_size // 16,
#             patch_size=3,
#             num_heads=num_heads[2],
#             mlp_ratios=4,
#             in_chans=embed_dims[1],
#             embed_dims=embed_dims[2],
#             stride=2,
#             sr_ratios=sr_ratios[2],
#             depths=3,
#             cross=stage3_cross,
#         )
#         self.AttentionBlock4 = AttentionBlock(
#             img_size=img_size // 32,
#             patch_size=3,
#             num_heads=num_heads[3],
#             mlp_ratios=4,
#             in_chans=embed_dims[2],
#             embed_dims=embed_dims[3],
#             stride=2,
#             sr_ratios=sr_ratios[3],
#             depths=3,
#             cross=stage4_cross,
#         )

#         # FPN
#         self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
#         self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[3], embed_dims[3]),
#             nn.BatchNorm2d(embed_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[3], embed_dims[2]),
#         )
#         self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[2], embed_dims[2]),
#             nn.BatchNorm2d(embed_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[2], embed_dims[1]),
#         )
#         self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[1], embed_dims[1]),
#             nn.BatchNorm2d(embed_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[1], embed_dims[0]),
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # stage 1 # 1/4
#         x = self.AttentionBlock1(x)
#         out1 = x
#         # stage 2 # 1/8
#         x = self.AttentionBlock2(x)
#         out2 = x
#         # stage 3 # 1/16
#         x = self.AttentionBlock3(x)
#         out3 = x
#         # stage 3 # 1/32
#         x = self.AttentionBlock4(x)
#         out4 = x

#         # FPN
#         c4_out = self.layer4_outconv(out4)
#         _, _, H, W = out3.shape
#         c4_out_2x = F.interpolate(c4_out, size=(H, W), mode="bilinear", align_corners=True)
#         c3_out = self.layer3_outconv(out3)
#         _, _, H, W = out2.shape
#         c3_out = self.layer3_outconv2(c3_out + c4_out_2x)
#         c3_out_2x = F.interpolate(c3_out, size=(H, W), mode="bilinear", align_corners=True)
#         c2_out = self.layer2_outconv(out2)
#         _, _, H, W = out1.shape
#         c2_out = self.layer2_outconv2(c2_out + c3_out_2x)
#         c2_out_2x = F.interpolate(c2_out, size=(H, W), mode="bilinear", align_corners=True)
#         c1_out = self.layer1_outconv(out1)
#         c1_out = self.layer1_outconv2(c1_out + c2_out_2x)

#         return c3_out, c1_out


# #############################################################
# ### Matchformer_SEA_lite
# #############################################################
# class Matchformer_SEA_lite(nn.Module):
#     def __init__(
#         self,
#         img_size=224,
#         in_chans=1,
#         embed_dims=[128, 192, 256, 512],
#         num_heads=[1, 2, 4, 8],
#         sr_ratios=[8, 4, 2, 1],
#         stage1_cross=[False, False, True],
#         stage2_cross=[False, False, True],
#         stage3_cross=[False, True, True],
#         stage4_cross=[False, True, True],
#     ):
#         super().__init__()
#         # Attention
#         self.AttentionBlock1 = AttentionBlock(
#             img_size=img_size // 2,
#             patch_size=7,
#             num_heads=num_heads[0],
#             mlp_ratios=4,
#             in_chans=in_chans,
#             embed_dims=embed_dims[0],
#             stride=4,
#             sr_ratios=sr_ratios[0],
#             depths=3,
#             cross=stage1_cross,
#         )
#         self.AttentionBlock2 = AttentionBlock(
#             img_size=img_size // 4,
#             patch_size=3,
#             num_heads=num_heads[1],
#             mlp_ratios=4,
#             in_chans=embed_dims[0],
#             embed_dims=embed_dims[1],
#             stride=2,
#             sr_ratios=sr_ratios[1],
#             depths=3,
#             cross=stage2_cross,
#         )
#         self.AttentionBlock3 = AttentionBlock(
#             img_size=img_size // 16,
#             patch_size=3,
#             num_heads=num_heads[2],
#             mlp_ratios=4,
#             in_chans=embed_dims[1],
#             embed_dims=embed_dims[2],
#             stride=2,
#             sr_ratios=sr_ratios[2],
#             depths=3,
#             cross=stage3_cross,
#         )
#         self.AttentionBlock4 = AttentionBlock(
#             img_size=img_size // 32,
#             patch_size=3,
#             num_heads=num_heads[3],
#             mlp_ratios=4,
#             in_chans=embed_dims[2],
#             embed_dims=embed_dims[3],
#             stride=2,
#             sr_ratios=sr_ratios[3],
#             depths=3,
#             cross=stage4_cross,
#         )

#         # FPN
#         self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
#         self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[3], embed_dims[3]),
#             nn.BatchNorm2d(embed_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[3], embed_dims[2]),
#         )
#         self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[2], embed_dims[2]),
#             nn.BatchNorm2d(embed_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[2], embed_dims[1]),
#         )
#         self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(embed_dims[1], embed_dims[1]),
#             nn.BatchNorm2d(embed_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(embed_dims[1], embed_dims[0]),
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # stage 1 # 1/4
#         x = self.AttentionBlock1(x)
#         out1 = x
#         # stage 2 # 1/8
#         x = self.AttentionBlock2(x)
#         out2 = x
#         # stage 3 # 1/16
#         x = self.AttentionBlock3(x)
#         out3 = x
#         # stage 3 # 1/32
#         x = self.AttentionBlock4(x)
#         out4 = x

#         # FPN
#         c4_out = self.layer4_outconv(out4)
#         _, _, H, W = out3.shape
#         c4_out_2x = F.interpolate(c4_out, size=(H, W), mode="bilinear", align_corners=True)
#         c3_out = self.layer3_outconv(out3)
#         _, _, H, W = out2.shape
#         c3_out = self.layer3_outconv2(c3_out + c4_out_2x)
#         c3_out_2x = F.interpolate(c3_out, size=(H, W), mode="bilinear", align_corners=True)
#         c2_out = self.layer2_outconv(out2)
#         _, _, H, W = out1.shape
#         c2_out = self.layer2_outconv2(c2_out + c3_out_2x)
#         c2_out_2x = F.interpolate(c2_out, size=(H, W), mode="bilinear", align_corners=True)
#         c1_out = self.layer1_outconv(out1)
#         c1_out = self.layer1_outconv2(c1_out + c2_out_2x)

#         return c2_out, c1_out
