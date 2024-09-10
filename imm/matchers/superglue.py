from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import nn

from imm.base import MatcherModel
from imm.misc import _cfg
from imm.registry.factory import load_model_weights

from ._helper import MATCHERS_REGISTRY


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image image_shape"""

    # we modified image_shape to be a tensor of shape (2,),
    width, height = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [layer(x).view(batch_dim, self.dim, self.num_heads, -1) for layer, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(MatcherModel):
    #
    required_inputs = [
        "kpts0",
        "kpts1",
        "scores0",
        "scores1",
        "desc0",
        "desc1",
        "size0",
        "size1",
    ]

    def __init__(self, cfg: Union[Dict[str, Any], DictConfig] = {}):
        super().__init__(cfg)

        self.kenc = KeypointEncoder(self.cfg["descriptor_dim"], self.cfg["keypoint_encoder"])

        self.gnn = AttentionalGNN(
            feature_dim=self.cfg["descriptor_dim"],
            layer_names=self.cfg["GNN_layers"],
        )

        self.final_proj = nn.Conv1d(
            self.cfg["descriptor_dim"],
            self.cfg["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )

        # Sinkhorn parameters
        self.bin_score: torch.Tensor = torch.nn.Parameter(torch.tensor(1.0))

    def transform_inputs(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                if isinstance(data[k][0], torch.Tensor):
                    data[k] = torch.stack(data[k])
        return data

    def process_matches(self, data: Dict[str, Any], preds: torch.Tensor) -> Dict[str, Any]:
        # mutuals
        indices0 = preds["indices0"][0]
        mscores0 = preds["mscores0"][0]

        kpts0 = data["kpts0"][0]
        kpts1 = data["kpts1"][0]

        valid = torch.where(indices0 != -1)[0]

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[indices0[valid]]

        scores = mscores0[valid]

        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mscores": scores,
            "kpts0": kpts0,
            "kpts1": kpts1,
        }

    def forward(self, data: dict, **kwargs: Any) -> dict:
        # unpack
        kpts0, kpts1 = data["kpts0"], data["kpts1"]
        desc0, desc1 = data["desc0"], data["desc1"]
        scores0, scores1 = data["scores0"], data["scores1"]
        size0, size1 = data["size0"], data["size1"]

        #
        assert kpts0.dim() == 3 and kpts1.dim() == 3, "Keypoints must have 3 dimensions"
        assert desc0.dim() == 3 and desc1.dim() == 3, "Descriptors must have 3 dimensions"
        assert scores0.dim() == 2 and scores1.dim() == 2, "Scores must have 2 dimensions"

        # # permute desc0 and desc1 to B d N
        # desc0 = desc0.permute(0, 2, 1)
        # desc1 = desc1.permute(0, 2, 1)

        # print shape of all inputs
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]  # noqa: F841
            return {
                "matches": kpts0.new_full(shape0, -1, dtype=torch.int),
                "matches_scores": kpts0.new_zeros(shape0),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, size0)
        kpts1 = normalize_keypoints(kpts1, size1)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.cfg["descriptor_dim"] ** 0.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=self.cfg["sinkhorn_iterations"])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)  # noqa: F841
        valid0 = mutual0 & (mscores0 > self.cfg["match_threshold"])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # return
        out = {"indices0": indices0, "mscores0": mscores0}

        return out


default_cfgs = {
    "superglue_indoor": _cfg(
        drive="https://drive.google.com/uc?id=1kuo7a0qYvx28Rjor0-BWVTB6t8FWToGT",
        descriptor_dim=256,
        weights="indoor",
        keypoint_encoder=[32, 64, 128, 256],
        GNN_layers=["self", "cross"] * 9,
        sinkhorn_iterations=20,
        match_threshold=0.2,
    ),
    "superglue_outdoor": _cfg(
        drive="https://drive.google.com/uc?id=1nNRn-V0oa66EEgYFqrOdrQOiYTShxtMC",
        descriptor_dim=256,
        weights="outdoor",
        keypoint_encoder=[32, 64, 128, 256],
        GNN_layers=["self", "cross"] * 9,
        sinkhorn_iterations=20,
        match_threshold=0.2,
    ),
}


def _make_model(
    name,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    # create model
    model = SuperGlue(cfg=cfg)

    # load pretrained
    if pretrained:
        load_model_weights(model, name, cfg)

    return model


@MATCHERS_REGISTRY.register(name="superglue_indoor", default_cfg=default_cfgs["superglue_indoor"])
def superglue_indoor(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="superglue_indoor", cfg=cfg, **kwargs)


@MATCHERS_REGISTRY.register(name="superglue_outdoor", default_cfg=default_cfgs["superglue_outdoor"])
def superglue_outdoor(cfg: Dict[str, Any] = {}, **kwargs):
    return _make_model(name="superglue_outdoor", cfg=cfg, **kwargs)
