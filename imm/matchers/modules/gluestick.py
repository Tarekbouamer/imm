import warnings
from copy import deepcopy

import torch
import torch.utils.checkpoint
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning)

ETH_EPS = 1e-8


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)

    # TODO: Check if this is correct
    size = size.unsqueeze(0)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class EndPtEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, endpoints, scores):
        # endpoints should be [B, N, 2, 2]
        # output is [B, feature_dim, N * 2]
        b_size, n_pts, _, _ = endpoints.shape
        assert tuple(endpoints.shape[-2:]) == (2, 2)
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0]).unsqueeze(2)
        endpt_offset = torch.cat([endpt_offset, -endpt_offset], dim=2)
        endpt_offset = endpt_offset.reshape(b_size, 2 * n_pts, 2).transpose(1, 2)
        inputs = [endpoints.flatten(1, 2).transpose(1, 2), endpt_offset, scores.repeat(1, 2).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.prob = []

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1) for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob.mean(dim=1))
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        if skip_init:
            self.register_parameter("scaling", nn.Parameter(torch.tensor(0.0)))
        else:
            self.scaling = 1.0

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type, skip_init):
        super().__init__()
        assert layer_type in ["cross", "self"]
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4, skip_init)

    def forward(self, desc0, desc1):
        if self.type == "cross":
            src0, src1 = desc1, desc0
        elif self.type == "self":
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(nn.Module):
    def __init__(self, feature_dim, line_attention=False):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP([self.dim * 3, self.dim * 2, self.dim], do_bn=True)
        self.line_attention = line_attention
        if line_attention:
            self.proj_node = nn.Conv1d(self.dim, self.dim, kernel_size=1)
            self.proj_neigh = nn.Conv1d(2 * self.dim, self.dim, kernel_size=1)

    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size = lines_junc_idx.shape[0]
        line_desc = torch.gather(ldesc, 2, lines_junc_idx[:, None].repeat(1, self.dim, 1))
        message = torch.cat(
            [line_desc, line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(), line_enc], dim=1
        )
        return self.mlp(message)  # [b_size, D, n_lines * 2]

    def get_endpoint_attention(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        b_size = lines_junc_idx.shape[0]
        expanded_lines_junc_idx = lines_junc_idx[:, None].repeat(1, self.dim, 1)

        # Query: desc of the current node
        query = self.proj_node(ldesc)  # [b_size, D, n_junc]
        query = torch.gather(query, 2, expanded_lines_junc_idx)
        # query is [b_size, D, n_lines * 2]

        # Key: combination of neighboring desc and line encodings
        line_desc = torch.gather(ldesc, 2, expanded_lines_junc_idx)
        key = self.proj_neigh(
            torch.cat([line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(), line_enc], dim=1)
        )  # [b_size, D, n_lines * 2]

        # Compute the attention weights with a custom softmax per junction
        prob = (query * key).sum(dim=1) / self.dim**0.5  # [b_size, n_lines * 2]
        prob = torch.exp(prob - prob.max())
        denom = torch.zeros_like(ldesc[:, 0]).scatter_reduce_(
            dim=1, index=lines_junc_idx, src=prob, reduce="sum", include_self=False
        )  # [b_size, n_junc]
        denom = torch.gather(denom, 1, lines_junc_idx)  # [b_size, n_lines * 2]
        prob = prob / (denom + ETH_EPS)
        return prob  # [b_size, n_lines * 2]

    def forward(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1):
        # Gather the endpoint updates
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        update0, update1 = torch.zeros_like(ldesc0), torch.zeros_like(ldesc1)
        dim = ldesc0.shape[1]
        if self.line_attention:
            # Compute an attention for each neighbor and do a weighted average
            prob0 = self.get_endpoint_attention(ldesc0, line_enc0, lines_junc_idx0)
            lupdate0 = lupdate0 * prob0[:, None]
            update0 = update0.scatter_reduce_(
                dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1), src=lupdate0, reduce="sum", include_self=False
            )
            prob1 = self.get_endpoint_attention(ldesc1, line_enc1, lines_junc_idx1)
            lupdate1 = lupdate1 * prob1[:, None]
            update1 = update1.scatter_reduce_(
                dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1), src=lupdate1, reduce="sum", include_self=False
            )
        else:
            # Average the updates for each junction (requires torch > 1.12)
            update0 = update0.scatter_reduce_(
                dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1), src=lupdate0, reduce="mean", include_self=False
            )
            update1 = update1.scatter_reduce_(
                dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1), src=lupdate1, reduce="mean", include_self=False
            )

        # Update
        ldesc0 = ldesc0 + update0
        ldesc1 = ldesc1 + update1

        return ldesc0, ldesc1


class AttentionalGNN(nn.Module):
    def __init__(
        self,
        feature_dim,
        layer_types,
        checkpointed=False,
        skip=False,
        inter_supervision=None,
        num_line_iterations=1,
        line_attention=False,
    ):
        super().__init__()
        self.checkpointed = checkpointed
        self.inter_supervision = inter_supervision
        self.num_line_iterations = num_line_iterations
        self.inter_layers = {}
        self.layers = nn.ModuleList([GNNLayer(feature_dim, layer_type, skip) for layer_type in layer_types])
        self.line_layers = nn.ModuleList([LineLayer(feature_dim, line_attention) for _ in range(len(layer_types) // 2)])

    def forward(self, desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1):
        for i, layer in enumerate(self.layers):
            if self.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(layer, desc0, desc1, preserve_rng_state=False)
            else:
                desc0, desc1 = layer(desc0, desc1)
            if layer.type == "self" and lines_junc_idx0.shape[1] > 0 and lines_junc_idx1.shape[1] > 0:
                # Add line self attention layers after every self layer
                for _ in range(self.num_line_iterations):
                    if self.checkpointed:
                        desc0, desc1 = torch.utils.checkpoint.checkpoint(
                            self.line_layers[i // 2],
                            desc0,
                            desc1,
                            line_enc0,
                            line_enc1,
                            lines_junc_idx0,
                            lines_junc_idx1,
                            preserve_rng_state=False,
                        )
                    else:
                        desc0, desc1 = self.line_layers[i // 2](
                            desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1
                        )

            # Optionally store the line descriptor at intermediate layers
            if self.inter_supervision is not None and (i // 2) in self.inter_supervision and layer.type == "cross":
                self.inter_layers[i // 2] = (desc0.clone(), desc1.clone())
        return desc0, desc1


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
