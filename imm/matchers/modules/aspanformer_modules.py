import copy
import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from torch.nn import Module


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), pre_scaling=None):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released imm.models.
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pre_scaling = pre_scaling

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

        if pre_scaling[0] is not None and pre_scaling[1] is not None:
            train_res, test_res = pre_scaling[0], pre_scaling[1]
            x_position, y_position = (
                x_position * train_res[1] / test_res[1],
                y_position * train_res[0] / test_res[0],
            )

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x, scaling=None):
        """
        Args:
            x: [N, C, H, W]
        """
        if scaling is None:  # onliner scaling overwrites pre_scaling
            return x + self.pe[:, :, : x.size(2), : x.size(3)], self.pe[:, :, : x.size(2), : x.size(3)]
        else:
            pe = torch.zeros((self.d_model, *self.max_shape))
            y_position = torch.ones(self.max_shape).cumsum(0).float().unsqueeze(0) * scaling[0]
            x_position = torch.ones(self.max_shape).cumsum(1).float().unsqueeze(0) * scaling[1]

            div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
            div_term = div_term[:, None, None]  # [C//4, 1, 1]
            pe[0::4, :, :] = torch.sin(x_position * div_term)
            pe[1::4, :, :] = torch.cos(x_position * div_term)
            pe[2::4, :, :] = torch.sin(y_position * div_term)
            pe[3::4, :, :] = torch.cos(y_position * div_term)
            pe = pe.unsqueeze(0).to(x.device)
            return x + pe[:, :, : x.size(2), : x.size(3)], pe[:, :, : x.size(2), : x.size(3)]


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return [x3_out, x1_out]


class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        return [x4_out, x2_out]


def build_backbone(config):
    if config["backbone_type"] == "ResNetFPN":
        if config["resolution"] == (8, 2):
            return ResNetFPN_8_2(config["resnetfpn"])
        elif config["resolution"] == (16, 4):
            return ResNetFPN_16_4(config["resnetfpn"])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")


class messageLayer_ini(nn.Module):
    def __init__(self, d_model, d_flow, d_value, nhead):
        super().__init__()
        super(messageLayer_ini, self).__init__()

        self.d_model = d_model
        self.d_flow = d_flow
        self.d_value = d_value
        self.nhead = nhead
        self.attention = FullAttention(d_model, nhead)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1, bias=False)
        self.merge_head = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.merge_f = self.merge_f = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model * 2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
        )

        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model)

    def forward(self, x0, x1, pos0, pos1, mask0=None, mask1=None):
        # x1,x2: b*d*L
        x0, x1 = (
            self.update(x0, x1, pos1, mask0, mask1),
            self.update(x1, x0, pos0, mask1, mask0),
        )
        return x0, x1

    def update(self, f0, f1, pos1, mask0, mask1):
        """
        Args:
            f0: [N, D, H, W]
            f1: [N, D, H, W]
        Returns:
            f0_new: (N, d, h, w)
        """
        bs, h, w = f0.shape[0], f0.shape[2], f0.shape[3]

        f0_flatten, f1_flatten = (
            f0.view(bs, self.d_model, -1),
            f1.view(bs, self.d_model, -1),
        )
        pos1_flatten = pos1.view(bs, self.d_value - self.d_model, -1)
        f1_flatten_v = torch.cat([f1_flatten, pos1_flatten], dim=1)

        queries, keys = self.q_proj(f0_flatten), self.k_proj(f1_flatten)
        values = self.v_proj(f1_flatten_v).view(bs, self.nhead, self.d_model // self.nhead, -1)

        queried_values = self.attention(queries, keys, values, mask0, mask1)
        msg = self.merge_head(queried_values).view(bs, -1, h, w)
        msg = self.norm2(self.merge_f(torch.cat([f0, self.norm1(msg)], dim=1)))
        return f0 + msg


class messageLayer_gla(nn.Module):
    def __init__(
        self,
        d_model,
        d_flow,
        d_value,
        nhead,
        radius_scale,
        nsample,
        update_flow=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_flow = d_flow
        self.d_value = d_value
        self.nhead = nhead
        self.radius_scale = radius_scale
        self.update_flow = update_flow
        self.flow_decoder = nn.Sequential(
            nn.Conv1d(d_flow, d_flow // 2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_flow // 2, 4, kernel_size=1, bias=False),
        )
        self.attention = HierachicalAttention(d_model, nhead, nsample, radius_scale)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1, bias=False)

        d_extra = d_flow if update_flow else 0
        self.merge_f = nn.Sequential(
            nn.Conv2d(
                d_model * 2 + d_extra,
                d_model + d_flow,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                d_model + d_flow,
                d_model + d_extra,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model + d_extra)

    def forward(
        self,
        x0,
        x1,
        flow_feature0,
        flow_feature1,
        pos0,
        pos1,
        mask0=None,
        mask1=None,
        ds0=[4, 4],
        ds1=[4, 4],
    ):
        """
        Args:
            x0 (torch.Tensor): [B, C, H, W]
            x1 (torch.Tensor): [B, C, H, W]
            flow_feature0 (torch.Tensor): [B, C', H, W]
            flow_feature1 (torch.Tensor): [B, C', H, W]
        """
        flow0, flow1 = (
            self.decode_flow(flow_feature0, flow_feature1.shape[2:]),
            self.decode_flow(flow_feature1, flow_feature0.shape[2:]),
        )
        x0_new, flow_feature0_new = self.update(x0, x1, flow0.detach(), flow_feature0, pos1, mask0, mask1, ds0, ds1)
        x1_new, flow_feature1_new = self.update(x1, x0, flow1.detach(), flow_feature1, pos0, mask1, mask0, ds1, ds0)
        return (
            x0_new,
            x1_new,
            flow_feature0_new,
            flow_feature1_new,
            flow0,
            flow1,
        )

    def update(self, x0, x1, flow0, flow_feature0, pos1, mask0, mask1, ds0, ds1):
        bs = x0.shape[0]
        queries, keys = (
            self.q_proj(x0.view(bs, self.d_model, -1)),
            self.k_proj(x1.view(bs, self.d_model, -1)),
        )
        x1_pos = torch.cat([x1, pos1], dim=1)
        values = self.v_proj(x1_pos.view(bs, self.d_value, -1))
        msg = self.attention(
            queries,
            keys,
            values,
            flow0,
            x0.shape[2:],
            x1.shape[2:],
            mask0,
            mask1,
            ds0,
            ds1,
        )

        if self.update_flow:
            update_feature = torch.cat([x0, flow_feature0], dim=1)
        else:
            update_feature = x0
        msg = self.norm2(self.merge_f(torch.cat([update_feature, self.norm1(msg)], dim=1)))
        update_feature = update_feature + msg

        x0_new, flow_feature0_new = (
            update_feature[:, : self.d_model],
            update_feature[:, self.d_model :],
        )
        return x0_new, flow_feature0_new

    def decode_flow(self, flow_feature, kshape):
        bs, h, w = (
            flow_feature.shape[0],
            flow_feature.shape[2],
            flow_feature.shape[3],
        )
        scale_factor = torch.tensor([kshape[1], kshape[0]]).cuda()[None, None, None]
        flow = self.flow_decoder(flow_feature.view(bs, -1, h * w)).permute(0, 2, 1).view(bs, h, w, 4)
        flow_coordinates = torch.sigmoid(flow[:, :, :, :2]) * scale_factor
        flow_var = flow[:, :, :, 2:]
        flow = torch.cat([flow_coordinates, flow_var], dim=-1)  # B*H*W*4
        return flow


class flow_initializer(nn.Module):
    def __init__(self, dim, dim_flow, nhead, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.dim = dim
        self.dim_flow = dim_flow

        encoder_layer = messageLayer_ini(dim, dim_flow, dim + dim_flow, nhead)
        self.layers_coarse = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(layer_num)])
        self.decoupler = nn.Conv2d(self.dim, self.dim + self.dim_flow, kernel_size=1)
        self.up_merge = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def forward(
        self,
        feat0,
        feat1,
        pos0,
        pos1,
        mask0=None,
        mask1=None,
        ds0=[4, 4],
        ds1=[4, 4],
    ):
        # feat0: [B, C, H0, W0]
        # feat1: [B, C, H1, W1]
        # use low-res MHA to initialize flow feature
        bs = feat0.size(0)
        h0, w0, h1, w1 = (
            feat0.shape[2],
            feat0.shape[3],
            feat1.shape[2],
            feat1.shape[3],
        )

        # coarse level
        sub_feat0, sub_feat1 = (
            F.avg_pool2d(feat0, ds0, stride=ds0),
            F.avg_pool2d(feat1, ds1, stride=ds1),
        )

        sub_pos0, sub_pos1 = (
            F.avg_pool2d(pos0, ds0, stride=ds0),
            F.avg_pool2d(pos1, ds1, stride=ds1),
        )

        if mask0 is not None:
            mask0, mask1 = (
                -F.max_pool2d(-mask0.view(bs, 1, h0, w0), ds0, stride=ds0).view(bs, -1),
                -F.max_pool2d(-mask1.view(bs, 1, h1, w1), ds1, stride=ds1).view(bs, -1),
            )

        for layer in self.layers_coarse:
            sub_feat0, sub_feat1 = layer(sub_feat0, sub_feat1, sub_pos0, sub_pos1, mask0, mask1)
        # decouple flow and visual features
        decoupled_feature0, decoupled_feature1 = (
            self.decoupler(sub_feat0),
            self.decoupler(sub_feat1),
        )

        sub_feat0, sub_flow_feature0 = (
            decoupled_feature0[:, : self.dim],
            decoupled_feature0[:, self.dim :],
        )
        sub_feat1, sub_flow_feature1 = (
            decoupled_feature1[:, : self.dim],
            decoupled_feature1[:, self.dim :],
        )
        update_feat0, flow_feature0 = (
            F.upsample(sub_feat0, scale_factor=ds0, mode="bilinear"),
            F.upsample(sub_flow_feature0, scale_factor=ds0, mode="bilinear"),
        )
        update_feat1, flow_feature1 = (
            F.upsample(sub_feat1, scale_factor=ds1, mode="bilinear"),
            F.upsample(sub_flow_feature1, scale_factor=ds1, mode="bilinear"),
        )

        feat0 = feat0 + self.up_merge(torch.cat([feat0, update_feat0], dim=1))
        feat1 = feat1 + self.up_merge(torch.cat([feat1, update_feat1], dim=1))

        return feat0, feat1, flow_feature0, flow_feature1  # b*c*h*w


class LocalFeatureTransformer_Flow(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer_Flow, self).__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]

        self.pos_transform = nn.Conv2d(config["d_model"], config["d_flow"], kernel_size=1, bias=False)
        self.ini_layer = flow_initializer(
            self.d_model,
            config["d_flow"],
            config["nhead"],
            config["ini_layer_num"],
        )

        encoder_layer = messageLayer_gla(
            config["d_model"],
            config["d_flow"],
            config["d_flow"] + config["d_model"],
            config["nhead"],
            config["radius_scale"],
            config["nsample"],
        )
        encoder_layer_last = messageLayer_gla(
            config["d_model"],
            config["d_flow"],
            config["d_flow"] + config["d_model"],
            config["nhead"],
            config["radius_scale"],
            config["nsample"],
            update_flow=False,
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config["layer_num"] - 1)] + [encoder_layer_last])
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if "temp" in name or "sample_offset" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feat0,
        feat1,
        pos0,
        pos1,
        mask0=None,
        mask1=None,
        ds0=[4, 4],
        ds1=[4, 4],
    ):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            pos1,pos2:  [N, C, H, W]
        Outputs:
            feat0: [N,-1,C]
            feat1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """
        bs = feat0.size(0)

        pos0, pos1 = self.pos_transform(pos0), self.pos_transform(pos1)
        pos0, pos1 = pos0.expand(bs, -1, -1, -1), pos1.expand(bs, -1, -1, -1)
        assert self.d_model == feat0.size(1), "the feature number of src and transformer must be equal"

        flow_list = [[], []]  # [px,py,sx,sy]
        if mask0 is not None:
            mask0, mask1 = mask0[:, None].float(), mask1[:, None].float()
        feat0, feat1, flow_feature0, flow_feature1 = self.ini_layer(feat0, feat1, pos0, pos1, mask0, mask1, ds0, ds1)
        for layer in self.layers:
            feat0, feat1, flow_feature0, flow_feature1, flow0, flow1 = layer(
                feat0,
                feat1,
                flow_feature0,
                flow_feature1,
                pos0,
                pos1,
                mask0,
                mask1,
                ds0,
                ds1,
            )
            flow_list[0].append(flow0)
            flow_list[1].append(flow1)
        flow_list[0] = torch.stack(flow_list[0], dim=0)
        flow_list[1] = torch.stack(flow_list[1], dim=0)
        feat0, feat1 = (
            feat0.permute(0, 2, 3, 1).view(bs, -1, self.d_model),
            feat1.permute(0, 2, 3, 1).view(bs, -1, self.d_model),
        )
        return feat0, feat1, flow_list


class layernorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.affine = nn.parameter.Parameter(torch.ones(dim), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        # x: B*C*H*W
        mean, std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        return self.affine[None, :, None, None] * (x - mean) / (std + 1e-6) + self.bias[None, :, None, None]


class HierachicalAttention(Module):
    def __init__(self, d_model, nhead, nsample, radius_scale, nlevel=3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nsample = nsample
        self.nlevel = nlevel
        self.radius_scale = radius_scale
        self.merge_head = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
        )
        self.fullattention = FullAttention(d_model, nhead)
        self.temp = nn.parameter.Parameter(torch.tensor(1.0), requires_grad=True)
        sample_offset = torch.tensor(
            [[pos[0] - nsample[1] / 2 + 0.5, pos[1] - nsample[1] / 2 + 0.5] for pos in product(range(nsample[1]), range(nsample[1]))]
        )  # r^2*2
        self.sample_offset = nn.parameter.Parameter(sample_offset, requires_grad=False)

    def forward(
        self,
        query,
        key,
        value,
        flow,
        size_q,
        size_kv,
        mask0=None,
        mask1=None,
        ds0=[4, 4],
        ds1=[4, 4],
    ):
        """
        Args:
            q,k,v (torch.Tensor): [B, C, L]
            mask (torch.Tensor): [B, L]
            flow (torch.Tensor): [B, H, W, 4]
        Return:
            all_message (torch.Tensor): [B, C, H, W]
        """

        variance = flow[:, :, :, 2:]
        offset = flow[:, :, :, :2]  # B*H*W*2
        bs = query.shape[0]
        h0, w0 = size_q[0], size_q[1]
        h1, w1 = size_kv[0], size_kv[1]
        variance = torch.exp(0.5 * variance) * self.radius_scale  # b*h*w*2(pixel scale)
        span_scale = torch.clamp((variance * 2 / self.nsample[1]), min=1)  # b*h*w*2

        sub_sample0, sub_sample1 = [ds0, 2, 1], [ds1, 2, 1]
        q_list = [
            F.avg_pool2d(
                query.view(bs, -1, h0, w0),
                kernel_size=sub_size,
                stride=sub_size,
            )
            for sub_size in sub_sample0
        ]
        k_list = [F.avg_pool2d(key.view(bs, -1, h1, w1), kernel_size=sub_size, stride=sub_size) for sub_size in sub_sample1]
        v_list = [
            F.avg_pool2d(
                value.view(bs, -1, h1, w1),
                kernel_size=sub_size,
                stride=sub_size,
            )
            for sub_size in sub_sample1
        ]  # n_level

        offset_list = [
            F.avg_pool2d(
                offset.permute(0, 3, 1, 2),
                kernel_size=sub_size * self.nsample[0],
                stride=sub_size * self.nsample[0],
            ).permute(0, 2, 3, 1)
            / sub_size
            for sub_size in sub_sample0[1:]
        ]  # n_level-1
        span_list = [
            F.avg_pool2d(
                span_scale.permute(0, 3, 1, 2),
                kernel_size=sub_size * self.nsample[0],
                stride=sub_size * self.nsample[0],
            ).permute(0, 2, 3, 1)
            for sub_size in sub_sample0[1:]
        ]  # n_level-1

        if mask0 is not None:
            mask0, mask1 = mask0.view(bs, 1, h0, w0), mask1.view(bs, 1, h1, w1)
            mask0_list = [-F.max_pool2d(-mask0, kernel_size=sub_size, stride=sub_size) for sub_size in sub_sample0]
            mask1_list = [-F.max_pool2d(-mask1, kernel_size=sub_size, stride=sub_size) for sub_size in sub_sample1]
        else:
            mask0_list = mask1_list = [None, None, None]

        message_list = []
        # full attention at coarse scale
        mask0_flatten = mask0_list[0].view(bs, -1) if mask0 is not None else None
        mask1_flatten = mask1_list[0].view(bs, -1) if mask1 is not None else None
        message_list.append(
            self.fullattention(
                q_list[0],
                k_list[0],
                v_list[0],
                mask0_flatten,
                mask1_flatten,
                self.temp,
            ).view(bs, self.d_model, h0 // ds0[0], w0 // ds0[1])
        )

        for index in range(1, self.nlevel):
            q, k, v = q_list[index], k_list[index], v_list[index]
            mask0, mask1 = mask0_list[index], mask1_list[index]
            s, o = span_list[index - 1], offset_list[index - 1]  # B*h*w(*2)
            q, k, v, sample_pixel, mask_sample = self.partition_token(q, k, v, o, s, mask0)  # B*Head*D*G*N(G*N=H*W for q)
            message_list.append(
                self.group_attention(q, k, v, 1, mask_sample).view(
                    bs,
                    self.d_model,
                    h0 // sub_sample0[index],
                    w0 // sub_sample0[index],
                )
            )
        # fuse
        all_message = torch.cat(
            [
                F.upsample(
                    message_list[idx],
                    scale_factor=sub_sample0[idx],
                    mode="nearest",
                )
                for idx in range(self.nlevel)
            ],
            dim=1,
        ).view(bs, -1, h0 * w0)  # b*3d*H*W

        all_message = self.merge_head(all_message).view(bs, -1, h0, w0)  # b*d*H*W
        return all_message

    def partition_token(self, q, k, v, offset, span_scale, maskv):
        # q,k,v: B*C*H*W
        # o: B*H/2*W/2*2
        # span_scale:B*H*W
        bs = q.shape[0]
        h, w = q.shape[2], q.shape[3]
        hk, wk = k.shape[2], k.shape[3]
        offset = offset.view(bs, -1, 2)
        span_scale = span_scale.view(bs, -1, 1, 2)
        # B*G*2
        offset_sample = self.sample_offset[None, None] * span_scale
        sample_pixel = offset[:, :, None] + offset_sample  # B*G*r^2*2
        sample_norm = sample_pixel / torch.tensor([wk / 2, hk / 2]).cuda()[None, None, None] - 1

        q = (
            q.view(
                bs,
                -1,
                h // self.nsample[0],
                self.nsample[0],
                w // self.nsample[0],
                self.nsample[0],
            )
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(
                bs,
                self.nhead,
                self.d_model // self.nhead,
                -1,
                self.nsample[0] ** 2,
            )
        )  # B*head*D*G*N(G*N=H*W for q)
        # sample token
        k = F.grid_sample(k, grid=sample_norm, align_corners=True).view(
            bs, self.nhead, self.d_model // self.nhead, -1, self.nsample[1] ** 2
        )  # B*head*D*G*r^2

        v = F.grid_sample(v, grid=sample_norm, align_corners=True).view(
            bs, self.nhead, self.d_model // self.nhead, -1, self.nsample[1] ** 2
        )  # B*head*D*G*r^2

        # import pdb;pdb.set_trace()
        if maskv is not None:
            mask_sample = (
                F.grid_sample(
                    maskv.view(bs, -1, h, w).float(),
                    grid=sample_norm,
                    mode="nearest",
                )
                == 1
            )  # B*1*G*r^2
        else:
            mask_sample = None
        return q, k, v, sample_pixel, mask_sample

    def group_attention(self, query, key, value, temp, mask_sample=None):
        # q,k,v: B*Head*D*G*N(G*N=H*W for q)
        bs = query.shape[0]
        # import pdb;pdb.set_trace()
        QK = torch.einsum("bhdgn,bhdgm->bhgnm", query, key)
        if mask_sample is not None:
            num_head, number_n = QK.shape[1], QK.shape[3]
            QK.masked_fill_(
                ~(mask_sample[:, :, :, None]).expand(-1, num_head, -1, number_n, -1).bool(),
                float(-1e8),
            )
        # Compute the attention and the weighted average
        softmax_temp = temp / query.size(2) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("bhgnm,bhdgm->bhdgn", A, value).contiguous().view(bs, self.d_model, -1)
        return queried_values


class FullAttention(Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, q, k, v, mask0=None, mask1=None, temp=1):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            q,k,v: [N, D, L]
            mask: [N, L]
        Returns:
            msg: [N,L]
        """
        bs = q.shape[0]
        q, k, v = (
            q.view(bs, self.nhead, self.d_model // self.nhead, -1),
            k.view(bs, self.nhead, self.d_model // self.nhead, -1),
            v.view(bs, self.nhead, self.d_model // self.nhead, -1),
        )
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nhdl,nhds->nhls", q, k)
        if mask0 is not None:
            QK.masked_fill_(
                ~(mask0[:, None, :, None] * mask1[:, None, None]).bool(),
                float(-1e8),
            )
        # Compute the attention and the weighted average
        softmax_temp = temp / q.size(2) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("nhls,nhds->nhdl", A, v).contiguous().view(bs, self.d_model, -1)
        return queried_values


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, type=None, index=0):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        encoder_layer = LoFTREncoderLayer(config["d_model"], config["nhead"], config["attention"])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        index = 0
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0, type="self", index=index)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0, type="cross", index=index)
                index += 1
            else:
                raise KeyError
        return feat0, feat1


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config["fine_concat_coarse_feat"]
        self.W = self.config["fine_window_size"]

        d_model_c = self.config["coarse"]["d_model"]
        d_model_f = self.config["fine"]["d_model"]
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data["hw0_f"][0] // data["hw0_c"][0]

        data.update({"W": W})
        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f0_unfold = rearrange(feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f1_unfold = rearrange(feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2)

        # 2. select only the predicted matches
        # [n, ww, cf]
        feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]
        feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(
                torch.cat(
                    [
                        feat_c0[data["b_ids"], data["i_ids"]],
                        feat_c1[data["b_ids"], data["j_ids"]],
                    ],
                    0,
                )
            )  # [2n, c]
            feat_cf_win = self.merge_feat(
                torch.cat(
                    [
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        repeat(feat_c_win, "n c -> n ww c", ww=W**2),  # [2n, ww, cf]
                    ],
                    -1,
                )
            )
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold


INF = 1e9


def mask_border(m, b: int, v):
    """Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

        # we provide 2 options for differentiable matching
        self.match_type = config["match_type"]
        if self.match_type == "dual_softmax":
            self.temperature = nn.parameter.Parameter(torch.tensor(10.0), requires_grad=True)
        elif self.match_type == "sinkhorn":
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(torch.tensor(config["skh_init_bin_score"], requires_grad=True))
            self.skh_iters = config["skh_iters"]
            self.skh_prefilter = config["skh_prefilter"]
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, flow_list, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            offset: [layer, B, H, W, 4] (*2)
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = (
            feat_c0.size(0),
            feat_c0.size(  # noqa: F841
                1
            ),
            feat_c1.size(1),
            feat_c0.size(2),
        )
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

        if self.match_type == "dual_softmax":
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) * self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == "sinkhorn":
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config["sparse_spvs"]:
                data.update({"conf_matrix_with_bin": assign_matrix.clone()})

        data.update({"conf_matrix": conf_matrix})
        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

        # update predicted offset
        if flow_list[0].shape[2] == flow_list[1].shape[2] and flow_list[0].shape[3] == flow_list[1].shape[3]:
            flow_list = torch.stack(flow_list, dim=0)
        data.update({"predict_flow": flow_list})  # [2*L*B*H*W*4]
        self.get_offset_match(flow_list, data, mask_c0, mask_c1)

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", **axes_lengths)
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
        mask = rearrange(mask, "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)", **axes_lengths)

        # 2. mutual nearest
        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min,),
                    device=_device,
                )

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (
                    max(
                        num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min,
                    ),
                ),
                device=_device,
            )
            # set conf of gt paddings to all zero
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                ),
            )

        # These matches select patches that feed into fine-level network
        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = torch.stack([i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data["hw1_c"][1], j_ids // data["hw1_c"][1]], dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update(
            {
                "gt_mask": mconf == 0,
                "m_bids": b_ids[mconf != 0],  # mconf == 0 => gt matches
                "mkpts0_c": mkpts0_c[mconf != 0],
                "mkpts1_c": mkpts1_c[mconf != 0],
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches

    @torch.no_grad()
    def get_offset_match(self, flow_list, data, mask1, mask2):
        """
        Args:
            offset (torch.Tensor): [L, B, H, W, 2]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        offset1 = flow_list[0]
        bs, layer_num = offset1.shape[1], offset1.shape[0]

        # left side
        offset1 = offset1.view(layer_num, bs, -1, 4)
        conf1 = offset1[:, :, :, 2:].mean(dim=-1)
        if mask1 is not None:
            conf1.masked_fill_(~mask1.bool()[None].expand(layer_num, -1, -1), 100)
        offset1 = offset1[:, :, :, :2]
        self.get_offset_match_work(offset1, conf1, data, "left")

        # rihgt side
        if len(flow_list) == 2:
            offset2 = flow_list[1].view(layer_num, bs, -1, 4)
            conf2 = offset2[:, :, :, 2:].mean(dim=-1)
            if mask2 is not None:
                conf2.masked_fill_(~mask2.bool()[None].expand(layer_num, -1, -1), 100)
            offset2 = offset2[:, :, :, :2]
            self.get_offset_match_work(offset2, conf2, data, "right")

    @torch.no_grad()
    def get_offset_match_work(self, offset, conf, data, side):
        bs, layer_num = offset.shape[1], offset.shape[0]  # noqa: F841
        # 1. confidence thresholding
        mask_conf = conf < 2
        for index in range(bs):
            # safe guard in case that no match survives
            mask_conf[:, index, 0] = True
        # 3. find offset matches
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        l_ids, b_ids, i_ids = torch.where(mask_conf)
        j_coor = offset[l_ids, b_ids, i_ids, :2] * scale  # [N,2]
        i_coor = torch.stack([i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]], dim=1) * scale
        # i_coor=torch.as_tensor([[index%data['hw0_c'][1],index//data['hw0_c'][1]] for index in i_ids]).cuda().float()*scale #[N,2]
        # These matches is the current prediction (for visualization)
        data.update(
            {
                "offset_bids_" + side: b_ids,  # mconf == 0 => gt matches
                "offset_lids_" + side: l_ids,
                "conf" + side: conf[mask_conf],
            }
        )

        if side == "right":
            data.update(
                {
                    "offset_kpts0_f_" + side: j_coor.detach(),
                    "offset_kpts1_f_" + side: i_coor,
                }
            )
        else:
            data.update(
                {
                    "offset_kpts0_f_" + side: i_coor,
                    "offset_kpts1_f_" + side: j_coor.detach(),
                }
            )


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training is False, "M is always >0, when training, see coarse_imm.py"
            # logger.warning('No matches found in coarse-level.')
            data.update(
                {
                    "expec_f": torch.empty(0, 3, device=feat_f0.device),
                    "mkpts0_f": data["mkpts0_c"],
                    "mkpts1_f": data["mkpts1_c"],
                }
            )
            return

        feat_f0_picked = feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum("mc,mrc->mr", feat_f0_picked, feat_f1)
        softmax_temp = 1.0 / C**0.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        # [M]  clamp needed for numerical stability
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)

        # for fine-level supervision
        data.update({"expec_f": torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale  # noqa: F841

        # mkpts0_f and mkpts1_f
        mkpts0_f = data["mkpts0_c"]
        scale1 = scale * data["scale1"][data["b_ids"]] if "scale0" in data else scale
        mkpts1_f = data["mkpts1_c"] + (coords_normed * (W // 2) * scale1)[: len(data["mconf"])]

        data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})
