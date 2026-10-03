"""Layer subset ported from fastai v1.0.60 (Apache-2.0).

Ported by need, in file order: NormType, init_default, SelfAttention, conv
helpers, conv_layer, SequentialEx, MergeLayer, res_block, sigmoid_range,
SigmoidRange, icnr, PixelShuffle_ICNR. Structure and forward behaviour are
copied from the 1.0.60 sources so state-dict keys line up with checkpoints
saved against fastai 1.0.60.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from torch.nn.utils import spectral_norm, weight_norm


class NormType(Enum):
    Batch = 1
    BatchZero = 2
    Weight = 3
    Spectral = 4
    Instance = 5
    InstanceZero = 6


def ifnone(a, b):
    return b if a is None else a


def init_default(m, func=nn.init.kaiming_normal_):
    if func:
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            with torch.no_grad():
                func(m.weight, gain=1)
                m.bias.data.zero_()
        else:
            if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
                func(m.weight)
            if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
                m.bias.data.zero_()
    return m


def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
    return nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)


def conv2d(ni, nf, ks=3, stride=1, padding=None, bias=False, init=nn.init.kaiming_normal_):
    if padding is None:
        padding = ks // 2
    return init_default(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)


def relu(inplace=False, leaky=None):
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


def conv_layer(ni, nf, ks=3, stride=1, padding=None, bias=None, is_1d=False,
               norm_type=NormType.Batch, use_activ=True, leaky=None,
               transpose=False, init=nn.init.kaiming_normal_, self_attention=False):
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    """Self attention layer, https://arxiv.org/pdf/1805.08318.pdf

    The query/key/value convs carry spectral_norm — the published DeOldify
    checkpoints store weight_orig/weight_u/weight_v for them, so the
    parametrization must be present at load time.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.query = spectral_norm(conv1d(n_channels, n_channels // 8))
        self.key = spectral_norm(conv1d(n_channels, n_channels // 8))
        self.value = spectral_norm(conv1d(n_channels, n_channels))
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class SequentialEx(nn.Module):
    """Like nn.Sequential, but with ModuleList semantics, and can access module input."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, l):
        return self.layers.append(l)


class MergeLayer(nn.Module):
    """Merge a shortcut with the result of the module by adding them or concatenating them if dense=True."""

    def __init__(self, dense=False):
        super().__init__()
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


def res_block(nf, dense=False, norm_type=NormType.Batch, bottle=False, **conv_kwargs):
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch):
        norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(
        conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
        MergeLayer(dense))


def sigmoid_range(x, low, high):
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle_ICNR(nn.Module):
    """Upsample by scale from ni filters to nf (default ni), using nn.PixelShuffle, icnr init, and weight_norm."""

    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, leaky=None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


def custom_conv_layer(ni, nf, ks=3, stride=1, padding=None, bias=None, is_1d=False,
                      norm_type=NormType.Batch, use_activ=True, leaky=None,
                      transpose=False, init=nn.init.kaiming_normal_, self_attention=False,
                      extra_bn=False):
    """Ported from DeOldify deoldify/layers.py (MIT): fastai's conv_layer with
    an extra_bn flag — DeOldify's Spectral-norm layers still carry BatchNorm."""
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn is True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)
