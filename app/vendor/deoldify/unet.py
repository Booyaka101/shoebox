"""DeOldify DynamicUnetDeep, ported for inference (MIT, jantic/DeOldify).

fastai's hook_outputs/model_sizes machinery is replaced with plain
register_forward_hook captures and a dummy forward, but the module tree —
and therefore the state-dict keys of the published checkpoints — is unchanged.
Only DynamicUnetDeep is ported: that is what get_image_colorizer uses for the
artistic image model (see DeOldify deoldify/visualize.py).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastai_layers import (NormType, SequentialEx, MergeLayer, SigmoidRange,
                            PixelShuffle_ICNR, custom_conv_layer, ifnone, icnr,
                            relu, res_block)


def batchnorm_2d(nf):
    return nn.BatchNorm2d(nf)


def _get_sfs_idxs(sizes) -> List[int]:
    """Get the indexes of the layers where the size of the activation changes."""
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class _SaveFeatures:
    """Forward-hook capture replacing fastai's hook_outputs(detach=False)."""

    def __init__(self, module: nn.Module):
        self.stored = None
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.stored = output

    def remove(self):
        self._hook.remove()


def _children(m: nn.Module):
    return list(m.children())


def _in_channels(m: nn.Module) -> int:
    for layer in _children(m):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            return layer.in_channels
        for child in _children(layer):
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                return child.in_channels
    raise ValueError('No Conv2d found in encoder')


def _model_sizes(encoder: nn.Module, size: Tuple[int, int] = (256, 256)) -> list:
    """Output shape of every top-level encoder module for a dummy `size` input."""
    hooks = [_SaveFeatures(m) for m in _children(encoder)]
    with torch.no_grad():
        encoder.eval()(torch.randn(1, _in_channels(encoder), *size))
    sizes = [tuple(h.stored.shape) for h in hooks]
    for h in hooks:
        h.remove()
    return sizes


class _SaveFeaturesList:
    def __init__(self, modules):
        self.sfs = [_SaveFeatures(m) for m in modules]

    def __getitem__(self, i):
        return self.sfs[i]

    def remove(self):
        for h in self.sfs:
            h.remove()


class CustomPixelShuffle_ICNR(nn.Module):
    """Upsample by scale from ni filters to nf (default ni), using nn.PixelShuffle, icnr init, and weight_norm."""

    def __init__(self, ni, nf=None, scale=2, blur=False, leaky=None, **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class UnetBlockDeep(nn.Module):
    """A quasi-UNet block, using PixelShuffle_ICNR upsampling."""

    def __init__(self, up_in_c, x_in_c, hook, final_div=True, blur=False,
                 leaky=None, self_attention=False, nf_factor=1.0, **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class DynamicUnetDeep(SequentialEx):
    """Create a U-Net from a given architecture."""

    def __init__(self, encoder: nn.Module, n_classes: int, blur=False, blur_final=True,
                 self_attention=False, y_range: Optional[Tuple[float, float]] = None,
                 last_cross=True, bottle=False,
                 norm_type=NormType.Batch, nf_factor=1.0, **kwargs):
        super().__init__()
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = _model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = _SaveFeaturesList([encoder[i] for i in sfs_idxs])
        with torch.no_grad():
            x = encoder.eval()(torch.randn(1, _in_channels(encoder), *imsize)).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
            custom_conv_layer(ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
        ).eval()
        with torch.no_grad():
            x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = UnetBlockDeep(
                up_in_c, x_in_c, self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                nf_factor=nf_factor,
                **kwargs
            ).eval()
            layers.append(unet_block)
            with torch.no_grad():
                x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += _in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        # rebuild the SequentialEx module list in one go (same key layout as
        # fastai's incremental append)
        self.layers = nn.ModuleList(layers)

    def __del__(self):
        if hasattr(self, 'sfs'):
            self.sfs.remove()
