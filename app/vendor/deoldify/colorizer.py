"""DeOldify artistic image colorization — inference port (MIT, jantic/DeOldify).

The filter math in this file mirrors deoldify/filters.py (ModelImageVisualizer
/ BaseFilter / ColorizerFilter) with fastai's Learner replaced by a bare
nn.Module. Key contract:

  1. input photo -> grayscale (LA -> RGB), so only luminance is colorized
  2. stretch to a render_factor*16 square, normalize with ImageNet stats
  3. forward through DynamicUnetDeep, denormalize
  4. resize back to the original size, then transplant only the chroma
     channels (YUV) onto the original's own luminance
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image as PilImage
import torchvision

from .unet import DynamicUnetDeep
from .fastai_layers import NormType

# From DeOldify deoldify/filters.py, BaseFilter(stats=imagenet_stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

ARTISTIC_WEIGHTS_URL = 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth'
WEIGHTS_NAME = 'ColorizeArtistic_gen.pth'

logger = logging.getLogger(__name__)


def _load_state_dict(weights_path: Path) -> dict:
    sd = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']  # fastai v1 Learner.save wraps the state dict
    clean = {}
    for k, v in sd.items():
        clean[k[len('module.'):] if k.startswith('module.') else k] = v
    return clean


def build_artistic_model(weights_path: Path, device: torch.device) -> nn.Module:
    """DynamicUnetDeep exactly as gen_inference_deep builds it (resnet34 body,
    nf_factor=1.5, Spectral norm, self-attention, y_range=(-3, 3))."""
    body = nn.Sequential(*list(torchvision.models.resnet34().children())[:-2])
    net = DynamicUnetDeep(
        body,
        n_classes=3,
        blur=True,
        blur_final=True,
        self_attention=True,
        y_range=(-3.0, 3.0),
        norm_type=NormType.Spectral,
        nf_factor=1.5,
    )
    sd = _load_state_dict(weights_path)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net.to(device)


class ArtisticColorizer:
    """Port of ModelImageVisualizer(filtr=ColorizerFilter(gen_inference_deep(...)))"""

    RENDER_BASE = 16  # DeOldify ColorizerFilter.render_base

    def __init__(self, weights_path: Path, device: torch.device):
        self.device = device
        self.model = build_artistic_model(weights_path, device)

    def colorize(self, rgb_array: np.ndarray, render_factor: int = 35) -> np.ndarray:
        """rgb_array: HxWx3 uint8 RGB. Returns HxWx3 uint8 RGB, colorized."""
        orig = PilImage.fromarray(rgb_array)
        filtered = orig.copy()
        render_sz = render_factor * self.RENDER_BASE
        model_image = self._model_process(filtered, render_sz)
        raw_color = self._unsquare(model_image, orig)
        return np.asarray(self._post_process(raw_color, orig))

    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        # a simple stretch to fit a square really makes a big difference in
        # rendering quality/consistency (DeOldify BaseFilter._scale_to_square)
        return orig.resize((targ, targ), resample=PilImage.BILINEAR)

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._scale_to_square(orig, sz).convert('LA').convert('RGB')
        x = torch.from_numpy(np.asarray(model_image, dtype=np.float32))
        x = x.permute(2, 0, 1).contiguous()
        x = x.to(self.device)
        x.div_(255)
        x = (x - torch.from_numpy(IMAGENET_MEAN).to(x.device)) / torch.from_numpy(IMAGENET_STD).to(x.device)
        with torch.no_grad():
            out = self.model(x[None])[0]
        out = out.float().cpu()
        out = out * torch.from_numpy(IMAGENET_STD) + torch.from_numpy(IMAGENET_MEAN)
        out = out.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
        return PilImage.fromarray(out)

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        return image.resize(orig.size, resample=PilImage.BILINEAR)

    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        # keep the original's luminance, take only chroma from the model:
        # human eyes are far less sensitive to chroma error than luma error
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        return PilImage.fromarray(cv2.cvtColor(hires, cv2.COLOR_YUV2RGB))
