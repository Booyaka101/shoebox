"""shoebox shim for basicsr.utils: the pieces facelib/vqgan import at module
load time. tensor2img mirrors the upstream implementation closely enough for
the inference paths shoebox uses (img2tensor/imwrite are NOT duplicated here —
facelib's own misc module is the single source for those)."""

import logging

import numpy as np
import torch

from .download_util import load_file_from_url  # noqa: F401 (re-exported)


def get_root_logger(logger_name='basicsr', log_level=0, log_file=None):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if isinstance(tensor, list):
        return [tensor2img(t, rgb2bgr, out_type, min_max) for t in tensor]
    tensor = tensor.squeeze(0).float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy()
    if img_np.ndim == 1:
        raise TypeError('Only support 4D, 3D or 2D tensor')
    if img_np.ndim == 3 and img_np.shape[0] == 1:  # single-channel -> tile
        img_np = np.tile(img_np, (3, 1, 1))
    if rgb2bgr and img_np.ndim == 3:
        img_np = img_np[[2, 1, 0], :, :]
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


__all__ = ['get_root_logger', 'tensor2img', 'load_file_from_url']
