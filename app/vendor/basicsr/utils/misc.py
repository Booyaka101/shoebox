"""shoebox shim for basicsr.utils.misc."""

import torch


def gpu_is_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


__all__ = ['gpu_is_available', 'get_device']
