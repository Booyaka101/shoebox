import os
import torch
from torch import nn
from copy import deepcopy

from facelib.utils import load_file_from_url
from facelib.utils import download_pretrained_models

# shoebox edit: the vendored copy ships only the retinaface detector (the
# CodeFormer default). The yolov5face package and its two model branches are
# omitted; asking for them raises a clear error instead of an ImportError.

from .retinaface.retinaface import RetinaFace


def init_detection_model(model_name, half=False, device='cuda'):
    if model_name == 'retinaface_resnet50':
        model = init_retinaface_model(model_name, half, device)
    elif model_name == 'retinaface_mobile0.25':
        model = init_retinaface_model(model_name, half, device)
    else:
        raise NotImplementedError(
            f"{model_name} is not vendored in shoebox; use retinaface_resnet50 "
            "(the CodeFormer default) or retinaface_mobile0.25.")

    return model


def init_retinaface_model(model_name, half=False, device='cuda'):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='weights/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model


# shoebox edit: init_yolov5face_model and its Google-Drive fallback are removed
# along with the yolov5face package this vendored copy does not ship.
