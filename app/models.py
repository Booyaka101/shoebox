"""Lazy model singletons for shoebox.

One heavy model is resident at a time: the pipeline runs stage-by-stage
(colorize -> face restore -> background upscale -> QA detection) and calls
unload() between stages, with torch.cuda.empty_cache() to actually return the
memory. Weights auto-download on first use into <project>/weights/ from the
URLs pinned below (CodeFormer v0.1.0 release assets, as read from
inference_codeformer.py and facelib's own init code, plus DeOldify's
ColorizeArtistic_gen.pth from data.deepai.org).

GPU note: torch.cuda.is_available() can report True while the CUDA runtime is
actually unusable (missing/broken DLLs surface only at the first compute), so
get_device() runs a tiny real CUDA op before trusting the device and falls
back to CPU with a warning otherwise.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / 'vendor'
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import torch  # noqa: E402  (path setup must precede vendor imports)

log = logging.getLogger(__name__)

# Pinned weight sources. CodeFormer URLs are the ones its own inference script
# and facelib package download; the DeOldify URL is from the README
# "Completed Generator Weights" section (weights released under MIT).
WEIGHT_URLS = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'deoldify': 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth',
}

WEIGHT_FILES = {
    'codeformer': ('codeformer.pth', Path('CodeFormer')),
    'realesrgan': ('RealESRGAN_x2plus.pth', Path('CodeFormer')),
    'detection': ('detection_Resnet50_Final.pth', Path('facelib')),
    'parsing': ('parsing_parsenet.pth', Path('facelib')),
    'deoldify': ('ColorizeArtistic_gen.pth', Path('deoldify')),
}

_state_lock = threading.RLock()
_models = {}          # name -> loaded model object
_download_status = {}  # name -> {"done": int, "total": int} while downloading
_device_cache = None
_device_warning = None


def get_device():
    """torch.device for processing; forces a real CUDA op before trusting GPU."""
    global _device_cache, _device_warning
    with _state_lock:
        if _device_cache is not None:
            return _device_cache
        forced = os.environ.get('SHOEBOX_DEVICE', '').strip().lower()
        if forced == 'cpu':
            _device_cache = torch.device('cpu')
            return _device_cache
        if torch.cuda.is_available():
            try:
                probe = torch.zeros(4, 4, device='cuda')
                probe @ probe
                _device_cache = torch.device('cuda')
                return _device_cache
            except RuntimeError as err:
                _device_warning = (
                    'CUDA is present but broken (%s); running on CPU. Jobs will be '
                    'much slower.' % str(err).splitlines()[0])
                log.warning(_device_warning)
        else:
            _device_warning = ('No NVIDIA GPU with a working CUDA runtime found; '
                               'running on CPU. Jobs will be much slower.')
        _device_cache = torch.device('cpu')
        return _device_cache


def device_warning():
    return _device_warning


def device_name():
    if get_device().type == 'cuda':
        return torch.cuda.get_device_name(0)
    return 'CPU'


def weights_path(name):
    filename, subdir = WEIGHT_FILES[name]
    path = Path(os.environ.get('SHOEBOX_WEIGHTS', '')) if os.environ.get('SHOEBOX_WEIGHTS') \
        else Path(__file__).resolve().parents[1] / 'weights'
    return path / subdir / filename


def download_status():
    return dict(_download_status)


def ensure_weights(names):
    """Download any missing weights, streaming progress into _download_status.
    A zero-byte leftover from a crashed download counts as missing, and stale
    .part files are removed before a fresh attempt."""
    import requests
    for name in names:
        stale = weights_path(name)
        if stale.suffix == '.pth' and stale.exists() and stale.stat().st_size == 0:
            stale.unlink()
    missing = [n for n in names
               if not weights_path(n).exists() or weights_path(n).stat().st_size == 0]
    for name in missing:
        url = WEIGHT_URLS[name]
        target = weights_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.part')
        if tmp.exists():
            tmp.unlink()
        log.info('downloading %s weights: %s', name, url)
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                _download_status[name] = {'done': 0, 'total': total}
                done = 0
                with open(tmp, 'wb') as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        _download_status[name] = {'done': done, 'total': total}
            if target.exists():
                target.unlink()
            tmp.rename(target)
            log.info('downloaded %s -> %s', name, target)
        finally:
            _download_status.pop(name, None)
    return missing


def unload(*names):
    with _state_lock:
        for name in names:
            _models.pop(name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_all():
    with _state_lock:
        _models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_colorizer():
    """DeOldify artistic colorizer (ColorizeArtistic_gen.pth)."""
    with _state_lock:
        if 'colorizer' not in _models:
            ensure_weights(['deoldify'])
            from deoldify.colorizer import ArtisticColorizer
            t0 = time.time()
            _models['colorizer'] = ArtisticColorizer(weights_path('deoldify'), get_device())
            log.info('colorizer loaded in %.1fs', time.time() - t0)
        return _models['colorizer']


def get_codeformer():
    """CodeFormer net + FaceRestoreHelper (det retinaface_resnet50, parse
    parsenet), the exact construction from inference_codeformer.py."""
    with _state_lock:
        if 'face' not in _models:
            ensure_weights(['codeformer', 'detection', 'parsing'])
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
            import codeformer_arch
            device = get_device()
            t0 = time.time()
            net = codeformer_arch.CodeFormer(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=['32', '64', '128', '256']).to(device)
            ckpt = torch.load(str(weights_path('codeformer')))['params_ema']
            net.load_state_dict(ckpt)
            net.eval()
            helper = FaceRestoreHelper(
                1,  # upscale factor is set per job via set_upscale_factor
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=device)
            _models['face'] = (helper, net)
            log.info('face restoration models loaded in %.1fs', time.time() - t0)
        return _models['face']


def get_face_detector():
    """Detection-only model for counting faces in restored output (QA)."""
    with _state_lock:
        if 'detector' not in _models:
            ensure_weights(['detection'])
            from facelib.detection import init_detection_model
            _models['detector'] = init_detection_model(
                'retinaface_resnet50', half=False, device=get_device())
        return _models['detector']


def get_bg_upsampler(bg_tile=400):
    """RealESRGAN x2plus background upsampler, tiled, as inference_codeformer.py
    builds it for --bg_upsampler realesrgan."""
    with _state_lock:
        if 'bg' not in _models:
            ensure_weights(['realesrgan'])
            from rrdbnet_arch import RRDBNet
            from realesrgan_utils import RealESRGANer
            device = get_device()
            use_half = False
            if device.type == 'cuda':
                no_half = ('1650', '1660')  # f16-unsafe GPUs, per upstream
                if not any(g in torch.cuda.get_device_name(0) for g in no_half):
                    use_half = True
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                            num_grow_ch=32, scale=2)
            _models['bg'] = RealESRGANer(
                scale=2,
                model_path=str(weights_path('realesrgan')),
                model=model,
                tile=bg_tile,
                tile_pad=40,
                pre_pad=0,
                half=use_half,
                device=device)
        return _models['bg']
