"""Per-image restoration pipeline for shoebox.

Flow for one photo (originals are only ever opened for reading):

  1. read (unicode-safe via np.fromfile + imdecode, 8-bit BGR)
  2. saturation heuristic decides colorization (already-color photos skip it)
  3. DeOldify artistic colorization when needed
  4. CodeFormer face restore through FaceRestoreHelper; per-face inference
     failures are flagged for review — CodeFormer's own except-branch pastes
     the untouched face back and moves on silently, which is exactly what a
     restoration tool must not do
  5. optional RealESRGAN x2 background (tiled), feeding the paste-back like
     inference_codeformer.py does with --bg_upsampler realesrgan
  6. write restored JPEG/PNG + sidecar JSON metadata

Heavy models are loaded per stage and unloaded between stages; with
upscale_bg enabled the RealESRGAN tile runs inside the face stage because the
restored faces are pasted onto the upscaled background (the reference flow).
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import models, qa

log = logging.getLogger(__name__)

PRESETS = {
    # fidelity_weight: CodeFormer's w dial — higher = stay closer to the
    # original face, lower = let the model take over. render_factor drives
    # DeOldify's colorization resolution (in 16px units).
    'low': {'w': 0.2, 'render_factor': 21},
    'balanced': {'w': 0.5, 'render_factor': 32},
    'max': {'w': 0.9, 'render_factor': 45},
}
PRESET_ORDER = ['low', 'balanced', 'max']

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

PIPELINE_LOCK = threading.Lock()  # one job (or rerun) at a time


def required_metadata_keys():
    return {'source_hash', 'models', 'preset', 'w', 'w_final', 'face_count_in',
            'face_count_out', 'sharpness_in', 'sharpness_out', 'errors', 'flags'}


def find_images(folder: Path):
    """Sorted image files, ignoring macOS metadata junk and prior outputs."""
    files = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith('._'):
            continue
        if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith('_restored'):
            files.append(p)
    return files


def read_image_bgr(path: Path):
    """cv2.imread fails on non-ASCII Windows paths; go through the bytes."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError('file is empty')
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('unreadable or corrupt image data')
    return img


def sha256_of(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def output_name(job_dir: Path, stem: str, ext: str, src_hash: str):
    """<stem>_restored<ext>; content-hash suffix on collisions."""
    base = job_dir / 'restored' / f'{stem}_restored{ext}'
    if base.exists():
        base = job_dir / 'restored' / f'{stem}_restored_{src_hash[:8]}{ext}'
    return base


def encode_image(img_bgr, ext: str) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise ValueError(f'encoding failed for {ext}')
    return buf.tobytes()


def run_pipeline(paths, preset, colorize, upscale_bg, work_state=None, cancel=None):
    """Restore the given Paths stage by stage. Returns one result dict per path.

    paths must already be validated to exist. work_state is the shared job
    dict (progress keys are updated in place); cancel is a dict with a bool
    'flag' checked between images.
    """

    def cancelled():
        return bool(cancel and cancel.get('flag'))

    def report(stage, done, total):
        if work_state is not None:
            work_state['stage'] = stage
            work_state['stage_done'] = done
            work_state['stage_total'] = total

    presets = dict(PRESETS[preset])
    w = presets['w']
    render_factor = presets['render_factor']

    # stage 0: read + measure inputs
    infos = []
    for i, path in enumerate(paths):
        report('reading', i, len(paths))
        info = {
            'path': path,
            'name': path.name,
            'stem': path.stem,
            'ext': '.jpg' if path.suffix.lower() in ('.jpg', '.jpeg') else '.png',
            'src_hash': sha256_of(path),
            'errors': [],
        }
        try:
            info['img_in'] = read_image_bgr(path)
        except Exception as err:
            info['skip'] = f'unreadable: {err}'
            infos.append(info)
            continue
        h, w_in = info['img_in'].shape[:2]
        info['w_in'], info['h_in'] = int(w_in), int(h)
        info['was_gray'] = qa.chroma_variance(
            cv2.cvtColor(info['img_in'], cv2.COLOR_BGR2RGB)) <= 10.0
        info['colorize_needed'] = bool(colorize and info['was_gray'])
        info['sharpness_in'] = qa.sharpness(info['img_in'])
        infos.append(info)
    report('reading', len(paths), len(paths))

    # stage 1: colorization (DeOldify artistic)
    to_color = [n for n in infos if 'img_in' in n and n['colorize_needed']]
    if to_color:
        colorizer = models.get_colorizer()
        try:
            for i, info in enumerate(to_color):
                if cancelled():
                    break
                report('colorizing', i, len(to_color))
                rgb = cv2.cvtColor(info['img_in'], cv2.COLOR_BGR2RGB)
                colored = colorizer.colorize(rgb, render_factor=render_factor)
                info['img_current'] = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        finally:
            models.unload('colorizer')
    for info in infos:
        info.setdefault('img_current', info.get('img_in'))

    # stage 2: face restoration (+ optional RealESRGAN background inside the
    # paste-back, as CodeFormer's own script does)
    to_face = [n for n in infos if 'img_current' in n and not n.get('skip')]
    if to_face:
        helper, net = models.get_codeformer()
        bg = models.get_bg_upsampler() if upscale_bg else None
        device = models.get_device()
        try:
            for i, info in enumerate(to_face):
                if cancelled():
                    break
                report('restoring faces', i, len(to_face))
                _face_stage(info, helper, net, bg, device, w, upscale_bg)
        finally:
            models.unload('face', 'bg')
    report('restoring faces', len(to_face), len(to_face))

    # stage 3: QA — count faces in the final output with the detector alone
    to_qa = [n for n in infos if n.get('img_final') is not None]
    if to_qa:
        detector = models.get_face_detector()
        try:
            for i, info in enumerate(to_qa):
                report('quality check', i, len(to_qa))
                info['face_count_out'] = _count_faces(info['img_final'], detector)
        finally:
            models.unload('detector')
    for info in infos:
        info.setdefault('face_count_out', 0)

    # stage 4: metrics, flags, write outputs
    for i, info in enumerate(infos):
        report('writing', i, len(infos))
        if info.get('skip'):
            continue
        out_same_size = info['img_final']
        if out_same_size.shape[:2] != info['img_in'].shape[:2]:
            resized = cv2.resize(out_same_size, (info['w_in'], info['h_in']),
                                 interpolation=cv2.INTER_AREA)
        else:
            resized = out_same_size
        info['sharpness_out'] = qa.sharpness(resized)
        still_gray = qa.chroma_variance(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)) <= 10.0
        info['flags'] = qa.evaluate(
            face_errors=info['errors'],
            face_count_in=info['face_count_in'],
            face_count_out=info['face_count_out'],
            sharpness_in=info['sharpness_in'],
            sharpness_out=info['sharpness_out'],
            was_gray=info['was_gray'],
            colorize_requested=bool(colorize),
            restored_still_gray=still_gray)
        info['meta'] = {
            'source_hash': info['src_hash'],
            'models': {
                'colorize': 'DeOldify ColorizeArtistic_gen.pth' if info['colorize_needed'] else None,
                'face_restore': 'CodeFormer codeformer.pth (retinaface_resnet50 detection)',
                'bg_upscale': 'RealESRGAN_x2plus.pth x2' if upscale_bg else None,
            },
            'preset': preset,
            'w': w,
            'w_final': info.get('w_final', w),
            'render_factor': render_factor,
            'w_in': info['w_in'],
            'h_in': info['h_in'],
            'w_out': int(info['img_final'].shape[1]),
            'h_out': int(info['img_final'].shape[0]),
            'face_count_in': info['face_count_in'],
            'face_count_out': info['face_count_out'],
            'sharpness_in': round(info['sharpness_in'], 2),
            'sharpness_out': round(info['sharpness_out'], 2),
            'errors': info['errors'],
            'flags': info['flags'],
        }
    return infos


def _face_stage(info, helper, net, bg, device, w, upscale_bg):
    """Face restoration for one image, mirroring inference_codeformer.py's
    per-image body — except inference failures flag instead of passing the
    original face through silently."""
    import torch
    from torchvision.transforms.functional import normalize
    from facelib.utils.misc import img2tensor
    from basicsr.utils import tensor2img

    info['face_count_in'] = 0
    info['errors'] = []
    helper.clean_all()
    helper.set_upscale_factor(2 if upscale_bg else 1)
    try:
        helper.read_image(info['img_current'])
        num_det = helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5)
        info['face_count_in'] = int(max(num_det, 0))
        helper.align_warp_face()
    except Exception as err:
        log.warning('face detection failed for %s: %s', info['name'], err)
        info['errors'].append(f'face detection failed: {err}')
        info['face_count_in'] = 0

    for idx, cropped_face in enumerate(helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
            helper.add_restored_face(restored_face.astype('uint8'), cropped_face)
        except Exception as err:
            # the deliberate difference from upstream: record and flag, never
            # paste the original face back pretending it was restored
            log.warning('CodeFormer inference failed for %s face %d: %s',
                        info['name'], idx, err)
            info['errors'].append(f'face {idx} restoration failed: {err}')
            torch.cuda.empty_cache()

    if not helper.cropped_faces:
        # nothing detectable/fixable: carry the current image through
        info['img_final'] = info['img_current']
        return

    try:
        bg_img = bg.enhance(info['img_current'], outscale=2)[0] if bg is not None else None
        helper.get_inverse_affine(None)
        paste_kwargs = {}
        if bg is not None:
            paste_kwargs = {'upsample_img': bg_img, 'face_upsampler': bg}
        info['img_final'] = helper.paste_faces_to_input_image(**paste_kwargs)
    except Exception as err:
        log.warning('paste-back failed for %s: %s', info['name'], err)
        info['errors'].append(f'paste-back failed: {err}')
        info['img_final'] = info['img_current']


def _count_faces(img_bgr, detector):
    """Count detectable faces on a restored image (QA mirror of the
    detector's own preprocessing in FaceRestoreHelper)."""
    import numpy as np
    h, w = img_bgr.shape[:2]
    scale = 640 / min(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=interp)
    import torch
    with torch.no_grad():
        bboxes = detector.detect_faces(img)
    if bboxes is None or bboxes.shape[0] == 0:
        return 0
    count = 0
    for bbox in bboxes:
        eye_dist = float(np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]]))
        if eye_dist >= 5:
            count += 1
    return count


def next_preset(preset):
    i = PRESET_ORDER.index(preset) if preset in PRESET_ORDER else 0
    return PRESET_ORDER[min(i + 1, len(PRESET_ORDER) - 1)]
