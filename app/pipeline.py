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
  6. quality metrics and flags, then `sink(name, stem, ext, src_hash, meta,
     data)` hands the caller the encoded output bytes — main.py writes them
     to disk immediately, so a crash mid-job keeps every finished photo

Memory: only one photo's pixels are alive at a time. Inputs and intermediates
are dropped as soon as a photo's output is handed to the sink, so folder size
does not scale RAM. Cancellation is checked between photos and records the
unprocessed ones as status 'cancelled' rather than failing them.
"""

import hashlib
import io
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import models, qa

log = logging.getLogger(__name__)

PIL_FORMATS = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.tif': 'TIFF',
               '.tiff': 'TIFF', '.bmp': 'BMP', '.webp': 'WEBP'}

ORIENTATION_TAG = 274  # baked into the pixels by imdecode; must not survive


def source_exif(path) -> bytes | None:
    """EXIF of the source scan, minus Orientation.

    cv2.imdecode applies the EXIF rotation, so the restored pixels are
    upright — copying the tag would make other tools rotate them again.
    Everything else (dates, camera, GPS) is kept for archives and Immich
    timelines. Returns None for files without EXIF or on any read error.
    """
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            exif.pop(ORIENTATION_TAG, None)
            return exif.tobytes() or None
    except Exception as err:
        log.debug('no EXIF read from %s: %s', path, err)
        return None

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
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file() or p.name.startswith('._'):
            continue
        if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith('_restored'):
            files.append(p)
    return files


def read_image_bgr(path):
    """cv2.imread fails on non-ASCII Windows paths; go through the bytes."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError('file is empty')
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('unreadable or corrupt image data')
    return img


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def output_name(job_dir, stem: str, ext: str, src_hash: str):
    """<stem>_restored<ext>; content-hash suffix on collisions."""
    job_dir = Path(job_dir)
    base = job_dir / 'restored' / f'{stem}_restored{ext}'
    if base.exists():
        base = job_dir / 'restored' / f'{stem}_restored_{src_hash[:8]}{ext}'
    return base


def encode_image(img_bgr, ext: str, exif_bytes: bytes = None) -> bytes:
    """JPEG/TIFF go through Pillow so source EXIF (minus orientation) rides
    along; any Pillow failure falls back to the plain OpenCV encode."""
    if ext in ('.jpg', '.jpeg', '.tif', '.tiff') and exif_bytes:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            params = {'exif': exif_bytes}
            if ext in ('.jpg', '.jpeg'):
                params['quality'] = 95
                params['subsampling'] = 1  # 4:2:2, like cv2's default
            buf = io.BytesIO()
            im.save(buf, format=PIL_FORMATS[ext], **params)
            return buf.getvalue()
        except Exception as err:
            log.warning('EXIF-preserving encode failed (%s); writing without EXIF', err)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise ValueError(f'encoding failed for {ext}')
    return buf.tobytes()


def run_pipeline(paths, preset, colorize, upscale_bg, sink, work_state=None, cancel=None):
    """Restore the given paths stage by stage. Returns one public info dict
    per path: {name, status: done|skipped|cancelled, meta?, reason?}.

    sink(name, stem, ext, src_hash, meta, data) is called for every finished
    photo, as soon as it is finished. paths must already exist.
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
            'name': path.name,
            'stem': path.stem,
            'ext': '.jpg' if path.suffix.lower() in ('.jpg', '.jpeg') else '.png',
            'src_hash': sha256_of(path),
            'errors': [],
            'status': 'pending',
            'exif': source_exif(path),
        }
        try:
            img_in = read_image_bgr(path)
        except Exception as err:
            info['status'] = 'skipped'
            info['reason'] = f'unreadable: {err}'
            infos.append(info)
            continue
        h, w_px = img_in.shape[:2]
        info['w_in'], info['h_in'] = int(w_px), int(h)
        info['was_gray'] = qa.chroma_variance(
            cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)) <= 10.0
        info['colorize_needed'] = bool(colorize and info['was_gray'])
        info['sharpness_in'] = qa.sharpness(img_in)
        info['chroma_in'] = qa.chroma_variance(
            cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB))
        info['_img'] = img_in
        infos.append(info)
    report('reading', len(paths), len(paths))
    big = [n for n in infos if n['status'] == 'pending'
           and n['w_in'] * n['h_in'] > 24_000_000]
    if big and work_state is not None:
        names = ', '.join(n['name'] for n in big[:3])
        work_state.setdefault('warnings', []).append(
            f'{len(big)} photo(s) above 24 megapixels ({names}) will take '
            'noticeably longer, especially with background upscaling.')

    # stage 1: colorization (DeOldify artistic)
    to_color = [n for n in infos if n['status'] == 'pending' and n['colorize_needed']]
    if to_color:
        colorizer = models.get_colorizer()
        try:
            for i, info in enumerate(to_color):
                if cancelled():
                    break
                report('colorizing', i, len(to_color))
                rgb = cv2.cvtColor(info['_img'], cv2.COLOR_BGR2RGB)
                colored = colorizer.colorize(rgb, render_factor=render_factor)
                info['_img'] = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        finally:
            models.unload('colorizer')
    if cancelled():
        for info in infos:
            if info['status'] == 'pending':
                info['status'] = 'cancelled'
    report('colorizing', len(to_color), len(to_color))

    # stage 2: face restoration (+ optional RealESRGAN background inside the
    # paste-back, as CodeFormer's own script does), QA face count via the
    # helper's own detector, metrics, write through the sink
    to_face = [n for n in infos if n['status'] == 'pending']
    if to_face:
        helper, net = models.get_codeformer()
        bg = models.get_bg_upsampler() if upscale_bg else None
        device = models.get_device()
        try:
            for i, info in enumerate(to_face):
                if cancelled():
                    info['status'] = 'cancelled'
                    continue
                report('restoring faces', i, len(to_face))
                try:
                    _restore_one(info, helper, net, bg, device, w, upscale_bg)
                    _finish_one(info, preset, w, render_factor, colorize, upscale_bg)
                    data = encode_image(info.pop('_final'), info['ext'], info.get('exif'))
                    sink(info['name'], info['stem'], info['ext'],
                         info['src_hash'], info['meta'], data)
                    info['status'] = 'done'
                except Exception as err:
                    log.exception('processing failed for %s', info['name'])
                    info['status'] = 'skipped'
                    info['reason'] = f'processing failed: {err}'
                finally:
                    info.pop('_img', None)
        finally:
            models.unload('face', 'bg')
    report('restoring faces', len(to_face), len(to_face))
    if cancelled():
        for info in infos:
            if info['status'] == 'pending':
                info['status'] = 'cancelled'
    return _public(infos)


def _restore_one(info, helper, net, bg, device, w, upscale_bg):
    """Face restoration for one image, mirroring inference_codeformer.py's
    per-image body — except inference failures flag instead of passing the
    original face through silently."""
    import torch
    from torchvision.transforms.functional import normalize
    from facelib.utils.misc import img2tensor
    from basicsr.utils import tensor2img

    info['face_count_in'] = 0
    info['errors'] = []
    info['_det'] = helper.face_detector
    helper.clean_all()
    helper.set_upscale_factor(2 if upscale_bg else 1)
    try:
        helper.read_image(info['_img'])
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
        info['_final'] = info['_img']
        return

    try:
        bg_img = bg.enhance(info['_img'], outscale=2)[0] if bg is not None else None
        helper.get_inverse_affine(None)
        paste_kwargs = {}
        if bg is not None:
            paste_kwargs = {'upsample_img': bg_img, 'face_upsampler': bg}
        info['_final'] = helper.paste_faces_to_input_image(**paste_kwargs)
    except Exception as err:
        log.warning('paste-back failed for %s: %s', info['name'], err)
        info['errors'].append(f'paste-back failed: {err}')
        info['_final'] = info['_img']


def _finish_one(info, preset, w, render_factor, colorize, upscale_bg):
    """QA face count (the helper's own detector), metrics, flags, metadata."""
    final = info['_final']
    if final.shape[:2] != (info['h_in'], info['w_in']):
        resized = cv2.resize(final, (info['w_in'], info['h_in']),
                             interpolation=cv2.INTER_AREA)
    else:
        resized = final
    info['sharpness_out'] = qa.sharpness(resized)
    info['face_count_out'] = _count_faces(final, info['_det'])
    info['chroma_out'] = qa.chroma_variance(
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    info['flags'] = qa.evaluate(
        face_errors=info['errors'],
        face_count_in=info['face_count_in'],
        face_count_out=info['face_count_out'],
        sharpness_in=info['sharpness_in'],
        sharpness_out=info['sharpness_out'],
        was_gray=info['was_gray'],
        colorize_requested=bool(colorize),
        restored_still_gray=info['chroma_out'] <= 10.0)
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
        'w_out': int(final.shape[1]),
        'h_out': int(final.shape[0]),
        'face_count_in': info['face_count_in'],
        'face_count_out': info['face_count_out'],
        'sharpness_in': round(info['sharpness_in'], 2),
        'sharpness_out': round(info['sharpness_out'], 2),
        'chroma_in': round(info['chroma_in'], 2),
        'chroma_out': round(info['chroma_out'], 2),
        'errors': info['errors'],
        'flags': info['flags'],
    }


def _count_faces(img_bgr, detector):
    """Count detectable faces on a restored image (QA mirror of the
    detector's own preprocessing in FaceRestoreHelper)."""
    import torch
    h, w = img_bgr.shape[:2]
    scale = 640 / min(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=interp)
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


def _public(infos):
    out = []
    for info in infos:
        pub = {'name': info['name'], 'status': info['status'],
               'src_hash': info['src_hash']}
        if info['status'] == 'skipped':
            pub['reason'] = info.get('reason', 'unknown')
        if info['status'] == 'done':
            pub['meta'] = info['meta']
        out.append(pub)
    return out


def next_preset(preset):
    i = PRESET_ORDER.index(preset) if preset in PRESET_ORDER else 0
    return PRESET_ORDER[min(i + 1, len(PRESET_ORDER) - 1)]
