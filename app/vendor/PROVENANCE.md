# Vendored code provenance

Everything under `app/vendor/` is third-party code, copied so shoebox runs
without patching site-packages. Files are unmodified except where a
`shoebox edit:` comment marks a change.

## facelib/ — from CodeFormer (github.com/sczhou/CodeFormer, master)

- `facelib/utils/__init__.py`, `misc.py`, `face_restoration_helper.py`,
  `face_utils.py` — copied verbatim.
  - `misc.py`: one edit — `ROOT_DIR` points at the shoebox project root so
    weight downloads land in `<project>/weights/` (upstream used its repo
    root; the vendored copy sits two directories deeper).
- `facelib/detection/__init__.py` — copied with the yolov5face package
  removed (imports, `init_yolov5face_model`, and the Google-Drive fallback).
  shoebox ships the retinaface detectors only; requesting a YOLOv5 model
  raises a clear NotImplementedError.
- `facelib/detection/retinaface/*.py`, `align_trans.py`,
  `matlab_cp2tform.py` — verbatim.
- `facelib/parsing/__init__.py`, `bisenet.py`, `parsenet.py`, `resnet.py` —
  verbatim.
- License: S-Lab License 1.0 (non-commercial use permitted with notice;
  reproduced in the project LICENSE).

## basicsr/ — a shim, not the real package

The published `basicsr` package imports
`torchvision.transforms.functional_tensor`, removed in torchvision 0.17, so
`import basicsr` fails on any current torch. The shim provides only what the
vendored code imports:

- `basicsr/utils/download_util.py` — `load_file_from_url`, resolving against
  the shoebox project root.
- `basicsr/utils/misc.py` — `get_device`, `gpu_is_available`.
- `basicsr/utils/__init__.py` — `get_root_logger`, `img2tensor`, `tensor2img`,
  `imwrite` (upstream-equivalent implementations).
- `basicsr/utils/registry.py` — no-op `ARCH_REGISTRY` stub.
- `basicsr/paths.py` — project-root/weights location helper.
- `basicsr/archs/vqgan_arch.py` — copied verbatim from CodeFormer's basicsr
  (it keeps its `@ARCH_REGISTRY.register()` decorator, satisfied by the stub).

## codeformer_arch.py — from CodeFormer (basicsr/archs/)

Copied; edits: dropped the `get_root_logger`/`ARCH_REGISTRY` imports and the
registry decorator (class is instantiated directly).

## rrdbnet_arch.py — from CodeFormer's basicsr (basicsr/archs/)

Copied; edits: registry dropped, and the three helpers used
(`default_init_weights`, `make_layer`, `pixel_unshuffle`) are inlined because
upstream pulls them from `arch_util.py`, which imports compiled deform-conv
ops shoebox never uses.

## realesrgan_utils.py — from Real-ESRGAN (github.com/xinntao/Real-ESRGAN,
realesrgan/utils.py)

Copied verbatim (RealESRGANer, the tiled upsampler CodeFormer's own script
uses for `--bg_upsampler realesrgan`). Its `basicsr.utils.download_util`
import resolves to the shim. License: BSD-3-Clause (Real-ESRGAN).

## deoldify/ — inference port of DeOldify + fastai v1.0.60

DeOldify pins fastai 1.0.60, which does not run on current PyTorch. The port
keeps only inference and preserves the module tree so the published
`ColorizeArtistic_gen.pth` loads with `strict=True`:

- `fastai_layers.py` — subset of fastai v1.0.60 `layers.py` (Apache-2.0):
  NormType, init_default, SelfAttention, conv_layer, SequentialEx,
  MergeLayer, res_block, SigmoidRange, icnr, PixelShuffle_ICNR; plus
  DeOldify's `custom_conv_layer` (MIT) with its extra_bn flag.
- `unet.py` — DeOldify's `DynamicUnetDeep`/`UnetBlockDeep`/
  `CustomPixelShuffle_ICNR` (MIT) with fastai hook machinery replaced by
  plain `register_forward_hook` captures and a dummy forward for sizes.
- `colorizer.py` — DeOldify `filters.py` (MIT) inference math: LA→RGB
  grayscale input, render_factor square, ImageNet normalize/denormalize,
  YUV chroma transplant back onto the original luminance. Model construction
  mirrors `gen_inference_deep` (resnet34 body, nf_factor 1.5, Spectral norm,
  self-attention, y_range (-3, 3)).

Weight files are NOT in the repository; they auto-download on first use from
the URLs in `app/models.py` (CodeFormer v0.1.0 release assets, as read from
inference_codeformer.py and facelib's init code; DeOldify's
ColorizeArtistic_gen.pth from data.deepai.org — MIT-licensed per the DeOldify
README).
