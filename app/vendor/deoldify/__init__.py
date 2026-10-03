"""DeOldify artistic colorization, inference-only port.

Files here are derived from:
  * fastai v1.0.60 (Apache-2.0, https://github.com/fastai/fastai) — the layer
    building blocks in fastai_layers.py, ported so the published
    ColorizeArtistic_gen.pth checkpoint loads with exact key parity.
  * DeOldify (MIT, https://github.com/jantic/DeOldify) — the unet and the
    filter/post-process math in unet.py and colorizer.py, with fastai's
    Learner/hook machinery replaced by plain PyTorch equivalents.

DeOldify pins fastai 1.0.60, which does not run on current PyTorch, so this
port keeps only what inference needs. Layer order and module attribute names
match the originals exactly, verified by loading the checkpoint strict=True.
"""
