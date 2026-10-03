# Changelog

## 1.0.0 — 2026-10-03

First release.

- Batch restore a folder of scans: DeOldify artistic colorization (grayscale
  detected by saturation heuristic), CodeFormer face restoration, optional
  RealESRGAN x2 background upscale.
- Originals are only ever read; restored copies plus JSON sidecar metadata
  land in `results/<job id>/restored/`.
- Browser UI: folder picker (native dialog or typed path), preset radio
  (low / balanced / max fidelity), colorize and upscale toggles, live stage
  and progress, before/after comparison gallery, flagged-photo queue with
  re-run at higher fidelity, cancel.
- QA flags: face-restoration failure (no silent passthrough), face-count
  decrease, sharpness drop over 15%, colorization that stayed gray.
- Export folder per job; Immich push (find-or-create album, upload, retry
  queue for failed uploads).
- Edge cases: corrupt files skipped with a reason and the job continues,
  same-stem collisions get a content-hash suffix, CPU fallback with a
  warning, tiled background pass for large images.
- Vendored CodeFormer facelib with a basicsr shim, and a fastai-free
  inference port of DeOldify (loads ColorizeArtistic_gen.pth strict=True);
  see app/vendor/PROVENANCE.md.
- Tests: real end-to-end run over the LOC Families corpus plus Immich client
  tests against a mock server.
