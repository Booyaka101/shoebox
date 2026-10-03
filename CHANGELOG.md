# Changelog

## 1.0.3 — 2026-10-03

Third loop: the batch-workflow release.

- **Pause and resume.** Pause stops after the current photo and remembers
  what is left; Resume processes the remainder into the same job. Verified
  live: paused during colorization (5 remaining), resumed to done.
- **Queue multiple folders.** Submitting a job while another runs queues it
  (up to 8) instead of refusing; the worker runs them in order.
- **Delete old jobs** from the Previous-jobs dropdown (refuses while a job
  is queued/running).
- **Contact-sheet PDF** of the restored set (12 labelled thumbnails per
  page) next to the export button.
- The photo currently being processed shows in the progress bar; gallery
  metadata lines no longer render `undefined` for missing sidecars.
- Fixed a race found by the new endpoint tests: cancelling a job that the
  worker had just dequeued could be overwritten by the job's own start-up,
  so the job ran to completion. Cancelling a queued job now also raises the
  pipeline's cancel flag; the worker re-checks under the lock.
- Tests: 32 (pause/resume, queue ordering, queued-cancel race, delete,
  contact sheet).

## 1.0.2 — 2026-10-03

Second review loop: archive fidelity and launcher polish.

- Restored JPEGs and TIFFs now keep the source photo's EXIF (dates, camera,
  GPS) so Immich timelines survive the restore; the Orientation tag is
  stripped because the restore bakes the rotation into the pixels. PNG
  outputs stay EXIF-free by design.
- run.bat detects the GPU: no NVIDIA card means the CPU PyTorch build is
  installed instead of the 2.5 GB CUDA one; failed installs now explain
  themselves and pause instead of vanishing.
- "Port already in use" prints a friendly hint instead of a stack trace.
- Exports now include each photo's sidecar JSON next to the image.
- Jobs warn (in the UI) when photos above 24 megapixels will be slow.
- Tests: 26 (EXIF kept / orientation stripped / export sidecar cases).

## 1.0.1 — 2026-10-03

Review-pass fixes and enhancements.

- Photos are written to `restored/` the moment each one finishes; closing the
  app mid-job keeps everything already written, and the job shows up as
  "interrupted" on the next start (previously such a job blocked new ones
  with a "already running" error until its status file was deleted).
- Memory now stays flat per photo instead of growing with folder size —
  inputs and intermediates are released as soon as each photo is written.
- Filenames with `&`, `#`, `+` or spaces no longer break the gallery
  (URLs are percent-encoded).
- Previous jobs list in the UI; pick an old job to browse its gallery again.
- "Show in Explorer" button next to export; re-exporting replaces the export
  folder instead of leaving stale files.
- Elapsed time and ETA while a job runs; "not processed" counter when a job
  is stopped.
- Zero-byte leftover model files from a crashed first run are re-downloaded
  instead of failing model load; interrupted downloads clean up their
  `.part` files and progress entry.
- Immich upload timeout raised to 300 s for large PNGs over slow links.
- Tests: 25 (endpoint tests via FastAPI TestClient, crash-recovery, URL
  encoding, export wipe, retry-queue flake fixed with a threaded mock).

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
