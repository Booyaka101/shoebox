# PROGRESS — shoebox

Status: **v1.0.4 complete and verified end-to-end.** (2026-10-04)

## 1.0.4 review loop (this round)

- History across restarts: every past job opens from disk after a restart
  (gallery, export, contact sheet, Immich) — previously only the newest
  survived and older ones 404'd. Job ids are path-sanitized.
- Resume after a crash: `/resume` now also accepts interrupted jobs and
  computes the outstanding photos (results/skips subtracted from the
  folder). Verified by restart-simulation tests.
- `/status?job_id=` so the UI follows the selected job (a resumed old job is
  not the newest); Resume button offered for paused and interrupted jobs.
- Worker singleton (one for the process, not one per UI session);
  contact-sheet thumbnails decode large JPEGs at draft resolution and
  captions use a real font.
- Tests: 36/36 (restart survival, interrupted-resume, path-safety,
  empty-job contact sheet).

## 1.0.3 review loop

- Pause/resume: pause stops after the current photo and records the
  remainder; resume finishes it into the same job. Verified live with real
  models (paused during colorization with 5 remaining → resumed → done).
- Multi-folder queue: submissions while a job runs are queued (max 8) and
  run in order by a single worker.
- Delete old jobs (refuses while queued/running); contact-sheet PDF
  (12 labelled thumbnails/page); current-photo display; `undefined`
  metadata fallback in gallery cards.
- Fixed a real race the new tests exposed: cancelling a job the worker had
  just dequeued was overwritten by the job's own start-up (`cancelled` →
  `running`), so it ran to completion. Cancelling a queued job now also
  raises the pipeline cancel flag, and the worker re-checks the status
  under the lock. Stress-verified 5 consecutive green module runs (was
  failing ~1 in 3).
- Tests: 32/32.

## 1.0.2 review loop

- EXIF preservation: restored JPEG/TIFF keep source dates/camera/GPS (Immich
  timelines survive); Orientation tag stripped (imdecode bakes the rotation
  into pixels); Pillow encode with cv2 fallback. Verified by round-trip test.
- run.bat: GPU detection (nvidia-smi) picks CPU vs CUDA torch wheels; failed
  installs now explain themselves and pause
- Friendly "port already in use" message instead of a stack trace
- Export includes sidecar JSONs; job warning for >24MP scans
- Tests: 26/26

## 1.0.1 review loop

- Crash resilience: per-photo incremental writes to restored/; interrupted
  jobs recover cleanly on restart (verified live: mid-job kill → restart →
  job marked interrupted with warning, new job starts immediately — before
  the fix such a job 409-blocked every new job)
- Flat per-photo memory: inputs/intermediates dropped per photo; folder size
  no longer scales RAM
- Gallery URLs percent-encoded (filenames with `&`, `#`, `+` verified live
  with a file named `Oma & Opa 1961.jpg`)
- Job history dropdown (GET /jobs), Show-in-Explorer (POST /reveal, path
  pinned under results/), export wipe on re-export, elapsed/ETA, cancelled
  counter
- ensure_weights: zero-byte leftovers re-downloaded, .part cleanup,
  download-status reset on failure; Immich upload timeout 300 s
- Tests: 25/25 (new tests/test_main.py via FastAPI TestClient; threaded
  HTTP/1.1 mock server killed a recurring 10053 flake)

## What works (verified on this machine, RTX 4090, torch 2.14.1+cu126)

- `run.bat` one-click launcher (venv + CUDA torch + deps + serve + open browser)
- Full pipeline per photo: saturation heuristic → DeOldify artistic
  colorization → CodeFormer face restore (RetinaFace det + parsing) →
  optional RealESRGAN x2 tiled background → metrics → restored JPEG/PNG +
  JSON sidecar (`source_hash, models, preset, w, w_final, face_count_in/out,
  sharpness_in/out, errors, flags`, plus dimensions)
- Stage-based model residency with `torch.cuda.empty_cache()` between stages;
  CPU fallback with a warning surfaced as a UI banner
- RealESRGAN x2 tiled background verified end to end (restored faces pasted
  onto the upscaled background, outputs 2x)
- FastAPI on 127.0.0.1:8545 + single-page vanilla JS UI: folder picker
  (native dialog or typed path), presets, toggles, live stage/progress, model
  download progress, cancel, before/after slider gallery, flagged queue with
  re-run-at-higher-fidelity, export, Immich push
- QA flags: face restoration failed (never CodeFormer's silent passthrough),
  face-count decrease, sharpness drop >15%, colorization that stayed gray
- Edge cases verified: corrupt file skipped with reason while the job
  continues; `x.jpg`+`x.jpeg` stem collision gets a content-hash suffix;
  tinted nitrate scans skip colorization; Immich-unreachable produces a
  readable error (502) and failed uploads land in a retry queue that drains
  on the next push
- Immich client tested against a mock server (album find-or-create, upload,
  already-in-album skip, retry queue, drain)
- Packaging: pyproject (installable, console script `shoebox`), MIT LICENSE
  with third-party notices, vendored-code PROVENANCE, .gitignore, git repo
  committed, release zip at `dist/shoebox-1.0.0.zip`
- Tests: 17 passed (real E2E over the 5 LOC Families photos + unit + mock)

## Real end-to-end runs (the ones to beat)

- Job `20261003-124214-89d303` — 5 LOC scans + corrupt file, balanced
  preset, colorize + 2x upscale: **restored 5, flagged 3, skipped 1**;
  the 1939 sisters photo (grayscale, 2 faces) came out colorized, restored,
  1448×2048, faces 2→2, zero flags. Originals byte-identical.
- Job `20261003-122430-d1dd59` — same corpus without upscale, plus a
  same-stem collision: restored 6, flagged 2, skipped 1; the
  `x.jpg`/`x.jpeg` pair got the content-hash suffix; re-run at higher
  fidelity verified on the flagged photo.

Both galleries verified in the browser (headless Chrome + API), export
verified, Immich-unreachable error path verified. Output in `results/`.

## Next steps (concrete)

1. Owner ships it: create the GitHub repo `shoebox`, push `master`, cut a
   GitHub Release and attach `dist/shoebox-1.0.0.zip` (already built).
2. Owner self-verify on ~20 personal scans per the brief; the 5-photo public
   corpus (`examples/loc-families/`) is the offline stand-in.
3. README demo GIF is a cross-dissolve; a screen recording of a live job
   would be stronger.
4. Feature candidates (not built, in rough value order):
   - pause/resume for overnight jobs
   - sidecar EXIF embedding (write restoration metadata into the JPEG)
   - contact-sheet PDF of the restored set
   - auto render_factor/downscale for very large scans
   - background-removal-aware colorization mask (avoid tinting borders)
   - multi-folder queue
   - v2 per brief: scratch/dust inpainting pass, film-negative inversion

## Verification gaps (honest)

- The `pip install .` clean-path check used `--no-deps` (torch installs via
  run.bat's CUDA index; a full dep install in the check venv would pull the
  CPU torch from PyPI — the exact pitfall LESSONS warns about).
- Immich push has never touched a real Immich server (mock + unreachable
  error path only). First real push may surface auth-schema drift; the retry
  queue means nothing is lost if it does.
- The UI was driven via headless-Chrome screenshots + API calls; a human
  should drag the sliders once.
