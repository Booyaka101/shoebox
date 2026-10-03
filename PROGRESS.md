# PROGRESS — shoebox

Status: **v1.0.0 complete and verified end-to-end.** (2026-10-03)

## What works (verified on this machine, RTX 4090, torch 2.14.1+cu126)

- `run.bat` one-click launcher (venv + CUDA torch + deps + serve + open browser)
- Full pipeline per photo: saturation heuristic → DeOldify artistic
  colorization → CodeFormer face restore (RetinaFace det + parsing) →
  optional RealESRGAN x2 tiled background → metrics → restored JPEG/PNG +
  JSON sidecar (`source_hash, models, preset, w, w_final, face_count_in/out,
  sharpness_in/out, errors, flags`, plus dimensions)
- Stage-based model residency with `torch.cuda.empty_cache()` between stages;
  CPU fallback with a warning surfaced as a UI banner
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

## Real end-to-end run (the one to beat)

Job `20261003-122430-d1dd59`, folder of 5 LOC scans + corrupt file + stem
collision, balanced preset, colorize on: **restored 6, flagged 2, skipped 1,
originals byte-identical (SHA-256 checked), gallery + export verified in the
browser.** Output in `results/20261003-122430-d1dd59/`.

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
