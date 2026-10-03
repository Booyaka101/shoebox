"""shoebox — local batch photo-archive restorer.

FastAPI app: start a job over a folder of scanned photos, watch progress,
review the before/after gallery and the QA flag queue, export the restored
set, optionally push it to the user's own Immich server.

Endpoints (all JSON unless noted):
  POST /jobs    {folder, preset, colorize, upscale_bg}   start a batch job
  GET  /status  current or last job, model downloads, device
  GET  /pairs   [{name, before, after, meta, flags}]     gallery data
  POST /export  {job_id}                                 copy restored set out
  POST /immich  {job_id, base_url, api_key, album_name?} push to Immich
  POST /rerun   {job_id, name}                           re-run one photo at
                                                         the next preset
  POST /cancel  {job_id}                                 stop between photos
  GET  /img/before?job_id=&name= /img/after?job_id=&name=
  GET  /pick_folder                                      native folder dialog
  GET  /                                                 the UI
"""

import json
import logging
import shutil
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__, immich, models, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('shoebox')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
WEB_DIR = PROJECT_ROOT / 'web'


@asynccontextmanager
async def lifespan(_app):
    RESULTS_DIR.mkdir(exist_ok=True)
    _load_last_job()
    yield


app = FastAPI(title='shoebox', version=__version__, lifespan=lifespan,
              description='Free, local, batch photo-archive restorer.')

JOBS = {}            # job_id -> job dict (also mirrored to job.json)
JOBS_LOCK = threading.Lock()
_active_job_id = None


# ---------------------------------------------------------------- job store

def _job_dir(job_id: str) -> Path:
    return RESULTS_DIR / job_id


def _save_job(job, job_dir=None):
    """job_dir may be pinned by the caller: the job thread outlives the HTTP
    request that created it, so it must not re-resolve a directory global
    that could have changed between request and write."""
    if job_dir is None:
        job_dir = _job_dir(job['id'])
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {k: v for k, v in job.items() if k != 'cancel'}
    (job_dir / 'job.json').write_text(
        json.dumps(snapshot, indent=1, default=str), encoding='utf-8')


def _load_last_job():
    global _active_job_id
    if not RESULTS_DIR.exists():
        return
    candidates = []
    for d in RESULTS_DIR.iterdir():
        meta = d / 'job.json'
        if meta.is_file():
            try:
                candidates.append((meta.stat().st_mtime,
                                   json.loads(meta.read_text(encoding='utf-8'))))
            except (OSError, ValueError):
                continue
    if candidates:
        candidates.sort()
        job = candidates[-1][1]
        job.setdefault('cancel', {'flag': False})
        if job.get('status') in ('queued', 'running'):
            # the process died mid-job; nothing is running now
            job['status'] = 'interrupted'
            job.setdefault('warnings', []).append(
                'the app was closed while this job was running; photos already '
                'written to restored/ are kept')
        JOBS[job['id']] = job
        _active_job_id = job['id']


def _get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f'unknown job {job_id}')
    return job


def _latest_job() -> dict:
    with JOBS_LOCK:
        if _active_job_id and _active_job_id in JOBS:
            return JOBS[_active_job_id]
        if JOBS:
            return max(JOBS.values(), key=lambda j: j.get('started', 0))
    raise HTTPException(404, 'no jobs yet — start one from the front page')


# ---------------------------------------------------------------- job runner

def _make_sink(job, job_dir):
    """Returns the pipeline sink: writes each finished photo to disk the
    moment it is done, so a crash mid-job keeps the work."""
    restored_dir = Path(job_dir) / 'restored'
    restored_dir.mkdir(parents=True, exist_ok=True)

    def sink(name, stem, ext, src_hash, meta, data):
        out_path = pipeline.output_name(job_dir, stem, ext, src_hash)
        out_path.write_bytes(data)
        sidecar = out_path.parent / (out_path.stem + '.json')
        sidecar.write_text(json.dumps(meta, indent=1), encoding='utf-8')
        job['results'].append({'name': name, 'output': out_path.name,
                               'flags': meta.get('flags', [])})
        job['done'] += 1
        flagged = sum(1 for r in job['results'] if r['flags'])
        job['summary'] = {'restored': len(job['results']), 'flagged': flagged,
                          'skipped': len(job['skipped']),
                          'cancelled': len(job['cancelled'])}
        if job['done'] % 5 == 0:
            _save_job(job, job_dir)

    return sink


def _run_job_thread(job, job_dir):
    global _active_job_id
    try:
        folder = Path(job['folder'])
        paths = pipeline.find_images(folder)
        if not paths:
            job['status'] = 'error'
            job['warnings'].append(f'no supported images (jpg/png/tif/bmp/webp) in {folder}')
            return
        job['total'] = len(paths)
        job['status'] = 'running'
        job['skipped'] = []
        job['cancelled'] = []
        job['results'] = []

        def progress(stage, done, total):
            job['stage'], job['stage_done'], job['stage_total'] = stage, done, total
            _save_job(job, job_dir)

        infos = pipeline.run_pipeline(
            paths, job['preset'], job['colorize'], job['upscale_bg'],
            sink=_make_sink(job, job_dir), work_state=job, cancel=job['cancel'])

        job['status'] = 'cancelled' if job['cancel']['flag'] else 'done'
        for info in infos:
            if info['status'] == 'skipped':
                job['skipped'].append({'name': info['name'], 'reason': info['reason']})
            elif info['status'] == 'cancelled':
                job['cancelled'].append(info['name'])
        flagged = sum(1 for r in job['results'] if r['flags'])
        job['summary'] = {'restored': len(job['results']), 'flagged': flagged,
                          'skipped': len(job['skipped']),
                          'cancelled': len(job['cancelled'])}
        if models.device_warning():
            job['warnings'].insert(0, models.device_warning())
    except Exception as err:
        log.exception('job failed')
        job['status'] = 'error'
        job['warnings'].append(f'job failed: {err}')
    finally:
        job['finished'] = time.time()
        job['stage'] = 'finished'
        models.unload_all()
        _save_job(job, job_dir)


# ---------------------------------------------------------------- endpoints

class JobRequest(BaseModel):
    folder: str
    preset: str = 'balanced'
    colorize: bool = True
    upscale_bg: bool = False


@app.post('/jobs')
def create_job(req: JobRequest):
    global _active_job_id
    folder = Path(req.folder)
    if not folder.is_dir():
        raise HTTPException(400, f'not a folder: {folder}')
    if req.preset not in pipeline.PRESETS:
        raise HTTPException(400, f'preset must be one of {pipeline.PRESET_ORDER}')
    with pipeline.PIPELINE_LOCK:
        running = [j for j in JOBS.values() if j['status'] in ('queued', 'running')]
        if running:
            raise HTTPException(409, f'job {running[0]["id"]} is already running')
        job = {
            'id': time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6],
            'folder': str(folder),
            'folder_name': folder.name,
            'preset': req.preset,
            'colorize': req.colorize,
            'upscale_bg': req.upscale_bg,
            'status': 'queued',
            'stage': 'starting', 'stage_done': 0, 'stage_total': 0,
            'done': 0, 'total': 0,
            'started': time.time(), 'finished': None,
            'device': models.device_name(),
            'device_warning': models.device_warning(),
            'warnings': [], 'skipped': [], 'cancelled': [], 'results': [],
            'cancel': {'flag': False},
            'version': __version__,
        }
        JOBS[job['id']] = job
        _active_job_id = job['id']
        job_dir = _job_dir(job['id'])
        _save_job(job, job_dir)
        threading.Thread(target=_run_job_thread, args=(job, job_dir),
                         daemon=True).start()
        return {'job_id': job['id'], 'status': job['status']}


@app.get('/status')
def status():
    try:
        job = _latest_job()
    except HTTPException:
        job = None
    payload = {
        'version': __version__,
        'device': models.device_name(),
        'device_warning': models.device_warning(),
        'downloading': models.download_status(),
        'job': job,
    }
    return JSONResponse(payload)


class RerunRequest(BaseModel):
    job_id: str
    name: str


@app.post('/rerun')
def rerun(req: RerunRequest):
    job = _get_job(req.job_id)
    if job['status'] in ('queued', 'running'):
        raise HTTPException(409, 'a job is running; re-run after it finishes')
    src = Path(job['folder']) / req.name
    if not src.is_file():
        raise HTTPException(404, f'{req.name} not found in {job["folder"]}')
    entry = next((r for r in job['results'] if r['name'] == req.name), None)
    sidecar = None
    if entry:
        sc = _job_dir(job['id']) / 'restored' / (Path(entry['output']).stem + '.json')
        if sc.exists():
            sidecar = json.loads(sc.read_text(encoding='utf-8'))
    current_preset = (sidecar or {}).get('preset', job['preset'])
    new_preset = pipeline.next_preset(current_preset)
    if new_preset == current_preset:
        raise HTTPException(400, 'already at the highest-fidelity preset')
    with pipeline.PIPELINE_LOCK:
        old_output = entry['output'] if entry else None
        written = {}

        def sink(name, stem, ext, src_hash, meta, data):
            job_dir = _job_dir(job['id'])
            out_path = pipeline.output_name(job_dir, stem, ext, src_hash)
            out_path.write_bytes(data)
            (job_dir / 'restored' / (out_path.stem + '.json')).write_text(
                json.dumps(meta, indent=1), encoding='utf-8')
            written.update({'output': out_path.name, 'flags': meta.get('flags', [])})

        infos = pipeline.run_pipeline([src], new_preset, job['colorize'],
                                      job['upscale_bg'], sink=sink)
        info = infos[0]
        if info['status'] != 'done':
            raise HTTPException(500,
                                f're-run failed: {info.get("reason", "processing failed")}')
        job_dir = _job_dir(job['id'])
        if old_output and old_output != written['output']:
            (job_dir / 'restored' / old_output).unlink(missing_ok=True)
            (job_dir / 'restored' / (Path(old_output).stem + '.json')).unlink(missing_ok=True)
        new_entry = {'name': req.name, 'output': written['output'],
                     'flags': written['flags']}
        if entry:
            job['results'].remove(entry)
        job['results'].append(new_entry)
        job['rerun'] = {'name': req.name, 'from': current_preset, 'to': new_preset,
                        'at': time.time()}
        _save_job(job, job_dir)
        return {'name': req.name, 'preset': new_preset, 'output': written['output'],
                'flags': written['flags']}


class CancelRequest(BaseModel):
    job_id: str


@app.post('/cancel')
def cancel(req: CancelRequest):
    job = _get_job(req.job_id)
    job['cancel']['flag'] = True
    return {'job_id': job['id'], 'status': job['status']}


@app.get('/pairs')
def pairs(job_id: str = None):
    from urllib.parse import quote
    job = _get_job(job_id) if job_id else _latest_job()
    job_dir = _job_dir(job['id'])
    restored_dir = job_dir / 'restored'
    folder = Path(job['folder'])
    out = []
    for entry in job.get('results', []):
        sidecar_path = restored_dir / (Path(entry['output']).stem + '.json')
        meta = {}
        if sidecar_path.exists():
            try:
                meta = json.loads(sidecar_path.read_text(encoding='utf-8'))
            except ValueError:
                meta = {'errors': ['sidecar unreadable']}
        out.append({
            'name': entry['name'],
            # quote() not quote_plus: spaces stay readable, & and # are escaped
            'before': f'/img/before?job_id={job["id"]}&name={quote(Path(entry["name"]).name)}',
            'after': f'/img/after?job_id={job["id"]}&name={quote(Path(entry["output"]).name)}',
            'flags': entry['flags'],
            'meta': meta,
        })
    return {'job_id': job['id'], 'folder': str(folder), 'pairs': out, 'summary': job.get('summary')}


@app.get('/img/before')
def img_before(job_id: str, name: str):
    job = _get_job(job_id)
    safe = Path(name).name  # no traversal
    allowed = {r['name'] for r in job.get('results', [])} | \
              {s['name'] for s in job.get('skipped', [])}
    path = Path(job['folder']) / safe
    if safe not in allowed or not path.is_file():
        raise HTTPException(404, 'no such source photo in this job')
    return FileResponse(path)


@app.get('/img/after')
def img_after(job_id: str, name: str):
    job = _get_job(job_id)
    safe = Path(name).name
    path = _job_dir(job['id']) / 'restored' / safe
    allowed = {r['output'] for r in job.get('results', [])}
    if safe not in allowed or not path.is_file():
        raise HTTPException(404, 'no such restored photo in this job')
    return FileResponse(path)


class ExportRequest(BaseModel):
    job_id: str


@app.post('/export')
def export(req: ExportRequest):
    job = _get_job(req.job_id)
    job_dir = _job_dir(job['id'])
    restored_dir = job_dir / 'restored'
    export_dir = job_dir / 'export'
    if export_dir.exists():
        # a previous export may hold files a re-run has replaced
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in job.get('results', []):
        src = restored_dir / entry['output']
        if src.is_file():
            shutil.copy2(src, export_dir / entry['output'])
            copied += 1
    return {'exported': copied, 'to': str(export_dir)}


@app.get('/jobs')
def jobs_history():
    """Every job on disk, newest first, for the UI's history dropdown."""
    items = []
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            meta = d / 'job.json'
            if not meta.is_file():
                continue
            try:
                j = json.loads(meta.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            items.append({
                'id': j.get('id', d.name),
                'folder': j.get('folder'),
                'folder_name': j.get('folder_name'),
                'status': j.get('status'),
                'started': j.get('started'),
                'summary': j.get('summary'),
            })
    items.sort(key=lambda x: x.get('started') or 0, reverse=True)
    return {'jobs': items}


class RevealRequest(BaseModel):
    job_id: str
    target: str = 'export'  # 'export' or 'restored'


@app.post('/reveal')
def reveal(req: RevealRequest):
    """Open a job's folder in Windows Explorer (path pinned under results/)."""
    import subprocess
    if req.target not in ('export', 'restored'):
        raise HTTPException(400, "target must be 'export' or 'restored'")
    job_dir = _job_dir(req.job_id).resolve()
    target = (job_dir / req.target).resolve()
    if not str(target).startswith(str(job_dir)):
        raise HTTPException(400, 'bad target')
    if not target.is_dir():
        raise HTTPException(404, f'{req.target} folder does not exist yet')
    subprocess.Popen(['explorer', str(target)])
    return {'opened': str(target)}


class ImmichRequest(BaseModel):
    job_id: str
    base_url: str
    api_key: str = Field(min_length=1, description='Immich API key')
    album_name: str = None


@app.post('/immich')
def immich_push(req: ImmichRequest):
    job = _get_job(req.job_id)
    job_dir = _job_dir(job['id'])
    files = [job_dir / 'restored' / r['output'] for r in job.get('results', [])]
    files = [f for f in files if f.is_file()]
    if not files:
        raise HTTPException(400, 'nothing restored yet for this job')
    album_name = req.album_name or f'Restored {job["folder_name"]}'
    client = immich.ImmichClient(req.base_url, req.api_key)
    try:
        uploaded, already, queued, added = immich.push_to_immich(
            client, files, album_name, job_dir / 'immich_queue.json')
    except immich.ImmichError as err:
        raise HTTPException(502, str(err))
    return {'album': album_name, 'uploaded': uploaded, 'already_in_album': already,
            'retry_queued': queued, 'added_to_album': added}


@app.get('/pick_folder')
def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askdirectory(title='Choose the folder of scans to restore')
        root.destroy()
        return {'folder': path or None}
    except Exception as err:  # no display / tkinter missing
        raise HTTPException(503, f'folder dialog unavailable ({err}); type the path instead')


@app.get('/')
def index():
    return FileResponse(WEB_DIR / 'index.html')


def main():
    import os
    import threading
    import uvicorn
    port = int(os.environ.get('SHOEBOX_PORT', '8545'))
    # SelectorEventLoop avoids the Windows proactor ConnectionResetError spam
    # when browsers drop image range requests
    threading.Timer(1.2, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    uvicorn.run(app, host='127.0.0.1', port=port, loop='asyncio:SelectorEventLoop', log_level='info')


if __name__ == '__main__':
    main()
