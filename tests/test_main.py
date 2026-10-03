"""Endpoint tests with FastAPI's TestClient against an isolated results dir.

No models are loaded here: the heavy pipeline is monkeypatched where a job
would run, and the rest exercises routing, validation, recovery and export
on real files.
"""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    results.mkdir()
    monkeypatch.setattr(app_main, 'RESULTS_DIR', results)
    monkeypatch.setattr(app_main, 'JOBS', {})
    monkeypatch.setattr(app_main, '_active_job_id', None)
    # never let a test touch the GPU pipeline
    monkeypatch.setattr(app_main.pipeline, 'run_pipeline', _fake_run_pipeline)
    with TestClient(app_main.app) as c:
        yield c


def _fake_run_pipeline(paths, preset, colorize, upscale_bg, sink,
                       work_state=None, cancel=None):
    import numpy as np
    out = []
    for i, p in enumerate(paths):
        if work_state is not None:
            work_state['stage'], work_state['stage_done'], work_state['stage_total'] = \
                'restoring faces', i, len(paths)
        img = np.zeros((8, 8, 3), dtype='uint8')
        img[2:6, 2:6] = (0, 128, 255)
        meta = {
            'source_hash': 'f' * 64, 'models': {'colorize': None,
            'face_restore': 'x', 'bg_upscale': None}, 'preset': preset,
            'w': 0.5, 'w_final': 0.5, 'face_count_in': 0, 'face_count_out': 0,
            'sharpness_in': 1.0, 'sharpness_out': 1.0, 'errors': [], 'flags': [],
        }
        import cv2
        ok, buf = cv2.imencode('.png', img)
        sink(p.name, p.stem, '.png', 'f' * 64, meta, buf.tobytes())
        out.append({'name': p.name, 'status': 'done', 'src_hash': 'f' * 64,
                    'meta': meta})
    return out


def _fake_folder(tmp_path):
    folder = tmp_path / 'scans'
    folder.mkdir(exist_ok=True)
    (folder / 'a.jpg').write_bytes(b'x' * 10)
    return folder


def test_create_job_rejects_missing_folder(client):
    r = client.post('/jobs', json={'folder': 'Z:\\definitely\\not\\here'})
    assert r.status_code == 400
    assert 'not a folder' in r.json()['detail']


def test_create_job_rejects_bad_preset(client, tmp_path):
    folder = _fake_folder(tmp_path)
    r = client.post('/jobs', json={'folder': str(folder), 'preset': 'turbo'})
    assert r.status_code == 400


def test_job_runs_and_pairs_are_url_encoded(client, tmp_path):
    folder = tmp_path / 'scans'
    folder.mkdir()
    # a name that would break a naive query string
    (folder / 'Oma & Opa 1961.jpg').write_bytes(b'x' * 10)
    r = client.post('/jobs', json={'folder': str(folder), 'preset': 'low'})
    assert r.status_code == 200
    job_id = r.json()['job_id']
    deadline = time.time() + 15
    while time.time() < deadline:
        status = client.get('/status').json()['job']['status']
        if status in ('done', 'error', 'cancelled'):
            break
        time.sleep(0.1)
    assert status == 'done'
    pairs = client.get('/pairs', params={'job_id': job_id}).json()['pairs']
    assert len(pairs) == 1
    assert '%26' in pairs[0]['before'], 'the & in the filename must be percent-encoded'
    img = client.get(pairs[0]['before'])
    assert img.status_code == 200
    img = client.get(pairs[0]['after'])
    assert img.status_code == 200


def test_interrupted_job_marked_on_startup(client, tmp_path):
    results = Path(app_main.RESULTS_DIR)
    job_dir = results / '20260101-000000-deadbe'
    job_dir.mkdir()
    (job_dir / 'job.json').write_text(json.dumps({
        'id': '20260101-000000-deadbe', 'folder': str(tmp_path),
        'folder_name': 'scans', 'status': 'running', 'started': 1.0,
        'results': [], 'skipped': [], 'warnings': [],
    }))
    app_main._load_last_job()
    job = client.get('/status').json()['job']
    assert job['status'] == 'interrupted'
    assert any('closed' in w for w in job['warnings'])
    # and starting a new job must NOT be rejected with 409
    folder = _fake_folder(tmp_path)
    r = client.post('/jobs', json={'folder': str(folder), 'preset': 'low'})
    assert r.status_code == 200


def test_jobs_history_lists_on_disk_jobs(client, tmp_path):
    results = Path(app_main.RESULTS_DIR)
    for jid, started in (('b-second', 2.0), ('a-first', 1.0)):
        d = results / jid
        d.mkdir()
        (d / 'job.json').write_text(json.dumps({
            'id': jid, 'folder': 'X', 'folder_name': 'X', 'status': 'done',
            'started': started, 'summary': {'restored': 1},
        }))
    jobs = client.get('/jobs').json()['jobs']
    assert [j['id'] for j in jobs] == ['b-second', 'a-first']


def test_export_wipes_stale_files(client, tmp_path):
    folder = _fake_folder(tmp_path)
    job_id = client.post('/jobs', json={'folder': str(folder), 'preset': 'low'}).json()['job_id']
    deadline = time.time() + 15
    while time.time() < deadline:
        if client.get('/status').json()['job']['status'] == 'done':
            break
        time.sleep(0.1)
    export_dir = Path(app_main.RESULTS_DIR) / job_id / 'export'
    export_dir.mkdir(parents=True)
    stale = export_dir / 'gone_restored.jpg'
    stale.write_bytes(b'stale')
    r = client.post('/export', json={'job_id': job_id})
    assert r.status_code == 200
    assert r.json()['exported'] == 1
    assert not stale.exists(), 'stale export files must not survive a re-export'
    exported = sorted(p.name for p in export_dir.iterdir())
    assert 'a_restored.png' in exported
    assert 'a_restored.json' in exported, 'sidecar metadata should travel with the export'


def test_reveal_rejects_bad_target_and_missing_folder(client, tmp_path):
    folder = _fake_folder(tmp_path)
    job_id = client.post('/jobs', json={'folder': str(folder), 'preset': 'low'}).json()['job_id']
    deadline = time.time() + 15
    while time.time() < deadline:
        if client.get('/status').json()['job']['status'] == 'done':
            break
        time.sleep(0.1)
    r = client.post('/reveal', json={'job_id': job_id, 'target': '..\\..\\'})
    assert r.status_code == 400
    r = client.post('/reveal', json={'job_id': job_id, 'target': 'nope'})
    assert r.status_code == 400
    # export/ does not exist until an export runs
    r = client.post('/reveal', json={'job_id': job_id, 'target': 'export'})
    assert r.status_code == 404


def test_cancel_of_unknown_job_is_404(client):
    r = client.post('/cancel', json={'job_id': 'nope'})
    assert r.status_code == 404
