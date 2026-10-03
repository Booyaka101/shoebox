"""Immich client tests against a mock Immich server (no network, no key)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.immich import ImmichClient, ImmichError, push_to_immich


class MockImmich(BaseHTTPRequestHandler):
    """Just enough of the Immich REST API for the push flow.

    HTTP/1.1 with correct Content-Length lets the client's session reuse one
    connection for the whole test, which keeps Windows loopback churn (and
    its occasional 10053 resets) out of the picture."""

    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/api/server/ping':
            self._json(200, {'res': 'pong'})
        elif self.path == '/api/albums':
            self._json(200, MockImmich.albums)
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        if self.path == '/api/albums':
            name = json.loads(body)['albumName']
            existing = next((a for a in MockImmich.albums if a['albumName'] == name), None)
            if existing:
                self._json(201, existing)
            else:
                album = {'id': f'album-{len(MockImmich.albums) + 1}', 'albumName': name}
                MockImmich.albums.append(album)
                self._json(201, album)
        elif self.path == '/api/search/metadata':
            names = sorted({a['filename'] for a in MockImmich.assets})
            items = [{'originalFilename': n} for n in names]
            self._json(200, {'assets': {'items': items, 'total': len(items)}})
        elif self.path == '/api/assets':
            # find the filename in the multipart body; names starting with
            # 'bad' fail exactly once (transient outage), exercising the
            # client's retry-queue path and its drain on a later push
            filename = None
            for part in body.split(b'filename="')[1:]:
                filename = part.split(b'"')[0].decode()
                break
            if filename and filename.startswith('bad') \
                    and MockImmich.upload_attempts.get(filename, 0) < 3:
                # survive the client's own three in-push retries, so the file
                # only succeeds on a later push (the retry-queue drain path)
                MockImmich.upload_attempts[filename] = MockImmich.upload_attempts.get(filename, 0) + 1
                self._json(500, {'error': 'boom'})
                return
            received.append(filename)
            MockImmich.assets.append({'filename': filename})
            self._json(201, {'id': f'asset-{len(MockImmich.assets)}'})
        else:
            self._json(404, {'error': 'not found'})

    def do_PUT(self):
        if self.path.startswith('/api/albums/') and self.path.endswith('/assets'):
            self._json(200, {'successfullyAdded': len(MockImmich.pending_ids)})
        else:
            self._json(404, {'error': 'not found'})


class QuietServer(ThreadingHTTPServer):
    """Threaded so uploads never serialize behind a half-closed connection;
    daemon_threads keep pytest from hanging on teardown."""

    daemon_threads = True


received = []


@pytest.fixture(scope='module')
def server():
    MockImmich.albums = []
    MockImmich.assets = []
    MockImmich.pending_ids = []
    MockImmich.upload_attempts = {}
    received.clear()
    httpd = QuietServer(('127.0.0.1', 0), MockImmich)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def reset_mock(server):
    MockImmich.albums = []
    MockImmich.assets = []
    MockImmich.pending_ids = []
    MockImmich.upload_attempts = {}
    received.clear()


def _photo(tmp_path, name, content=b'fake jpeg bytes'):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_ping_failure_is_readable():
    client = ImmichClient('http://127.0.0.1:9', 'key')
    with pytest.raises(ImmichError) as exc:
        client.ping()
    assert 'cannot reach Immich' in str(exc.value)


def test_push_uploads_and_creates_album(server, reset_mock, tmp_path):
    photo = _photo(tmp_path, 'a_restored.jpg', b'AAA')
    queue = tmp_path / 'queue.json'
    uploaded, already, failed, added = push_to_immich(
        ImmichClient(server, 'key'), [photo], 'Restored scans', queue)
    assert uploaded == 1 and already == 0 and failed == 0
    assert received == ['a_restored.jpg']
    assert any(a['albumName'] == 'Restored scans' for a in MockImmich.albums)
    assert not queue.exists()


def test_push_skips_photos_already_in_album(server, reset_mock, tmp_path):
    photo = _photo(tmp_path, 'a_restored.jpg', b'AAA')
    queue = tmp_path / 'queue.json'
    push_to_immich(ImmichClient(server, 'key'), [photo], 'Restored scans', queue)
    uploaded, already, failed, added = push_to_immich(
        ImmichClient(server, 'key'), [photo], 'Restored scans', queue)
    assert uploaded == 0 and already == 1 and failed == 0


def test_failed_upload_lands_in_retry_queue(server, reset_mock, tmp_path):
    good = _photo(tmp_path, 'good_restored.jpg', b'AAA')
    bad = _photo(tmp_path, 'bad_restored.jpg', b'BBB')
    queue = tmp_path / 'queue.json'
    uploaded, already, failed, added = push_to_immich(
        ImmichClient(server, 'key'), [good, bad], 'Restored scans', queue)
    assert uploaded == 1 and failed == 1
    assert queue.exists()
    assert json.loads(queue.read_text()) == [str(bad)]


def test_retry_queue_is_drained_on_next_push(server, reset_mock, tmp_path):
    good = _photo(tmp_path, 'good_restored.jpg', b'AAA')
    bad = _photo(tmp_path, 'bad_restored.jpg', b'BBB')
    queue = tmp_path / 'queue.json'
    push_to_immich(ImmichClient(server, 'key'), [good, bad], 'Restored scans', queue)
    MockImmich.pending_ids = ['asset-x']
    uploaded, already, failed, added = push_to_immich(
        ImmichClient(server, 'key'), [good], 'Restored scans', queue)
    assert uploaded == 1, 'queued file should be retried and succeed'
    assert 'bad_restored.jpg' in received
    assert failed == 0 and not queue.exists()
