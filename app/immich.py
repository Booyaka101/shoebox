"""Immich push client for shoebox.

Talks to the user's own Immich server with their base URL and API key. Flow:

  1. ping the server (fail fast with a readable message)
  2. find or create the album 'Restored <folder name>'
  3. upload each restored file via POST /api/assets (multipart)
  4. add the uploaded asset ids to the album via PUT /api/albums/{id}/assets

Anything that fails mid-run is recorded in a retry queue file (JSON next to
the job) so nothing is lost; the next push with the same details drains the
queue first. Album membership is checked through POST /api/search/metadata,
not GET /api/albums/<id> — recent Immich servers return `assets: null` there
with HTTP 200.
"""

import json
import logging
import mimetypes
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

TIMEOUT = 60
UPLOAD_TIMEOUT = 300  # multi-hundred-MB PNGs over wifi to a NAS
UPLOAD_RETRIES = 3
RETRY_BACKOFF = 2.0


class ImmichError(Exception):
    pass


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, session=None):
        base = base_url.strip().rstrip('/')
        if not base.startswith(('http://', 'https://')):
            base = 'http://' + base
        self.base_url = base
        self.session = session or requests.Session()
        self.session.headers.update({'x-api-key': api_key, 'Accept': 'application/json'})

    def _url(self, path):
        return self.base_url + path

    def ping(self):
        try:
            resp = self.session.get(self._url('/api/server/ping'), timeout=TIMEOUT)
        except requests.RequestException as err:
            raise ImmichError(f'cannot reach Immich at {self.base_url}: {err}') from err
        if resp.status_code != 200:
            raise ImmichError(
                f'Immich ping returned HTTP {resp.status_code} — check the base URL '
                f'(it should be your server root, e.g. http://nas:2283)')
        return True

    def find_or_create_album(self, album_name: str) -> str:
        resp = self.session.get(self._url('/api/albums'), timeout=TIMEOUT)
        if resp.status_code != 200:
            raise ImmichError(f'listing albums failed: HTTP {resp.status_code} {resp.text[:200]}')
        for album in resp.json():
            if album.get('albumName') == album_name:
                return album['id']
        resp = self.session.post(self._url('/api/albums'), json={'albumName': album_name},
                                 timeout=TIMEOUT)
        if resp.status_code not in (200, 201):
            raise ImmichError(f'creating album failed: HTTP {resp.status_code} {resp.text[:200]}')
        return resp.json()['id']

    def album_asset_filenames(self, album_id: str) -> set:
        """originalFilename set for everything already in the album, via
        POST /api/search/metadata (GET /api/albums/<id> returns assets: null
        on current servers)."""
        body = {'albumIds': [album_id], 'size': 1000, 'page': 1}
        names = set()
        while True:
            resp = self.session.post(self._url('/api/search/metadata'), json=body,
                                     timeout=TIMEOUT)
            if resp.status_code == 400:
                # newer schema shape
                body2 = {'filter': {'albumIds': {'any': [album_id]}},
                         'size': 1000, 'page': body['page']}
                resp = self.session.post(self._url('/api/search/metadata'), json=body2,
                                         timeout=TIMEOUT)
            if resp.status_code != 200:
                raise ImmichError(
                    f'album membership search failed: HTTP {resp.status_code} {resp.text[:200]}')
            data = resp.json()
            items = data.get('assets', {}).get('items', data.get('assets', []))
            for asset in items:
                name = asset.get('originalFilename')
                if name:
                    names.add(name)
            total = data.get('assets', {}).get('total', len(items))
            if body['page'] * 1000 >= total or not items:
                return names
            body['page'] += 1

    def upload(self, file_path: Path, device_id: str = 'shoebox') -> str:
        stat = file_path.stat()
        mime = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
        if not mime.startswith(('image/',)):
            mime = 'image/' + mime
        data = {
            'deviceAssetId': f'{file_path.name}-{int(stat.st_mtime)}-{stat.st_size}',
            'deviceId': device_id,
            'fileCreatedAt': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(stat.st_mtime)),
            'fileModifiedAt': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(stat.st_mtime)),
            'fileSize': str(stat.st_size),
            'mimeType': mime,
        }
        last_err = None
        for attempt in range(1, UPLOAD_RETRIES + 1):
            # reopen per attempt: a file object left at EOF would upload
            # empty bodies on the retries
            with open(file_path, 'rb') as fh:
                files = [('assetData', (file_path.name, fh, mime))]
                try:
                    resp = self.session.post(self._url('/api/assets'), data=data,
                                             files=files, timeout=UPLOAD_TIMEOUT)
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        return body.get('id') or body.get('asset', {}).get('id')
                    last_err = ImmichError(
                        f'upload of {file_path.name} failed: HTTP {resp.status_code} {resp.text[:200]}')
                except requests.RequestException as err:
                    last_err = ImmichError(f'upload of {file_path.name} failed: {err}')
            if attempt < UPLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
        raise last_err

    def add_to_album(self, album_id: str, asset_ids: list) -> int:
        if not asset_ids:
            return 0
        resp = self.session.put(self._url(f'/api/albums/{album_id}/assets'),
                                json={'ids': asset_ids}, timeout=TIMEOUT)
        if resp.status_code != 200:
            raise ImmichError(
                f'adding to album failed: HTTP {resp.status_code} {resp.text[:200]}')
        added = resp.json().get('successfullyAdded', 0)
        return int(added)


def push_to_immich(client: ImmichClient, files: list, album_name: str,
                   queue_path: Path, progress=None):
    """Upload files into album_name, draining the retry queue first.

    Returns (uploaded, skipped_already_there, queued_now). Files that still
    fail are appended to queue_path so a later run picks them up.
    """
    pending = []
    if queue_path.exists():
        try:
            pending = json.loads(queue_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pending = []

    done_paths = {str(p) for p in files}
    pending = [entry for entry in pending if entry not in done_paths]
    todo = pending + [str(p) for p in files]

    client.ping()
    album_id = client.find_or_create_album(album_name)
    already = client.album_asset_filenames(album_id)

    uploaded, already_there, failed = 0, 0, []
    asset_ids = []
    for i, entry in enumerate(todo):
        if progress:
            progress(i, len(todo))
        path = Path(entry)
        if not path.exists():
            failed.append(entry)
            continue
        if path.name in already:
            already_there += 1
            continue
        try:
            asset_id = client.upload(path)
            asset_ids.append(asset_id)
            uploaded += 1
        except ImmichError as err:
            log.warning('%s', err)
            failed.append(entry)

    added = client.add_to_album(album_id, asset_ids)
    if failed:
        queue_path.write_text(json.dumps(sorted(set(failed)), indent=1), encoding='utf-8')
    elif queue_path.exists():
        queue_path.unlink()
    return uploaded, already_there, len(failed), added
