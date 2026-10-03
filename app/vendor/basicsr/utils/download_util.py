"""shoebox shim for basicsr.utils.download_util.

Same contract as upstream load_file_from_url: cached download of `url` into
`model_dir`, resolved against the shoebox project root. facelib calls it with
model_dir='weights/facelib', so the join lands in <project>/weights/facelib —
the same folder models.py manages.
"""

import os
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from ..paths import PROJECT_ROOT, weights_root


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:
        target_dir = weights_root() / 'checkpoints'
    else:
        target_dir = PROJECT_ROOT / model_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = target_dir / filename
    if not cached_file.exists():
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, str(cached_file), hash_prefix=None, progress=progress)
    return str(cached_file)


__all__ = ['load_file_from_url']
