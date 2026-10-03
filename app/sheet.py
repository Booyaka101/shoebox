"""Contact-sheet PDF: one page per 12 restored photos, labelled thumbnails.

Pillow draws the grid and writes the multi-page PDF; no other dependency.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

COLS, ROWS = 3, 4
CELL_W, CELL_H = 340, 300
CAPTION = 34
MARGIN = 18
PAGE_W = MARGIN * 2 + COLS * CELL_W
PAGE_H = MARGIN * 2 + ROWS * (CELL_H + CAPTION)


def build(files, out_path: Path, per_page: int = COLS * ROWS) -> int:
    """Write a paginated PDF of labelled thumbnails; returns the page count."""
    pages = []
    for start in range(0, len(files), per_page):
        page = Image.new('RGB', (PAGE_W, PAGE_H), (18, 20, 24))
        draw = ImageDraw.Draw(page)
        chunk = files[start:start + per_page]
        for idx, path in enumerate(chunk):
            col, row = idx % COLS, idx // COLS
            x = MARGIN + col * CELL_W
            y = MARGIN + row * (CELL_H + CAPTION)
            try:
                with Image.open(path) as im:
                    thumb = im.convert('RGB')
                    thumb.thumbnail((CELL_W - 8, CELL_H - 8))
                    px = x + (CELL_W - thumb.width) // 2
                    py = y + (CELL_H - thumb.height) // 2
                    page.paste(thumb, (px, py))
            except Exception as err:
                log.warning('contact sheet skipped %s: %s', path, err)
            caption = Path(path).name
            if len(caption) > 42:
                caption = caption[:39] + '...'
            draw.text((x + 4, y + CELL_H + 8), caption, fill=(224, 168, 60))
        pages.append(page)
    if not pages:
        raise ValueError('no pages to write')
    pages[0].save(str(out_path), save_all=True,
                  append_images=pages[1:], resolution=120.0)
    return len(pages)
