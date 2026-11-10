"""Local HLS playlist entries used as a camera fallback."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEntry:
    path: Path
    title: str
    credit: str


def load_playlist(path):
    """Read segment paths and the #CREDIT belonging to each EXTINF entry."""
    playlist = Path(path)
    if not playlist.is_file():
        return []

    entries = []
    title = None
    credit = ''
    for raw_line in playlist.read_text(encoding='utf-8-sig').splitlines():
        line = raw_line.strip()
        if line.startswith('#EXTINF:'):
            title = line.partition(',')[2].strip()
            credit = ''
        elif line.startswith('#CREDIT:'):
            credit = line.removeprefix('#CREDIT:').strip()
        elif line and not line.startswith('#'):
            if title is not None:
                entries.append(VideoEntry(playlist.parent / line, title or line, credit))
            title = None
            credit = ''
    return entries
