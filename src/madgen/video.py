"""Phase 4 (video): cut the source video at the chosen segments and concat on the target timeline.

The picture is decided frame by frame over several voices in priority order: a frame shows
the highest-priority voice that covers it. A voice covers its notes, and also the gap after a
note when that gap is shorter than `hold_sec` (so short rests do not flicker to another voice).
A frame goes black only when no voice covers it, i.e. nothing sounds for `hold_sec` or longer.

Everything is quantized to whole frames on an absolute timeline, so rounding never accumulates
and the video stays locked to the audio.
"""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ffmpeg
from .progress import progress


@dataclass
class VideoNote:
    start_sec: float
    duration_sec: float
    video_ref: str | None
    seg_start: float


@dataclass
class Span:
    first_frame: int
    frames: int
    video_ref: str | None   # None = black
    source_start: float = 0.0


def build_spans(voices: list[list[VideoNote]], total_sec: float, fps: int,
                hold_sec: float = 0.5) -> list[Span]:
    """voices: note lists (each sorted by start), highest priority first."""
    total_frames = round(total_sec * fps)
    # owner[f] = (voice, note) shown at frame f; -1 = black
    owner_voice = np.full(total_frames, -1, dtype=np.int64)
    owner_note = np.full(total_frames, -1, dtype=np.int64)
    # Paint lowest priority first so higher priorities overwrite.
    for vi in range(len(voices) - 1, -1, -1):
        notes = voices[vi]
        for ni, note in enumerate(notes):
            if note.video_ref is None:
                continue
            end = note.start_sec + note.duration_sec
            next_start = notes[ni + 1].start_sec if ni + 1 < len(notes) else total_sec
            if next_start - end < hold_sec:
                end = next_start
            f0, f1 = round(note.start_sec * fps), min(round(end * fps), total_frames)
            owner_voice[f0:f1] = vi
            owner_note[f0:f1] = ni

    spans: list[Span] = []
    f = 0
    while f < total_frames:
        g = f + 1
        while g < total_frames and owner_voice[g] == owner_voice[f] and owner_note[g] == owner_note[f]:
            g += 1
        if owner_voice[f] < 0:
            spans.append(Span(f, g - f, None))
        else:
            note = voices[owner_voice[f]][owner_note[f]]
            # A clip that takes over mid-note starts from the matching point inside the segment.
            offset = max(0.0, f / fps - note.start_sec)
            spans.append(Span(f, g - f, note.video_ref, note.seg_start + offset))
        f = g
    return spans


def render_video(spans: list[Span], audio: Path, out: Path, clip_cache: Path,
                 width: int = 1280, height: int = 720, fps: int = 30, jobs: int = 6) -> None:
    """Render and concat the spans, then mux `audio`. Clips are cached in `clip_cache` by
    content, so several videos from one render (the mix and each part) share their clips."""
    clip_cache.mkdir(parents=True, exist_ok=True)

    def clip_path(span: Span) -> Path:
        key = f"{span.video_ref}|{span.source_start:.3f}|{span.frames}|{width}x{height}@{fps}"
        return clip_cache / (hashlib.sha1(key.encode()).hexdigest() + ".mp4")

    clips = [clip_path(s) for s in spans]
    todo = sorted({c: s for c, s in zip(clips, spans, strict=True) if not c.exists()}.items())

    def make(item: tuple[Path, Span]) -> None:
        path, span = item
        tmp = path.with_suffix(".part.mp4")
        if span.video_ref is None:
            ffmpeg.make_black_clip(span.frames, tmp, width, height, fps)
        else:
            ffmpeg.extract_clip(Path(span.video_ref), span.source_start, span.frames, tmp, width, height, fps)
        tmp.rename(path)

    progress.stage(f"video {out.name}: clips", len(todo))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, _ in enumerate(pool.map(make, todo), 1):
            progress.update(n)
            print(f"\r  video clips {n}/{len(todo)}", end="", file=sys.stderr, flush=True)
    if todo:
        print(file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress.stage(f"video {out.name}: concat + mux")
    ffmpeg.concat_with_audio(clips, audio, out, clip_cache / f"{out.stem}.list.txt")
