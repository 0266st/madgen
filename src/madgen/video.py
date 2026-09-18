"""Phase 4 (video): cut the source video at the chosen segments and lay the parts out on screen.

Within one layer the picture is decided frame by frame over several voices in priority order: a
frame shows the highest-priority voice that covers it. A voice covers its notes, and also the gap
after a note when that gap is shorter than `hold_sec` (so short rests do not flicker to another
voice). A frame is empty when no voice covers it, i.e. nothing sounds for `hold_sec` or longer.

Layered layout (the default) stacks those layers, the way 音MAD usually look:
- background: full screen, the parts that keep sounding (drums, bass)
- panels: the other accompaniment parts, small, above and below the middle
- lead: the sung part (or the main melody), centred and larger
Each layer becomes its own video plus a black/white mask saying when it is showing, and the
layers are composed in one ffmpeg pass. The fullscreen layout keeps everything on one plane.

Everything is quantized to whole frames on an absolute timeline, so rounding never accumulates
and the video stays locked to the audio.
"""

from __future__ import annotations

import hashlib
import random
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
    video_ref: str | None   # None = nothing showing (black, or transparent in a layer)
    source_start: float = 0.0


@dataclass
class Layer:
    """One video plane: its spans, and where it sits on the canvas."""

    name: str
    spans: list[Span]
    x: int
    y: int
    width: int
    height: int
    transparent: bool = True   # False for the background layer, which always covers the canvas


def build_spans(voices: list[list[VideoNote]], total_sec: float, fps: int,
                hold_sec: float = 0.5) -> list[Span]:
    """voices: note lists (each sorted by start), highest priority first."""
    total_frames = round(total_sec * fps)
    # owner[f] = (voice, note) shown at frame f; -1 = nothing
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
    return _merge_repeats(spans)


def _merge_repeats(spans: list[Span]) -> list[Span]:
    """Notes that reuse the same segment play on instead of restarting it: without this a part that
    picks one segment over and over (drums do) shows the same few frames again and again, which
    looks like a frozen picture."""
    merged: list[Span] = []
    for span in spans:
        last = merged[-1] if merged else None
        if (last is not None and last.video_ref == span.video_ref
                and (span.video_ref is None or abs(last.source_start - span.source_start) < 0.02)):
            last.frames += span.frames
        else:
            merged.append(Span(span.first_frame, span.frames, span.video_ref, span.source_start))
    return merged


def _merge_empty(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in spans:
        if merged and span.video_ref is None and merged[-1].video_ref is None:
            merged[-1].frames += span.frames
        else:
            merged.append(Span(span.first_frame, span.frames, span.video_ref, span.source_start))
    return merged


def split_spans_over_slots(spans: list[Span], slots: int, seed: int = 0) -> list[list[Span]]:
    """Random layout: every showing span goes to one randomly picked slot, the rest stay empty."""
    rng = random.Random(seed)
    out: list[list[Span]] = [[] for _ in range(slots)]
    for span in spans:
        pick = rng.randrange(slots) if span.video_ref is not None else -1
        for i, slot in enumerate(out):
            slot.append(span if i == pick else Span(span.first_frame, span.frames, None))
    return [_merge_empty(slot) for slot in out]


def lead_box(width: int, height: int, scale: float = 0.55) -> tuple[int, int, int, int]:
    w = int(width * scale) // 2 * 2
    h = (w * height // width) // 2 * 2
    return ((width - w) // 2, (height - h) // 2, w, h)


def panel_slots(count: int, width: int, height: int,
                lead: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Positions for `count` panels: a top row and a bottom row, clear of the lead panel."""
    count = max(1, count)
    per_row = -(-count // 2)
    pw = (min(width // max(3, per_row), int(width * 0.28))) // 2 * 2
    ph = (pw * height // width) // 2 * 2
    band = lead[1] - 8   # the free band above (and below) the lead panel
    if ph > band:        # shrink so panels do not slide under the lead panel
        ph = max(2, band // 2 * 2)
        pw = (ph * width // height) // 2 * 2
    margin = max(4, (lead[1] - ph) // 2)
    rows = [margin, height - ph - margin]
    gap = (width - per_row * pw) // (per_row + 1)
    slots = []
    for i in range(count):
        row, col = divmod(i, per_row)
        slots.append((gap + col * (pw + gap), rows[min(row, 1)], pw, ph))
    return slots


def _clip_paths(spans: list[Span], cache: Path, width: int, height: int, fps: int,
                key: str) -> list[Path]:
    paths = []
    for span in spans:
        name = f"{span.video_ref}|{span.source_start:.3f}|{span.frames}|{width}x{height}@{fps}|{key}"
        paths.append(cache / (hashlib.sha1(name.encode()).hexdigest() + ".mp4"))
    return paths


def _make_clips(items: list[tuple[Path, Span]], width: int, height: int, fps: int, key: str,
                jobs: int, label: str) -> None:
    def make(item: tuple[Path, Span]) -> None:
        path, span = item
        tmp = path.with_suffix(".part.mp4")
        if span.video_ref is None:
            ffmpeg.make_color_clip(span.frames, tmp, width, height, fps, key)
        else:
            ffmpeg.extract_clip(Path(span.video_ref), span.source_start, span.frames, tmp,
                                width, height, fps, pad_color=key)
        tmp.rename(path)

    if not items:
        return
    progress.stage(label, len(items))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, _ in enumerate(pool.map(make, items), 1):
            progress.update(n)
            print(f"\r  {label} {n}/{len(items)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


def _build_plane(spans: list[Span], out: Path, cache: Path, width: int, height: int, fps: int,
                 jobs: int, key: str, label: str) -> Path:
    """Render the clips of one plane and return the concat list that plays them in order."""
    clips = _clip_paths(spans, cache, width, height, fps, key)
    todo = sorted({c: s for c, s in zip(clips, spans, strict=True) if not c.exists()}.items())
    _make_clips(todo, width, height, fps, key, jobs, label)
    list_file = out.with_suffix(".list.txt")
    ffmpeg.write_concat_list(clips, list_file)
    return list_file


def render_video(spans: list[Span], audio: Path, out: Path, clip_cache: Path,
                 width: int = 1280, height: int = 720, fps: int = 30, jobs: int = 6,
                 key: str = "black", background: str | None = None) -> None:
    """Fullscreen layout: one plane, clips concatenated, audio muxed on. Clips are cached by
    content, so several videos from one render share them. `background` overrides the colour of
    the empty parts (the mix uses black, so the finished video has no key colour left in it)."""
    clip_cache.mkdir(parents=True, exist_ok=True)
    key = background or key
    clips = _clip_paths(spans, clip_cache, width, height, fps, key)
    todo = sorted({c: s for c, s in zip(clips, spans, strict=True) if not c.exists()}.items())
    _make_clips(todo, width, height, fps, key, jobs, f"video {out.name}: clips")
    out.parent.mkdir(parents=True, exist_ok=True)
    progress.stage(f"video {out.name}: concat + mux")
    ffmpeg.concat_with_audio(clips, audio, out, clip_cache / f"{out.stem}.list.txt")


def render_layered_video(layers: list[Layer], audio: Path, out: Path, clip_cache: Path,
                         width: int = 1280, height: int = 720, fps: int = 30, jobs: int = 6,
                         key: str = "black", background: str | None = None) -> None:
    """Layered layout: each layer becomes its own video, then one keyed compositing pass.

    `background` overrides the colour behind everything (the mix uses black, so no key colour is
    left in the finished video); the overlay layers still use the key colour to be keyed out."""
    clip_cache.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_layer = next((lyr for lyr in layers if not lyr.transparent), None)
    overlays = [lyr for lyr in layers if lyr is not base_layer]
    frames = max((s.first_frame + s.frames for layer in layers for s in layer.spans), default=0)
    if base_layer is None:
        # Nothing plays underneath: a plain black plane becomes the canvas.
        base_layer = Layer("canvas", [Span(0, frames, None)], 0, 0, width, height, transparent=False)
    base = _build_plane(base_layer.spans, clip_cache / f"{out.stem}-base.mp4", clip_cache,
                        width, height, fps, jobs, background or key,
                        f"video {out.name}: {base_layer.name} clips")

    composed = []
    for i, layer in enumerate(overlays):
        stem = f"{out.stem}-{i}-{hashlib.sha1(layer.name.encode()).hexdigest()[:8]}"
        video = _build_plane(layer.spans, clip_cache / f"{stem}.mp4", clip_cache,
                             layer.width, layer.height, fps, jobs, key,
                             f"video {out.name}: {layer.name} clips")
        composed.append({"video": video, "x": layer.x, "y": layer.y})
    progress.stage(f"video {out.name}: composing {len(layers)} layers")
    silent = clip_cache / f"{out.stem}-composed.mp4"
    key_hex = key_color_hex(key) if key in KEY_COLORS else key
    ffmpeg.compose_layers(base, composed, silent, key_hex, clip_cache, KEY_SIMILARITY, fps, frames)
    progress.stage(f"video {out.name}: muxing audio")
    ffmpeg.mux_audio(silent, audio, out)


# --- chroma key -------------------------------------------------------------------------------
#
# "Nothing here" (a layer's rests, the padding beside a clip, the empty canvas) is painted in a
# key colour instead of black, so the videos can be keyed in a video editor, and so the layers
# can be composed by keying that colour out instead of carrying a separate mask.

KEY_COLORS = {
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
}
KEY_SIMILARITY = 0.12


def pick_key_color(samples: np.ndarray, candidates: dict[str, tuple[int, int, int]] | None = None) -> str:
    """The candidate colour that the material uses least (max distance from its pixels)."""
    candidates = candidates or KEY_COLORS
    if samples.size == 0:
        return "magenta"
    pixels = samples.reshape(-1, 3).astype(np.int16)
    scores = {}
    for name, rgb in candidates.items():
        distance = np.abs(pixels - np.array(rgb, dtype=np.int16)).max(axis=1)
        # How much of the material sits near this colour (and would be keyed out by mistake).
        scores[name] = float(np.mean(distance < 255 * KEY_SIMILARITY * 2))
    return min(scores, key=lambda name: (scores[name], list(candidates).index(name)))


def key_color_hex(name: str) -> str:
    r, g, b = KEY_COLORS[name]
    return f"0x{r:02X}{g:02X}{b:02X}"
