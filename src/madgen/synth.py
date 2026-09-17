"""Phase 4 (audio): WORLD analysis -> f0 rewrite -> resynthesis, placed on the target timeline.

Silence rules:
- A rest in the target stays silent: nothing is ever placed outside a note's own span
  (except the short fade-out tail that doubles as the crossfade into a legato neighbour).
- A segment shorter than its note is not stretched or looped: the remainder of the note is silence.

Lyrics mode instead reproduces the note faithfully for sustained units (vowel / N): it takes the
*core* of the segment -- the longest run of frames that are voiced and within CORE_RANGE_DB of the
segment's loudest frame, i.e. the vowel body without its decay, breath or silence -- and fits it to
exactly the note length: cut when longer, time-stretched when shorter (WORLD frames resampled,
STRETCH_EDGE_SEC kept as is at each end). Unvoiced frames inside the core are given a pitch, so
the whole note sounds.

Pitch correction: a segment is resynthesized at the note's pitch only when its own pitch is off
by more than `threshold_cents`; otherwise the original audio is used untouched.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyworld
import soundfile as sf

from .progress import progress

SR = 44100
FADE_IN_SEC = 0.005
XFADE_SEC = 0.02      # fade-out tail; overlaps the next note's fade-in when legato
PAD_SEC = 0.03        # extra context read around a segment so WORLD's edges are clean
TARGET_RMS_DB = -18.0
STRETCH_EDGE_SEC = 0.02
CORE_RANGE_DB = 12.0
CORE_MAX_GAP_FRAMES = 2   # quiet/unvoiced blips this short do not break the core
CORE_MIN_FRAMES = 4
# Faithful sustain: a sung vowel should sound pitched and even, so below SUSTAIN_PERIODIC_HZ the
# aperiodicity is capped (breathy material becomes voiced), and each frame's energy is pulled to
# the core's median (within +-SUSTAIN_LEVEL_RANGE_DB).
SUSTAIN_PERIODIC_HZ = 4000.0
SUSTAIN_MAX_APERIODICITY = 0.2
SUSTAIN_LEVEL_RANGE_DB = 12.0
FRAME_PERIOD = 5.0


@dataclass
class NoteJob:
    audio_path: str
    seg_start: float
    seg_end: float
    seg_f0: float
    target_f0: float
    note_start: float
    note_dur: float
    velocity: int
    flatten: float
    threshold_cents: float | None  # None = never correct the pitch
    level_db: float = TARGET_RMS_DB  # loudness the unit is normalized to (consonants sit lower)
    # Judge and shift pitch by the f0 of the part actually used, not the whole segment's median.
    # Needed for speech material (lyrics mode), whose pitch drifts within a segment.
    measure_used_f0: bool = False
    # Lyrics mode: fit the segment's voiced core to exactly the note length (see module docstring).
    sustain_to_note: bool = False


def needs_correction(seg_f0: float, target_f0: float, threshold_cents: float | None) -> bool:
    if threshold_cents is None or np.isnan(seg_f0):  # NaN: unvoiced material, no pitch to move
        return False
    return abs(1200 * np.log2(target_f0 / seg_f0)) > threshold_cents


def _db_to_amp(db: float) -> float:
    return 10 ** (db / 20)


def _stretch_frames(a: np.ndarray, n_new: int, edge: int) -> np.ndarray:
    """Resample frames along time to n_new frames (linear interpolation between neighbours),
    keeping `edge` frames at each end untouched."""
    n = len(a)
    edge = max(0, min(edge, n // 4, n_new // 4))
    mid_new = n_new - 2 * edge
    pos = np.concatenate([
        np.arange(edge, dtype=np.float64),
        np.linspace(edge, n - 1 - edge, mid_new) if mid_new > 0 else np.zeros(0),
        np.arange(n - edge, n, dtype=np.float64),
    ])
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = pos - i0
    if a.ndim == 2:
        w = w[:, None]
    return a[i0] * (1 - w) + a[i1] * w


def _frame_rms_db(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Loudness per WORLD frame (frame k is centred on sample k * hop)."""
    hop = int(SR * FRAME_PERIOD / 1000)
    padded = np.pad(x, (hop, hop * (n_frames + 2)))
    idx = np.arange(n_frames)[:, None] * hop + np.arange(2 * hop)[None, :]
    return 20 * np.log10(np.sqrt(np.mean(padded[idx] ** 2, axis=1)) + 1e-12)


def _longest_run(mask: np.ndarray, max_gap: int) -> tuple[int, int]:
    """[start, end) of the longest run of True, bridging gaps of up to max_gap frames."""
    best = (0, 0)
    start = None
    gap = 0
    for i, m in enumerate(np.append(mask, False)):
        if m:
            if start is None:
                start = i
            gap = 0
            end = i + 1
            if end - start > best[1] - best[0]:
                best = (start, end)
        elif start is not None:
            gap += 1
            if gap > max_gap or i == len(mask):
                start, gap = None, 0
    return best


def _fill_unvoiced(f0: np.ndarray) -> np.ndarray:
    voiced = np.nonzero(f0 > 0)[0]
    if voiced.size == 0:
        return f0
    return np.interp(np.arange(len(f0)), voiced, f0[voiced])


def _render_sustained(job: NoteJob, x: np.ndarray, offset: int) -> tuple[np.ndarray, float, bool, float]:
    """-> (audio of exactly note_dur + XFADE_SEC, reference f0, corrected, stretch ratio)"""
    f0, t = pyworld.dio(x, SR, frame_period=FRAME_PERIOD, f0_floor=60.0, f0_ceil=1100.0)
    f0 = pyworld.stonemask(x, f0, t, SR)
    fps = 1000.0 / FRAME_PERIOD
    seg_a = int(offset * fps) // SR
    seg_b = min(len(f0), int((offset + (job.seg_end - job.seg_start) * SR) * fps) // SR)
    if seg_b - seg_a < CORE_MIN_FRAMES:
        seg_a, seg_b = 0, len(f0)
    loud = _frame_rms_db(x, len(f0))
    seg_loud = loud[seg_a:seg_b]
    is_loud = seg_loud > seg_loud.max() - CORE_RANGE_DB
    a, b = _longest_run(is_loud & (f0[seg_a:seg_b] > 0), CORE_MAX_GAP_FRAMES)
    if b - a < CORE_MIN_FRAMES:
        a, b = _longest_run(is_loud, CORE_MAX_GAP_FRAMES)
    if b - a < CORE_MIN_FRAMES:
        a, b = 0, seg_b - seg_a
    core = slice(seg_a + a, seg_a + b)

    voiced_core = f0[core][f0[core] > 0]
    ref_f0 = float(np.median(voiced_core)) if voiced_core.size >= 3 else float("nan")
    corrected = needs_correction(ref_f0, job.target_f0, job.threshold_cents)

    sp = pyworld.cheaptrick(x, f0, t, SR)[core]
    ap = pyworld.d4c(x, f0, t, SR)[core]
    low = np.arange(ap.shape[1]) * SR / (2 * (ap.shape[1] - 1)) < SUSTAIN_PERIODIC_HZ
    ap[:, low] = np.minimum(ap[:, low], SUSTAIN_MAX_APERIODICITY)
    energy = sp.sum(axis=1)
    limit = 10 ** (SUSTAIN_LEVEL_RANGE_DB / 10)
    sp = sp * np.clip(np.median(energy) / energy, 1 / limit, limit)[:, None]
    core_f0 = _fill_unvoiced(f0[core])
    if corrected:
        core_f0 = job.target_f0 * (core_f0 / ref_f0) ** (1.0 - job.flatten)
    elif np.isnan(ref_f0) and job.threshold_cents is not None:
        core_f0 = np.full(len(core_f0), job.target_f0)  # no pitch in the material at all: give it the note's
        corrected = True

    out_len = job.note_dur + XFADE_SEC
    n_out = max(1, int(np.ceil(out_len * fps)) + 1)
    n_core = len(core_f0)
    if n_core >= n_out:
        f0_o, sp_o, ap_o = core_f0[:n_out], sp[:n_out], ap[:n_out]
    else:
        edge = int(round(STRETCH_EDGE_SEC * fps))
        f0_o = _stretch_frames(core_f0, n_out, edge)
        sp_o = _stretch_frames(sp, n_out, edge)
        ap_o = _stretch_frames(ap, n_out, edge)
    y = pyworld.synthesize(np.ascontiguousarray(f0_o), np.ascontiguousarray(sp_o), np.ascontiguousarray(ap_o),
                           SR, frame_period=FRAME_PERIOD)
    return y[: int(out_len * SR)], ref_f0, corrected, max(1.0, n_out / n_core)


def render_note(job: NoteJob) -> tuple[int, np.ndarray, float, bool, float]:
    """Return (start sample on the output timeline, audio, reference f0 used, pitch corrected,
    stretch ratio)."""
    info = sf.info(job.audio_path)
    seg_len = job.seg_end - job.seg_start
    # Take at most the note length plus the crossfade tail, and never beyond the segment,
    # so the tail never drags in the silence/breath that follows the segment in the source.
    use_len = min(seg_len, job.note_dur + XFADE_SEC)
    read_start = max(0.0, job.seg_start - PAD_SEC)
    # A sustained lyrics unit looks for its core over the whole segment.
    read_end = min(info.duration, job.seg_start + (seg_len if job.sustain_to_note else use_len) + PAD_SEC)
    x, _ = sf.read(job.audio_path, start=int(read_start * SR), stop=int(read_end * SR), dtype="float64")
    if x.size < 64:
        return int(job.note_start * SR), np.zeros(0, dtype=np.float32), job.seg_f0, False, 1.0

    offset = int((job.seg_start - read_start) * SR)
    stretch_ratio = 1.0
    if job.sustain_to_note:
        y, ref_f0, corrected, stretch_ratio = _render_sustained(job, x, offset)
        return _finish(job, y, ref_f0, corrected, stretch_ratio)
    frames_per_sec = 1000.0 / FRAME_PERIOD
    first_frame = int(offset * frames_per_sec) // SR
    last_frame = int((offset + int(use_len * SR)) * frames_per_sec) // SR
    ref_f0 = job.seg_f0
    f0 = t = None
    if job.measure_used_f0:
        f0, t = pyworld.dio(x, SR, frame_period=FRAME_PERIOD, f0_floor=60.0, f0_ceil=1100.0)
        f0 = pyworld.stonemask(x, f0, t, SR)
        used = f0[first_frame:last_frame]
        voiced_used = used[used > 0]
        ref_f0 = float(np.median(voiced_used)) if voiced_used.size >= 3 else float("nan")

    corrected = needs_correction(ref_f0, job.target_f0, job.threshold_cents)
    if corrected:
        if f0 is None:
            f0, t = pyworld.dio(x, SR, frame_period=FRAME_PERIOD, f0_floor=60.0, f0_ceil=1100.0)
            f0 = pyworld.stonemask(x, f0, t, SR)
        sp = pyworld.cheaptrick(x, f0, t, SR)
        ap = pyworld.d4c(x, f0, t, SR)
        new_f0 = f0.copy()
        voiced = f0 > 0
        # Keep (1 - flatten) of the natural contour around the target pitch.
        new_f0[voiced] = job.target_f0 * (f0[voiced] / ref_f0) ** (1.0 - job.flatten)
        y = pyworld.synthesize(new_f0, sp, ap, SR, frame_period=FRAME_PERIOD)
    else:
        y = x

    y = y[offset: offset + int(use_len * SR)]
    return _finish(job, y, ref_f0, corrected, stretch_ratio)


def _finish(job: NoteJob, y: np.ndarray, ref_f0: float, corrected: bool,
            stretch_ratio: float) -> tuple[int, np.ndarray, float, bool, float]:
    """Loudness normalization and fades."""
    if y.size == 0:
        return int(job.note_start * SR), np.zeros(0, dtype=np.float32), ref_f0, corrected, stretch_ratio

    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (_db_to_amp(job.level_db) / rms) * (job.velocity / 127)

    fade_in = min(int(FADE_IN_SEC * SR), y.size // 2)
    fade_out = min(int(XFADE_SEC * SR), y.size // 2)
    if fade_in:
        y[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out:
        y[-fade_out:] *= np.linspace(1, 0, fade_out)
    return int(round(job.note_start * SR)), y.astype(np.float32), ref_f0, corrected, stretch_ratio


def render_voice(jobs: list[NoteJob], total_sec: float,
                 workers: int | None = None) -> tuple[np.ndarray, list[tuple[float, bool, float]]]:
    """-> (audio, per job: (reference f0, pitch corrected, stretch ratio))"""
    out = np.zeros(int(total_sec * SR) + 1, dtype=np.float32)
    info: list[tuple[float, bool, float]] = []
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (start, y, ref_f0, corrected, ratio) in enumerate(pool.map(render_note, jobs, chunksize=8), 1):
            progress.update(n)
            info.append((ref_f0, corrected, ratio))
            end = min(out.size, start + y.size)
            out[start:end] += y[: end - start]
    return out, info


def sum_tracks(tracks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(max(t.size for t in tracks), dtype=np.float32)
    for t in tracks:
        out[: t.size] += t
    return out


def normalize_gain(audio: np.ndarray, peak_db: float = -1.0) -> float:
    """Gain that brings `audio`'s peak to `peak_db`. Apply the mix's gain to the parts too,
    so the exported parts add back up to the mix."""
    peak = float(np.max(np.abs(audio))) or 1.0
    return _db_to_amp(peak_db) / peak
