"""Phase 1: analyze sources into pitched segments and add them to the corpus DB.

Melody mode only needs pitch, so a source is cut into *voiced, pitch-stable* segments.
Silence handling on the source side happens here: frames that are quiet (silence) or
unvoiced (breath, noise, consonant hiss) never become part of a segment, so they can
never be picked as material.
"""

from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

from . import db, ffmpeg
from .progress import progress

ANALYSIS_SR = 16000
SYNTH_SR = 44100
FRAME_PERIOD_MS = 5.0
CHUNK_SEC = 120.0

MEDIA_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
              ".mp4", ".mkv", ".webm", ".mov", ".avi"}


@dataclass
class SegmentParams:
    silence_db: float = -45.0      # absolute floor (dBFS): quieter frames are silence
    silence_rel_db: float = 35.0   # also silence if this far below the source's loud level
    min_sec: float = 0.06          # shorter voiced pieces are dropped
    max_sec: float = 1.5           # long notes are split so they stay reusable
    pitch_tol_semitones: float = 1.0  # a segment ends when the pitch drifts this far


def file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def iter_media(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in MEDIA_EXTS))
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"source not found: {p}")
    return files


def _frame_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f0 (Hz, 0 = unvoiced) and rms (dBFS) per 5 ms frame."""
    x64 = x.astype(np.float64)
    f0, t = pyworld.dio(x64, ANALYSIS_SR, frame_period=FRAME_PERIOD_MS, f0_floor=60.0, f0_ceil=1100.0)
    f0 = pyworld.stonemask(x64, f0, t, ANALYSIS_SR)
    hop = int(ANALYSIS_SR * FRAME_PERIOD_MS / 1000)
    win = hop * 4
    padded = np.pad(x64, (win // 2, win))
    n = len(f0)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    rms = np.sqrt(np.mean(padded[idx] ** 2, axis=1) + 1e-12)
    return f0, 20 * np.log10(rms)


def _analyze_chunk(args: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    path, start, frames = args
    x, _ = sf.read(path, start=start, frames=frames, dtype="float32")
    return _frame_features(x)


def analyze_frames(wav16: Path, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame features for a whole (possibly hours long) file, chunked across processes."""
    total = sf.info(str(wav16)).frames
    chunk = int(CHUNK_SEC * ANALYSIS_SR)
    hop = int(ANALYSIS_SR * FRAME_PERIOD_MS / 1000)
    jobs = [(str(wav16), s, min(chunk, total - s)) for s in range(0, total, chunk)]
    f0s, rmss = [], []
    progress.stage("pitch analysis (chunks)", len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (f0, rms) in enumerate(pool.map(_analyze_chunk, jobs)):
            # Each chunk yields frames at 0, hop, ...; keep exactly the frames that belong to it.
            keep = -(-jobs[i][2] // hop)
            f0s.append(f0[:keep])
            rmss.append(rms[:keep])
            progress.update(i + 1)
            print(f"\r  analyzed {min((i + 1) * CHUNK_SEC, total / ANALYSIS_SR):.0f}"
                  f"/{total / ANALYSIS_SR:.0f} s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return np.concatenate(f0s), np.concatenate(rmss)


def segment_frames(f0: np.ndarray, rms_db: np.ndarray, p: SegmentParams) -> list[tuple[int, int]]:
    """Cut frame features into [start, end) frame ranges of voiced, audible, pitch-stable sound."""
    loud_level = np.percentile(rms_db, 95)
    threshold = max(p.silence_db, loud_level - p.silence_rel_db)
    usable = (f0 > 0) & (rms_db > threshold)

    frame_sec = FRAME_PERIOD_MS / 1000
    min_len = int(round(p.min_sec / frame_sec))
    max_len = int(round(p.max_sec / frame_sec))
    semis = np.zeros_like(f0)
    semis[f0 > 0] = 12 * np.log2(f0[f0 > 0] / 440.0)

    # Runs of usable frames.
    edges = np.diff(np.concatenate([[0], usable.astype(np.int8), [0]]))
    run_starts = np.nonzero(edges == 1)[0]
    run_ends = np.nonzero(edges == -1)[0]

    segments: list[tuple[int, int]] = []
    for rs, re_ in zip(run_starts, run_ends, strict=True):
        if re_ - rs < min_len:
            continue
        s = rs
        total = semis[rs]
        count = 1
        for i in range(rs + 1, re_):
            mean = total / count
            if abs(semis[i] - mean) > p.pitch_tol_semitones or i - s >= max_len:
                if i - s >= min_len:
                    segments.append((s, i))
                s, total, count = i, semis[i], 1
            else:
                total += semis[i]
                count += 1
        if re_ - s >= min_len:
            segments.append((s, re_))
    return segments


PHONEME_ANALYZERS = ("wav2vec2",)


def _add_phoneme_segments(conn, digest: str, path: Path, wav16: Path, video_ref: str | None,
                          device: str = "auto", whisper_model: str = "large-v3") -> None:
    from .phoneme_analysis import analyze

    audio, _ = sf.read(str(wav16), dtype="float32")
    segs = analyze(audio, None if device == "auto" else device, whisper_model)
    progress.stage("writing phoneme segments to DB")
    conn.execute("DELETE FROM phoneme_candidates WHERE segment_id IN "
                 "(SELECT id FROM segments WHERE source_id = ? AND kind = 'phoneme')", (digest,))
    conn.execute("DELETE FROM segments WHERE source_id = ? AND kind = 'phoneme'", (digest,))
    for seg in segs:
        cur = conn.execute(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db, "
            "kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'phoneme')",
            (digest, seg.start_sec, seg.end_sec, seg.phoneme, seg.f0_hz, video_ref, seg.f0_std_cents, seg.rms_db),
        )
        conn.executemany(
            "INSERT INTO phoneme_candidates (segment_id, rank, phoneme, confidence) VALUES (?, ?, ?, ?)",
            [(cur.lastrowid, rank, ph, conf) for rank, (ph, conf) in enumerate(seg.candidates, 1)],
        )
    conn.execute("UPDATE sources SET phonemes_analyzer = ? WHERE source_id = ?", ("wav2vec2", digest))
    conn.commit()
    print(f"  {len(segs)} phoneme segments", file=sys.stderr)
    progress.log(f"{path}: {len(segs)} phoneme segments")


def build_corpus(sources: list[Path], db_path: Path, workers: int | None = None,
                 params: SegmentParams | None = None, phonemes: str = "none",
                 device: str = "auto", whisper_model: str = "large-v3") -> None:
    if phonemes != "none" and phonemes not in PHONEME_ANALYZERS:
        raise SystemExit(f"unknown phoneme analyzer {phonemes!r}; choose from {PHONEME_ANALYZERS}")
    params = params or SegmentParams()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    cache_dir = db_path.with_suffix(db_path.suffix + ".cache")
    conn = db.connect(db_path)

    for path in iter_media(sources):
        print(f"[build-corpus] {path}", file=sys.stderr)
        progress.log(f"source {path}")
        progress.stage("hashing source")
        digest = file_hash(path)
        row = conn.execute("SELECT has_video, phonemes_analyzer FROM sources WHERE source_id = ?",
                           (digest,)).fetchone()
        if row:
            has_video, analyzed = row
            if phonemes == "none" or analyzed == phonemes:
                print("  unchanged, skipping (cached)", file=sys.stderr)
                continue
            # Pitch analysis is cached; only the phoneme analysis is missing.
            print("  pitch analysis cached; adding phoneme analysis", file=sys.stderr)
            wav16 = cache_dir / f"{digest}.16k.wav"
            progress.stage("decoding audio (16 kHz)")
            ffmpeg.extract_audio(path, wav16, ANALYSIS_SR)
            _add_phoneme_segments(conn, digest, path, wav16,
                                  str(path.resolve()) if has_video else None, device, whisper_model)
            wav16.unlink()
            continue
        # The same path with different content: drop the stale analysis.
        for (old,) in conn.execute("SELECT source_id FROM sources WHERE path = ?", (str(path.resolve()),)).fetchall():
            conn.execute("DELETE FROM phoneme_candidates WHERE segment_id IN "
                         "(SELECT id FROM segments WHERE source_id = ?)", (old,))
            conn.execute("DELETE FROM segments WHERE source_id = ?", (old,))
            conn.execute("DELETE FROM sources WHERE source_id = ?", (old,))

        info = ffmpeg.probe(path)
        wav44 = cache_dir / f"{digest}.wav"
        wav16 = cache_dir / f"{digest}.16k.wav"
        print(f"  decoding audio ({info['duration']:.0f} s)", file=sys.stderr)
        progress.stage(f"decoding audio ({info['duration']:.0f} s)")
        ffmpeg.extract_audio(path, wav44, SYNTH_SR)
        ffmpeg.extract_audio(path, wav16, ANALYSIS_SR)

        f0, rms_db = analyze_frames(wav16, workers)
        segs = segment_frames(f0, rms_db, params)
        frame_sec = FRAME_PERIOD_MS / 1000
        video_ref = str(path.resolve()) if info["has_video"] else None
        records = []
        for s, e in segs:
            voiced = f0[s:e][f0[s:e] > 0]
            cents = 1200 * np.log2(voiced / np.median(voiced))
            records.append((
                digest, s * frame_sec, e * frame_sec, "", float(np.median(voiced)), video_ref,
                float(np.std(cents)), float(np.mean(rms_db[s:e])),
            ))
        conn.executemany(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        conn.execute(
            "INSERT INTO sources (source_id, file_hash, path, audio_cache, has_video) VALUES (?, ?, ?, ?, ?)",
            (digest, digest, str(path.resolve()), str(wav44.resolve()), int(info["has_video"])),
        )
        conn.commit()
        usable_sec = sum(e - s for s, e in segs) * frame_sec
        print(f"  {len(records)} segments, {usable_sec:.0f} s of usable voiced sound", file=sys.stderr)
        progress.log(f"{path}: {len(records)} pitch segments")
        if phonemes != "none":
            _add_phoneme_segments(conn, digest, path, wav16, video_ref, device, whisper_model)
        wav16.unlink()
    conn.close()
