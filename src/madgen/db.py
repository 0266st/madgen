"""Corpus DB (SQLite): one row per source segment, plus the analyzed-source cache table."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,      -- content hash of the source file
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    phoneme TEXT NOT NULL,        -- 1-best phoneme ('' for pitch segments)
    f0_hz REAL,                   -- median f0 of the segment
    video_ref TEXT,               -- path of the source video, NULL for audio-only sources
    f0_std_cents REAL NOT NULL,   -- pitch stability within the segment
    rms_db REAL NOT NULL,         -- mean loudness (dBFS)
    kind TEXT NOT NULL DEFAULT 'pitch'  -- 'pitch' (melody mode) | 'phoneme' (lyrics mode)
);
CREATE TABLE IF NOT EXISTS phoneme_candidates (
    segment_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,        -- 1 = 1-best
    phoneme TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (segment_id, rank)
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,      -- change detection
    path TEXT NOT NULL,
    audio_cache TEXT NOT NULL,    -- decoded mono wav used for synthesis
    has_video INTEGER NOT NULL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phonemes_analyzer TEXT        -- NULL until the phoneme analysis has run
);
"""

# Columns added after the first release, for DBs created before them.
_MIGRATIONS = [
    ("segments", "kind", "TEXT NOT NULL DEFAULT 'pitch'"),
    ("sources", "phonemes_analyzer", "TEXT"),
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    for table, column, decl in _MIGRATIONS:
        exists = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
        if exists and column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.executescript(SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS segments_kind ON segments (kind, source_id, start_sec)")
    return conn


@dataclass
class Corpus:
    """Column arrays of every segment, in source/time order, ready for vectorized matching."""

    ids: np.ndarray
    source_id: np.ndarray
    start: np.ndarray
    end: np.ndarray
    phoneme: np.ndarray
    f0: np.ndarray
    f0_std: np.ndarray
    rms_db: np.ndarray
    video_ref: list[str | None]
    # next_idx[i] == i + 1 when segment i+1 directly follows segment i in the same source,
    # else -1. This is what the concatenation cost checks.
    next_idx: np.ndarray
    audio_cache: dict[str, str]
    source_path: dict[str, str]
    # Lyrics mode only: top-3 phoneme candidates per segment ('' = none) and their confidences.
    cand_phoneme: np.ndarray | None = None
    cand_conf: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)


def load_corpus(conn: sqlite3.Connection, kind: str = "pitch") -> Corpus:
    rows = conn.execute(
        "SELECT id, source_id, start_sec, end_sec, phoneme, f0_hz, f0_std_cents, rms_db, video_ref "
        "FROM segments WHERE kind = ? ORDER BY source_id, start_sec",
        (kind,),
    ).fetchall()
    if not rows:
        if kind == "phoneme":
            raise SystemExit("corpus has no phoneme analysis: run `madgen build-corpus --phonemes wav2vec2` first")
        raise SystemExit("corpus is empty: run `madgen build-corpus` first")
    ids, sid, st, en, ph, f0, sd, rms, vref = zip(*rows, strict=True)
    start = np.asarray(st, dtype=np.float64)
    end = np.asarray(en, dtype=np.float64)
    source_id = np.asarray(sid)
    same_source = source_id[1:] == source_id[:-1]
    touching = np.abs(start[1:] - end[:-1]) < 1e-3
    next_idx = np.full(len(rows), -1, dtype=np.int64)
    follow = np.nonzero(same_source & touching)[0]
    next_idx[follow] = follow + 1
    sources = conn.execute("SELECT source_id, audio_cache, path FROM sources").fetchall()
    cand_phoneme = cand_conf = None
    if kind == "phoneme":
        pos = {int(i): n for n, i in enumerate(ids)}
        cand_phoneme = np.full((len(rows), 3), "", dtype=object)
        cand_conf = np.zeros((len(rows), 3))
        for seg_id, rank, cph, conf in conn.execute(
            "SELECT c.segment_id, c.rank, c.phoneme, c.confidence FROM phoneme_candidates c "
            "JOIN segments s ON s.id = c.segment_id WHERE s.kind = 'phoneme' AND c.rank <= 3"
        ):
            n = pos[seg_id]
            cand_phoneme[n, rank - 1] = cph
            cand_conf[n, rank - 1] = conf
    return Corpus(
        ids=np.asarray(ids, dtype=np.int64),
        source_id=source_id,
        start=start,
        end=end,
        phoneme=np.asarray(ph),
        f0=np.asarray([np.nan if v is None else v for v in f0], dtype=np.float64),
        f0_std=np.asarray(sd, dtype=np.float64),
        rms_db=np.asarray(rms, dtype=np.float64),
        video_ref=list(vref),
        next_idx=next_idx,
        audio_cache={s: a for s, a, _ in sources},
        source_path={s: p for s, _, p in sources},
        cand_phoneme=cand_phoneme,
        cand_conf=cand_conf,
    )
