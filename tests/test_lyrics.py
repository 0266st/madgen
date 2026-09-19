"""Lyrics mode: kana table, UST/USTX parsing, lexicographic matching, DB migration, end to end."""

import json
import sqlite3

import numpy as np
import pytest
import soundfile as sf

from madgen import db
from madgen.cli import build_parser
from madgen.corpus import build_corpus
from madgen.match import LyricsWeights, select_units_lyrics
from madgen.phonemes import distance, fill_candidates, kana_to_morae
from madgen.render import render
from madgen.target import TargetUnit
from madgen.ust import load_ust


def test_kana_to_morae():
    morae, unknown = kana_to_morae("きゃっとーシャン")
    assert morae == [["ky", "a"], ["cl"], ["t", "o"], ["o"], ["sh", "a"], ["N"]]
    assert unknown == []
    assert kana_to_morae("ちょ")[0] == [["ch", "o"]]
    assert kana_to_morae("か↑")[1] == ["↑"]


def test_distance_table():
    assert distance("a", "a") == 0
    assert distance("i", "e") < distance("i", "o")          # close vowels are closer
    assert distance("k", "g") < distance("k", "m")          # voicing only < manner + voicing
    assert distance("a", "k") == 1.0
    cands = fill_candidates("k", 0.9)
    assert [c[0] for c in cands][0] == "k" and len(cands) == 3
    assert all(c[1] < 0.1 for c in cands[1:])


USTX = """\
resolution: 480
tempos:
- position: 0
  bpm: 120
tracks:
- track_name: Lead
  mute: false
  volume: 0
- track_name: Muted
  mute: true
  volume: 0
voice_parts:
- track_no: 0
  position: 480
  notes:
  - {position: 0, duration: 480, tone: 69, lyric: か}
  - {position: 480, duration: 480, tone: 71, lyric: +}
  - {position: 960, duration: 480, tone: 69, lyric: R}
  - {position: 1440, duration: 480, tone: 69, lyric: a ん}
- track_no: 1
  position: 0
  notes:
  - {position: 0, duration: 480, tone: 60, lyric: あ}
"""


def test_ustx_parsing(tmp_path):
    path = tmp_path / "song.ustx"
    path.write_text(USTX, encoding="utf-8")
    voices = load_ust(path)
    assert [v.track_name for v in voices] == ["Lead"]          # muted track skipped
    units = [(u.phoneme, round(u.start_sec, 3), round(u.duration_sec, 3), u.note_index) for u in voices[0].units]
    # 120 BPM, 480 tpb -> 0.5 s per beat; the part starts at beat 1 (0.5 s)
    assert units == [
        ("k", 0.5, 0.06, 0), ("a", 0.56, 0.44, 0),   # か
        ("a", 1.0, 0.5, 1),                          # + : melisma on the next pitch
        ("N", 2.0, 0.5, 3),                          # R at 1.5 s is a rest; "a ん" -> ん
    ]
    assert load_ust(path, "Muted")[0].track_name == "Muted"   # explicit selection includes it


def test_ust_classic(tmp_path):
    path = tmp_path / "song.ust"
    path.write_bytes(
        "[#SETTING]\nTempo=120\n[#0000]\nLength=480\nLyric=R\nNoteNum=60\n"
        "[#0001]\nLength=960\nLyric=さ\nNoteNum=62\n[#TRACKEND]\n".encode("cp932"))
    units = load_ust(path)[0].units
    assert [(u.phoneme, round(u.start_sec, 3)) for u in units] == [("s", 0.5), ("a", 0.56)]


def _corpus(phonemes, conf, f0, start=None, dur=0.3):
    n = len(phonemes)
    start = np.arange(n) * 1.0 if start is None else np.asarray(start, dtype=float)
    return db.Corpus(
        ids=np.arange(n), source_id=np.array(["s"] * n), start=start, end=start + dur,
        phoneme=np.array([p[0] for p in phonemes]), f0=np.asarray(f0, dtype=float),
        f0_std=np.zeros(n), rms_db=np.zeros(n), video_ref=[None] * n,
        next_idx=np.array([i + 1 if i + 1 < n and abs(start[i + 1] - start[i] - dur) < 1e-6 else -1
                           for i in range(n)]),
        audio_cache={"s": ""}, source_path={"s": ""},
        cand_phoneme=np.array(phonemes, dtype=object), cand_conf=np.asarray(conf, dtype=float),
    )


def _unit(ph, f0=440.0, start=0.0, dur=0.3):
    return TargetUnit(0, start, dur, f0, 100, phoneme=ph, note_index=0)


def test_lexicographic_phoneme_beats_pitch():
    corpus = _corpus(
        [["e", "a", "o"],     # 0: "a" only as 2nd candidate, exact pitch
         ["a", "e", "o"]],    # 1: "a" as 1-best, pitch 5 semitones off
        [[0.9, 0.9, 0.0], [0.9, 0.05, 0.05]],
        [440.0, 587.3])
    assert list(select_units_lyrics([_unit("a")], corpus)) == [1]


def test_lexicographic_pitch_breaks_phoneme_ties():
    corpus = _corpus(
        [["a", "", ""], ["a", "", ""], ["a", "", ""]],
        [[0.95, 0, 0], [0.95, 0, 0], [0.95, 0, 0]],
        [520.0, 445.0, 300.0])
    assert list(select_units_lyrics([_unit("a", 440.0)], corpus)) == [1]


def test_lexicographic_concatenation_breaks_pitch_ties():
    # Segments 0 and 1 are adjacent in the source; 2 is elsewhere. All equally good otherwise.
    corpus = _corpus(
        [["k", "", ""], ["a", "", ""], ["a", "", ""]],
        [[0.95, 0, 0]] * 3,
        [np.nan, 440.0, 440.0],
        start=[0.0, 0.3, 5.0])
    units = [_unit("k", start=0.0, dur=0.06), _unit("a", start=0.06, dur=0.3)]
    assert list(select_units_lyrics(units, corpus, LyricsWeights(top_k=3))) == [0, 1]


def test_no_match_falls_back_to_distance():
    corpus = _corpus([["o", "", ""], ["i", "", ""]], [[0.9, 0, 0], [0.9, 0, 0]], [440.0, 440.0])
    assert list(select_units_lyrics([_unit("e")], corpus)) == [1]   # e is closer to i than to o


def test_old_db_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE segments (id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, start_sec REAL NOT NULL,
            end_sec REAL NOT NULL, phoneme TEXT NOT NULL, f0_hz REAL, video_ref TEXT,
            f0_std_cents REAL NOT NULL, rms_db REAL NOT NULL);
        CREATE TABLE sources (source_id TEXT PRIMARY KEY, file_hash TEXT NOT NULL, path TEXT NOT NULL,
            audio_cache TEXT NOT NULL, has_video INTEGER NOT NULL, analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO segments VALUES (1, 'x', 0, 0.2, '', 220, NULL, 5, -20);
        INSERT INTO sources (source_id, file_hash, path, audio_cache, has_video) VALUES ('x', 'x', 'p', 'a', 0);
    """)
    conn.commit()
    conn.close()
    corpus = db.load_corpus(db.connect(path))
    assert len(corpus) == 1 and corpus.f0[0] == 220
    with pytest.raises(SystemExit):
        db.load_corpus(db.connect(path), "phoneme")


def test_lyrics_render_end_to_end(tmp_path):
    """A phoneme corpus inserted by hand (the real analyzer needs the GPU models)."""
    sr = 44100
    t = np.arange(sr) / sr
    src = tmp_path / "src.wav"
    sf.write(src, np.concatenate([0.3 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)]).astype(np.float32), sr)
    db_path = tmp_path / "corpus.sqlite"
    build_corpus([src], db_path, workers=1)
    conn = db.connect(db_path)
    sid = conn.execute("SELECT source_id FROM sources").fetchone()[0]
    for ph, s, e, f0 in [("k", 0.0, 0.05, None), ("a", 0.05, 0.9, 440.0)]:
        cur = conn.execute(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db, "
            "kind) "
            "VALUES (?, ?, ?, ?, ?, NULL, 0, -10, 'phoneme')", (sid, s, e, ph, f0))
        conn.executemany("INSERT INTO phoneme_candidates VALUES (?, ?, ?, ?)",
                         [(cur.lastrowid, 1, ph, 0.9), (cur.lastrowid, 2, "g" if ph == "k" else "o", 0.05)])
    conn.commit()

    ustx = tmp_path / "song.ustx"
    ustx.write_text(USTX, encoding="utf-8")
    out_dir = tmp_path / "out"
    render(build_parser().parse_args(
        ["render", "--db", str(db_path), "--ust", str(ustx), "--out-dir", str(out_dir), "--split-parts",
         "--workers", "1"]))
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    assert [(e["phoneme"], e["matched_rank"]) for e in plan] == [("k", 1), ("a", 1), ("a", 1), ("N", None)]
    assert plan[0]["pitch_corrected"] is False            # consonant: never corrected
    assert plan[2]["pitch_corrected"] is True              # + melisma on B4 from an A4 segment
    y, _ = sf.read(out_dir / "mix.wav")
    assert np.max(np.abs(y[int(1.53 * sr): int(2.0 * sr)])) == 0.0   # the R rest stays silent
    assert (out_dir / "parts" / "ust00_Lead.wav").exists()


def _short_vowel_corpus(tmp_path):
    """A phoneme corpus whose only vowel segment (0.3-0.8 s) is 0.2 s of a 440 Hz tone followed by
    0.3 s of near silence -- long on paper, short in what actually sounds."""
    sr = 44100
    t = np.arange(sr) / sr
    x = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 11)) * 0.15  # harmonic, voice-like
    x[int(0.5 * sr):] *= 0.001
    src = tmp_path / "src.wav"
    sf.write(src, x.astype(np.float32), sr)
    db_path = tmp_path / "corpus.sqlite"
    build_corpus([src], db_path, workers=1)
    conn = db.connect(db_path)
    sid = conn.execute("SELECT source_id FROM sources").fetchone()[0]
    cur = conn.execute(
        "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db, kind) "
        "VALUES (?, 0.3, 0.8, 'a', 440, NULL, 0, -10, 'phoneme')", (sid,))
    conn.execute("INSERT INTO phoneme_candidates VALUES (?, 1, 'a', 0.9)", (cur.lastrowid,))
    conn.commit()
    ustx = tmp_path / "long.ustx"
    ustx.write_text(
        "resolution: 480\ntempos:\n- {position: 0, bpm: 60}\ntracks:\n- {track_name: Lead}\n"
        "voice_parts:\n- track_no: 0\n  position: 0\n  notes:\n"
        "  - {position: 0, duration: 480, tone: 69, lyric: あ}\n", encoding="utf-8")  # a 1 s note
    return db_path, ustx


@pytest.mark.parametrize("sustain", [True, False])
def test_vowel_sustained_for_the_whole_note(tmp_path, sustain):
    import pyworld

    db_path, ustx = _short_vowel_corpus(tmp_path)
    out_dir = tmp_path / "out"
    extra = [] if sustain else ["--no-lyrics-stretch"]
    render(build_parser().parse_args(
        ["render", "--db", str(db_path), "--ust", str(ustx), "--out-dir", str(out_dir), "--workers", "1", *extra]))
    y, sr = sf.read(out_dir / "mix.wav")
    peak = np.max(np.abs(y[: sr]))
    late = np.max(np.abs(y[int(0.6 * sr): int(0.95 * sr)]))
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    if sustain:
        # The 0.2 s core is stretched over the 1 s note (not the quiet tail): still loud near the end...
        assert late > 0.5 * peak
        assert plan[0]["stretch_ratio"] == pytest.approx(1.02 / 0.2, rel=0.05)
        # ...and on the note's pitch (A4) throughout.
        f0, _ = pyworld.harvest(y[: sr], sr, frame_period=5.0, f0_floor=100, f0_ceil=1000)
        voiced = f0[20:180]
        assert np.mean(voiced > 0) > 0.95
        assert np.all(np.abs(1200 * np.log2(voiced[voiced > 0] / 440.0)) < 30)
    else:
        assert late < 0.05 * peak               # the material as it is: 0.2 s of tone, then near silence
        assert plan[0]["stretch_ratio"] == 1.0


def test_gains_balance_and_groups(tmp_path):
    import mido

    db_path, ustx = _short_vowel_corpus(tmp_path)
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="Backing"))
    tr.append(mido.MetaMessage("set_tempo", tempo=1000000))
    tr.append(mido.Message("note_on", note=69, velocity=127, time=0))
    tr.append(mido.Message("note_off", note=69, time=480))
    mid_path = tmp_path / "backing.mid"
    mid.save(mid_path)

    from madgen.render import active_loudness_db

    def run(name, *extra):
        out_dir = tmp_path / name
        render(build_parser().parse_args(
            ["render", "--db", str(db_path), "--ust", str(ustx), "--melody", str(mid_path),
             "--out-dir", str(out_dir), "--workers", "1", *extra]))
        return {k: active_loudness_db(sf.read(out_dir / f"{k}.wav")[0]) for k in ("vocals", "accompaniment")}, out_dir

    levels, out_dir = run("auto")
    assert levels["vocals"] - levels["accompaniment"] == pytest.approx(2.0, abs=0.3)
    mix, _ = sf.read(out_dir / "mix.wav")
    both = sf.read(out_dir / "vocals.wav")[0] + sf.read(out_dir / "accompaniment.wav")[0]
    assert np.max(np.abs(both - mix)) < 1e-3     # the groups add up to the mix

    levels, _ = run("manual", "--no-auto-balance", "--gain", "Backing=-10")
    raw, _ = run("raw", "--no-auto-balance")
    lowered = levels["vocals"] - levels["accompaniment"]
    assert (raw["vocals"] - raw["accompaniment"]) - lowered == pytest.approx(-10.0, abs=0.3)

    levels, _ = run("group", "--vocal-boost", "12", "--ust-gain", "-2")
    assert levels["vocals"] - levels["accompaniment"] == pytest.approx(10.0, abs=0.3)

    with pytest.raises(SystemExit):
        run("bad", "--gain", "Nope=3")
