"""Drum tracks (GM channel 10): one borrowed phoneme per instrument, and the older behaviours."""

import json

import mido
import numpy as np
import pytest
import soundfile as sf

from madgen import db
from madgen.cli import build_parser
from madgen.corpus import build_corpus
from madgen.phonemes import drum_for
from madgen.render import render
from madgen.target import PERCUSSION_CHANNEL, load_midi

SR = 44100
# (phoneme, voiced): the kit needs noise for the hats and something low and pitched for the kick.
MATERIAL = [("ts", False), ("sh", False), ("s", False), ("t", False), ("k", False),
            ("b", True), ("d", True), ("g", True)]


def _corpus(tmp_path):
    """A source holding one short burst per phoneme, and a DB describing them."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 0.3)) / SR
    pieces, rows = [], []
    for i, (phoneme, voiced) in enumerate(MATERIAL):
        if voiced:
            wave = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 6)) * 0.3
        else:
            wave = rng.standard_normal(t.size) * 0.2
        pieces.append(wave)
        rows.append((phoneme, i * 0.3, i * 0.3 + 0.12, 110.0 if voiced else None))
    src = tmp_path / "src.wav"
    sf.write(src, np.concatenate(pieces).astype(np.float32), SR)

    db_path = tmp_path / "corpus.sqlite"
    build_corpus([src], db_path, workers=1)
    conn = db.connect(db_path)
    sid = conn.execute("SELECT source_id FROM sources").fetchone()[0]
    for phoneme, start, end, f0 in rows:
        cur = conn.execute(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, "
            "f0_std_cents, rms_db, kind) VALUES (?, ?, ?, ?, ?, NULL, 0, -12, 'phoneme')",
            (sid, start, end, phoneme, f0))
        conn.execute("INSERT INTO phoneme_candidates VALUES (?, 1, ?, 0.9)", (cur.lastrowid, phoneme))
    conn.commit()
    return db_path


def _drum_midi(tmp_path, notes=(36, 42, 38, 42, 36, 42)):
    """A drum track: the notes name instruments, and they are long on purpose (1 beat each)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Drums"))
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=100, channel=PERCUSSION_CHANNEL, time=0))
        track.append(mido.Message("note_off", note=note, channel=PERCUSSION_CHANNEL, time=480))
    path = tmp_path / "drums.mid"
    mid.save(path)
    return path


def _render(db_path, mid, tmp_path, name, *extra):
    out_dir = tmp_path / name
    render(build_parser().parse_args(
        ["render", "--db", str(db_path), "--melody", str(mid), "--out-dir", str(out_dir),
         "--workers", "1", *extra]))
    return json.loads((out_dir / "plan.json").read_text()), out_dir


def test_one_sound_per_instrument(tmp_path):
    db_path = _corpus(tmp_path)
    mid = _drum_midi(tmp_path)
    plan, _ = _render(db_path, mid, tmp_path, "samples")

    assert all("instrument" in e for e in plan)
    by_instrument = {}
    for e in plan:
        by_instrument.setdefault(e["instrument"], set()).add(e["segment_id"])
    # Every hit of an instrument is the same sound...
    assert all(len(v) == 1 for v in by_instrument.values()), by_instrument
    # ...and no two instruments share one.
    chosen = [next(iter(v)) for v in by_instrument.values()]
    assert len(set(chosen)) == len(chosen)
    assert set(by_instrument) == {"キック", "ハイハット", "スネア"}


def test_material_suits_the_instrument(tmp_path):
    db_path = _corpus(tmp_path)
    plan, _ = _render(db_path, _drum_midi(tmp_path), tmp_path, "suits")
    picked = {e["instrument"]: e for e in plan}
    # The kick takes a pitched, low sound; the hats and the snare take noise.
    assert picked["キック"]["f0_hz"] is not None
    assert picked["ハイハット"]["f0_hz"] is None
    assert picked["スネア"]["f0_hz"] is None
    # A drum hit is never pitch-corrected: its pitch is not what is heard.
    assert not any(e["pitch_corrected"] for e in plan)


def test_hits_are_cut_to_the_instrument_length(tmp_path):
    db_path = _corpus(tmp_path)
    plan, out_dir = _render(db_path, _drum_midi(tmp_path, notes=(42, 42)), tmp_path, "short")
    audio, sr = sf.read(out_dir / "mix.wav")
    hat = drum_for(42)
    # The notes last a beat (0.5 s) but a closed hi-hat is a short tick.
    tail = audio[int((hat.max_sec + 0.05) * sr): int(0.5 * sr)]
    assert np.max(np.abs(tail)) == 0.0
    assert np.max(np.abs(audio[: int(hat.max_sec * sr)])) > 0.1


@pytest.mark.parametrize("mode,expect_sound", [("off", False), ("samples", True)])
def test_percussion_off_is_silent(tmp_path, mode, expect_sound):
    db_path = _corpus(tmp_path)
    _, out_dir = _render(db_path, _drum_midi(tmp_path), tmp_path, mode, "--percussion", mode)
    audio, _ = sf.read(out_dir / "mix.wav")
    assert bool(np.max(np.abs(audio)) > 0.1) is expect_sound


def test_pitched_mode_keeps_the_old_behaviour(tmp_path):
    db_path = _corpus(tmp_path)
    plan, _ = _render(db_path, _drum_midi(tmp_path), tmp_path, "pitched", "--percussion", "pitched")
    # The old path treats the note numbers as pitches, so it says nothing about instruments...
    assert not any("instrument" in e for e in plan)
    # ...and picks from the pitch corpus, where the segments are the pitched ones.
    assert all(e["f0_hz"] is not None for e in plan)


def test_without_a_phoneme_corpus_it_falls_back(tmp_path):
    """A corpus analyzed before this feature existed still renders, the way it used to."""
    rng = np.random.default_rng(1)
    t = np.arange(SR) / SR
    src = tmp_path / "plain.wav"
    sf.write(src, (sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 6)) * 0.3
                   + rng.standard_normal(t.size) * 0.01).astype(np.float32), SR)
    db_path = tmp_path / "plain.sqlite"
    build_corpus([src], db_path, workers=1)
    plan, _ = _render(db_path, _drum_midi(tmp_path), tmp_path, "fallback")
    assert not any("instrument" in e for e in plan)


def test_drum_tracks_are_still_matched_by_track(tmp_path):
    mid = _drum_midi(tmp_path)
    voices = load_midi(mid)
    assert all(v.percussion for v in voices)
