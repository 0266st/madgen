"""End-to-end checks on synthetic material, focused on the silence rules."""

import json
import sqlite3

import mido
import numpy as np
import soundfile as sf

from madgen.cli import build_parser
from madgen.corpus import build_corpus
from madgen.render import render
from madgen.target import load_midi
from madgen.video import VideoNote, build_spans


def _render_args(db_path, mid, *extra):
    return build_parser().parse_args(["render", "--db", str(db_path), "--melody", str(mid), "--workers", "1", *extra])


def _source_wav(path, sr=44100):
    """0-1 s: 220 Hz tone | 1-2 s: silence | 2-3 s: noise (breath) | 3-4 s: 330 Hz tone."""
    t = np.arange(sr) / sr
    rng = np.random.default_rng(0)
    x = np.concatenate([
        0.3 * np.sin(2 * np.pi * 220 * t),
        np.zeros(sr),
        0.05 * rng.standard_normal(sr),
        0.3 * np.sin(2 * np.pi * 330 * t),
    ])
    sf.write(path, x.astype(np.float32), sr)


def _melody_mid(path):
    """120 BPM, 480 tpb: note 57 (220 Hz) 0-0.5 s, rest 0.5-1.5 s, note 64 (330 Hz) 1.5-2.5 s."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))
    tr.append(mido.Message("note_on", note=57, velocity=100, time=0))
    tr.append(mido.Message("note_off", note=57, time=480))
    tr.append(mido.Message("note_on", note=64, velocity=100, time=960))
    tr.append(mido.Message("note_off", note=64, time=960))
    mid.save(path)


def test_silence_rules(tmp_path):
    src = tmp_path / "src.wav"
    mid = tmp_path / "melody.mid"
    db_path = tmp_path / "corpus.sqlite"
    _source_wav(src)
    _melody_mid(mid)

    build_corpus([src], db_path, workers=1)
    segs = sqlite3.connect(db_path).execute("SELECT start_sec, end_sec FROM segments").fetchall()
    assert segs
    # Source silence (1-2 s) and noise (2-3 s) must never become material.
    for start, end in segs:
        assert end <= 1.0 + 0.02 or start >= 3.0 - 0.02, (start, end)

    # Cached: a second build adds nothing.
    build_corpus([src], db_path, workers=1)
    assert len(sqlite3.connect(db_path).execute("SELECT id FROM segments").fetchall()) == len(segs)

    out = tmp_path / "out.wav"
    render(_render_args(db_path, mid, "--out", str(out)))
    y, sr = sf.read(out)
    # The rest between the notes stays silent (allowing the 20 ms fade-out tail).
    assert np.max(np.abs(y[int(0.53 * sr): int(1.5 * sr)])) == 0.0
    # Both notes sound.
    assert np.max(np.abs(y[int(0.1 * sr): int(0.4 * sr)])) > 0.1
    assert np.max(np.abs(y[int(1.6 * sr): int(2.4 * sr)])) > 0.1


def test_polyphony_split(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.Message("note_on", note=60, velocity=90, time=0))
    tr.append(mido.Message("note_on", note=64, velocity=90, time=0))
    tr.append(mido.Message("note_off", note=60, time=480))
    tr.append(mido.Message("note_off", note=64, time=0))
    path = tmp_path / "chord.mid"
    mid.save(path)
    voices = load_midi(path)
    assert len(voices) == 2
    assert voices[0].units[0].target_f0_hz > voices[1].units[0].target_f0_hz


def test_pitch_threshold(tmp_path):
    src, mid, db_path = tmp_path / "src.wav", tmp_path / "melody.mid", tmp_path / "corpus.sqlite"
    _source_wav(src)
    _melody_mid(mid)  # notes exactly on the source pitches (220 / 330 Hz, within 2 cents)
    build_corpus([src], db_path, workers=1)

    def corrected(*extra):
        out_dir = tmp_path / ("_".join(extra) or "default")
        render(_render_args(db_path, mid, "--out-dir", str(out_dir), *extra))
        return [e["pitch_corrected"] for e in json.loads((out_dir / "plan.json").read_text())]

    assert corrected() == [False, False]                          # in tune: untouched by default
    assert corrected("--pitch-threshold", "0") == [True, True]    # 0 = always correct
    assert corrected("--no-pitch-correct") == [False, False]


def test_split_parts_add_up_to_mix(tmp_path):
    src, db_path = tmp_path / "src.wav", tmp_path / "corpus.sqlite"
    _source_wav(src)
    build_corpus([src], db_path, workers=1)
    mid = mido.MidiFile(ticks_per_beat=480)
    for name, note in (("Lead", 57), ("Bass", 64)):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=name))
        tr.append(mido.Message("note_on", note=note, velocity=100, time=0))
        tr.append(mido.Message("note_off", note=note, time=480))
    mid_path = tmp_path / "two.mid"
    mid.save(mid_path)

    out_dir = tmp_path / "out"
    render(_render_args(db_path, mid_path, "--out-dir", str(out_dir), "--split-parts"))
    mix, _ = sf.read(out_dir / "mix.wav")
    parts = sorted((out_dir / "parts").glob("*.wav"))
    assert [p.name for p in parts] == ["00_Lead.wav", "01_Bass.wav"]
    total = sum(sf.read(p)[0] for p in parts)
    assert np.max(np.abs(total - mix)) < 1e-3


def test_video_falls_back_to_other_voices():
    lead = [VideoNote(2.0, 1.0, "lead.mp4", 10.0)]
    backing = [VideoNote(0.5, 4.0, "back.mp4", 50.0), VideoNote(8.0, 1.0, "back.mp4", 60.0)]
    spans = build_spans([lead, backing], total_sec=10.0, fps=10)
    shown = [(s.first_frame, s.frames, s.video_ref) for s in spans]
    assert shown == [
        (0, 5, None),            # nothing sounds yet
        (5, 15, "back.mp4"),     # lead is resting: backing is shown, not black
        (20, 10, "lead.mp4"),    # lead takes priority
        (30, 15, "back.mp4"),    # lead rests again while backing still sounds
        (45, 35, None),          # nothing sounds for >= 0.5 s
        (80, 10, "back.mp4"),    # backing note
        (90, 10, None),          # trailing 1 s of silence
    ]
    # Taking over mid-note starts from the matching point in the segment.
    assert spans[3].source_start == 50.0 + (3.0 - 0.5)
