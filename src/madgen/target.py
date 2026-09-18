"""Phase 2: target definition -- turn a MIDI melody into target units.

Rests are not represented as units at all: a rest is simply time with no unit, and the
renderer leaves it silent (the previous note is never stretched into it).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mido

PERCUSSION_CHANNEL = 9


@dataclass
class TargetUnit:
    index: int
    start_sec: float
    duration_sec: float
    target_f0_hz: float
    velocity: int
    phoneme: str | None = None  # lyrics mode (UST/USTX) only
    note_index: int = -1        # the note this unit belongs to (a note may hold consonant + vowel)


@dataclass
class Voice:
    """A monophonic line. A polyphonic track is split into several voices."""

    track_index: int
    track_name: str
    voice_index: int
    units: list[TargetUnit] = field(default_factory=list)
    origin: str = "midi"   # "midi" (melody mode) or "ust" (lyrics mode)
    gain: float = 1.0      # track volume (USTX)
    channel: int | None = None   # MIDI channel the notes came from

    @property
    def percussion(self) -> bool:
        """GM channel 10 (0-based 9) is the drum kit: pitches are instruments, not notes."""
        return self.channel == PERCUSSION_CHANNEL

    @property
    def lyrics(self) -> bool:
        return self.origin == "ust"

    @property
    def label(self) -> str:
        prefix = "ust" if self.lyrics else ""
        return f"{prefix}{self.track_index}:{self.track_name}/v{self.voice_index}"


def midi_to_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _decode_name(msg: mido.MetaMessage) -> str:
    # mido decodes meta text as latin-1; Japanese DAWs usually write Shift_JIS.
    raw = msg.name.encode("latin-1", errors="replace")
    for enc in ("utf-8", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return msg.name


def _tempo_map(mid: mido.MidiFile) -> list[tuple[int, int]]:
    changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                changes.append((tick, msg.tempo))
    changes.sort()
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500000))
    return changes


def tick_to_sec(tempo_map: list[tuple[int, int]], tpb: int):
    """tempo_map: sorted (tick, microseconds per beat), starting at tick 0."""
    # Precompute seconds at each tempo change, then convert by segment.
    starts = []
    sec = 0.0
    for i, (tick, tempo) in enumerate(tempo_map):
        if i > 0:
            prev_tick, prev_tempo = tempo_map[i - 1]
            sec += (tick - prev_tick) * prev_tempo / 1e6 / tpb
        starts.append((tick, sec, tempo))

    def convert(tick: int) -> float:
        base = starts[0]
        for s in starts:
            if s[0] <= tick:
                base = s
            else:
                break
        return base[1] + (tick - base[0]) * base[2] / 1e6 / tpb

    return convert


def split_voices(notes: list[tuple[float, float, int, int]]) -> list[list[tuple[float, float, int, int]]]:
    """Greedy voice allocation: highest simultaneous note goes to voice 0, and so on."""
    voices: list[list[tuple[float, float, int, int]]] = []
    ends: list[float] = []
    for note in sorted(notes, key=lambda n: (n[0], -n[2])):
        start = note[0]
        for vi, end in enumerate(ends):
            if end <= start + 1e-6:
                voices[vi].append(note)
                ends[vi] = note[1]
                break
        else:
            voices.append([note])
            ends.append(note[1])
    return voices


def load_midi(path: Path, tracks: str | None = None) -> list[Voice]:
    mid = mido.MidiFile(path)
    convert = tick_to_sec(_tempo_map(mid), mid.ticks_per_beat)
    wanted = None if tracks is None else {t.strip() for t in tracks.split(",") if t.strip()}

    voices: list[Voice] = []
    available: list[str] = []
    for ti, track in enumerate(mid.tracks):
        name = ""
        tick = 0
        open_notes: dict[tuple[int, int], tuple[int, int]] = {}
        notes: list[tuple[float, float, int, int]] = []
        channels: Counter[int] = Counter()
        for msg in track:
            tick += msg.time
            if msg.type == "track_name" and not name.strip("_ "):
                # Some DAWs write a placeholder name ("__") first and the real one later.
                name = _decode_name(msg)
            elif msg.type == "note_on" and msg.velocity > 0:
                channels[msg.channel] += 1
                key = (msg.channel, msg.note)
                if key in open_notes:  # retrigger without note_off: close the old one
                    s, v = open_notes.pop(key)
                    notes.append((convert(s), convert(tick), msg.note, v))
                open_notes[key] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                key = (msg.channel, msg.note)
                if key in open_notes:
                    s, v = open_notes.pop(key)
                    notes.append((convert(s), convert(tick), msg.note, v))
        if not notes:
            continue
        channel = channels.most_common(1)[0][0] if channels else None
        kind = "打楽器" if channel == PERCUSSION_CHANNEL else "楽器"
        available.append(f"{ti}:{name or f'track{ti}'}（{len(notes)}音, {kind}）")
        if wanted is not None and str(ti) not in wanted and name not in wanted:
            continue
        for vi, vnotes in enumerate(split_voices([n for n in notes if n[1] > n[0]])):
            voice = Voice(ti, name or f"track{ti}", vi, channel=channel)
            for i, (s, e, note, vel) in enumerate(vnotes):
                voice.units.append(TargetUnit(i, s, e - s, midi_to_hz(note), vel, note_index=i))
            voices.append(voice)
    if not voices:
        raise SystemExit(no_tracks_message(path, tracks, available, "--tracks"))
    return voices


def no_tracks_message(path: Path, wanted: str | None, available: list[str], option: str) -> str:
    if not available:
        return f"{path} にノートがありません"
    listing = "\n".join(f"  {name}" for name in available)
    if wanted is None:
        return f"{path} に使えるトラックがありません:\n{listing}"
    return (f"{path} に {option} {wanted!r} に当てはまるトラックがありません。"
            f"\n名前か番号で指定してください:\n{listing}")

