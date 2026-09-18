"""Phase 2 (lyrics mode): UTAU .ust / OpenUtau .ustx -> phoneme-level target units.

Each note's lyric is split into morae (see phonemes.kana_to_morae), each mora into a short
consonant unit followed by a vowel unit. UTAU conventions handled:
- "R", "r", "pau", "sil", "br", "息", "吸", "" -> rest (no unit, stays silent)
- "+", "+~", "+*", "-", "ー" (a whole note)     -> melisma: the previous vowel continues on this note
- "a か" / "- か" (VCV / CVVC style prefixes)   -> the part after the last space is used
- "っ" -> a short silence, not a unit
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .phonemes import CLOSURE, VOWELS, is_voiced_sustained, kana_to_morae
from .target import TargetUnit, Voice, midi_to_hz, no_tracks_message, split_voices, tick_to_sec

CONSONANT_SEC = 0.06            # consonant length at the head of a note
CONSONANT_MAX_RATIO = 0.4       # ...but never more than this share of a short note
REST_LYRICS = {"", "r", "pau", "sil", "br", "息", "吸"}
MELISMA_LYRICS = {"+", "+~", "+*", "-", "ー"}
DEFAULT_VELOCITY = 100


@dataclass
class UstNote:
    start_tick: int
    duration_tick: int
    tone: int
    lyric: str


@dataclass
class UstTrack:
    index: int
    name: str
    muted: bool
    volume_db: float
    notes: list[UstNote]


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_ustx(path: Path) -> tuple[list[UstTrack], list[tuple[int, int]], int]:
    """-> (tracks, tempo map as (tick, microseconds per beat), ticks per beat)"""
    data = yaml.load(_read_text(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    resolution = int(data.get("resolution", 480))
    tempos = data.get("tempos") or [{"position": 0, "bpm": data.get("bpm", 120)}]
    tempo_map = sorted((int(t["position"]), round(60e6 / float(t["bpm"]))) for t in tempos)
    if tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, tempo_map[0][1]))

    track_info = data.get("tracks") or []
    tracks: dict[int, UstTrack] = {}
    for part in data.get("voice_parts") or []:
        no = int(part.get("track_no", 0))
        if no not in tracks:
            info = track_info[no] if no < len(track_info) else {}
            tracks[no] = UstTrack(
                index=no,
                name=str(info.get("track_name") or f"Track{no + 1}"),
                muted=bool(info.get("mute", False)),
                volume_db=float(info.get("volume", 0.0)),
                notes=[],
            )
        base = int(part.get("position", 0))
        for n in part.get("notes") or []:
            tracks[no].notes.append(UstNote(
                base + int(n["position"]), int(n["duration"]), int(n["tone"]), str(n.get("lyric", ""))))
    return [tracks[k] for k in sorted(tracks)], tempo_map, resolution


def parse_ust(path: Path) -> tuple[list[UstTrack], list[tuple[int, int]], int]:
    """Classic UTAU .ust: one track, notes back to back, tempo in [#SETTING] or on a note."""
    resolution = 480
    tempo_map: list[tuple[int, int]] = []
    notes: list[UstNote] = []
    name = path.stem
    tick = 0
    section: dict[str, str] | None = None
    sections: list[tuple[str, dict[str, str]]] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if line.startswith("[#") and line.endswith("]"):
            section = {}
            sections.append((line[2:-1], section))
        elif section is not None and "=" in line:
            k, v = line.split("=", 1)
            section[k.strip()] = v.strip()
    for tag, fields in sections:
        if tag == "SETTING":
            if "Tempo" in fields:
                tempo_map.append((0, round(60e6 / float(fields["Tempo"]))))
            name = fields.get("ProjectName") or name
        elif tag.isdigit() or tag in ("INSERT",):
            if "Tempo" in fields:
                tempo_map.append((tick, round(60e6 / float(fields["Tempo"]))))
            length = int(float(fields.get("Length", "0")))
            notes.append(UstNote(tick, length, int(fields.get("NoteNum", "60")), fields.get("Lyric", "R")))
            tick += length
    if not tempo_map or tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, tempo_map[0][1] if tempo_map else 500000))
    return [UstTrack(0, name, False, 0.0, notes)], sorted(set(tempo_map)), resolution


def _clean_lyric(lyric: str) -> str:
    lyric = lyric.strip()
    if " " in lyric:  # VCV "a か" / CVVC "- か"
        lyric = lyric.rsplit(" ", 1)[1]
    return lyric


def _note_units(morae: list[list[str]], start: float, dur: float) -> list[tuple[str, float, float]]:
    """Lay the morae of one note out in time: (phoneme, start, duration)."""
    out: list[tuple[str, float, float]] = []
    step = dur / len(morae)
    for mi, mora in enumerate(morae):
        t0 = start + mi * step
        if mora == [CLOSURE]:
            continue  # sokuon: silence
        heads, tail = mora[:-1], mora[-1]
        cons = min(CONSONANT_SEC, step * CONSONANT_MAX_RATIO) if heads else 0.0
        for h in heads:
            out.append((h, t0, cons / len(heads)))
            t0 += cons / len(heads)
        out.append((tail, t0, step - cons))
    return out


def load_ust(path: Path, tracks: str | None = None) -> list[Voice]:
    tracks_all, tempo_map, tpb = (parse_ustx if path.suffix.lower() == ".ustx" else parse_ust)(path)
    convert = tick_to_sec(tempo_map, tpb)
    wanted = None if tracks is None else {t.strip() for t in tracks.split(",") if t.strip()}

    voices: list[Voice] = []
    available = [f"{tr.index}:{tr.name}（{len(tr.notes)}音{'、ミュート' if tr.muted else ''}）"
                 for tr in tracks_all]
    for tr in tracks_all:
        if wanted is None and tr.muted:
            print(f"  ust track {tr.name} is muted, skipping", file=sys.stderr)
            continue
        if wanted is not None and str(tr.index) not in wanted and tr.name not in wanted:
            continue
        notes = sorted(
            (convert(n.start_tick), convert(n.start_tick + n.duration_tick), n.tone, i)
            for i, n in enumerate(tr.notes) if n.duration_tick > 0
        )
        for vi, vnotes in enumerate(split_voices(notes)):
            voice = Voice(tr.index, tr.name, vi, origin="ust", gain=10 ** (tr.volume_db / 20))
            last_vowel: str | None = None
            prev_end = None
            unknown: set[str] = set()
            for ni, (s, e, tone, src_i) in enumerate(vnotes):
                lyric = _clean_lyric(tr.notes[src_i].lyric)
                if lyric.lower() in REST_LYRICS:
                    last_vowel = None
                    continue
                if lyric in MELISMA_LYRICS:
                    # Only continues a vowel that was actually sounding right before.
                    if last_vowel is None or prev_end is None or s - prev_end > 1e-3:
                        continue
                    morae = [[last_vowel]]
                else:
                    morae, bad = kana_to_morae(lyric)
                    unknown.update(bad)
                    if not morae:
                        last_vowel = None
                        continue
                for ph, t0, d in _note_units(morae, s, e - s):
                    voice.units.append(TargetUnit(
                        index=len(voice.units), start_sec=t0, duration_sec=d,
                        target_f0_hz=midi_to_hz(tone), velocity=DEFAULT_VELOCITY,
                        phoneme=ph, note_index=ni,
                    ))
                tail = morae[-1][-1]
                last_vowel = tail if is_voiced_sustained(tail) else None
                prev_end = e
            if unknown:
                print(f"  ust track {tr.name}: ignored lyric characters {''.join(sorted(unknown))}",
                      file=sys.stderr)
            if voice.units:
                voices.append(voice)
    if not voices:
        raise SystemExit(no_tracks_message(path, tracks, available, "--ust-tracks"))
    return voices


__all__ = ["load_ust", "parse_ust", "parse_ustx", "VOWELS"]
