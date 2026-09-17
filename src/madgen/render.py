"""render: phases 2-4 wired together.

--melody (MIDI) tracks go through melody mode (phoneme-agnostic, pitch corpus);
--ust (UST/USTX) tracks go through lyrics mode (phoneme corpus). Both can be used at once:
the MIDI then holds the accompaniment and the UST the sung parts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import db
from .match import CostWeights, LyricsWeights, select_units, select_units_lyrics
from .phonemes import is_voiced_sustained
from .progress import progress
from .synth import SR, TARGET_RMS_DB, NoteJob, normalize_gain, render_voice, sum_tracks
from .target import Voice, load_midi
from .video import VideoNote, build_spans, render_video

TAIL_SEC = 1.0
FPS = 30
CONSONANT_LEVEL_DB = TARGET_RMS_DB - 6.0


@dataclass
class RenderedVoice:
    voice: Voice
    audio: np.ndarray
    video_notes: list[VideoNote]

    @property
    def sounding_sec(self) -> float:
        return sum(u.duration_sec for u in self.voice.units)

    @property
    def track_key(self) -> tuple[str, int]:
        return self.voice.origin, self.voice.track_index


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_") or "track"


def _track_matches(voice: Voice, wanted: str) -> bool:
    prefix = "ust" if voice.lyrics else ""
    return wanted in (voice.track_name, f"{prefix}{voice.track_index}")


def _video_priority(voices: list[RenderedVoice], wanted: str | None) -> list[RenderedVoice]:
    """Order voices for the mix video: the lead track first, then the rest by sounding time.
    The lead is --video-track if given, else the longest-sounding UST (sung) track if any,
    else a MIDI track named *main*, else the longest-sounding MIDI track."""
    tracks = {r.track_key: r.voice for r in voices}
    time = {key: sum(r.sounding_sec for r in voices if r.track_key == key) for key in tracks}
    if wanted is not None:
        lead = [key for key, v in tracks.items() if _track_matches(v, wanted)]
        if not lead:
            names = [f"{'ust' if v.lyrics else ''}{v.track_index}:{v.track_name}" for v in tracks.values()]
            raise SystemExit(f"--video-track {wanted!r} not found; tracks: {names}")
        lead_key = lead[0]
    else:
        sung = [key for key, v in tracks.items() if v.lyrics]
        named_main = [key for key, v in tracks.items() if not v.lyrics and "main" in v.track_name.lower()]
        if sung:
            lead_key = max(sung, key=lambda k: time[k])
        elif named_main:
            lead_key = named_main[0]
        else:
            lead_key = max(tracks, key=lambda k: time[k])
    return sorted(voices, key=lambda r: (r.track_key != lead_key, r.voice.voice_index, -r.sounding_sec))


def active_loudness_db(audio: np.ndarray, block_sec: float = 0.1) -> float:
    """Loudness (dBFS RMS) over the blocks that actually sound, so long rests don't count."""
    h = int(SR * block_sec)
    n = len(audio) // h
    if n == 0:
        return -120.0
    rms = np.sqrt((audio[: n * h].astype(np.float64).reshape(n, h) ** 2).mean(axis=1))
    # "Sounding" is relative to the loudest block, so the measure does not depend on overall level.
    active = rms[rms > max(1e-7, rms.max() * 10 ** (-50 / 20))]
    return -120.0 if active.size == 0 else float(20 * np.log10(np.sqrt((active ** 2).mean())))


def _parse_track_gains(specs: list[str] | None) -> list[tuple[str, float]]:
    out = []
    for spec in specs or []:
        name, sep, value = spec.rpartition("=")
        try:
            out.append((name, float(value)))
        except ValueError:
            sep = ""
        if not sep or not name:
            raise SystemExit(f"--gain expects NAME=DB (e.g. --gain Bass=-3 or --gain ust0=+2), got {spec!r}")
    return out


def _apply_gains(rendered: list[RenderedVoice], args: argparse.Namespace) -> None:
    """Volume: automatic vocal/accompaniment balance, then the group gains, then per-track gains.
    Nothing is touched when every gain is 0 dB (so melody-only renders are unchanged)."""
    factors = [0.0] * len(rendered)   # dB per voice
    sung = [i for i, r in enumerate(rendered) if r.voice.lyrics]
    backing = [i for i, r in enumerate(rendered) if not r.voice.lyrics]
    if sung and backing and args.vocal_boost is not None:
        sung_db = active_loudness_db(sum_tracks([rendered[i].audio for i in sung]))
        backing_db = active_loudness_db(sum_tracks([rendered[i].audio for i in backing]))
        adjust = sung_db - args.vocal_boost - backing_db
        for i in backing:
            factors[i] += adjust
        msg = (f"balance: vocals {sung_db:.1f} dB, accompaniment {backing_db:.1f} dB "
               f"-> accompaniment {adjust:+.1f} dB (vocals {args.vocal_boost:g} dB louder)")
        print(f"  {msg}", file=sys.stderr)
        progress.log(msg)
    for i in sung:
        factors[i] += args.ust_gain
    for i in backing:
        factors[i] += args.melody_gain
    for name, db_ in _parse_track_gains(args.gain):
        hit = [i for i, r in enumerate(rendered) if _track_matches(r.voice, name)]
        if not hit:
            names = sorted({f"{'ust' if r.voice.lyrics else ''}{r.voice.track_index}:{r.voice.track_name}"
                            for r in rendered})
            raise SystemExit(f"--gain: track {name!r} not found; tracks: {names}")
        for i in hit:
            factors[i] += db_
    for r, f in zip(rendered, factors, strict=True):
        if f != 0.0:
            r.audio = r.audio * np.float32(10 ** (f / 20))


def _resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    """(mix wav, mix mp4 or None, parts dir or None)"""
    if (args.out is None) == (args.out_dir is None):
        raise SystemExit("give exactly one of --out or --out-dir")
    if args.out_dir is not None:
        if args.video_out is not None:
            raise SystemExit("--video-out is for --out; with --out-dir use --video")
        d = args.out_dir
        return d / "mix.wav", (d / "mix.mp4" if args.video else None), (d / "parts" if args.split_parts else None)
    if args.split_parts:
        raise SystemExit("--split-parts needs --out-dir")
    if args.video:
        raise SystemExit("--video is for --out-dir; with --out use --video-out FILE")
    return args.out, args.video_out, None


def _video_notes(voice: Voice, jobs: list[NoteJob], refs: list[str | None]) -> list[VideoNote]:
    """One picture per note. A sung note holds several units (consonant + vowel): it shows the
    longest one, shifted so that unit lines up with where it sounds."""
    if not voice.lyrics:
        return [VideoNote(j.note_start, j.note_dur, ref, j.seg_start) for j, ref in zip(jobs, refs, strict=True)]
    notes: dict[int, list[int]] = {}
    for i, unit in enumerate(voice.units):
        notes.setdefault(unit.note_index, []).append(i)
    out = []
    for members in notes.values():
        first, last = voice.units[members[0]], voice.units[members[-1]]
        main = max(members, key=lambda i: voice.units[i].duration_sec)
        lead_in = voice.units[main].start_sec - first.start_sec
        out.append(VideoNote(
            first.start_sec, last.start_sec + last.duration_sec - first.start_sec,
            refs[main], max(0.0, jobs[main].seg_start - lead_in),
        ))
    return out


def render(args: argparse.Namespace) -> None:
    if args.lyrics is not None:
        raise NotImplementedError("--lyrics (plain dialogue text) is not implemented yet; use --ust for sung lyrics")
    if args.melody is None and args.ust is None:
        raise SystemExit("give --melody (MIDI), --ust (UST/USTX), or both")
    for name in ("pitch_flatten", "lyrics_pitch_flatten"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be within 0..1")
    for name in ("pitch_threshold", "lyrics_pitch_threshold"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    mix_wav, mix_mp4, parts_dir = _resolve_outputs(args)
    threshold = None if args.no_pitch_correct else args.pitch_threshold
    lyrics_threshold = None if args.no_pitch_correct else args.lyrics_pitch_threshold
    # Without correction, a wrong pitch is heard as is: make the matcher strict about it.
    weights = CostWeights(near_semitones=0.0) if threshold is None else CostWeights()
    lyrics_weights = (LyricsWeights(pitch_free_cents=0.0, pitch_step_cents=25.0) if lyrics_threshold is None
                      else LyricsWeights(pitch_free_cents=lyrics_threshold))

    conn = db.connect(args.db)
    voices: list[Voice] = []
    if args.melody is not None:
        voices += load_midi(args.melody, args.tracks)
    if args.ust is not None:
        from .ust import load_ust
        voices += load_ust(args.ust, args.ust_tracks)
    corpora = {}
    if any(not v.lyrics for v in voices):
        corpora["pitch"] = db.load_corpus(conn, "pitch")
    if any(v.lyrics for v in voices):
        corpora["phoneme"] = db.load_corpus(conn, "phoneme")
    total_sec = max(u.start_sec + u.duration_sec for v in voices for u in v.units) + TAIL_SEC
    sizes = ", ".join(f"{len(c)} {kind} segments" for kind, c in corpora.items())
    print(f"[render] corpus {sizes}, {len(voices)} voices, {total_sec:.1f} s", file=sys.stderr)
    progress.log(f"corpus {sizes}, {len(voices)} voices, {total_sec:.1f} s")

    rendered: list[RenderedVoice] = []
    plan = []
    corrected = 0
    for voice in voices:
        corpus = corpora["phoneme" if voice.lyrics else "pitch"]
        progress.stage(f"matching {voice.label} ({len(voice.units)} units)")
        if voice.lyrics:
            path = select_units_lyrics(voice.units, corpus, lyrics_weights)
        else:
            path = select_units(voice.units, corpus, weights)
        jobs, refs = [], []
        for unit, si in zip(voice.units, path, strict=True):
            sid = corpus.source_id[si]
            sustained = not voice.lyrics or is_voiced_sustained(unit.phoneme)
            job = NoteJob(
                audio_path=corpus.audio_cache[sid],
                seg_start=float(corpus.start[si]),
                seg_end=float(corpus.end[si]),
                seg_f0=float(corpus.f0[si]),
                target_f0=unit.target_f0_hz,
                note_start=unit.start_sec,
                note_dur=unit.duration_sec,
                velocity=unit.velocity,
                flatten=args.lyrics_pitch_flatten if voice.lyrics else args.pitch_flatten,
                # Consonants are left alone: their pitch is not what is heard.
                threshold_cents=(lyrics_threshold if voice.lyrics else threshold) if sustained else None,
                level_db=TARGET_RMS_DB if sustained else CONSONANT_LEVEL_DB,
                measure_used_f0=voice.lyrics and sustained,
                sustain_to_note=voice.lyrics and sustained and not args.no_lyrics_stretch,
            )
            jobs.append(job)
            refs.append(corpus.video_ref[si])
            entry = {
                "voice": voice.label, "index": unit.index,
                "note_start_sec": round(unit.start_sec, 4), "note_duration_sec": round(unit.duration_sec, 4),
                "target_f0_hz": round(unit.target_f0_hz, 2),
                "segment_id": int(corpus.ids[si]), "source": corpus.source_path[sid],
                "start_sec": round(job.seg_start, 4), "end_sec": round(job.seg_end, 4),
                "f0_hz": None if np.isnan(job.seg_f0) else round(job.seg_f0, 2),
                "pitch_offset_cents": None,   # filled in after synthesis
                "pitch_corrected": False,
            }
            if voice.lyrics:
                cands = [(str(p), round(float(c), 3))
                         for p, c in zip(corpus.cand_phoneme[si], corpus.cand_conf[si], strict=True) if p]
                rank = next((r + 1 for r, (p, _) in enumerate(cands) if p == unit.phoneme), None)
                entry.update({"phoneme": unit.phoneme, "matched_rank": rank, "segment_candidates": cands})
            plan.append(entry)
        print(f"  synthesizing {voice.label} ({len(jobs)} {'phonemes' if voice.lyrics else 'notes'})",
              file=sys.stderr)
        progress.stage(f"synthesizing {voice.label}", len(jobs))
        audio, synth_info = render_voice(jobs, total_sec, args.workers)
        for entry, job, (ref_f0, is_corrected, ratio) in zip(plan[len(plan) - len(jobs):], jobs, synth_info,
                                                             strict=True):
            corrected += is_corrected
            entry["pitch_offset_cents"] = (None if np.isnan(ref_f0)
                                           else round(float(1200 * np.log2(job.target_f0 / ref_f0)), 1))
            entry["pitch_corrected"] = bool(is_corrected)
            if job.measure_used_f0:
                entry["used_f0_hz"] = None if np.isnan(ref_f0) else round(ref_f0, 2)
            if voice.lyrics:
                entry["stretch_ratio"] = round(ratio, 2)
        if voice.gain != 1.0:
            audio = audio * np.float32(voice.gain)
        rendered.append(RenderedVoice(voice, audio, _video_notes(voice, jobs, refs)))
    print(f"  pitch corrected {corrected}/{len(plan)} units "
          f"({'off' if threshold is None else f'threshold {threshold:g} cents'})", file=sys.stderr)
    if any(v.lyrics for v in voices) and lyrics_threshold is not None:
        print(f"  lyrics: pitch threshold {lyrics_threshold:g} cents, flatten {args.lyrics_pitch_flatten:g}",
              file=sys.stderr)
    if any(v.lyrics for v in voices):
        sung = [e for e in plan if "phoneme" in e]
        ranks = [e["matched_rank"] for e in sung]
        print(f"  phoneme match: 1-best {ranks.count(1)}, 2nd {ranks.count(2)}, 3rd {ranks.count(3)}, "
              f"none {ranks.count(None)} (of {len(sung)})", file=sys.stderr)

    progress.stage("mixing and writing audio")
    _apply_gains(rendered, args)
    mix = sum_tracks([r.audio for r in rendered])
    gain = normalize_gain(mix)
    mix_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mix_wav, mix * gain, SR, subtype="PCM_16")
    print(f"  wrote {mix_wav}", file=sys.stderr)

    plan_path = args.plan or (args.out_dir / "plan.json" if args.out_dir else None)
    if plan_path:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    # Parts: one per track (its voices summed), at the mix's gain so parts add up to the mix.
    parts: list[tuple[Path, list[RenderedVoice]]] = []
    if parts_dir is not None:
        for key in sorted({r.track_key for r in rendered}, key=lambda k: (k[0] != "midi", k[1])):
            members = [r for r in rendered if r.track_key == key]
            prefix = "ust" if key[0] == "ust" else ""
            stem = parts_dir / f"{prefix}{key[1]:02d}_{_safe_name(members[0].voice.track_name)}"
            parts.append((stem, members))
            parts_dir.mkdir(parents=True, exist_ok=True)
            sf.write(stem.with_suffix(".wav"), sum_tracks([r.audio for r in members]) * gain, SR,
                     subtype="PCM_16")
            print(f"  wrote {stem.with_suffix('.wav')}", file=sys.stderr)

    # Groups: accompaniment only / vocals only, when a render has both (again at the mix's gain).
    groups: list[tuple[Path, list[RenderedVoice]]] = []
    if args.out_dir is not None and any(r.voice.lyrics for r in rendered) and any(not r.voice.lyrics for r in rendered):
        for name, lyrics in (("accompaniment", False), ("vocals", True)):
            members = [r for r in rendered if r.voice.lyrics == lyrics]
            stem = args.out_dir / name
            groups.append((stem, members))
            sf.write(stem.with_suffix(".wav"), sum_tracks([r.audio for r in members]) * gain, SR, subtype="PCM_16")
            print(f"  wrote {stem.with_suffix('.wav')}", file=sys.stderr)

    videos: list[tuple[list[RenderedVoice], Path, Path]] = []
    if mix_mp4 is not None:
        videos.append((_video_priority(rendered, args.video_track), mix_wav, mix_mp4))
    if groups and args.video:
        for stem, members in groups:
            wanted = args.video_track
            if wanted is not None and not any(_track_matches(r.voice, wanted) for r in members):
                wanted = None
            videos.append((_video_priority(members, wanted), stem.with_suffix(".wav"), stem.with_suffix(".mp4")))
    if parts and args.video:
        for stem, members in parts:
            members = sorted(members, key=lambda r: r.voice.voice_index)
            videos.append((members, stem.with_suffix(".wav"), stem.with_suffix(".mp4")))
    if not videos:
        return
    if all(n.video_ref is None for r in rendered for n in r.video_notes):
        raise SystemExit("video output: the chosen segments have no video source")

    cache = Path(tempfile.mkdtemp(prefix="madgen-clips-", dir=mix_wav.parent))
    try:
        for ordered, audio, out in videos:
            print(f"  video {out.name} follows {' > '.join(r.voice.label for r in ordered)}", file=sys.stderr)
            spans = build_spans([r.video_notes for r in ordered], total_sec, FPS)
            render_video(spans, audio, out, cache, fps=FPS)
            print(f"  wrote {out}", file=sys.stderr)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
