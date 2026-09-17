"""Command line: build-corpus (phase 1), render (phases 2-4), auto (both)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--melody", type=Path,
                   help="MIDI (.mid): rendered in melody mode, phonemes ignored. With --ust, the accompaniment "
                        "(leave the sung tracks out of it)")
    p.add_argument("--ust", type=Path,
                   help="UTAU .ust / OpenUtau .ustx: the sung tracks, rendered in lyrics mode "
                        "(needs `build-corpus --phonemes wav2vec2`)")
    p.add_argument("--ust-tracks", default=None,
                   help="comma-separated UST track names or numbers to use (default: all unmuted)")
    p.add_argument("--lyrics", type=Path,
                   help="plain dialogue text without melody (not implemented yet)")
    p.add_argument("--tracks", default=None,
                   help="comma-separated MIDI track names or indices to use (default: all with notes)")
    p.add_argument("--out", type=Path, default=None, help="output audio file (.wav)")
    p.add_argument("--video-out", type=Path, default=None,
                   help="with --out: also render a video (.mp4)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output folder instead of --out: writes mix.wav, plan.json (and mix.mp4 with --video)")
    p.add_argument("--video", action="store_true", help="with --out-dir: also render videos")
    p.add_argument("--split-parts", action="store_true",
                   help="with --out-dir: also write each MIDI track as parts/NN_<track>.wav (.mp4 with --video)")
    p.add_argument("--video-track", default=None,
                   help="track the mix video prefers: a track name, a MIDI track index, or ustN for a UST track "
                        "(default: the longest-sounding UST track, else a MIDI track named *main*, else the "
                        "longest-sounding); other tracks fill in while it rests")
    p.add_argument("--pitch-threshold", type=float, default=25.0, metavar="CENTS",
                   help="melody mode: correct a note's pitch only when the material is off by more than this "
                        "(default 25 cents; 0 = always correct)")
    p.add_argument("--no-pitch-correct", action="store_true",
                   help="never pitch-correct: use the material as is (matching then favours exact pitch)")
    p.add_argument("--lyrics-pitch-threshold", type=float, default=0.0, metavar="CENTS",
                   help="lyrics mode: the same for sung vowels (default 0 = always put them on the note's pitch)")
    p.add_argument("--lyrics-pitch-flatten", type=float, default=1.0,
                   help="lyrics mode: the same as --pitch-flatten for sung vowels (default 1 = flat on the note)")
    p.add_argument("--no-lyrics-stretch", action="store_true",
                   help="lyrics mode: use the vowel material as it is (up to the note length, the rest silent) "
                        "instead of fitting its voiced core to exactly the note length")
    p.add_argument("--vocal-boost", type=float, default=6.0, metavar="DB",
                   help="with both --ust and --melody: level the sung tracks this many dB above the "
                        "accompaniment, by measured loudness (default 6)")
    p.add_argument("--no-auto-balance", dest="vocal_boost", action="store_const", const=None,
                   help="do not balance vocals against the accompaniment automatically")
    p.add_argument("--ust-gain", type=float, default=0.0, metavar="DB", help="gain for all UST tracks (dB)")
    p.add_argument("--melody-gain", type=float, default=0.0, metavar="DB", help="gain for all MIDI tracks (dB)")
    p.add_argument("--gain", action="append", metavar="TRACK=DB",
                   help="gain for one track (repeatable): a track name, a MIDI track index, or ustN; "
                        "e.g. --gain Bass=-3 --gain ust0=+2")
    p.add_argument("--pitch-flatten", type=float, default=0.6,
                   help="melody mode, corrected notes: 0 = keep the source's natural pitch contour, "
                        "1 = flat on the note pitch")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--log", type=Path, default=None,
                   help="live progress log (default: <out-dir>/render.log or <out>.log)")
    p.add_argument("--plan", type=Path, default=None, help="write the chosen segments as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="madgen", description="音MAD auto generator")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-corpus", help="analyze sources into the corpus DB (diff only)")
    b.add_argument("--source", type=Path, action="append", required=True)
    b.add_argument("--db", type=Path, required=True)
    b.add_argument("--phonemes", choices=["none", "wav2vec2"], default="none",
                   help="also run the phoneme analysis lyrics mode needs (whisperX + wav2vec2 phoneme CTC; "
                        "GPU recommended, install with `uv sync --extra lyrics`)")
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--log", type=Path, default=None, help="live progress log (default: <db>.log)")

    r = sub.add_parser("render", help="match the target against the corpus and synthesize")
    r.add_argument("--db", type=Path, required=True)
    _add_render_args(r)

    a = sub.add_parser("auto", help="build-corpus + render in one go")
    a.add_argument("--source", type=Path, action="append", required=True)
    a.add_argument("--db", type=Path, default=Path("corpus.sqlite"))
    _add_render_args(a)

    return parser


def _log_path(args: argparse.Namespace) -> Path:
    if args.log is not None:
        return args.log
    if args.command == "build-corpus":
        return args.db.with_name(args.db.name + ".log")
    if getattr(args, "out_dir", None) is not None:
        return args.out_dir / "render.log"
    if getattr(args, "out", None) is not None:
        return args.out.with_name(args.out.name + ".log")
    return args.db.with_name(args.db.name + ".log")


def main(argv: list[str] | None = None) -> None:
    from .progress import progress

    args = build_parser().parse_args(argv)
    progress.open(_log_path(args))
    status = "failed"
    try:
        _run(args)
        status = "ok"
    except BaseException as e:
        progress.log(f"error: {type(e).__name__}: {e}")
        raise
    finally:
        progress.close(status)


def _run(args: argparse.Namespace) -> None:
    if args.command in ("build-corpus", "auto"):
        from .corpus import build_corpus
        # `auto` with a UST needs phonemes, so it asks for them itself.
        phonemes = getattr(args, "phonemes", None) or ("wav2vec2" if getattr(args, "ust", None) else "none")
        build_corpus(args.source, args.db, workers=args.workers, phonemes=phonemes)
    if args.command in ("render", "auto"):
        from .render import render
        render(args)
