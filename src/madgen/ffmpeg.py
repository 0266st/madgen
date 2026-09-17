"""ffmpeg wrapper. Uses the static binary shipped by imageio-ffmpeg, so no system install is needed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str]) -> None:
    """Run ffmpeg with the given arguments, raising on failure."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr.strip()}")


def probe(path: Path) -> dict:
    """Return {'duration': sec, 'has_video': bool} for a media file.

    imageio-ffmpeg ships no ffprobe, so the information is parsed out of ffmpeg's
    own report on the input.
    """
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    text = proc.stderr
    duration = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            duration = int(h) * 3600 + int(m) * 60 + float(s)
    has_video = any(
        "Video:" in line and "Stream #" in line for line in text.splitlines()
    )
    return {"duration": duration, "has_video": has_video}


def extract_audio(src: Path, dst: Path, sample_rate: int) -> None:
    """Decode the audio track of `src` to a mono PCM wav at `sample_rate`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(dst),
        ]
    )


def extract_clip(src: Path, start: float, frames: int, dst: Path, width: int, height: int, fps: float) -> None:
    """Cut a silent video clip out of `src`, normalized to one size and frame rate.

    Normalizing every clip is what lets them be concatenated without re-encoding
    the whole timeline later.
    """
    run(
        [
            "-ss", f"{start:.4f}",
            "-i", str(src),
            "-an",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},"
                   # near the end of a source there may be too few frames: hold the last one
                   f"tpad=stop_mode=clone:stop=-1",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-frames:v", str(frames),
            str(dst),
        ]
    )


def make_black_clip(frames: int, dst: Path, width: int, height: int, fps: float) -> None:
    """Render a black clip -- used for the rests, where the target has no note."""
    run(
        [
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
            "-frames:v", str(frames),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(dst),
        ]
    )


def concat_with_audio(clips: list[Path], audio: Path, dst: Path, list_file: Path) -> None:
    """Concat the clips (demuxer, no re-encode) and mux the finished audio onto them."""
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in clips), encoding="utf-8"
    )
    run(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-i", str(audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(dst),
        ]
    )
