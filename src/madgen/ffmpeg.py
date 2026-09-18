"""ffmpeg wrapper. Uses the static binary shipped by imageio-ffmpeg, so no system install is needed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str]) -> None:
    """Run ffmpeg with the given arguments, raising on failure."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {' '.join(args)}\n"
                           f"{proc.stderr.strip() or '(no stderr; killed by a signal?)'}")


def probe(path: Path) -> dict:
    """Return {'duration': sec, 'has_video': bool} for a media file.

    imageio-ffmpeg ships no ffprobe, so the information is parsed out of ffmpeg's
    own report on the input.
    """
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def extract_clip(src: Path, start: float, frames: int, dst: Path, width: int, height: int, fps: float,
                 pad_color: str = "black") -> None:
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
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:{pad_color},fps={fps},"
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


def make_color_clip(frames: int, dst: Path, width: int, height: int, fps: float,
                    color: str = "black") -> None:
    """A solid-colour clip: the rests of a layer, and the black/white frames of a layer's mask."""
    run(
        [
            "-f", "lavfi",
            "-i", f"color=c={color}:s={width}x{height}:r={fps}",
            "-frames:v", str(frames),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(dst),
        ]
    )


def write_concat_list(clips: list[Path], list_file: Path) -> None:
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in clips), encoding="utf-8")


def concat_with_audio(clips: list[Path], audio: Path, dst: Path, list_file: Path) -> None:
    """Concat the clips (demuxer, no re-encode) and mux the finished audio onto them."""
    write_concat_list(clips, list_file)
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


def sample_frames(src: Path, count: int, duration: float, width: int = 64) -> np.ndarray:
    """`count` frames spread over the file, as an (n, h, w, 3) uint8 array (for picking a key colour)."""
    height = width * 9 // 16
    frames = []
    for i in range(count):
        at = duration * (i + 0.5) / count
        proc = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
             "-ss", f"{at:.3f}", "-i", str(src), "-frames:v", "1",
             "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
        )
        if len(proc.stdout) == width * height * 3:
            frames.append(np.frombuffer(proc.stdout, np.uint8).reshape(height, width, 3))
    return np.stack(frames) if frames else np.zeros((0, height, width, 3), np.uint8)


def concat_input(list_file: Path) -> list[str]:
    """Input arguments that read a concat list as one stream. Decoding through the concat demuxer
    gives continuous timestamps; concatenating to a file with -c copy does not, and the composite
    then holds single frames for seconds at a time."""
    return ["-f", "concat", "-safe", "0", "-i", str(list_file)]


def compose_layers(base: Path, layers: list[dict], dst: Path, key_color: str, work_dir: Path,
                   similarity: float = 0.12, fps: int = 30, frames: int | None = None) -> None:
    """Stack `layers` onto the `base` plane (no audio anywhere) and encode the result.

    `base` and each layer's "video" are concat lists of clips; decoding through the concat demuxer
    keeps timestamps continuous (concatenating to a file with -c copy does not, and the composite
    then holds single frames for seconds). Wherever a layer shows the key colour it is transparent,
    so its rests and the padding beside its clips let the layers below through.

    The layers are composed one at a time rather than in a single graph with every input at once:
    with many concat inputs ffmpeg stalls some of them and duplicates frames into the output, which
    looks like the video freezing for tens of seconds (and where it freezes varies per run).
    """
    stage_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p"]
    final_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    limit = ["-frames:v", str(frames)] if frames is not None else []
    if not layers:
        run(concat_input(base) + final_args + limit + [str(dst)])
        return

    current: list[str] = concat_input(base)
    stage: Path | None = None
    for i, layer in enumerate(layers):
        last = i == len(layers) - 1
        out = dst if last else work_dir / f"{dst.stem}-stage{i}.mp4"
        run(current + concat_input(layer["video"]) + [
            "-filter_complex",
            f"[1:v]colorkey={key_color}:{similarity}:0.05[key];"
            f"[0:v][key]overlay={layer['x']}:{layer['y']}:eof_action=pass[out]",
            "-map", "[out]",
            *(final_args if last else stage_args),
            *limit,
            str(out),
        ])
        if stage is not None:
            stage.unlink(missing_ok=True)
        stage, current = (None, []) if last else (out, ["-i", str(out)])


def mux_audio(video: Path, audio: Path, dst: Path) -> None:
    """Put `audio` on `video` without touching the picture."""
    run(["-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", str(dst)])
