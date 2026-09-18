import wave

import pytest

from madgen import ffmpeg


def test_probe_japanese_filename(tmp_path):
    source = tmp_path / "テスト音声「サンプル」.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)

    assert ffmpeg.probe(source) == {"duration": 1.0, "has_video": False}


def test_run_reports_japanese_missing_filename(tmp_path):
    source = tmp_path / "存在しない音声.wav"
    with pytest.raises(RuntimeError) as error:
        ffmpeg.run(["-i", str(source), "-f", "null", "-"])
    assert "ffmpeg failed" in str(error.value)
    assert source.name in str(error.value)
