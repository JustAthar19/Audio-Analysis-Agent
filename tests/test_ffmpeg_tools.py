"""
Tests for ffmpeg_tools.py.
Use pytest-mock to patch subprocess.run — never call real ffmpeg in tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tools import ffmpeg_tools as ft


WAV_JSON = {
    "format": {
        "duration": "120.5",
        "size": "1048576",
        "format_name": "wav",
        "format_long_name": "WAV / WAVE (Waveform Audio)",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "mjpeg",
        },
        {
            "codec_type": "audio",
            "codec_name": "pcm_s16le",
            "sample_rate": "48000",
            "channels": "2",
            "bits_per_raw_sample": "16",
        },
    ],
}

MP3_JSON = {
    "format": {
        "duration": "60.0",
        "size": "500000",
        "format_name": "mp3",
    },
    "streams": [
        {
            "codec_type": "audio",
            "codec_name": "mp3",
            "sample_rate": "44100",
            "channels": "1",
        }
    ],
}


def _mock_run(mocker, returncode: int, stdout: str = "", stderr: str = ""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return mocker.patch("subprocess.run", return_value=m)


def test_extract_metadata_wav_picks_audio_stream(mocker):
    m = _mock_run(mocker, 0, stdout=json.dumps(WAV_JSON), stderr="")
    meta = ft.extract_metadata("/tmp/x.wav")
    assert m.call_count == 1
    assert meta.codec == "pcm_s16le"
    assert meta.is_lossless is True
    assert meta.sample_rate_hz == 48000
    assert abs(meta.file_size_mb - 1.0) < 0.01
    assert meta.bit_depth == 16


def test_extract_metadata_mp3_missing_bit_depth(mocker):
    m = _mock_run(mocker, 0, stdout=json.dumps(MP3_JSON), stderr="")
    meta = ft.extract_metadata("/a/b/file.mp3")
    assert m.call_count == 1
    assert meta.codec == "mp3"
    assert meta.is_lossless is False
    assert meta.bit_depth is None


def test_extract_metadata_runtime_error(mocker):
    _mock_run(mocker, 1, stdout="", stderr="boom")
    with pytest.raises(RuntimeError, match="boom"):
        ft.extract_metadata("/nope")


def test_run_silence_detection_parses_segments(mocker):
    stderr = """
    silence_start: 1.0
    silence_end: 3.0 | silence_duration: 2.0
    silence_start: 10.0
    silence_end: 12.0 | silence_duration: 2.0
    """
    _mock_run(mocker, 0, stdout="", stderr=stderr)
    segs = ft.run_silence_detection("/f", -40, 0.5, file_duration_sec=100.0)
    assert len(segs) == 2
    assert segs[0]["start_sec"] == 1.0
    assert segs[0]["end_sec"] == 3.0


def test_run_silence_detection_eof_silence(mocker):
    stderr = "silence_start: 50.0\n"
    _mock_run(mocker, 0, stdout="", stderr=stderr)
    segs = ft.run_silence_detection("/f", -40, 0.5, file_duration_sec=60.0)
    assert len(segs) == 1
    assert segs[0]["end_sec"] == 60.0


def test_run_silence_detection_fails(mocker):
    _mock_run(mocker, 1, stderr="bad")
    with pytest.raises(RuntimeError, match="bad"):
        ft.run_silence_detection("/f", -40, 0.5)


def test_run_volume_detect_parses_histogram(mocker):
    stderr = """
    mean_volume: -20.0 dB
    max_volume: -1.0 dB
    histogram_0db: 42
    """
    _mock_run(mocker, 0, stderr=stderr)
    out = ft.run_volume_detect("/f")
    assert out["mean_db"] == -20.0
    assert out["peak_db"] == -1.0
    assert out["histogram_0db"] == 42


def test_run_volume_detect_histogram_default_zero(mocker):
    stderr = """
    mean_volume: -30.0 dB
    max_volume: -6.0 dB
    """
    _mock_run(mocker, 0, stderr=stderr)
    out = ft.run_volume_detect("/f")
    assert out["histogram_0db"] == 0


def test_run_astats_clipping_groups_segments(mocker):
    stderr = """
    pts_time: 1.0
    lavfi.astats.Overall.Peak_level=-0.5 dB
    pts_time: 1.05
    lavfi.astats.Overall.Peak_level=-0.4 dB
    pts_time: 5.0
    lavfi.astats.Overall.Peak_level=-12.0 dB
    pts_time: 5.05
    lavfi.astats.Overall.Peak_level=-0.2 dB
    """
    _mock_run(mocker, 0, stderr=stderr)
    segs = ft.run_astats_clipping("/f", window_sec=0.5, clip_threshold_db=-1.0)
    assert len(segs) == 2


def test_run_volume_detect_nonzero_code(mocker):
    _mock_run(mocker, 1, stderr="volfail")
    with pytest.raises(RuntimeError, match="volfail"):
        ft.run_volume_detect("/f")


def test_run_astats_nonzero_code(mocker):
    _mock_run(mocker, 1, stderr="astfail")
    with pytest.raises(RuntimeError, match="astfail"):
        ft.run_astats_clipping("/f", 1.0, -1.0)


def test_run_astats_rms_samples_parse(mocker):
    stderr = """
    Overall.RMS_level: -40.0 dB
    Overall.RMS_level: -41.0 dB
    """
    _mock_run(mocker, 0, stderr=stderr)
    vals = ft.run_astats_rms_samples("/f", window_sec=0.5)
    assert len(vals) == 2
    assert vals[0] == -40.0
