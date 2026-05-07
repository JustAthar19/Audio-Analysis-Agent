from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Optional

from models.report import FileMetadata

logger = logging.getLogger(__name__)

LOSSLESS_CODECS = {
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
    "flac",
}

_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<start>[\d.]+)")
_SILENCE_END_LINE_RE = re.compile(
    r"silence_end:\s*(?P<end>[\d.]+)(?:\s*\|\s*silence_duration:\s*(?P<dur1>[\d.]+))?"
)
_SILENCE_DUR_STANDALONE_RE = re.compile(r"silence_duration:\s*(?P<dur2>[\d.]+)")

_MEAN_VOL_RE = re.compile(r"mean_volume:\s*(?P<mean>[-\d.]+)\s*dB")
_MAX_VOL_RE = re.compile(r"max_volume:\s*(?P<max>[-\d.]+)\s*dB")
_HIST_RE = re.compile(r"histogram_(?P<hist>\d+)db:\s*(?P<count>\d+)")

_PTS_TIME_RE = re.compile(r"pts_time:\s*(?P<pts>[\d.]+)")
_PEAK_LEVEL_RE = re.compile(
    r"Peak_level:\s*(?P<peak>[-\d.]+)\s*dB|"
    r"lavfi\.astats\.Overall\.Peak_level\s*=\s*(?P<peak2>[-\d.]+)|"
    r"lavfi\.astats\.Overall\.Peak_level\s+(?P<peak3>[-\d.]+)\s+dB"
)


def _parse_astats_peak_db(raw: str) -> Optional[float]:
    """Peak_level from astats/ametadata; ffmpeg may print '-' when unknown."""
    s = raw.strip()
    if not s or s == "-":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v

_RMS_LINE_RE = re.compile(
    r"Overall\.RMS_level:\s*(?P<rms>[-\d.]+)\s*dB|rms_level:\s*(?P<rms2>[-\d.]+)\s*dB"
)


def extract_metadata(filepath: str, *, timeout: float = 120.0) -> FileMetadata:
    """
    Run: ffprobe -v quiet -print_format json -show_format -show_streams <file>
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffprobe failed")

    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise RuntimeError("No audio stream found in file")
    stream = audio_streams[0]

    fmt = data.get("format") or {}
    codec = stream.get("codec_name") or "unknown"

    duration_raw = fmt.get("duration")
    if duration_raw is None:
        duration_sec = float(stream.get("duration") or 0.0)
    else:
        duration_sec = float(duration_raw)

    sample_rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)

    bit_depth: Optional[int] = None
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        if stream.get(key) is not None:
            bit_depth = int(stream[key])
            break

    size_raw = fmt.get("size")
    file_size_mb = float(size_raw) / (1024**2) if size_raw is not None else 0.0

    format_name = fmt.get("format_long_name") or fmt.get("format_name") or "unknown"

    path = Path(filepath)
    return FileMetadata(
        file_name=path.name,
        file_path=str(path.resolve()),
        file_size_mb=file_size_mb,
        duration_sec=duration_sec,
        codec=codec,
        sample_rate_hz=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        is_lossless=codec in LOSSLESS_CODECS,
        format_name=format_name,
    )


def run_silence_detection(
    filepath: str,
    noise_threshold_db: float,
    min_duration_sec: float,
    *,
    file_duration_sec: Optional[float] = None,
    timeout: float = 120.0,
) -> list[dict]:
    """
    Run: ffmpeg -i <file> -af "silencedetect=noise=<db>dB:duration=<sec>" -f null - 2>&1
    """
    af = f"silencedetect=noise={noise_threshold_db}dB:duration={min_duration_sec}"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        filepath,
        "-af",
        af,
        "-f",
        "null",
        "-",
    ]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "silencedetect failed")

    text = result.stderr or ""
    segments: list[dict] = []
    current_start: Optional[float] = None
    last_end_line_had_duration = False

    for line in text.splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            current_start = float(m_start.group("start"))
            last_end_line_had_duration = False
            continue

        m_end = _SILENCE_END_LINE_RE.search(line)
        if m_end and current_start is not None:
            end = float(m_end.group("end"))
            dur_s = m_end.group("dur1")
            if dur_s is not None:
                duration = float(dur_s)
                last_end_line_had_duration = True
            else:
                m_d2 = _SILENCE_DUR_STANDALONE_RE.search(line)
                duration = float(m_d2.group("dur2")) if m_d2 else max(0.0, end - current_start)
                last_end_line_had_duration = m_d2 is not None
            segments.append(
                {
                    "start_sec": current_start,
                    "end_sec": end,
                    "duration_sec": duration,
                }
            )
            current_start = None
            continue

        m_dur = _SILENCE_DUR_STANDALONE_RE.search(line)
        if m_dur and current_start is not None and not last_end_line_had_duration:
            # Some builds emit silence_duration on the next line only
            last_end_line_had_duration = True

    if current_start is not None and file_duration_sec is not None:
        end = file_duration_sec
        segments.append(
            {
                "start_sec": current_start,
                "end_sec": end,
                "duration_sec": max(0.0, end - current_start),
            }
        )

    return segments


def run_volume_detect(filepath: str, *, timeout: float = 120.0) -> dict:
    """
    Run: ffmpeg -i <file> -af "volumedetect" -f null - 2>&1
    """
    cmd = ["ffmpeg", "-nostdin", "-i", filepath, "-af", "volumedetect", "-f", "null", "-"]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "volumedetect failed")

    text = result.stderr or ""
    mean_db = 0.0
    peak_db = 0.0
    histogram_0db = 0

    mm = _MEAN_VOL_RE.search(text)
    if mm:
        mean_db = float(mm.group("mean"))
    xm = _MAX_VOL_RE.search(text)
    if xm:
        peak_db = float(xm.group("max"))

    for hm in _HIST_RE.finditer(text):
        label = hm.group("hist")
        count = int(hm.group("count"))
        if label == "0":
            histogram_0db = count
            break

    return {
        "mean_db": mean_db,
        "peak_db": peak_db,
        "histogram_0db": histogram_0db,
    }


def run_astats_clipping(
    filepath: str,
    window_sec: float,
    clip_threshold_db: float,
    *,
    timeout: float = 120.0,
) -> list[dict]:
    """
    Run per-frame peak detection:
    ffmpeg -i <file> -af "astats=metadata=1:reset=<window>,
    ametadata=print:key=lavfi.astats.Overall.Peak_level" -f null - 2>&1
    """
    stats = f"astats=metadata=1:reset={window_sec}"
    meta = "ametadata=print:key=lavfi.astats.Overall.Peak_level"
    af = f"{stats},{meta}"
    cmd = ["ffmpeg", "-nostdin", "-i", filepath, "-af", af, "-f", "null", "-"]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "astats clipping failed")

    text = result.stderr or ""
    frames: list[tuple[float, float]] = []
    current_pts: Optional[float] = None

    for line in text.splitlines():
        pt = _PTS_TIME_RE.search(line)
        if pt:
            current_pts = float(pt.group("pts"))
        pk = _PEAK_LEVEL_RE.search(line)
        if pk and current_pts is not None:
            peak_s = None
            for key in ("peak", "peak2", "peak3"):
                try:
                    g = pk.group(key)
                except IndexError:
                    continue
                if g is not None:
                    peak_s = g
                    break
            if peak_s is None:
                continue
            peak_val = _parse_astats_peak_db(peak_s)
            if peak_val is None:
                continue
            frames.append((current_pts, peak_val))

    clipping_frames = [(t, p) for t, p in frames if p >= clip_threshold_db]
    if not clipping_frames:
        return []

    segments: list[dict] = []
    seg_start = clipping_frames[0][0]
    seg_end = clipping_frames[0][0]
    seg_peak = clipping_frames[0][1]

    for i in range(1, len(clipping_frames)):
        t, p = clipping_frames[i]
        prev_t, _ = clipping_frames[i - 1]
        if t - prev_t <= window_sec * 2 + 0.05:
            seg_end = t
            seg_peak = max(seg_peak, p)
        else:
            segments.append(
                {
                    "start_sec": seg_start,
                    "end_sec": seg_end,
                    "peak_db": seg_peak,
                }
            )
            seg_start = t
            seg_end = t
            seg_peak = p

    segments.append(
        {
            "start_sec": seg_start,
            "end_sec": seg_end,
            "peak_db": seg_peak,
        }
    )
    return segments


def run_band_volume(
    filepath: str,
    low_hz: int,
    high_hz: int,
    *,
    timeout: float = 120.0,
) -> Optional[float]:
    """
    Measure mean volume in a frequency band using highpass + lowpass + volumedetect.
    """
    af = f"highpass=f={low_hz},lowpass=f={high_hz},volumedetect"
    cmd = ["ffmpeg", "-nostdin", "-i", filepath, "-af", af, "-f", "null", "-"]
    logger.debug("subprocess: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("run_band_volume timed out for %s", filepath)
        return None

    if result.returncode != 0:
        logger.warning("run_band_volume failed: %s", (result.stderr or "")[:200])
        return None

    text = result.stderr or ""
    mm = _MEAN_VOL_RE.search(text)
    if not mm:
        return None
    return float(mm.group("mean"))


def run_astats_rms_samples(
    filepath: str,
    *,
    window_sec: float = 0.5,
    timeout: float = 120.0,
) -> list[float]:
    """
    Collect per-window RMS level (dB) via astats + ametadata for fallback noise floor.
    """
    stats = f"astats=metadata=1:reset={window_sec}"
    meta = "ametadata=print:key=lavfi.astats.Overall.RMS_level"
    af = f"{stats},{meta}"
    cmd = ["ffmpeg", "-nostdin", "-i", filepath, "-af", af, "-f", "null", "-"]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "astats RMS failed")

    text = result.stderr or ""
    rms_values: list[float] = []
    for line in text.splitlines():
        m = _RMS_LINE_RE.search(line)
        if m:
            val = m.group("rms") or m.group("rms2")
            if val:
                rms_values.append(float(val))
    return rms_values


def sample_window_mean_db(
    filepath: str,
    start_sec: float,
    duration_sec: float,
    *,
    timeout: float = 60.0,
) -> Optional[float]:
    """
    Run volumedetect on an extracted segment [start_sec, start_sec + duration_sec).
    """
    if duration_sec <= 0:
        return None
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-ss",
        str(start_sec),
        "-i",
        filepath,
        "-t",
        str(duration_sec),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    logger.debug("subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return None
    text = result.stderr or ""
    mm = _MEAN_VOL_RE.search(text)
    if not mm:
        return None
    return float(mm.group("mean"))
