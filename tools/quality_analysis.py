from __future__ import annotations

import logging

from pydantic import BaseModel

from models.report import (
    Category,
    ClippingSegment,
    Issue,
    NoiseMetrics,
    Severity,
    SilenceMetrics,
    SilenceSegment,
    VolumeMetrics,
)

from tools.ffmpeg_tools import (
    run_astats_clipping,
    run_astats_rms_samples,
    run_band_volume,
    run_silence_detection,
    run_volume_detect,
    sample_window_mean_db,
)

logger = logging.getLogger(__name__)


class SilenceReport(BaseModel):
    """Structured silence detection output."""

    metrics: SilenceMetrics
    issues: list[Issue]


class VolumeReport(BaseModel):
    """Structured volume and clipping analysis."""

    metrics: VolumeMetrics
    issues: list[Issue]


class NoiseProfile(BaseModel):
    """Structured SNR / noise-floor analysis."""

    metrics: NoiseMetrics
    issues: list[Issue]


def _percentile_5(values: list[float]) -> float:
    if not values:
        return -96.0
    sorted_v = sorted(values)
    k = max(0, int(round(0.05 * (len(sorted_v) - 1))))
    return sorted_v[k]


def _clipping_severity(
    histogram_0db: int,
    segments: list[ClippingSegment],
) -> str:
    if histogram_0db == 0 and not segments:
        return "none"
    if histogram_0db < 100 and len(segments) <= 1:
        return "minor"
    if histogram_0db < 5000 or len(segments) <= 3:
        return "moderate"
    return "severe"


def _volume_issues(
    metrics: VolumeMetrics,
    total_duration_sec: float,
    *,
    detector: str = "volume",
) -> list[Issue]:
    issues: list[Issue] = []
    sev = metrics.clipping_severity
    if sev == "severe":
        issues.append(
            Issue(
                severity=Severity.critical,
                category=Category.clipping,
                message="Severe clipping detected; audio may be distorted or inadmissible.",
                detector=detector,
            )
        )
    elif sev == "moderate":
        for seg in metrics.clipping_segments[:5]:
            issues.append(
                Issue(
                    severity=Severity.high,
                    category=Category.clipping,
                    message=f"Moderate clipping between {seg.start_sec:.2f}s and {seg.end_sec:.2f}s "
                    f"(peak {seg.peak_db:.1f} dB).",
                    detector=detector,
                    timestamp_sec=seg.start_sec,
                )
            )
    if metrics.mean_db < -30.0:
        issues.append(
            Issue(
                severity=Severity.medium,
                category=Category.volume,
                message=f"Mean level is low ({metrics.mean_db:.1f} dB); speech intelligibility may suffer.",
                detector=detector,
            )
        )
    if metrics.headroom_db < 1.0:
        issues.append(
            Issue(
                severity=Severity.low,
                category=Category.volume,
                message=f"Low headroom ({metrics.headroom_db:.1f} dB); signal is very hot.",
                detector=detector,
            )
        )
    if metrics.dynamic_range_db < 10.0:
        issues.append(
            Issue(
                severity=Severity.low,
                category=Category.integrity,
                message=f"Low dynamic range ({metrics.dynamic_range_db:.1f} dB); possible processing or tampering.",
                detector=detector,
            )
        )
    return issues


def build_volume_metrics_from_pass1(
    mean_db: float,
    peak_db: float,
    histogram_0db: int,
    clipping_segments: list[ClippingSegment],
) -> VolumeMetrics:
    headroom_db = 0.0 - peak_db
    dynamic_range_db = abs(peak_db - mean_db)
    sev = _clipping_severity(histogram_0db, clipping_segments)
    clip_detected = sev != "none"
    return VolumeMetrics(
        mean_db=mean_db,
        peak_db=peak_db,
        headroom_db=headroom_db,
        histogram_0db=histogram_0db,
        clipping_detected=clip_detected,
        clipping_severity=sev,
        clipping_segments=clipping_segments,
        dynamic_range_db=dynamic_range_db,
    )


def analyse_volume_and_clipping(
    filepath: str,
    total_duration_sec: float,
    *,
    window_sec: float = 1.0,
    clip_threshold_db: float = -1.0,
    force_astats: bool = False,
) -> VolumeReport:
    """
    Two-pass analysis:
    Pass 1: run_volume_detect — mean_db, peak_db, histogram_0db
    Pass 2: if peak_db >= -1.0 or histogram_0db > 0 or force_astats, run run_astats_clipping
    """
    vd = run_volume_detect(filepath)
    mean_db = float(vd["mean_db"])
    peak_db = float(vd["peak_db"])
    histogram_0db = int(vd["histogram_0db"])

    segments: list[ClippingSegment] = []
    run_second = force_astats or peak_db >= -1.0 or histogram_0db > 0
    if run_second:
        raw = run_astats_clipping(filepath, window_sec, clip_threshold_db)
        segments = [
            ClippingSegment(start_sec=r["start_sec"], end_sec=r["end_sec"], peak_db=r["peak_db"])
            for r in raw
        ]

    metrics = build_volume_metrics_from_pass1(mean_db, peak_db, histogram_0db, segments)
    issues = _volume_issues(metrics, total_duration_sec)
    return VolumeReport(metrics=metrics, issues=issues)


def merge_astats_clipping(
    base: VolumeMetrics,
    filepath: str,
    total_duration_sec: float,
    *,
    window_sec: float = 1.0,
    clip_threshold_db: float = -1.0,
) -> VolumeReport:
    """Run astats-only pass and merge clipping segments into existing volume metrics."""
    raw = run_astats_clipping(filepath, window_sec, clip_threshold_db)
    segments = [
        ClippingSegment(start_sec=r["start_sec"], end_sec=r["end_sec"], peak_db=r["peak_db"])
        for r in raw
    ]
    metrics = build_volume_metrics_from_pass1(
        base.mean_db,
        base.peak_db,
        base.histogram_0db,
        segments,
    )
    issues = _volume_issues(metrics, total_duration_sec, detector="clipping_astats")
    return VolumeReport(metrics=metrics, issues=issues)


def detect_silence(
    filepath: str,
    total_duration_sec: float,
    noise_threshold_db: float = -40.0,
    min_duration_sec: float = 2.0,
) -> SilenceReport:
    """
    Wraps run_silence_detection. Computes silence_ratio, longest_gap, and issues.
    """
    raw_segments = run_silence_detection(
        filepath,
        noise_threshold_db,
        min_duration_sec,
        file_duration_sec=total_duration_sec if total_duration_sec > 0 else None,
    )
    segs = [
        SilenceSegment(
            start_sec=s["start_sec"],
            end_sec=s["end_sec"],
            duration_sec=s["duration_sec"],
        )
        for s in raw_segments
    ]
    total_silence = sum(s.duration_sec for s in segs)
    ratio = (total_silence / total_duration_sec) if total_duration_sec > 0 else 0.0
    longest = max((s.duration_sec for s in segs), default=0.0)

    metrics = SilenceMetrics(
        ratio=ratio,
        total_sec=total_silence,
        longest_gap_sec=longest,
        segment_count=len(segs),
        segments=segs,
        threshold_db=noise_threshold_db,
        min_duration_sec=min_duration_sec,
    )

    issues: list[Issue] = []
    if ratio > 0.30:
        issues.append(
            Issue(
                severity=Severity.medium,
                category=Category.silence,
                message=f"High silence ratio ({ratio:.0%}); verify proceedings coverage.",
                detector="silence",
            )
        )
    for s in segs:
        if s.duration_sec > 300.0:
            issues.append(
                Issue(
                    severity=Severity.high,
                    category=Category.silence,
                    message=f"Very long silence gap ({s.duration_sec:.0f}s starting at {s.start_sec:.1f}s).",
                    detector="silence",
                    timestamp_sec=s.start_sec,
                )
            )
        elif s.duration_sec > 60.0:
            issues.append(
                Issue(
                    severity=Severity.medium,
                    category=Category.silence,
                    message=f"Long silence gap ({s.duration_sec:.0f}s starting at {s.start_sec:.1f}s).",
                    detector="silence",
                    timestamp_sec=s.start_sec,
                )
            )

    return SilenceReport(metrics=metrics, issues=issues)


def _snr_quality_label(snr_db: float) -> str:
    if snr_db >= 40.0:
        return "excellent"
    if snr_db >= 25.0:
        return "good"
    if snr_db >= 15.0:
        return "fair"
    if snr_db >= 8.0:
        return "poor"
    return "unusable"


def analyse_snr(
    filepath: str,
    mean_signal_db: float,
    silence_segments: list[dict],
    run_spectral: bool = True,
) -> NoiseProfile:
    """
    Estimate SNR using silence-window RMS samples, with astats RMS percentile fallback.
    Optional spectral tilt for noise typing.
    """
    per_window_db: list[float] = []
    measured_from_n = 0

    for seg in silence_segments:
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", start))
        inner_start = start + 0.5
        inner_end = end - 0.5
        usable = max(0.0, inner_end - inner_start)
        dur = min(usable, 10.0)
        if dur <= 0.05:
            continue
        mdb = sample_window_mean_db(filepath, inner_start, dur)
        if mdb is not None:
            per_window_db.append(mdb)
            measured_from_n += 1

    if per_window_db:
        noise_floor_db = sum(per_window_db) / len(per_window_db)
    else:
        try:
            rms_frames = run_astats_rms_samples(filepath)
        except RuntimeError as exc:
            logger.warning("astats RMS fallback failed: %s", exc)
            rms_frames = []
        noise_floor_db = _percentile_5(rms_frames) if rms_frames else -96.0
        per_window_db = rms_frames[:]
        measured_from_n = 0

    snr_db = mean_signal_db - noise_floor_db
    snr_quality = _snr_quality_label(snr_db)

    spectral_tilt: float | None = None
    noise_type = "unknown"
    if run_spectral:
        low_band = run_band_volume(filepath, 20, 500)
        high_band = run_band_volume(filepath, 2000, 8000)
        if low_band is not None and high_band is not None:
            spectral_tilt = high_band - low_band
            if spectral_tilt < -6.0:
                noise_type = "hum"
            elif spectral_tilt > 8.0:
                noise_type = "hiss"
            else:
                noise_type = "broadband"
        else:
            noise_type = "unknown"

    issues: list[Issue] = []
    if snr_quality == "unusable":
        issues.append(
            Issue(
                severity=Severity.critical,
                category=Category.noise,
                message=f"Noise floor dominates signal (SNR ~{snr_db:.1f} dB).",
                detector="snr",
            )
        )
    elif snr_quality == "poor":
        issues.append(
            Issue(
                severity=Severity.high,
                category=Category.noise,
                message=f"Poor SNR (~{snr_db:.1f} dB); transcription reliability may be impaired.",
                detector="snr",
            )
        )
    if measured_from_n == 0:
        issues.append(
            Issue(
                severity=Severity.low,
                category=Category.noise,
                message="Noise floor estimated without dedicated silence windows (lower confidence).",
                detector="snr",
            )
        )

    metrics = NoiseMetrics(
        floor_db=noise_floor_db,
        snr_db=snr_db,
        snr_quality=snr_quality,
        noise_type=noise_type,
        spectral_tilt_db=spectral_tilt,
        measured_from_n=measured_from_n,
        per_window_db=per_window_db[:100],
    )
    return NoiseProfile(metrics=metrics, issues=issues)


def clipping_overlaps_silence(
    clipping_segments: list[ClippingSegment],
    silence_segments: list[SilenceSegment],
) -> bool:
    """True if any clipping segment starts inside a silence interval."""
    for c in clipping_segments:
        for s in silence_segments:
            if s.start_sec <= c.start_sec <= s.end_sec:
                return True
    return False


def chain_of_custody_issue() -> Issue:
    return Issue(
        severity=Severity.high,
        category=Category.integrity,
        message=(
            "Chain-of-custody concern: clipping begins during a labelled silence segment "
            "(possible gain staging, second source, or monitoring artefact)."
        ),
        detector="overlap",
    )


def collect_issues(
    silence: SilenceMetrics,
    volume: VolumeMetrics,
    _noise: NoiseMetrics,
    *,
    clipping_segments: list[ClippingSegment] | None = None,
) -> list[Issue]:
    """Cross-cutting issues (e.g. clipping/silence overlap). Detector-specific issues are merged separately."""
    issues: list[Issue] = []

    # Re-run issue helpers would duplicate — callers merge SilenceReport/VolumeReport/NoiseProfile issues.
    # This function only adds cross-cutting overlap detection.
    clips = clipping_segments if clipping_segments is not None else volume.clipping_segments
    if clipping_overlaps_silence(clips, silence.segments):
        issues.append(chain_of_custody_issue())
    return issues


def collect_all_detector_issues(
    silence_issues: list[Issue],
    volume_issues: list[Issue],
    noise_issues: list[Issue],
    extra: list[Issue] | None = None,
) -> list[Issue]:
    out = [*silence_issues, *volume_issues, *noise_issues]
    if extra:
        out.extend(extra)
    return out


def compute_overall_score(
    silence: SilenceMetrics,
    volume: VolumeMetrics,
    noise: NoiseMetrics,
    issues: list[Issue],
) -> tuple[int, dict[str, int]]:
    """
    Weighted score: silence 25%, volume 40%, noise 35%.
    Subscores 0–100 penalized by severity within each domain.
    """
    # Baseline subscores from raw metrics
    silence_sub = max(0, min(100, int(100 * (1.0 - min(silence.ratio / 0.5, 1.0)))))
    clip_penalty = {"none": 0, "minor": 15, "moderate": 35, "severe": 70}.get(
        volume.clipping_severity, 0
    )
    vol_sub = max(0, 100 - clip_penalty - max(0, int(-20 - volume.mean_db)))
    snr_map = {"excellent": 100, "good": 85, "fair": 65, "poor": 40, "unusable": 10}
    noise_sub = snr_map.get(noise.snr_quality, 50)

    weighted = int(
        0.25 * silence_sub + 0.40 * vol_sub + 0.35 * noise_sub
    )
    weighted = max(0, min(100, weighted))

    for issue in issues:
        if issue.severity == Severity.critical:
            weighted = min(weighted, 39)
        elif issue.severity == Severity.high:
            weighted = min(weighted, weighted - 5)

    breakdown = {
        "silence": silence_sub,
        "volume": vol_sub,
        "noise": noise_sub,
    }
    return weighted, breakdown


def grade_from_score(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def admissibility_from_score_and_issues(score: int, issues: list[Issue]) -> str:
    critical = any(i.severity == Severity.critical for i in issues)
    if critical or score < 40:
        return "fail"
    if score < 70:
        return "review"
    return "pass"


