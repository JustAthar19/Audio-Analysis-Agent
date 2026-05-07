from __future__ import annotations

from models.report import ClippingSegment, SilenceSegment, SilenceMetrics
from tools.quality_analysis import (
    clipping_overlaps_silence,
    collect_issues,
    compute_overall_score,
)


def test_clipping_overlaps_silence_detects_nested_start():
    clips = [ClippingSegment(start_sec=5.5, end_sec=6.0, peak_db=-0.5)]
    silences = [SilenceSegment(start_sec=5.0, end_sec=7.0, duration_sec=2.0)]
    assert clipping_overlaps_silence(clips, silences) is True


def test_collect_issues_adds_chain_flag():
    silence = SilenceMetrics(
        ratio=0.1,
        total_sec=10.0,
        longest_gap_sec=5.0,
        segment_count=1,
        segments=[
            SilenceSegment(start_sec=0.0, end_sec=10.0, duration_sec=10.0),
        ],
        threshold_db=-40.0,
        min_duration_sec=1.0,
    )
    from models.report import NoiseMetrics, VolumeMetrics

    volume = VolumeMetrics(
        mean_db=-20.0,
        peak_db=-2.0,
        headroom_db=2.0,
        histogram_0db=10,
        clipping_detected=True,
        clipping_severity="moderate",
        clipping_segments=[
            ClippingSegment(start_sec=2.0, end_sec=2.5, peak_db=-0.5),
        ],
        dynamic_range_db=18.0,
    )
    noise = NoiseMetrics(
        floor_db=-50.0,
        snr_db=30.0,
        snr_quality="good",
        noise_type="broadband",
        spectral_tilt_db=None,
        measured_from_n=1,
        per_window_db=[],
    )
    extra = collect_issues(silence, volume, noise)
    assert any("Chain-of-custody" in i.message for i in extra)


def test_compute_overall_score_bounds():
    silence = SilenceMetrics(
        ratio=0.1,
        total_sec=5.0,
        longest_gap_sec=2.0,
        segment_count=1,
        segments=[],
        threshold_db=-40.0,
        min_duration_sec=1.0,
    )
    from models.report import NoiseMetrics, VolumeMetrics

    volume = VolumeMetrics(
        mean_db=-22.0,
        peak_db=-3.0,
        headroom_db=3.0,
        histogram_0db=0,
        clipping_detected=False,
        clipping_severity="none",
        clipping_segments=[],
        dynamic_range_db=19.0,
    )
    noise = NoiseMetrics(
        floor_db=-60.0,
        snr_db=35.0,
        snr_quality="good",
        noise_type="clean",
        spectral_tilt_db=None,
        measured_from_n=2,
        per_window_db=[],
    )
    score, br = compute_overall_score(silence, volume, noise, [])
    assert 0 <= score <= 100
    assert set(br.keys()) == {"silence", "volume", "noise"}
