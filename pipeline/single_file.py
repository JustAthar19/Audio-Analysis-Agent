from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from models.report import (
    AudioQuality,
    DepositionReport,
    FileMetadata,
    Issue,
    LLMInsights,
    NoiseMetrics,
    OverallScore,
    ReportMetadata,
    SilenceMetrics,
    VolumeMetrics,
)
from tools.ffmpeg_tools import extract_metadata
from tools.quality_analysis import (
    admissibility_from_score_and_issues,
    analyse_volume_and_clipping,
    collect_all_detector_issues,
    collect_issues,
    compute_overall_score,
    detect_silence,
    grade_from_score,
)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.4.1"


def _default_noise_metrics() -> NoiseMetrics:
    return NoiseMetrics(
        floor_db=-96.0,
        snr_db=0.0,
        snr_quality="fair",
        noise_type="unknown",
        spectral_tilt_db=None,
        measured_from_n=0,
        per_window_db=[],
    )


def _default_silence_metrics() -> SilenceMetrics:
    return SilenceMetrics(
        ratio=0.0,
        total_sec=0.0,
        longest_gap_sec=0.0,
        segment_count=0,
        segments=[],
        threshold_db=-40.0,
        min_duration_sec=2.0,
    )


def _default_volume_metrics() -> VolumeMetrics:
    return VolumeMetrics(
        mean_db=-30.0,
        peak_db=-6.0,
        headroom_db=6.0,
        histogram_0db=0,
        clipping_detected=False,
        clipping_severity="none",
        clipping_segments=[],
        dynamic_range_db=15.0,
    )


def build_report(
    filepath: str,
    collected: dict[str, Any],
    summary: str,
    verdict: str,
    actions: list[str],
    pipeline_start: float,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, Any]:
    """
    Assemble DepositionReport from agent outputs.

    Overall score weights: silence 25%, volume/clipping 40%, noise/SNR 35%.
    """
    warnings: list[str] = []

    file_meta_raw = collected.get("get_audio_metadata")
    if file_meta_raw:
        file_metadata = FileMetadata.model_validate(file_meta_raw)
    else:
        try:
            file_metadata = extract_metadata(filepath)
            warnings.append("File metadata recovered outside agent-collected results.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_metadata fallback failed: %s", exc)
            path_only = filepath
            file_metadata = FileMetadata(
                file_name=path_only.split("/")[-1],
                file_path=filepath,
                file_size_mb=0.0,
                duration_sec=0.0,
                codec="unknown",
                sample_rate_hz=0,
                channels=0,
                bit_depth=None,
                is_lossless=False,
                format_name="unknown",
            )
            warnings.append(f"Metadata incomplete: {exc}")

    silence_raw = collected.get("detect_silence")
    if silence_raw and silence_raw.get("metrics"):
        silence_metrics = SilenceMetrics.model_validate(silence_raw["metrics"])
    else:
        silence_metrics = _default_silence_metrics()
        warnings.append("Silence metrics missing; defaults used.")

    vol_clip = collected.get("detect_clipping")
    vol_measure = collected.get("measure_volume")
    vol_raw = vol_clip or vol_measure
    if vol_raw and vol_raw.get("metrics"):
        volume_metrics = VolumeMetrics.model_validate(vol_raw["metrics"])
    elif vol_measure and vol_measure.get("metrics"):
        volume_metrics = VolumeMetrics.model_validate(vol_measure["metrics"])
    else:
        volume_metrics = _default_volume_metrics()
        warnings.append("Volume metrics missing; defaults used.")

    noise_raw = collected.get("measure_snr")
    if noise_raw and noise_raw.get("metrics"):
        noise_metrics = NoiseMetrics.model_validate(noise_raw["metrics"])
    else:
        noise_metrics = _default_noise_metrics()
        warnings.append("Noise/SNR metrics missing; defaults used.")

    silence_issues = list(silence_raw.get("issues", [])) if silence_raw else []
    volume_issues: list[dict[str, Any]] = []
    if vol_measure:
        volume_issues.extend(list(vol_measure.get("issues", [])))
    if vol_clip:
        volume_issues.extend(list(vol_clip.get("issues", [])))
    if not volume_issues and vol_raw:
        volume_issues = list(vol_raw.get("issues", []))
    noise_issues = list(noise_raw.get("issues", [])) if noise_raw else []

    silence_issues_m = [Issue.model_validate(i) for i in silence_issues]
    volume_issues_m = [Issue.model_validate(i) for i in volume_issues]
    noise_issues_m = [Issue.model_validate(i) for i in noise_issues]

    overlap_issues = collect_issues(
        silence_metrics,
        volume_metrics,
        noise_metrics,
        clipping_segments=volume_metrics.clipping_segments,
    )
    issues = collect_all_detector_issues(
        silence_issues_m,
        volume_issues_m,
        noise_issues_m,
        extra=overlap_issues,
    )

    score, breakdown = compute_overall_score(silence_metrics, volume_metrics, noise_metrics, issues)
    grade = grade_from_score(score)
    admissibility = admissibility_from_score_and_issues(score, issues)

    overall = OverallScore(
        score=score,
        grade=grade,
        admissibility_flag=admissibility,
        score_breakdown=breakdown,
    )

    llm = LLMInsights(
        summary=summary,
        usability_verdict=verdict,
        recommended_actions=actions,
        confidence="high" if not warnings else "medium",
        model_used="gemini-2.5-flash-lite",
    )

    audio_quality = AudioQuality(
        silence=silence_metrics,
        volume=volume_metrics,
        noise=noise_metrics,
    )

    elapsed = max(0.0, time.perf_counter() - pipeline_start)
    report_meta = ReportMetadata(
        analysed_at=datetime.now(timezone.utc),
        pipeline_version=pipeline_version,
        analysis_duration_sec=elapsed,
        status="complete",
        warnings=warnings,
    )

    report = DepositionReport(
        file_metadata=file_metadata,
        report_metadata=report_meta,
        audio_quality=audio_quality,
        issues=issues,
        llm_insights=llm,
        overall_score=overall,
    )
    return report.model_dump(mode="json")


def run_single_file_pipeline(
    filepath: str,
    fast_mode: bool,
    agent_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """
    Orchestrates the full pipeline for a single file.

    Calls agent_fn(filepath, fast_mode), builds DepositionReport, returns model_dump().
    """
    pipeline_start = time.perf_counter()
    try:
        agent_out = agent_fn(filepath, fast_mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent invocation failed")
        partial_collected: dict[str, Any] = {}
        try:
            meta = extract_metadata(filepath)
            partial_collected["get_audio_metadata"] = meta.model_dump(mode="json")
            dur = float(meta.duration_sec)
            partial_collected["detect_silence"] = detect_silence(
                filepath,
                dur,
                noise_threshold_db=-45.0 if meta.is_lossless else -35.0,
                min_duration_sec=0.5,
            ).model_dump(mode="json")
            partial_collected["measure_volume"] = analyse_volume_and_clipping(
                filepath,
                dur,
                force_astats=False,
            ).model_dump(mode="json")
        except Exception as inner:  # noqa: BLE001
            logger.warning("Partial recovery failed: %s", inner)

        fail_report = build_report(
            filepath,
            partial_collected,
            summary="Pipeline failed before agent completed.",
            verdict="requires remediation",
            actions=["Re-run analysis", "Inspect audio file integrity"],
            pipeline_start=pipeline_start,
        )
        rep = DepositionReport.model_validate(fail_report)
        rep.report_metadata.status = "failed"
        rep.report_metadata.warnings.append(str(exc))
        return rep.model_dump(mode="json")

    if agent_out.get("compose_report"):
        collected = agent_out.get("collected") or {}
        payload = build_report(
            filepath,
            collected,
            summary=agent_out.get("summary", ""),
            verdict=agent_out.get("verdict", "usable with caveats"),
            actions=list(agent_out.get("actions") or []),
            pipeline_start=pipeline_start,
        )
        return payload

    warnings = []
    if agent_out.get("error"):
        warnings.append(agent_out.get("message") or "Unknown agent error")

    partial_collected = dict(agent_out.get("collected") or {})
    if not partial_collected.get("get_audio_metadata"):
        try:
            meta = extract_metadata(filepath)
            partial_collected["get_audio_metadata"] = meta.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"metadata: {exc}")

    summary = "Analysis incomplete — agent did not compose a final report."
    payload = build_report(
        filepath,
        partial_collected,
        summary=summary,
        verdict="requires remediation",
        actions=["Review logs", "Retry analysis"],
        pipeline_start=pipeline_start,
    )
    rep = DepositionReport.model_validate(payload)
    rep.report_metadata.status = "partial"
    rep.report_metadata.warnings.extend(warnings)
    rep.llm_insights.confidence = "low"
    return rep.model_dump(mode="json")

