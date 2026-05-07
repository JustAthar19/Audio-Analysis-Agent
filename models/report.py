from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class Category(str, Enum):
    silence = "silence"
    clipping = "clipping"
    volume = "volume"
    noise = "noise"
    metadata = "metadata"
    integrity = "integrity"


class Issue(BaseModel):
    severity: Severity
    category: Category
    message: str
    detector: str
    timestamp_sec: Optional[float] = None


class SilenceSegment(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float


class ClippingSegment(BaseModel):
    start_sec: float
    end_sec: float
    peak_db: float


class SilenceMetrics(BaseModel):
    ratio: float
    total_sec: float
    longest_gap_sec: float
    segment_count: int
    segments: list[SilenceSegment]
    threshold_db: float
    min_duration_sec: float


class VolumeMetrics(BaseModel):
    mean_db: float
    peak_db: float
    headroom_db: float
    histogram_0db: int
    clipping_detected: bool
    clipping_severity: str  # none|minor|moderate|severe
    clipping_segments: list[ClippingSegment]
    dynamic_range_db: float


class NoiseMetrics(BaseModel):
    floor_db: float
    snr_db: float
    snr_quality: str  # excellent|good|fair|poor|unusable
    noise_type: str  # clean|hum|hiss|broadband|unknown
    spectral_tilt_db: Optional[float]
    measured_from_n: int
    per_window_db: list[float]


class AudioQuality(BaseModel):
    silence: SilenceMetrics
    volume: VolumeMetrics
    noise: NoiseMetrics


class LLMInsights(BaseModel):
    summary: str
    usability_verdict: str
    recommended_actions: list[str]
    confidence: str  # high|medium|low
    model_used: str


class OverallScore(BaseModel):
    score: int
    grade: str
    admissibility_flag: str  # pass|review|fail
    score_breakdown: dict[str, int]


class FileMetadata(BaseModel):
    file_name: str
    file_path: str
    file_size_mb: float
    duration_sec: float
    codec: str
    sample_rate_hz: int
    channels: int
    bit_depth: Optional[int]
    is_lossless: bool
    format_name: str

class ReportMetadata(BaseModel):
    analysed_at: datetime
    schema_version: str = "1.0.0"
    pipeline_version: str
    analysis_duration_sec: float
    status: str  # complete|partial|failed
    warnings: list[str]

class DepositionReport(BaseModel):
    file_metadata: FileMetadata
    report_metadata: ReportMetadata
    audio_quality: AudioQuality
    issues: list[Issue]
    llm_insights: LLMInsights
    overall_score: OverallScore
