"""
Exposes ffmpeg tools via FastMCP so any MCP-compatible agent can use them.

Run standalone: python -m tools.mcp_server

Requires Python 3.10+ and `fastmcp` installed.
"""

from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastmcp is required for MCP server mode; install with pip install fastmcp (Python 3.10+)."
    ) from exc

from tools.ffmpeg_tools import (
    extract_metadata,
    run_astats_clipping,
    run_silence_detection,
)

mcp = FastMCP("audio-analysis-agent")


@mcp.tool()
def get_audio_metadata(filepath: str) -> dict:
    """Extract audio metadata using ffprobe."""
    return extract_metadata(filepath).model_dump()


@mcp.tool()
def detect_silence(
    filepath: str,
    noise_threshold_db: float = -40.0,
    min_duration_sec: float = 2.0,
) -> list[dict]:
    """Detect silence segments using ffmpeg silencedetect filter."""
    return run_silence_detection(filepath, noise_threshold_db, min_duration_sec)


@mcp.tool()
def detect_clipping(
    filepath: str,
    window_sec: float = 1.0,
    clip_threshold_db: float = -1.0,
) -> list[dict]:
    """Detect time-localised clipping using ffmpeg astats."""
    return run_astats_clipping(filepath, window_sec, clip_threshold_db)


if __name__ == "__main__":
    mcp.run()
