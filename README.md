# Audio Analysis Agent


Tools: **ffprobe/ffmpeg** extract quantitative metrics, a **Langgraph** agent selects tools adaptively, and the pipeline emits **structured JSON** plus optional **self-contained HTML** with admissibility-oriented scoring.

## Architecture

1. **Metadata (`ffprobe`)** — Container/stream facts (duration, codec, sample rate, lossless vs lossy).
2. **Quality detectors** — Silence (`silencedetect`), volume (`volumedetect`), clipping (`astats` + metadata), SNR (silence-window RMS or RMS percentile fallback), optional spectral tilt for noise typing.
3. **Structured metrics** — Normalised into Pydantic models (`SilenceMetrics`, `VolumeMetrics`, `NoiseMetrics`, `Issue`, etc.).
4. **Agent loop** — Langraphasd receives `SYSTEM_PROMPT` rules (thresholds, admissibility, chain-of-custody overlap). It calls tools in order (`get_audio_metadata` → `detect_silence`, `measure_volume`, conditional `detect_clipping` / `measure_snr`) then `**compose_report`**.
5. **Pipeline** — `run_single_file_pipeline` runs the agent, then `build_report` merges results, adds overlap-based integrity issues, computes weighted score (silence 25% / volume 40% / noise 35%), grades **A–F**, and sets admissibility **pass / review / fail**.
6. **Batch** — Thread pool per file, isolated failures, aggregate `batch_summary.json`.

Optional **FastMCP** server (`tools/mcp_server.py`) exposes raw ffmpeg helpers to MCP clients (requires **Python 3.10+** for `fastmcp`).

## Quick start

### Using uv 
[See uv documentation](https://docs.astral.sh/uv/guides/install-python/)
```bash
cd audio_agent
uv sync
cp .env.example .env
# Add GEMINIC_API_KEY to .env
```
Run commands with `uv run`:
```bash
uv run python main.py check
uv run python main.py analyse path/to/deposition.wav
uv run python main.py batch ./recordings/ --output-dir ./reports
```

### Using pip
```bash
cd audio_agent
# Create virtual environment
python -m venv .venv

# Activate virtual environment

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Add GEMINI_API_KEY to .env
```

Run commands with `pip`:
```bash
python main.py check
python main.py analyse path/to/deposition.wav
python main.py batch ./recordings/ --output-dir ./reports
```

Run tests (from `audio_agent/`):

```bash
pip install pytest pytest-mock
pytest tests
```

## CLI reference

| Command | Description |
| --- | --- |
| `python3 main.py check` | Verifies `ffmpeg` / `ffprobe` executables are available and callable. |
| `python3 main.py analyse AUDIO [--fast] [--format json\|html] [-o OUT]` | Analyses one file. Default output is `<audio_stem>_report.json` (or `.html` when `--format html`). |
| `python3 main.py batch DIRECTORY --output-dir DIR [--workers N] [--fast] [--format json\|html]` | Analyses matching audio files in parallel and writes per-file reports plus `batch_summary.json` under `--output-dir`. |


Environment variables (see `.env.example`):

- `AUDIO_AGENT_LOG_LEVEL` — logging level (default `INFO`).
- `AUDIO_AGENT_DEFAULT_FORMAT` — `json` or `html` when `--format` omitted.
- `AUDIO_AGENT_FAST_MODE` — if `true`, CLI behaves like `--fast` unless overridden.

## Design decisions

- **Agent vs fixed pipeline** — Deterministic ffmpeg passes live in `tools/`; the agent decides thresholds (e.g. −45 dB vs −35 dB silence noise floor from `is_lossless`), optional expensive clipping astats, and SNR spectral tilt. `run_single_file_pipeline` still builds a valid report if the agent stops early (partial/failed status).
- **Two-layer volume analysis** — Always **volumedetect** first; **astats** clipping run only if peaks/histogram suggest clipping (`peak_db >= -1` or `histogram_0db > 0`), or when `**detect_clipping`** is invoked explicitly.
- **Adaptive silence thresholds** — Prompt + user message instruct the model: lossless → −45 dB, compressed → −35 dB; `min_duration_sec` 0.5 vs 2.0 by context.
- **MCP** — Optional FastMCP wrapper for integrations without importing the full agent.
- **Admissibility scoring** — Critical issues cap score; thresholds map grade bands and **pass/review/fail** (`overall_score.admissibility_flag`).

## Example output (JSON excerpt)

```json
{
  "file_metadata": {
    "file_name": "moonlight-plaza.mp3",
    "file_path": "/Users/atharfathana/Documents/Machine Learning-AI/voicescript_ai_agent_assessment/data/moonlight-plaza.mp3",
    "file_size_mb": 0.0,
    "duration_sec": 0.0,
    "codec": "unknown",
    "sample_rate_hz": 0,
    "channels": 0,
    "bit_depth": null,
    "is_lossless": false,
    "format_name": "unknown"
  },
  "report_metadata": {
    "analysed_at": "2026-05-05T06:50:22.111523Z",
    "schema_version": "1.0.0",
    "pipeline_version": "0.4.1",
    "analysis_duration_sec": 153.52387824980542,
    "status": "partial",
    "warnings": [
      "Metadata incomplete: {\n\n}",
      "Silence metrics missing; defaults used.",
      "Volume metrics missing; defaults used.",
      "Noise/SNR metrics missing; defaults used.",
      "{\n\n}",
      "metadata: {\n\n}"
    ]
  },
  "audio_quality": {
    "silence": {
      "ratio": 0.0,
      "total_sec": 0.0,
      "longest_gap_sec": 0.0,
      "segment_count": 0,
      "segments": [],
      "threshold_db": -40.0,
      "min_duration_sec": 2.0
    },
    "volume": {
      "mean_db": -30.0,
      "peak_db": -6.0,
      "headroom_db": 6.0,
      "histogram_0db": 0,
      "clipping_detected": false,
      "clipping_severity": "none",
      "clipping_segments": [],
      "dynamic_range_db": 15.0
    },
    "noise": {
      "floor_db": -96.0,
      "snr_db": 0.0,
      "snr_quality": "fair",
      "noise_type": "unknown",
      "spectral_tilt_db": null,
      "measured_from_n": 0,
      "per_window_db": []
    }
  },
  "issues": [],
  "llm_insights": {
    "summary": "Analysis incomplete \u2014 agent did not compose a final report.",
    "usability_verdict": "requires remediation",
    "recommended_actions": [
      "Review logs",
      "Retry analysis"
    ],
    "confidence": "low",
    "model_used": "gemini-2.5-flash-lite"
  },
  "overall_score": {
    "score": 83,
    "grade": "B",
    "admissibility_flag": "pass",
    "score_breakdown": {
      "silence": 100,
      "volume": 90,
      "noise": 65
    }
  }
}
```

## Adding a new detector

1. Implement a subprocess helper in `tools/ffmpeg_tools.py` (timeouts, `capture_output=True`, `text=True`, **regex** parsing with named groups; raise `RuntimeError` on non-zero exit).
2. Wrap it in `tools/quality_analysis.py`: produce metric models + `Issue` rows.
3. Register a tool in `agent/agent.py` (`TOOLS` + `_dispatch_tool`).
4. Extend `build_report` in `pipeline/single_file.py` if the final `DepositionReport` should include new fields.
5. Add tests with **pytest-mock** patching `subprocess.run`.
