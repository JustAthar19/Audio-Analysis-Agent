from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _grade_style(grade: str) -> str:
    mapping = {
        "A": "#1b5e20",
        "B": "#33691e",
        "C": "#f57f17",
        "D": "#e65100",
        "F": "#b71c1c",
    }
    return mapping.get(grade, "#37474f")


def _severity_style(severity: str) -> str:
    mapping = {
        "critical": "#b71c1c",
        "high": "#e65100",
        "medium": "#f9a825",
        "low": "#546e7a",
    }
    return mapping.get(severity, "#455a64")


def render_html_report(report: dict[str, Any]) -> str:
    """Build a self-contained HTML document from a DepositionReport-shaped dict."""
    fm = report.get("file_metadata") or {}
    rm = report.get("report_metadata") or {}
    aq = report.get("audio_quality") or {}
    llm = report.get("llm_insights") or {}
    overall = report.get("overall_score") or {}
    issues = list(report.get("issues") or [])

    silence = aq.get("silence") or {}
    volume = aq.get("volume") or {}
    noise = aq.get("noise") or {}

    grade = str(overall.get("grade", ""))
    score = int(overall.get("score", 0))
    breakdown = overall.get("score_breakdown") or {}

    sil_br = int(breakdown.get("silence", 0))
    vol_br = int(breakdown.get("volume", 0))
    noise_br = int(breakdown.get("noise", 0))

    silence_segments = silence.get("segments") or []
    clip_segments = volume.get("clipping_segments") or []

    actions = llm.get("recommended_actions") or []

    issue_rows = []
    for iss in issues:
        sev = str(iss.get("severity", ""))
        issue_rows.append(
            f"<tr><td style='color:{_severity_style(sev)}'><strong>{html.escape(sev)}</strong></td>"
            f"<td>{html.escape(str(iss.get('category','')))}</td>"
            f"<td>{html.escape(str(iss.get('message','')))}</td>"
            f"<td>{html.escape(str(iss.get('detector','')))}</td></tr>"
        )

    timeline_parts: list[str] = []
    for s in silence_segments[:50]:
        timeline_parts.append(
            f"<li>Silence {float(s.get('start_sec',0)):.2f}s–{float(s.get('end_sec',0)):.2f}s "
            f"(duration {float(s.get('duration_sec',0)):.2f}s)</li>"
        )
    for c in clip_segments[:50]:
        timeline_parts.append(
            f"<li>Clipping {float(c.get('start_sec',0)):.2f}s–{float(c.get('end_sec',0)):.2f}s "
            f"(peak {float(c.get('peak_db',0)):.1f} dB)</li>"
        )

    actions_html = "".join(f"<li>{html.escape(str(a))}</li>" for a in actions)

    meta_rows = [
        ("File", str(fm.get("file_name", ""))),
        ("Duration (s)", str(fm.get("duration_sec", ""))),
        ("Codec", str(fm.get("codec", ""))),
        ("Sample rate (Hz)", str(fm.get("sample_rate_hz", ""))),
        ("Channels", str(fm.get("channels", ""))),
        ("Lossless", str(fm.get("is_lossless", ""))),
        ("Silence ratio", f"{float(silence.get('ratio',0))*100:.1f}%"),
        ("Mean level (dB)", str(volume.get("mean_db", ""))),
        ("Peak (dB)", str(volume.get("peak_db", ""))),
        ("SNR (dB)", str(noise.get("snr_db", ""))),
        ("SNR quality", str(noise.get("snr_quality", ""))),
        ("Noise type", str(noise.get("noise_type", ""))),
    ]

    meta_table = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in meta_rows
    )

    badge_color = _grade_style(grade)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Deposition Audio Report — {html.escape(str(fm.get('file_name','')))}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }}
.badge {{ font-size: 42px; font-weight: 700; padding: 12px 18px; border-radius: 12px; color: #fff;
 background: {badge_color}; }}
.bar-wrap {{ margin: 8px 0 16px; }}
.bar {{ height: 14px; border-radius: 8px; background: linear-gradient(90deg,#1565c0,#00897b); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.card {{ border: 1px solid #e0e0e0; border-radius: 10px; padding: 16px; background: #fafafa; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
ul {{ margin-top: 8px; }}
</style>
</head>
<body>
<header>
  <div class="badge">{html.escape(grade)}</div>
  <div>
    <h1 style="margin:0;">Audio quality report</h1>
    <div style="opacity:.75">{html.escape(str(fm.get('file_path','')))}</div>
    <div><strong>Overall score:</strong> {score}/100 &nbsp;
         <strong>Admissibility:</strong> {html.escape(str(overall.get('admissibility_flag','')))}</div>
  </div>
</header>

<section class="card">
  <h2>Executive summary</h2>
  <p>{html.escape(str(llm.get('summary','')))}</p>
  <p><strong>Verdict:</strong> {html.escape(str(llm.get('usability_verdict','')))}</p>
  <p><strong>Confidence:</strong> {html.escape(str(llm.get('confidence','')))}</p>
</section>

<h2>Score breakdown</h2>
<div class="bar-wrap"><div>Silence (25%)</div><div class="bar" style="width:{sil_br}%"></div><small>{sil_br}</small></div>
<div class="bar-wrap"><div>Volume / clipping (40%)</div><div class="bar" style="width:{vol_br}%"></div><small>{vol_br}</small></div>
<div class="bar-wrap"><div>Noise / SNR (35%)</div><div class="bar" style="width:{noise_br}%"></div><small>{noise_br}</small></div>

<div class="grid" style="margin-top:16px">
  <section class="card">
    <h2>Issues</h2>
    <table>
      <thead><tr><th>Severity</th><th>Category</th><th>Message</th><th>Detector</th></tr></thead>
      <tbody>{''.join(issue_rows) or '<tr><td colspan="4">No issues recorded.</td></tr>'}</tbody>
    </table>
  </section>
  <section class="card">
    <h2>Recommended actions</h2>
    <ol>{actions_html or '<li>No actions listed.</li>'}</ol>
  </section>
</div>

<h2>Technical metrics</h2>
<table>{meta_table}</table>

<h2>Timeline highlights</h2>
<ul>{''.join(timeline_parts) or '<li>No segments parsed.</li>'}</ul>

<footer style="margin-top:32px;font-size:12px;opacity:.7">
  Generated at {html.escape(str(rm.get('analysed_at','')))} · schema {html.escape(str(rm.get('schema_version','')))}
  · pipeline {html.escape(str(rm.get('pipeline_version','')))} · status {html.escape(str(rm.get('status','')))}
</footer>

<!-- Raw JSON for traceability -->
<script type="application/json" id="report-json">{html.escape(json.dumps(report, default=str))}</script>
</body>
</html>
"""


def save_report_html(report: dict[str, Any], path: Path) -> None:
    """Write self-contained HTML report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report), encoding="utf-8")
    logger.info("wrote HTML report to %s", path)


def save_mixed_report(report: dict[str, Any], path: Path, *, fmt: str) -> None:
    """Save JSON or HTML depending on fmt."""
    if fmt == "html":
        save_report_html(report, path)
    else:
        from output.json_writer import save_report_json

        save_report_json(report, path, fmt=fmt)
