from __future__ import annotations

import concurrent.futures
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def run_batch_pipeline(
    paths: list[Path],
    fast_mode: bool,
    workers: int,
    output_dir: Path,
    fmt: str,
    agent_fn: Callable[..., dict[str, Any]],
    save_fn: Callable[..., Any],
    logger_: logging.Logger,
) -> dict[str, Any]:
    """
    Fan-out across files using ThreadPoolExecutor.

    Each file is isolated in try/except; failures do not abort the batch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def _work(p: Path) -> dict[str, Any]:
        try:
            from pipeline.single_file import run_single_file_pipeline

            report = run_single_file_pipeline(str(p.resolve()), fast_mode, agent_fn)
            out_path = output_dir / f"{p.stem}_report.json"
            if fmt == "html":
                out_path = output_dir / f"{p.stem}_report.html"
            save_fn(report, out_path, fmt=fmt)
            return {"path": p, "report": report, "error": None}
        except Exception as exc:  # noqa: BLE001
            logger_.exception("Batch item failed: %s", p)
            return {"path": p, "report": None, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_work, p) for p in paths]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    errors = sum(1 for r in results if r.get("error"))

    scores: list[float] = []
    admissibility_counts: Counter[str] = Counter()
    issue_summary: Counter[str] = Counter()
    file_rows: list[dict[str, Any]] = []

    for item in results:
        path = item["path"]
        err = item["error"]
        rep = item["report"]
        if err or rep is None:
            file_rows.append(
                {
                    "file": str(path),
                    "score": 0,
                    "grade": "F",
                    "admissibility": "fail",
                    "verdict": "requires remediation",
                    "status": "error",
                }
            )
            continue

        overall = rep.get("overall_score") or {}
        llm = rep.get("llm_insights") or {}
        score = int(overall.get("score", 0))
        scores.append(float(score))
        adm_flag = str(overall.get("admissibility_flag", "fail"))
        admissibility_counts[adm_flag] += 1

        for issue in rep.get("issues") or []:
            sev = str(issue.get("severity", ""))
            if sev in ("high", "critical"):
                cat = str(issue.get("category", "unknown"))
                issue_summary[cat] += 1

        file_rows.append(
            {
                "file": str(path),
                "score": score,
                "grade": str(overall.get("grade", "F")),
                "admissibility": adm_flag,
                "verdict": str(llm.get("usability_verdict", "")),
                "status": str(rep.get("report_metadata", {}).get("status", "complete")),
            }
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(paths),
        "errors": errors,
        "avg_score": avg_score,
        "admissibility": dict(admissibility_counts),
        "issue_summary": dict(issue_summary),
        "files": file_rows,
    }

    batch_path = output_dir / "batch_summary.json"
    batch_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
