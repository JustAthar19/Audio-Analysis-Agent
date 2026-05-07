from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_report_json(report: dict[str, Any], path: Path, *, fmt: str = "json") -> None:
    """Serialise a report dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    logger.info("wrote JSON report to %s", path)


def save_report(report: dict[str, Any], path: Path, *, fmt: str = "json") -> None:
    """Dispatch save by format — json default; html delegated by caller."""
    save_report_json(report, path, fmt=fmt)
