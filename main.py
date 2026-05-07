from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level_name = os.environ.get("AUDIO_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _save_report(report: dict, path: Path, *, fmt: str = "json") -> None:
    if fmt == "html":
        from output.html_writer import save_report_html

        save_report_html(report, path)
    else:
        from output.json_writer import save_report_json

        save_report_json(report, path, fmt=fmt)


def cmd_check() -> int:
    """Verify ffmpeg and ffprobe are available."""
    ok = True
    for bin_name in ("ffmpeg", "ffprobe"):
        path = shutil.which(bin_name)
        if not path:
            logger.error("%s not found on PATH", bin_name)
            ok = False
        else:
            logger.info("%s -> %s", bin_name, path)
    if ok:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            logger.info("ffmpeg responds OK")
        except (subprocess.SubprocessError, OSError) as exc:
            logger.error("ffmpeg execution failed: %s", exc)
            ok = False
    return 0 if ok else 1


def cmd_analyse(args: argparse.Namespace) -> int:
    from agent.agent import run_agent
    from pipeline.single_file import run_single_file_pipeline

    fast = bool(args.fast) or os.environ.get("AUDIO_AGENT_FAST_MODE", "").lower() == "true"
    fmt = args.format or os.environ.get("AUDIO_AGENT_DEFAULT_FORMAT", "json")

    report = run_single_file_pipeline(args.audio_file, fast, run_agent)
    out = Path(args.output) if args.output else Path(f"{Path(args.audio_file).stem}_report.json")
    if fmt == "html":
        out = out.with_suffix(".html")
    _save_report(report, out, fmt=fmt)
    logger.info("Report written to %s", out)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from agent.agent import run_agent
    from pipeline.batch import run_batch_pipeline

    fast = bool(args.fast) or os.environ.get("AUDIO_AGENT_FAST_MODE", "").lower() == "true"
    fmt = args.format or os.environ.get("AUDIO_AGENT_DEFAULT_FORMAT", "json")
    root = Path(args.directory)
    patterns = ("*.wav", "*.mp3", "*.aac", "*.flac", "*.m4a", "*.ogg")
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(root.glob(pat)))
    if not paths:
        logger.warning("No audio files matched known extensions under %s", root)
        return 1

    run_batch_pipeline(
        paths=paths,
        fast_mode=fast,
        workers=int(args.workers),
        output_dir=Path(args.output_dir),
        fmt=fmt,
        agent_fn=run_agent,
        save_fn=_save_report,
        logger_=logging.getLogger("batch"),
    )
    logger.info("Batch complete; summary at %s", Path(args.output_dir) / "batch_summary.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio_agent", description="Forensic deposition audio analysis agent.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Verify ffmpeg/ffprobe availability")

    pa = sub.add_parser("analyse", help="Analyse a single audio file")
    pa.add_argument("audio_file", type=str)
    pa.add_argument("--fast", action="store_true", help="Prefer faster SNR path")
    pa.add_argument("--format", choices=("json", "html"), default=None)
    pa.add_argument("--output", "-o", type=str, default=None, help="Output path")

    pb = sub.add_parser("batch", help="Analyse many audio files in parallel")
    pb.add_argument("directory", type=str)
    pb.add_argument("--output-dir", required=True, type=str)
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--fast", action="store_true")
    pb.add_argument("--format", choices=("json", "html"), default=None)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check()
    if args.command == "analyse":
        return cmd_analyse(args)
    if args.command == "batch":
        return cmd_batch(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main()) ## what is this 
