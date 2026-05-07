from __future__ import annotations

import json
import logging
from typing import Any, Optional

from typing_extensions import TypedDict

# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.prompts import SYSTEM_PROMPT
from tools.ffmpeg_tools import extract_metadata
from tools.quality_analysis import (
    analyse_snr,
    analyse_volume_and_clipping,
    detect_silence,
    merge_astats_clipping,
)

logger = logging.getLogger(__name__)

# Max graph steps (model + tool nodes). ~10 LLM rounds with tools needs headroom.
_DEFAULT_RECURSION_LIMIT = 50


class SilenceSegment(TypedDict):
    """One interval from silencedetect, in seconds."""

    start_sec: float
    end_sec: float


def _tool_log_summary(name: str, params: dict[str, Any]) -> str:
    if name == "get_audio_metadata":
        return f"filepath={params.get('filepath')}"
    if name == "detect_silence":
        return (
            f"filepath={params.get('filepath')} "
            f"noise_threshold_db={params.get('noise_threshold_db')} "
            f"min_duration_sec={params.get('min_duration_sec')}"
        )
    if name == "measure_volume":
        return f"filepath={params.get('filepath')}"
    if name == "detect_clipping":
        return (
            f"filepath={params.get('filepath')} "
            f"window_sec={params.get('window_sec')} "
            f"clip_threshold_db={params.get('clip_threshold_db')}"
        )
    if name == "measure_snr":
        return (
            f"filepath={params.get('filepath')} "
            f"mean_signal_db={params.get('mean_signal_db')} "
            f"run_spectral={params.get('run_spectral')}"
        )
    if name == "compose_report":
        return f"filepath={params.get('filepath')}"
    return str(params)[:200]


def _dispatch_tool(
    name: str,
    params: dict[str, Any],
    ctx: dict[str, Any],
    collected_store: dict[str, Any],
) -> Any:
    """Execute one tool; updates ctx and collected_store."""
    logger.info("tool call %s (%s)", name, _tool_log_summary(name, params))
    if name == "get_audio_metadata":
        meta = extract_metadata(params["filepath"])
        ctx["duration_sec"] = float(meta.duration_sec)
        ctx["is_lossless"] = bool(meta.is_lossless)
        dumped = meta.model_dump(mode="json")
        collected_store[name] = dumped
        return dumped

    if name == "detect_silence":
        dur = float(ctx.get("duration_sec") or 0.0)
        report = detect_silence(
            params["filepath"],
            dur,
            noise_threshold_db=float(params["noise_threshold_db"]),
            min_duration_sec=float(params["min_duration_sec"]),
        )
        dumped = report.model_dump(mode="json")
        collected_store[name] = dumped
        return dumped

    if name == "measure_volume":
        dur = float(ctx.get("duration_sec") or 0.0)
        report = analyse_volume_and_clipping(
            params["filepath"],
            dur,
            force_astats=False,
        )
        ctx["volume_metrics"] = report.metrics
        dumped = report.model_dump(mode="json")
        collected_store[name] = dumped
        return dumped

    if name == "detect_clipping":
        dur = float(ctx.get("duration_sec") or 0.0)
        base = ctx.get("volume_metrics")
        if base is None:
            pre = analyse_volume_and_clipping(params["filepath"], dur, force_astats=False)
            ctx["volume_metrics"] = pre.metrics
            base = pre.metrics
        merged_report = merge_astats_clipping(
            base,
            params["filepath"],
            dur,
            window_sec=float(params.get("window_sec", 1.0)),
            clip_threshold_db=float(params.get("clip_threshold_db", -1.0)),
        )
        ctx["volume_metrics"] = merged_report.metrics
        dumped = merged_report.model_dump(mode="json")
        collected_store[name] = dumped
        return dumped

    if name == "measure_snr":
        seg_in = params.get("silence_segments") or []
        silence_segments: list[dict[str, float]] = []
        for s in seg_in:
            silence_segments.append(
                {
                    "start_sec": float(s.get("start_sec", 0.0)),
                    "end_sec": float(s.get("end_sec", 0.0)),
                }
            )
        profile = analyse_snr(
            params["filepath"],
            mean_signal_db=float(params["mean_signal_db"]),
            silence_segments=silence_segments,
            run_spectral=bool(params.get("run_spectral", True)),
        )
        dumped = profile.model_dump(mode="json")
        collected_store[name] = dumped
        return dumped

    if name == "compose_report":
        fp = params.get("filepath") or ctx.get("filepath")
        payload = {
            "filepath": fp,
            "collected": dict(collected_store),
            "summary": params["summary"],
            "verdict": params["verdict"],
            "actions": list(params.get("actions") or []),
            "compose_report": True,
        }
        collected_store["compose_report"] = payload
        return payload

    return {"is_error": True, "message": f"Unknown tool: {name}"}


def _build_tools(
    default_filepath: str,
    ctx: dict[str, Any],
    collected_store: dict[str, Any],
) -> list:
    """LangChain tools that delegate to `_dispatch_tool` (shared session state)."""

    @tool
    def get_audio_metadata(audio_path: Optional[str] = None) -> str:
        """Extract container and stream metadata using ffprobe. Always call first.

        Args:
            audio_path: Path to the audio file. If omitted, uses the analysis target file.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool("get_audio_metadata", {"filepath": fp}, ctx, collected_store)
        return json.dumps(out, default=str)

    @tool
    def detect_silence(
        noise_threshold_db: float,
        min_duration_sec: float,
        audio_path: Optional[str] = None,
    ) -> str:
        """Detect silence segments using ffmpeg silencedetect. Run before measure_snr when silence windows are expected.

        Args:
            noise_threshold_db: dB threshold. Lossless: -45, compressed: -35.
            min_duration_sec: Minimum gap length. Testimony: 0.5, formal: 2.0.
            audio_path: Optional path; defaults to the analysis target file.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool(
            "detect_silence",
            {
                "filepath": fp,
                "noise_threshold_db": noise_threshold_db,
                "min_duration_sec": min_duration_sec,
            },
            ctx,
            collected_store,
        )
        return json.dumps(out, default=str)

    @tool
    def measure_volume(audio_path: Optional[str] = None) -> str:
        """Measure mean dB, peak dB, and histogram via volumedetect. Always run.

        Args:
            audio_path: Optional path; defaults to the analysis target file.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool("measure_volume", {"filepath": fp}, ctx, collected_store)
        return json.dumps(out, default=str)

    @tool
    def detect_clipping(
        window_sec: float = 1.0,
        clip_threshold_db: float = -1.0,
        audio_path: Optional[str] = None,
    ) -> str:
        """Time-localised clipping detection via astats. EXPENSIVE on long files. Only if peak_db >= -1.0 or histogram_0db > 0.

        Args:
            window_sec: 0.5 for precise timestamps, 2.0 for speed.
            clip_threshold_db: Recommended -1.0 for inter-sample peak safety.
            audio_path: Optional path; defaults to the analysis target file.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool(
            "detect_clipping",
            {
                "filepath": fp,
                "window_sec": window_sec,
                "clip_threshold_db": clip_threshold_db,
            },
            ctx,
            collected_store,
        )
        return json.dumps(out, default=str)

    @tool
    def measure_snr(
        mean_signal_db: float,
        run_spectral: bool = True,
        silence_segments: Optional[list[SilenceSegment]] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        """Estimate SNR by sampling silence windows; falls back to percentile RMS if no silence.

        Args:
            mean_signal_db: From measure_volume (mean_db).
            run_spectral: If true, run spectral tilt for noise typing (~extra runtime).
            silence_segments: From detect_silence segments (start_sec/end_sec); use [] for fallback.
            audio_path: Optional path; defaults to the analysis target file.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool(
            "measure_snr",
            {
                "filepath": fp,
                "mean_signal_db": mean_signal_db,
                "silence_segments": silence_segments or [],
                "run_spectral": run_spectral,
            },
            ctx,
            collected_store,
        )
        return json.dumps(out, default=str)

    @tool
    def compose_report(
        summary: str,
        verdict: str,
        actions: list[str],
        audio_path: Optional[str] = None,
        collected: Optional[dict[str, Any]] = None,
    ) -> str:
        """Compose the final report when analysis is complete. Verdict: fully usable | usable with caveats | requires remediation | inadmissible.

        Args:
            summary: 2–3 sentence plain-English summary for a paralegal.
            verdict: One of: fully usable, usable with caveats, requires remediation, inadmissible.
            actions: Recommended actions, most urgent first.
            audio_path: Optional path; defaults to the analysis target file.
            collected: Ignored; server-side store is used automatically.
        """
        fp = audio_path or default_filepath
        out = _dispatch_tool(
            "compose_report",
            {
                "filepath": fp,
                "collected": collected or {},
                "summary": summary,
                "verdict": verdict,
                "actions": actions,
            },
            ctx,
            collected_store,
        )
        return json.dumps(out, default=str)

    return [
        get_audio_metadata,
        detect_silence,
        measure_volume,
        detect_clipping,
        measure_snr,
        compose_report,
    ]


def run_agent(
    filepath: str,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Run the LangGraph ReAct agent. 
    Using Gemini 2.5 Flash Lite via LangChain ChatGoogleGenerativeAI.
    Returns the compose_report payload (including merged collected tool results), or
    an error structure if the graph ends without compose_report or hits the recursion limit.
    """
    ctx: dict[str, Any] = {
        "duration_sec": None,
        "volume_metrics": None,
        "is_lossless": None,
        "filepath": filepath,
    }
    collected_store: dict[str, Any] = {}

    tools = _build_tools(filepath, ctx, collected_store)

    # model = ChatAnthropic(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=4096,
    #     temperature=0,
    # )

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.strip()),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        max_tokens=4096,
        temperature=0,
    )
    
    agent = create_tool_calling_agent(model, tools, prompt=agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=_DEFAULT_RECURSION_LIMIT,
    )
   
    # graph = create_react_agent( # deprecated
    #     model,
    #     tools,
    #     prompt=SYSTEM_PROMPT.strip(),
    # )
    
    

    user_extra = (
        "Use min_duration_sec=0.5 unless the recording is formal court-only session.\n"
        f"Target file path: {filepath}\n"
    )
    if fast_mode:
        user_extra += "FAST MODE: prefer measure_snr with run_spectral=false unless SNR is clearly poor.\n"

    user_content = (
        user_extra
        + "Analyse this deposition audio for forensic quality. "
        + "After metadata, choose silence noise_threshold_db: -45.0 dB if lossless else -35.0 dB."
    )

    config: dict[str, Any] = {"recursion_limit": _DEFAULT_RECURSION_LIMIT}

    try:
        ### deprecated
        # result = graph.invoke(
        #     {"messages": [HumanMessage(content=user_content)]},
        #     config=config,
        # )
        result = agent_executor.invoke({"input": user_content}, config=config)
    except GraphRecursionError:
        logger.warning("LangGraph recursion limit reached without compose_report.")
        return {
            "error": "max_iterations",
            "message": "Agent exceeded graph recursion limit without compose_report.",
            "collected": dict(collected_store),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("LangGraph agent failed")
        return {
            "error": "agent_exception",
            "message": str(exc),
            "collected": dict(collected_store),
        }

    compose_payload = collected_store.get("compose_report")
    if compose_payload:
        return compose_payload

    messages = result.get("messages") or []
    last_text = ""
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            last_text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif hasattr(block, "text"):
                    parts.append(str(getattr(block, "text", "")))
            last_text = "\n".join(parts)
    if not last_text:
        out = result.get("output")
        if isinstance(out, str):
            last_text = out

    return {
        "error": "agent_end_turn_without_compose",
        "message": last_text or "Model ended without compose_report.",
        "collected": dict(collected_store),
    }
