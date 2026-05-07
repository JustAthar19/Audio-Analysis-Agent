"""
Test the LangGraph agent without calling the real Anthropic API.
Mocks `create_react_agent` so the compiled graph never hits the network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent import agent as agent_mod


def test_agent_calls_metadata_first_and_composes(mocker):
    """Simulated graph run invokes tools in order; first dispatch is metadata."""
    calls: list[str] = []

    meta_dump = {
        "file_name": "t.wav",
        "file_path": "/t.wav",
        "file_size_mb": 1.0,
        "duration_sec": 10.0,
        "codec": "pcm_s16le",
        "sample_rate_hz": 48000,
        "channels": 1,
        "bit_depth": 16,
        "is_lossless": True,
        "format_name": "wav",
    }

    def fake_dispatch(name: str, params: dict[str, Any], ctx: dict[str, Any], store: dict[str, Any]):
        calls.append(name)
        if name == "get_audio_metadata":
            ctx["duration_sec"] = 10.0
            store[name] = meta_dump
            return meta_dump
        if name == "detect_silence":
            out = {
                "metrics": {
                    "ratio": 0.05,
                    "total_sec": 0.5,
                    "longest_gap_sec": 0.5,
                    "segment_count": 1,
                    "segments": [],
                    "threshold_db": -45.0,
                    "min_duration_sec": 0.5,
                },
                "issues": [],
            }
            store[name] = out
            return out
        if name == "measure_volume":
            out = {
                "metrics": {
                    "mean_db": -20.0,
                    "peak_db": -6.0,
                    "headroom_db": 6.0,
                    "histogram_0db": 0,
                    "clipping_detected": False,
                    "clipping_severity": "none",
                    "clipping_segments": [],
                    "dynamic_range_db": 14.0,
                },
                "issues": [],
            }
            store[name] = out
            return out
        if name == "compose_report":
            payload = {
                "filepath": params.get("filepath", "/t.wav"),
                "collected": dict(store),
                "summary": "ok",
                "verdict": "fully usable",
                "actions": [],
                "compose_report": True,
            }
            store["compose_report"] = payload
            return payload
        raise AssertionError(f"unexpected tool {name}")

    mocker.patch.object(agent_mod, "_dispatch_tool", side_effect=fake_dispatch)

    def fake_create_react_agent(model, tools, **kwargs):
        graph = MagicMock()

        def _invoke(state, config=None):
            by_name = {t.name: t for t in tools}
            by_name["get_audio_metadata"].invoke({})
            by_name["detect_silence"].invoke(
                {"noise_threshold_db": -45.0, "min_duration_sec": 0.5}
            )
            by_name["measure_volume"].invoke({})
            by_name["compose_report"].invoke(
                {
                    "summary": "ok",
                    "verdict": "fully usable",
                    "actions": [],
                }
            )
            return {"messages": []}

        graph.invoke.side_effect = _invoke
        return graph

    mocker.patch.object(agent_mod, "create_react_agent", side_effect=fake_create_react_agent)

    out = agent_mod.run_agent("/t.wav", fast_mode=False)
    assert out.get("compose_report") is True
    assert calls[0] == "get_audio_metadata"


def test_agent_graph_invoke_exception(mocker):
    mocker.patch.object(
        agent_mod,
        "create_react_agent",
        return_value=MagicMock(invoke=MagicMock(side_effect=RuntimeError("graph boom"))),
    )
    out = agent_mod.run_agent("/t.wav")
    assert out.get("error") == "agent_exception"
    assert "graph boom" in out.get("message", "")


def test_recursion_limit_maps_to_max_iterations_error(mocker):
    mocker.patch.object(
        agent_mod,
        "create_react_agent",
        return_value=MagicMock(
            invoke=MagicMock(side_effect=agent_mod.GraphRecursionError("recursion limit"))
        ),
    )
    out = agent_mod.run_agent("/t.wav")
    assert out.get("error") == "max_iterations"


def test_end_without_compose_returns_error_key(mocker):
    mocker.patch.object(
        agent_mod,
        "create_react_agent",
        return_value=MagicMock(invoke=MagicMock(return_value={"messages": []})),
    )
    out = agent_mod.run_agent("/t.wav")
    assert out.get("error") == "agent_end_turn_without_compose"
