"""
Unit tests for Benchmark Telemetry Engine.
Validates endpoint normalization, latency calculations (TTFT, TPS, TPOT),
and dynamic N-model compare streaming.
"""

import asyncio
import json
import pytest
from backend.benchmark_engine import BenchmarkEngine
from backend.models import ModelSlotConfig, CompareBenchmarkRequest


def test_endpoint_normalization():
    assert BenchmarkEngine.normalize_endpoint("https://api-inference.huggingface.co/v1/") == "https://api-inference.huggingface.co/v1/chat/completions"
    assert BenchmarkEngine.normalize_endpoint("https://integrate.api.nvidia.com/v1") == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert BenchmarkEngine.normalize_endpoint("http://localhost:8000/v1/chat/completions") == "http://localhost:8000/v1/chat/completions"


def test_compare_models_concurrent_execution():
    async def _run():
        req = CompareBenchmarkRequest(
            models=[
                ModelSlotConfig(
                    slot_id="slot_0",
                    model_id="Qwen/Qwen2.5-7B-Instruct",
                    api_base="mock://localhost",
                    api_key="demo-key"
                ),
                ModelSlotConfig(
                    slot_id="slot_1",
                    model_id="meta-llama/Llama-3-8b-instruct",
                    api_base="mock://localhost",
                    api_key="demo-key"
                )
            ],
            prompt="Explain quantum computing in 3 sentences.",
            temperature=0.7,
            max_tokens=128
        )
        return await BenchmarkEngine.compare_models(req)

    comp_resp = asyncio.run(_run())
    assert len(comp_resp.results) == 2
    assert comp_resp.results[0].slot_id == "slot_0"
    assert comp_resp.results[1].slot_id == "slot_1"
    assert comp_resp.results[0].ttft_ms > 0
    assert comp_resp.results[1].ttft_ms > 0
    assert comp_resp.results[0].tps > 0
    assert comp_resp.results[1].tps > 0


def test_stream_compare_events():
    async def _run():
        req = CompareBenchmarkRequest(
            models=[
                ModelSlotConfig(
                    slot_id="s1",
                    model_id="Qwen/Qwen2.5-7B-Instruct",
                    api_base="mock://localhost",
                    api_key="demo-key"
                ),
                ModelSlotConfig(
                    slot_id="s2",
                    model_id="google/gemma-2-9b-it",
                    api_base="mock://localhost",
                    api_key="demo-key"
                )
            ],
            prompt="Explain transformers."
        )
        events = []
        async for event in BenchmarkEngine.stream_compare_events(req):
            events.append(event)
        return events

    events = asyncio.run(_run())
    assert len(events) >= 6
    assert any("event: metadata" in e for e in events)
    assert any("event: ttft" in e for e in events)
    assert any("event: token" in e for e in events)
    assert any("event: completed" in e for e in events)
    assert any("event: all_completed" in e for e in events)
