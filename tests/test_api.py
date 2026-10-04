"""
Integration tests for FastAPI endpoints:
- GET /api/model/metadata
- POST /api/benchmark/compare
- POST /api/benchmark/compare/stream
"""

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app import app


def test_get_model_metadata_endpoint():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get("/api/model/metadata?model_id=meta-llama/Llama-3-8b-instruct&hf_token=test_token")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "meta-llama/Llama-3-8b-instruct"
    assert data["hidden_size"] == 4096
    assert data["num_hidden_layers"] == 32
    assert data["num_attention_heads"] == 32
    assert data["num_key_value_heads"] == 8
    assert data["gqa_ratio"] == 4.0
    assert "Grouped Query Attention (GQA 4:1)" in data["attention_type"]


def test_post_benchmark_compare_endpoint():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            payload = {
                "models": [
                    {
                        "slot_id": "slot_0",
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "api_base": "mock://localhost",
                        "api_key": "demo-key"
                    },
                    {
                        "slot_id": "slot_1",
                        "model_id": "meta-llama/Llama-3-8b-instruct",
                        "api_base": "mock://localhost",
                        "api_key": "demo-key"
                    }
                ],
                "prompt": "Test quantum mechanics",
                "system_prompt": "Answer concisely",
                "max_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.95
            }
            return await ac.post("/api/benchmark/compare", json=payload)

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    res0 = data["results"][0]
    res1 = data["results"][1]
    assert res0["slot_id"] == "slot_0"
    assert res1["slot_id"] == "slot_1"
    assert res0["ttft_ms"] > 0
    assert res1["ttft_ms"] > 0
    assert res0["tps"] > 0
    assert res1["tps"] > 0
    assert "metadata" in res0
    assert "metadata" in res1


def test_get_presets_endpoint():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get("/api/presets")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    assert "default_multi_slots" in data
    assert len(data["endpoints"]) >= 4
