"""
LLM Systems Lab - FastAPI Application & Inspection Engine
Endpoints for arbitrary multi-model comparisons, metadata fetching,
and concurrent SSE streaming latency telemetry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    ModelSlotConfig,
    CompareBenchmarkRequest,
    CompareBenchmarkResponse,
    CompareModelResult,
    ModelMetadata,
    # Backward compatibility aliases
    BenchmarkRequest,
    BenchmarkResponse,
    MultiBenchmarkRequest,
    MultiBenchmarkResponse,
)
from backend.hf_parser import ModelMetadataService, HFConfigParser
from backend.benchmark_engine import BenchmarkEngine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
    title="LLM Systems Lab - Dynamic Model Inspection & Benchmarking Engine",
    description="End-to-end telemetry platform extracting Hugging Face architecture metadata and measuring real-time TTFT/TPS latency across arbitrary concurrent models.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# -----------------------------------------------------------------------------
# 1. METADATA FETCHER ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/model/metadata", response_model=ModelMetadata)
async def get_model_metadata(
    model_id: str = Query(..., description="Hugging Face Model ID, e.g. Qwen/Qwen2.5-7B-Instruct"),
    hf_token: Optional[str] = Query(None, description="Optional Hugging Face access token for gated models")
):
    """
    Query Hugging Face Hub API and config.json to extract normalized architecture specifications.
    Diagnoses license acceptance/gated issues with explicit guidance.
    """
    try:
        metadata = await HFConfigParser.fetch_model_metadata(
            model_id=model_id,
            hf_token=hf_token
        )
        return metadata
    except PermissionError as pe:
        raise HTTPException(
            status_code=403,
            detail=str(pe)
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error extracting metadata for {model_id}: {str(exc)}")


@app.get("/api/model-config", response_model=ModelMetadata)
async def get_model_config_alias(
    model_id: str = Query(..., description="Hugging Face Model ID"),
    hf_token: Optional[str] = Query(None)
):
    """Backward-compatible alias for /api/model/metadata."""
    return await get_model_metadata(model_id=model_id, hf_token=hf_token)


# -----------------------------------------------------------------------------
# 2. INFERENCE ORCHESTRATOR & BENCHMARKING ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/benchmark/compare", response_model=CompareBenchmarkResponse)
async def compare_benchmark(req: CompareBenchmarkRequest):
    """
    Execute concurrent benchmarking across an arbitrary list of model configurations.
    Returns complete metrics (TTFT, TPS, latency) and extracted architecture metadata.
    """
    try:
        response = await BenchmarkEngine.compare_models(req)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison benchmark error: {str(exc)}")


@app.post("/api/benchmark/compare/stream")
async def stream_compare_benchmark(req: CompareBenchmarkRequest):
    """
    Multiplexed Server-Sent Events (SSE) endpoint concurrently streaming live tokens
    and real-time TTFT and TPS latency metrics across N user-configured models.
    """
    return StreamingResponse(
        BenchmarkEngine.stream_compare_events(req),
        media_type="text/event-stream"
    )


# Backward-compatible endpoints
@app.post("/api/benchmark-multi", response_model=CompareBenchmarkResponse)
async def run_benchmark_multi_alias(req: CompareBenchmarkRequest):
    return await compare_benchmark(req)


@app.post("/api/benchmark-multi/stream")
async def stream_benchmark_multi_alias(req: CompareBenchmarkRequest):
    return await stream_compare_benchmark(req)


@app.post("/api/benchmark", response_model=CompareModelResult)
async def run_single_benchmark_legacy(slot: ModelSlotConfig):
    req = CompareBenchmarkRequest(
        models=[slot],
        prompt="Explain quantum computing in 3 sentences."
    )
    resp = await BenchmarkEngine.compare_models(req)
    return resp.results[0]


@app.get("/api/presets")
async def get_presets():
    """Reference endpoints and sample model suggestions."""
    return {
        "endpoints": [
            {
                "name": "Hugging Face Inference API",
                "url": "https://api-inference.huggingface.co/v1/",
                "note": "Standard HF Serverless API"
            },
            {
                "name": "NVIDIA NIM",
                "url": "https://integrate.api.nvidia.com/v1",
                "note": "High throughput TensorRT-LLM"
            },
            {
                "name": "vLLM Local Server",
                "url": "http://localhost:8000/v1",
                "note": "Self-hosted vLLM engine"
            },
            {
                "name": "Ollama Local API",
                "url": "http://localhost:11434/v1",
                "note": "Local Ollama service"
            },
            {
                "name": "Local Mock Simulator",
                "url": "mock://localhost",
                "note": "Simulation mode (zero API keys needed)"
            }
        ],
        "default_multi_slots": [
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "api_base": "https://api-inference.huggingface.co/v1/"
            },
            {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "api_base": "https://api-inference.huggingface.co/v1/"
            },
            {
                "model_id": "deepseek-ai/DeepSeek-V3",
                "api_base": "https://api-inference.huggingface.co/v1/"
            }
        ]
    }


@app.get("/", response_class=HTMLResponse)
async def serve_root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>LLM Systems Lab API is running. Frontend not found.</h1>")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
