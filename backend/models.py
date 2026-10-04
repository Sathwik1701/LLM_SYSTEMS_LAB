"""
LLM Systems Lab - Dynamic Model Architecture & Benchmark Schemas
Strict Pydantic schemas supporting arbitrary multi-model slot configurations,
metadata extraction, and live streaming latency telemetry.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Normalized architecture specifications and static hardware estimation."""
    model_id: str
    model_type: str = "unknown"
    architectures: List[str] = Field(default_factory=list)
    parameter_count: Optional[str] = None
    total_parameters: Optional[str] = None
    active_parameters: Optional[str] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_type: str = "Multi-Head Attention (MHA)"
    gqa_ratio: Optional[float] = None
    kv_cache_kb_per_token: Optional[float] = None
    bytes_per_token_fp16: Optional[int] = None
    context_window: Optional[int] = None
    vocab_size: Optional[int] = None
    rope_theta: Optional[float] = None
    torch_dtype: Optional[str] = None
    decoder_topology: str = "Dense"
    layer_recipe: Optional[str] = None
    sliding_window: Optional[int] = None
    num_experts: Optional[int] = None
    num_active_experts: Optional[int] = None
    architecture_template: str = "generic-dense"
    weight_memory_fp16_gb: Optional[float] = None
    is_fallback: bool = False
    notes: List[str] = Field(default_factory=list)

    @property
    def attention_variant(self) -> str:
        return self.attention_type

    @property
    def max_position_embeddings(self) -> Optional[int]:
        return self.context_window


# Backward-compatible alias
NormalizedArchitecture = ModelMetadata


class ModelSlotConfig(BaseModel):
    """User-defined configuration for a single model slot in arbitrary N-slot comparisons."""
    slot_id: Optional[str] = None
    model_id: str = Field(..., description="Hugging Face Model ID / Repo Path, e.g. Qwen/Qwen2.5-7B-Instruct")
    api_base: str = Field("https://api-inference.huggingface.co/v1/", description="OpenAI-compatible inference base URL")
    api_key: Optional[str] = Field("", description="Inference API Key or HF Token for this endpoint")
    hf_token: Optional[str] = Field(None, description="Optional specific HF token for gated repository access")


class CompareBenchmarkRequest(BaseModel):
    """Payload for comparing N arbitrary models simultaneously."""
    models: List[ModelSlotConfig] = Field(..., min_length=1, description="List of 1 to N models to benchmark side-by-side")
    prompt: str = Field("Explain quantum computing in 3 sentences.", description="Benchmark input prompt")
    system_prompt: Optional[str] = Field(None, description="Optional system instructions")
    max_tokens: int = Field(256, ge=16, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    global_hf_token: Optional[str] = Field(None, description="Global Hugging Face token fallback")


class CompareModelResult(BaseModel):
    """Telemetry report and metadata for a single model in a comparison run."""
    slot_id: str
    model_id: str
    model_index: int
    metadata: ModelMetadata
    generated_text: str
    token_count: int
    ttft_ms: float               # Time to First Token (ms)
    tpot_ms: float               # Time per Output Token (ms)
    tps: float                   # Tokens Per Second (generation rate)
    total_latency_ms: float       # Total end-to-end latency (ms)
    error: Optional[str] = None


class CompareBenchmarkResponse(BaseModel):
    """Aggregated response for N-model comparison."""
    results: List[CompareModelResult]


# Legacy aliases for backward compatibility
BenchmarkRequest = ModelSlotConfig
BenchmarkResponse = CompareModelResult
MultiBenchmarkRequest = CompareBenchmarkRequest
MultiBenchmarkResponse = CompareBenchmarkResponse
ModelBenchmarkProfile = ModelSlotConfig
