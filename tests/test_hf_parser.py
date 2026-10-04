"""
Unit tests for Hugging Face Config Ingestion & Schema Normalization Service.
Validates key normalization, fallbacks, GQA ratio calculation, and MLA detection.
"""

import pytest
from backend.hf_parser import HFConfigParser
from backend.models import NormalizedArchitecture


def test_schema_normalization_standard_llama():
    raw_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16"
    }

    arch = HFConfigParser.normalize_architecture("meta-llama/Llama-3-8b-instruct", raw_config)

    assert arch.hidden_size == 4096
    assert arch.num_hidden_layers == 32
    assert arch.num_attention_heads == 32
    assert arch.num_key_value_heads == 8
    assert arch.gqa_ratio == 4.0
    assert "Grouped Query Attention (GQA 4:1)" in arch.attention_variant
    assert arch.max_position_embeddings == 8192
    assert arch.bytes_per_token_fp16 == 131072


def test_schema_normalization_fallbacks():
    """Verify fallback keys: d_model, n_layer, n_head, seq_length."""
    raw_config = {
        "model_type": "custom_transformer",
        "d_model": 2048,
        "n_layer": 24,
        "n_head": 16,
        # num_key_value_heads missing -> fallback to num_attention_heads (MHA)
        "seq_length": 4096,
        "vocab_size": 50257
    }

    arch = HFConfigParser.normalize_architecture("test/custom-model", raw_config)

    assert arch.hidden_size == 2048
    assert arch.num_hidden_layers == 24
    assert arch.num_attention_heads == 16
    assert arch.num_key_value_heads == 16  # Fallback to MHA
    assert arch.gqa_ratio == 1.0
    assert arch.attention_variant == "Multi-Head Attention (MHA)"
    assert arch.max_position_embeddings == 4096


def test_multi_query_attention_flag_fallback():
    """Verify multi_query flag normalizes num_key_value_heads to 1 (MQA)."""
    raw_config = {
        "model_type": "falcon",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "multi_query": True,
        "max_position_embeddings": 2048
    }

    arch = HFConfigParser.normalize_architecture("tiiuae/falcon-7b", raw_config)

    assert arch.num_key_value_heads == 1
    assert arch.gqa_ratio == 32.0
    assert arch.attention_variant == "Multi-Query Attention (MQA)"


def test_deepseek_mla_detection():
    """Verify DeepSeek MLA low-rank latent KV cache formulation."""
    raw_config = {
        "model_type": "deepseek_v3",
        "hidden_size": 5120,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 131072
    }

    arch = HFConfigParser.normalize_architecture("deepseek-ai/DeepSeek-V3", raw_config)

    assert "Multi-Head Latent Attention (MLA)" in arch.attention_variant
    # L * (d_c + d_r) * 2 = 60 * (512 + 64) * 2 = 69,120 bytes
    assert arch.bytes_per_token_fp16 == 69120


def test_fetch_config_json_preset_fallback():
    """Verify known preset fallback returns valid architecture even without active HF internet access."""
    import asyncio
    raw_cfg, is_fallback, notes = asyncio.run(
        HFConfigParser.fetch_config_json("meta-llama/Llama-3-8b-instruct")
    )
    assert raw_cfg is not None
    assert "hidden_size" in raw_cfg
    assert raw_cfg["hidden_size"] == 4096


def test_fetch_model_metadata_parameter_counts():
    """Verify metadata fetcher derives parameter count and context window."""
    import asyncio
    meta = asyncio.run(
        HFConfigParser.fetch_model_metadata("Qwen/Qwen2.5-7B-Instruct")
    )
    assert meta.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert meta.hidden_size == 3584
    assert meta.num_hidden_layers == 28
    assert meta.context_window in (32768, 131072)
    assert meta.parameter_count is not None


def test_model_metadata_service_deterministic_calculations():
    """Verify Tier 2 deterministic calculations: KV cache KB/token, MoE active params, and layer recipes."""
    from backend.hf_parser import ModelMetadataService

    # 1. Test MoE Active vs Total Parameter Derivation (DeepSeek-V3)
    deepseek_config = {
        "model_type": "deepseek_v3",
        "hidden_size": 5120,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 131072,
        "total_parameters": "671B",
        "active_parameters": "37B"
    }
    meta_ds = ModelMetadataService.normalize_architecture("deepseek-ai/DeepSeek-V3", deepseek_config)
    assert meta_ds.decoder_topology == "Mixture-of-Experts (MoE)"
    assert meta_ds.total_parameters == "671B"
    assert meta_ds.active_parameters == "37B"
    assert meta_ds.kv_cache_kb_per_token == 67.5  # 69120 bytes / 1024 = 67.5 KB/token
    assert meta_ds.architecture_template == "deepseek-mla-moe"

    # 2. Test Gemma Sliding-Window Layer Recipe
    gemma_config = {
        "model_type": "gemma2",
        "hidden_size": 3584,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "sliding_window": 4096,
        "max_position_embeddings": 8192
    }
    meta_gemma = ModelMetadataService.normalize_architecture("google/gemma-2-9b-it", gemma_config)
    assert "Sliding-Window" in meta_gemma.decoder_topology
    assert meta_gemma.architecture_template == "gemma-sliding-dense"
    assert meta_gemma.gqa_ratio == 2.0
    assert meta_gemma.sliding_window == 4096
    assert meta_gemma.kv_cache_kb_per_token is not None

