"""
LLM Systems Lab - Model Architecture Registry & Metadata Service
Ingests from Hugging Face Hub (config.json, generation_config.json, README.md),
executes a deterministic calculation engine for active parameters & KV cache footprint,
and provides a curated architectural fallback registry.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Any, Optional, Tuple, List
import httpx

from backend.models import ModelMetadata

logger = logging.getLogger("model_metadata_service")


class ModelMetadataService:
    """
    Tiered architectural inspection engine implementing:
    Tier 1: Hugging Face API Ingestion (config.json, generation_config.json, Hub API, README.md frontmatter)
    Tier 2: Deterministic Calculation Engine (Active vs Total params, KV cache KB/token, GQA ratio)
    Tier 3: Curated Architectural Registry Fallback (DeepSeek MLA, Gemma Sliding-Window, Mistral, Llama, Qwen)
    """

    HF_API_BASE = "https://huggingface.co/api/models/{model_id}"
    HF_RAW_CONFIG = "https://huggingface.co/{model_id}/raw/main/config.json"
    HF_RAW_GEN_CONFIG = "https://huggingface.co/{model_id}/raw/main/generation_config.json"
    HF_RAW_README = "https://huggingface.co/{model_id}/raw/main/README.md"

    # Tier 3: Curated Architectural Registry Fallback
    CURATED_REGISTRY: Dict[str, Dict[str, Any]] = {
        "meta-llama/llama-3.1-8b-instruct": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "8.03B",
            "active_parameters": "8.03B",
            "parameter_count": "8.03B",
            "decoder_topology": "Dense",
            "layer_recipe": "32 Dense GQA Layers (RoPE 500k)",
            "architecture_template": "llama-gqa-dense"
        },
        "meta-llama/meta-llama-3-8b-instruct": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "8.03B",
            "active_parameters": "8.03B",
            "parameter_count": "8.03B",
            "decoder_topology": "Dense",
            "layer_recipe": "32 Dense GQA Layers",
            "architecture_template": "llama-gqa-dense"
        },
        "meta-llama/llama-3-8b-instruct": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "8.03B",
            "active_parameters": "8.03B",
            "parameter_count": "8.03B",
            "decoder_topology": "Dense",
            "layer_recipe": "32 Dense GQA Layers",
            "architecture_template": "llama-gqa-dense"
        },
        "meta-llama/llama-3.1-70b-instruct": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "70.6B",
            "active_parameters": "70.6B",
            "parameter_count": "70.6B",
            "decoder_topology": "Dense",
            "layer_recipe": "80 Dense GQA Layers (8:1 compression)",
            "architecture_template": "llama-gqa-dense"
        },
        "mistralai/mistral-7b-instruct-v0.3": {
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32768,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "sliding_window": 4096,
            "total_parameters": "7.24B",
            "active_parameters": "7.24B",
            "parameter_count": "7.24B",
            "decoder_topology": "Dense (Sliding-Window)",
            "layer_recipe": "32 Sliding-Window Attention Layers (4096 window)",
            "architecture_template": "mistral-swa-dense"
        },
        "mistralai/mixtral-8x7b-instruct-v0.1": {
            "model_type": "mixtral",
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "torch_dtype": "bfloat16",
            "total_parameters": "46.7B",
            "active_parameters": "12.9B",
            "parameter_count": "12.9B Active / 46.7B MoE",
            "decoder_topology": "Mixture-of-Experts (MoE)",
            "layer_recipe": "32 MoE Layers (Top-2 routing of 8 experts)",
            "architecture_template": "generic-moe"
        },
        "google/gemma-2-9b-it": {
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 3584,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "bfloat16",
            "total_parameters": "9.24B",
            "active_parameters": "9.24B",
            "parameter_count": "9.24B",
            "decoder_topology": "Dense (Hybrid Sliding/Global)",
            "layer_recipe": "Alternating Sliding-Window (4096) + Full Attention",
            "architecture_template": "gemma-sliding-dense"
        },
        "google/gemma-2-27b-it": {
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 4608,
            "num_hidden_layers": 46,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "bfloat16",
            "total_parameters": "27.2B",
            "active_parameters": "27.2B",
            "parameter_count": "27.2B",
            "decoder_topology": "Dense (Hybrid Sliding/Global)",
            "layer_recipe": "46 Alternating Local/Global Layers",
            "architecture_template": "gemma-sliding-dense"
        },
        "qwen/qwen2.5-7b-instruct": {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "7.61B",
            "active_parameters": "7.61B",
            "parameter_count": "7.61B",
            "decoder_topology": "Dense",
            "layer_recipe": "28 Dense GQA Layers (7:1 ratio, 128k context)",
            "architecture_template": "llama-gqa-dense"
        },
        "qwen/qwen2.5-72b-instruct": {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 152064,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "total_parameters": "72.7B",
            "active_parameters": "72.7B",
            "parameter_count": "72.7B",
            "decoder_topology": "Dense",
            "layer_recipe": "80 Dense GQA Layers (8:1 ratio, 128k context)",
            "architecture_template": "llama-gqa-dense"
        },
        "deepseek-ai/deepseek-v3": {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "vocab_size": 129280,
            "max_position_embeddings": 131072,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "torch_dtype": "bfloat16",
            "total_parameters": "671B",
            "active_parameters": "37B",
            "parameter_count": "37B Active / 671B MoE",
            "decoder_topology": "Mixture-of-Experts (MoE)",
            "layer_recipe": "60 MLA Layers + 256 Fine-Grained Routed Experts (Top-8) + 1 Shared",
            "architecture_template": "deepseek-mla-moe"
        },
        "deepseek-ai/deepseek-r1": {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "vocab_size": 129280,
            "max_position_embeddings": 131072,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "torch_dtype": "bfloat16",
            "total_parameters": "671B",
            "active_parameters": "37B",
            "parameter_count": "37B Active / 671B MoE",
            "decoder_topology": "Mixture-of-Experts (MoE)",
            "layer_recipe": "60 MLA Layers + 256 Routed Experts (Top-8) + 1 Shared",
            "architecture_template": "deepseek-mla-moe"
        }
    }

    # Backward compatibility alias
    KNOWN_MODEL_PRESETS = CURATED_REGISTRY

    @classmethod
    async def fetch_model_metadata(
        cls,
        model_id: str,
        hf_token: Optional[str] = None,
        timeout: float = 12.0
    ) -> ModelMetadata:
        """
        Main entry point: fetches config.json, generation_config.json, Hub API stats,
        computes deterministic metrics, and applies curated registry fallback.
        """
        raw_config, is_fallback, notes, param_count_hint, total_p_str, active_p_str, gen_context = (
            await cls._query_tiered_hf(model_id, hf_token, timeout)
        )

        return cls.normalize_architecture(
            model_id=model_id,
            raw_config=raw_config,
            is_fallback=is_fallback,
            param_count_hint=param_count_hint,
            total_params_hint=total_p_str,
            active_params_hint=active_p_str,
            gen_context_hint=gen_context,
            extra_notes=notes
        )

    @classmethod
    async def fetch_config_json(
        cls,
        model_id: str,
        hf_token: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """Legacy helper for backwards compatibility."""
        meta = await cls.fetch_model_metadata(model_id, hf_token, timeout)
        return meta.model_dump(), meta.is_fallback, meta.notes

    @classmethod
    async def _query_tiered_hf(
        cls,
        model_id: str,
        hf_token: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[Dict[str, Any], bool, List[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
        """
        Tier 1: Multi-file HF Ingestion (Hub API, config.json, generation_config.json, README.md frontmatter).
        """
        model_clean = model_id.strip()
        notes: List[str] = []
        param_count: Optional[str] = None
        total_p_str: Optional[str] = None
        active_p_str: Optional[str] = None
        gen_context: Optional[int] = None

        effective_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        headers = {
            "User-Agent": "LLM-Systems-Lab-Architecture-Engine/3.0",
            "Accept": "application/json"
        }
        if effective_token:
            headers["Authorization"] = f"Bearer {effective_token.strip()}"

        gated_error: Optional[str] = None
        raw_config: Dict[str, Any] = {}

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                # 1. Fetch Hub API info
                api_url = cls.HF_API_BASE.format(model_id=model_clean)
                api_resp = await client.get(api_url, headers=headers)
                if api_resp.status_code == 200:
                    api_data = api_resp.json()
                    safetensors_info = api_data.get("safetensors", {})
                    if isinstance(safetensors_info, dict) and "total" in safetensors_info:
                        total_p = safetensors_info["total"]
                        total_p_str = f"{total_p / 1e9:.2f}B"
                        param_count = total_p_str
                    elif "transformersInfo" in api_data and "num_parameters" in api_data["transformersInfo"]:
                        num_p = api_data["transformersInfo"]["num_parameters"]
                        total_p_str = f"{num_p / 1e9:.2f}B"
                        param_count = total_p_str
                elif api_resp.status_code in (401, 403):
                    gated_error = (
                        "Access denied on Hugging Face Hub (HTTP 401/403). Ensure you have accepted "
                        "the model license agreement and provided a valid Hugging Face access token."
                    )
                    notes.append(gated_error)

                # 2. Fetch config.json
                raw_url = cls.HF_RAW_CONFIG.format(model_id=model_clean)
                resp = await client.get(raw_url, headers=headers)
                if resp.status_code == 200:
                    raw_config = resp.json()
                    notes.append(f"Ingested config.json directly from Hugging Face Hub for {model_clean}.")
                elif resp.status_code in (401, 403):
                    gated_error = (
                        "Access denied on Hugging Face Hub (HTTP 401/403). Ensure you have accepted "
                        "the model license agreement and provided a valid Hugging Face access token."
                    )
                    notes.append(gated_error)

                # 3. Fetch generation_config.json for context windows or generation defaults
                if raw_config:
                    try:
                        gen_url = cls.HF_RAW_GEN_CONFIG.format(model_id=model_clean)
                        gen_resp = await client.get(gen_url, headers=headers)
                        if gen_resp.status_code == 200:
                            gen_data = gen_resp.json()
                            if "max_length" in gen_data and isinstance(gen_data["max_length"], int):
                                gen_context = gen_data["max_length"]
                    except Exception:
                        pass

                # 4. Fetch README.md frontmatter if needed for parameter tags
                if raw_config and not param_count:
                    try:
                        readme_url = cls.HF_RAW_README.format(model_id=model_clean)
                        readme_resp = await client.get(readme_url, headers=headers)
                        if readme_resp.status_code == 200:
                            readme_txt = readme_resp.text[:3000]
                            # Check tags or model-index
                            p_match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*(?:parameters|params)?", readme_txt)
                            if p_match:
                                param_count = f"{p_match.group(1)}B"
                    except Exception:
                        pass

                if raw_config:
                    return raw_config, False, notes, param_count, total_p_str, active_p_str, gen_context

        except Exception as exc:
            notes.append(f"Network error querying Hugging Face: {str(exc)}")

        # Tier 3: Curated Architectural Registry Fallback Lookup
        slug = model_clean.lower()
        for reg_key, reg_data in cls.CURATED_REGISTRY.items():
            if slug == reg_key or slug.endswith("/" + reg_key.split("/")[-1]) or slug == reg_key.split("/")[-1]:
                notes.append(f"Loaded curated architectural registry profile for '{model_clean}'.")
                return (
                    reg_data,
                    True,
                    notes,
                    reg_data.get("parameter_count"),
                    reg_data.get("total_parameters"),
                    reg_data.get("active_parameters"),
                    reg_data.get("max_position_embeddings")
                )

        if gated_error:
            raise PermissionError(gated_error)

        if "/" in model_clean:
            notes.append(f"Inferred baseline transformer structural parameters for '{model_clean}'.")
            return cls._generate_generic_fallback(model_clean), True, notes, None, None, None, None

        raise ValueError(
            f"Unable to extract metadata for '{model_clean}'. " + " | ".join(notes)
        )

    @classmethod
    def _generate_generic_fallback(cls, model_id: str) -> Dict[str, Any]:
        """Generate safe baseline transformer architecture."""
        slug = model_id.lower()
        if "70b" in slug:
            return {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}
        elif "8b" in slug or "7b" in slug:
            return {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 8192, "torch_dtype": "bfloat16"}
        elif "14b" in slug or "13b" in slug:
            return {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 8, "max_position_embeddings": 32768, "torch_dtype": "bfloat16"}
        return {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 4096, "torch_dtype": "bfloat16"}

    @classmethod
    def normalize_architecture(
        cls,
        model_id: str,
        raw_config: Dict[str, Any],
        is_fallback: bool = False,
        param_count_hint: Optional[str] = None,
        total_params_hint: Optional[str] = None,
        active_params_hint: Optional[str] = None,
        gen_context_hint: Optional[int] = None,
        extra_notes: Optional[List[str]] = None
    ) -> ModelMetadata:
        """
        Tier 2: Deterministic Calculation Engine
        - Computes KV cache footprint (KB / token and Bytes / token)
        - Calculates exact MQA / GQA / MHA compression ratios
        - Derives Total vs Active parameter footprint
        - Analyzes layer recipe & sliding window distribution
        - Maps to architectural block template
        """
        notes = list(extra_notes or [])

        # Model type & architectures
        model_type = raw_config.get("model_type") or raw_config.get("type", "custom")
        architectures = raw_config.get("architectures", [])
        if not architectures and model_type:
            architectures = [f"{str(model_type).capitalize()}ForCausalLM"]

        # Hidden Size (Primary: hidden_size | Fallbacks: d_model, n_embd, dim, model_dim)
        hidden_size = (
            raw_config.get("hidden_size") or
            raw_config.get("d_model") or
            raw_config.get("n_embd") or
            raw_config.get("dim") or
            raw_config.get("model_dim") or
            4096
        )

        # Number of Layers (Primary: num_hidden_layers | Fallbacks: n_layer, num_layers, layers)
        num_hidden_layers = (
            raw_config.get("num_hidden_layers") or
            raw_config.get("n_layer") or
            raw_config.get("num_layers") or
            raw_config.get("layers") or
            32
        )

        # Attention Heads (Primary: num_attention_heads | Fallbacks: n_head, num_heads, heads)
        num_attention_heads = (
            raw_config.get("num_attention_heads") or
            raw_config.get("n_head") or
            raw_config.get("num_heads") or
            raw_config.get("heads") or
            32
        )

        # Key/Value Heads
        num_key_value_heads = raw_config.get("num_key_value_heads")
        if num_key_value_heads is None:
            num_key_value_heads = raw_config.get("num_kv_heads") or raw_config.get("n_head_kv")

        if num_key_value_heads is None:
            if raw_config.get("multi_query") is True or raw_config.get("multi_query_attention") is True:
                num_key_value_heads = 1
                notes.append("Detected multi_query flag -> Normalized to 1 KV Head (MQA).")
            else:
                num_key_value_heads = num_attention_heads

        hidden_size = int(hidden_size)
        num_hidden_layers = int(num_hidden_layers)
        num_attention_heads = int(num_attention_heads)
        num_key_value_heads = int(num_key_value_heads)

        # Head dimension
        head_dim = raw_config.get("head_dim")
        if not head_dim and num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads
        else:
            head_dim = int(head_dim) if head_dim else 128

        # GQA Ratio Calculation
        if num_key_value_heads > 0:
            gqa_ratio = round(num_attention_heads / num_key_value_heads, 2)
        else:
            gqa_ratio = 1.0

        # Precision Bytes (Default 2 bytes for FP16/BF16, 1 for FP8, 4 for FP32)
        torch_dtype = str(raw_config.get("torch_dtype", "bfloat16")).lower()
        if "fp8" in torch_dtype:
            precision_bytes = 1
        elif "float32" in torch_dtype or "fp32" in torch_dtype:
            precision_bytes = 4
        else:
            precision_bytes = 2  # standard 16-bit (bfloat16 / float16)

        # Attention Variant & KV Cache Formulation
        is_mla = (
            "kv_lora_rank" in raw_config or
            "q_lora_rank" in raw_config or
            "deepseek" in str(model_type).lower() or
            "deepseek" in model_id.lower() or
            "mla" in str(raw_config).lower()
        )

        if is_mla:
            attention_type = "Multi-Head Latent Attention (MLA)"
            dc = raw_config.get("kv_lora_rank", 512)
            dr = raw_config.get("qk_rope_head_dim", 64)
            # In MLA, per layer KV cache stores compressed latent vector (d_c) + decoupled key RoPE (d_r)
            bytes_per_token = int(num_hidden_layers * (dc + dr) * precision_bytes)
            notes.append(f"MLA Active: Low-rank latent KV compression (d_c={dc}, d_r={dr}).")
        elif num_key_value_heads == 1 and num_attention_heads > 1:
            attention_type = "Multi-Query Attention (MQA)"
            # KV Cache = 2 * L * 1 * head_dim * bytes_per_element
            bytes_per_token = int(2 * num_hidden_layers * 1 * head_dim * precision_bytes)
        elif num_key_value_heads < num_attention_heads:
            attention_type = f"Grouped Query Attention (GQA {int(gqa_ratio)}:1)"
            # KV Cache = 2 * L * kv_heads * head_dim * bytes_per_element
            bytes_per_token = int(2 * num_hidden_layers * num_key_value_heads * head_dim * precision_bytes)
        else:
            attention_type = "Multi-Head Attention (MHA)"
            # KV Cache = 2 * L * num_heads * head_dim * bytes_per_element
            bytes_per_token = int(2 * num_hidden_layers * num_attention_heads * head_dim * precision_bytes)

        kv_cache_kb_per_token = round(bytes_per_token / 1024.0, 3)

        # Context Window
        context_window = (
            raw_config.get("max_position_embeddings") or
            raw_config.get("seq_length") or
            raw_config.get("n_positions") or
            raw_config.get("max_sequence_length") or
            raw_config.get("max_seq_len") or
            gen_context_hint
        )
        if context_window:
            context_window = int(context_window)

        # Vocab Size
        vocab_size = raw_config.get("vocab_size")
        if vocab_size:
            vocab_size = int(vocab_size)

        # RoPE Theta
        rope_theta = raw_config.get("rope_theta")
        if rope_theta is None and isinstance(raw_config.get("rope_scaling"), dict):
            rope_theta = raw_config["rope_scaling"].get("original_max_position_embeddings") or raw_config["rope_scaling"].get("factor")
        if rope_theta is not None:
            rope_theta = float(rope_theta)

        # Sliding Window Attention
        sliding_window = raw_config.get("sliding_window")
        if sliding_window is not None:
            sliding_window = int(sliding_window)

        # MoE Topology & Expert Count Derivation
        num_local_experts = (
            raw_config.get("num_local_experts") or
            raw_config.get("n_routed_experts") or
            raw_config.get("num_experts")
        )
        num_active_experts = (
            raw_config.get("num_experts_per_tok") or
            raw_config.get("num_active_experts")
        )
        if num_local_experts:
            num_local_experts = int(num_local_experts)
        if num_active_experts:
            num_active_experts = int(num_active_experts)

        is_moe = bool(num_local_experts and num_local_experts > 1) or "mixtral" in str(model_type).lower()
        if is_moe:
            decoder_topology = "Mixture-of-Experts (MoE)"
        elif sliding_window:
            decoder_topology = "Dense (Sliding-Window)"
        else:
            decoder_topology = "Dense"

        # Deterministic Total vs Active Parameter Derivation
        intermediate_size = raw_config.get("intermediate_size") or (hidden_size * 4)
        intermediate_size = int(intermediate_size)

        total_parameters = total_params_hint or raw_config.get("total_parameters")
        active_parameters = active_params_hint or raw_config.get("active_parameters")
        parameter_count = param_count_hint or raw_config.get("parameter_count")

        if not total_parameters or not active_parameters:
            if is_moe:
                # Active params = Embeddings + Attention + (Active Experts * FFN Block)
                # FFN Block per expert ≈ 3 * hidden_size * intermediate_size (for SwiGLU)
                n_active = num_active_experts or 2
                n_total = num_local_experts or 8
                ffn_per_expert = 3 * hidden_size * intermediate_size
                shared_experts = raw_config.get("n_shared_experts", 0)
                shared_ffn = shared_experts * ffn_per_expert
                embed_params = (vocab_size or 32000) * hidden_size
                attn_params = num_hidden_layers * (4 * hidden_size * hidden_size)

                active_p_val = (embed_params + attn_params + (num_hidden_layers * (n_active * ffn_per_expert + shared_ffn))) / 1e9
                total_p_val = (embed_params + attn_params + (num_hidden_layers * (n_total * ffn_per_expert + shared_ffn))) / 1e9

                if not active_parameters:
                    active_parameters = f"{active_p_val:.1f}B"
                if not total_parameters:
                    total_parameters = f"{total_p_val:.1f}B"
                if not parameter_count:
                    parameter_count = f"{active_parameters} Active / {total_parameters} MoE"
            else:
                # Standard dense transformer
                # Approx params = 12 * L * hidden_size^2 (approximate standard scaling)
                # or calculated: embed = V * H, attn = 4 * H^2 * L, mlp = 3 * H * I * L
                embed_p = (vocab_size or 32000) * hidden_size
                attn_p = num_hidden_layers * (4 * hidden_size * hidden_size)
                mlp_p = num_hidden_layers * (3 * hidden_size * intermediate_size)
                dense_p_val = (embed_p + attn_p + mlp_p) / 1e9

                if not total_parameters:
                    total_parameters = f"{dense_p_val:.2f}B" if dense_p_val > 0.5 else None
                if not active_parameters:
                    active_parameters = total_parameters
                if not parameter_count:
                    parameter_count = total_parameters

        # Layer Recipe Synthesis
        layer_recipe = raw_config.get("layer_recipe")
        if not layer_recipe:
            if is_moe:
                layer_recipe = f"{num_hidden_layers} Layers (Top-{num_active_experts or 2} of {num_local_experts or 8} routed experts)"
            elif sliding_window:
                if "gemma" in str(model_type).lower():
                    layer_recipe = f"{num_hidden_layers} Hybrid Layers (Sliding {sliding_window} + Global MHA)"
                else:
                    layer_recipe = f"{num_hidden_layers} Sliding-Window Layers (Window: {sliding_window})"
            else:
                ratio_str = f" ({int(gqa_ratio)}:1 GQA)" if gqa_ratio > 1 else " (MHA)"
                layer_recipe = f"{num_hidden_layers} Uniform Dense Layers{ratio_str}"

        # Architectural Block Template Mapping
        if is_mla:
            architecture_template = "deepseek-mla-moe"
        elif "gemma" in str(model_type).lower() or "sliding" in decoder_topology.lower():
            architecture_template = "gemma-sliding-dense"
        elif is_moe:
            architecture_template = "generic-moe"
        elif "mistral" in str(model_type).lower():
            architecture_template = "mistral-swa-dense"
        else:
            architecture_template = "llama-gqa-dense"

        # Memory footprint in FP16 (2 bytes per total param)
        weight_memory_fp16_gb = None
        p_ref = total_parameters or parameter_count
        if p_ref and "B" in str(p_ref):
            try:
                num_match = re.findall(r"[\d\.]+", str(p_ref))
                if num_match:
                    params_b = float(num_match[0])
                    weight_memory_fp16_gb = round(params_b * 2.0, 1)
            except Exception:
                pass

        return ModelMetadata(
            model_id=model_id,
            model_type=str(model_type),
            architectures=[str(a) for a in architectures],
            parameter_count=str(parameter_count) if parameter_count else None,
            total_parameters=str(total_parameters) if total_parameters else None,
            active_parameters=str(active_parameters) if active_parameters else None,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_type=attention_type,
            gqa_ratio=gqa_ratio,
            kv_cache_kb_per_token=kv_cache_kb_per_token,
            bytes_per_token_fp16=bytes_per_token,
            context_window=context_window,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            torch_dtype=torch_dtype,
            decoder_topology=decoder_topology,
            layer_recipe=layer_recipe,
            sliding_window=sliding_window,
            num_experts=num_local_experts,
            num_active_experts=num_active_experts,
            architecture_template=architecture_template,
            weight_memory_fp16_gb=weight_memory_fp16_gb,
            is_fallback=is_fallback,
            notes=notes
        )


# Backward-compatible alias
HFConfigParser = ModelMetadataService
