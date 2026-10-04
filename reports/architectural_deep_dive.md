# Comparative Architectural & Latency Telemetry Report

This report documents the comparative architectural characteristics, memory footprint dynamics, and latency telemetry profiles of modern open-source language models analyzed using the LLM Architecture & Telemetry Engine.

---

## 1. Architectural Taxonomy & Topology Comparison

Modern open-source LLMs employ diverse attention variants and feed-forward topologies designed to balance representation capacity with inference efficiency:

| Model ID | Attention Type | KV Compression Ratio | Layer Topology | Active / Total Params | Context Window |
|---|---|---|---|---|---|
| **meta-llama/Llama-3.1-8B-Instruct** | Grouped-Query Attention (GQA) | 4:1 (32 Q / 8 KV) | Dense RMSNorm SwiGLU | 8.03B / 8.03B (100%) | 131,072 |
| **Qwen/Qwen2.5-7B-Instruct** | Grouped-Query Attention (GQA) | 7:1 (28 Q / 4 KV) | Dense RMSNorm SwiGLU | 7.61B / 7.61B (100%) | 131,072 |
| **mistralai/Mistral-7B-Instruct-v0.3** | Sliding-Window Attention (SWA) | 4:1 (32 Q / 8 KV) | Dense SwiGLU (Window 4096) | 7.25B / 7.25B (100%) | 32,768 |
| **google/gemma-2-9b-it** | Hybrid Sliding & Global Attention | 2:1 (16 Q / 8 KV) | Alternating SWA / Global Dense | 9.24B / 9.24B (100%) | 8,192 |
| **deepseek-ai/DeepSeek-V3** | Multi-Head Latent Attention (MLA) | Low-Rank Latent ($d_c=512$, $d_r=64$) | Sparse MoE (Top-8 / 256 + Shared) | 37B / 671B (~5.5%) | 128,000 |

---

## 2. KV Cache Footprint Calculations Across Precision Tiers

The Key-Value (KV) cache memory footprint per token is dictated by layer count, key-value heads, head dimension, and numerical precision:

$$\text{KV Cache Bytes/Token} = 2 \times \text{Layers} \times \text{KV Heads} \times \text{Head Dim} \times \text{Bytes per Precision Element}$$

### Footprint Per Token:

* **Llama-3.1-8B (32 Layers, 8 KV heads, Head Dim 128):**
  * FP16 / BF16 (2 bytes): $2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} \approx 128.0 \text{ KB/token}$
  * FP8 (1 byte): $64.0 \text{ KB/token}$
  * INT4 (0.5 bytes): $32.0 \text{ KB/token}$

* **Qwen2.5-7B (28 Layers, 4 KV heads, Head Dim 128):**
  * FP16 / BF16 (2 bytes): $2 \times 28 \times 4 \times 128 \times 2 = 57,344 \text{ bytes} \approx 56.0 \text{ KB/token}$
  * FP8 (1 byte): $28.0 \text{ KB/token}$
  * INT4 (0.5 bytes): $14.0 \text{ KB/token}$

* **DeepSeek-V3 (61 Layers, MLA Latent Compression):**
  * MLA projects KV vectors into a compressed latent vector of dimension $d_c = 512$ with decoupled RoPE dimension $d_r = 64$.
  * FP16 / BF16: $2 \times 61 \times (512 + 64) \times 2 \approx 140,544 \text{ bytes} \approx 137.25 \text{ KB/token}$
  * Despite having 671B parameters and 61 layers, MLA bounds KV memory to values comparable to 8B dense models.

---

## 3. Real-Time Telemetry & Hardware-Independent Latency Metrics

Inference performance across provider endpoints is continuously instrumented via two primary latency indicators:

1. **Time to First Token (TTFT):** Measures the duration between prompt dispatch and reception of the first streamed token chunk:
   $$\text{TTFT} = T_{\text{first chunk}} - T_{\text{dispatch}}$$
   * Indicates prompt processing/prefill throughput and provider queuing latency.

2. **Time Per Output Token (TPOT):** Measures the average generation duration across decoded tokens:
   $$\text{TPOT} = \frac{T_{\text{final}} - T_{\text{first chunk}}}{\max(1, N_{\text{tokens}} - 1)}$$
   * Reflects autoregressive decode speed and memory-bandwidth saturation.

3. **Decode Throughput (TPS):**
   $$\text{Throughput} = \frac{N_{\text{tokens}}}{T_{\text{final}} - T_{\text{first chunk}}} \quad (\text{tokens/sec})$$

---

## 4. Key Architectural Takeaways

1. **GQA Dominance in Dense Models:** 4:1 and 7:1 compression ratios in Llama 3.1 and Qwen 2.5 substantially reduce KV memory bandwidth bottlenecks during generation while maintaining full context quality.
2. **MoE Parameter Decoupling:** Frontier models like DeepSeek-V3 achieve 671B knowledge capacity while requiring only ~37B active parameters per token, drastically lowering inference FLOPs.
3. **Sliding Window Cache Eviction:** Mistral and Gemma utilize local sliding window attention to enforce an upper bound on cache size for long generation sequences.
