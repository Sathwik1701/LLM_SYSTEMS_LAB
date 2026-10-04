"""
LLM Systems Lab - Live Streaming Inference & Latency Telemetry Engine
Orchestrates concurrent asynchronous inference across arbitrary N-model configurations,
measuring TTFT, TPS, and Total Latency with SSE multiplexing and robust licensing error diagnostics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, Tuple, List
import httpx

from backend.models import (
    ModelSlotConfig,
    CompareBenchmarkRequest,
    CompareBenchmarkResponse,
    CompareModelResult,
    ModelMetadata,
)
from backend.hf_parser import HFConfigParser

logger = logging.getLogger("benchmark_engine")


class BenchmarkEngine:
    """
    Inference orchestrator for concurrent multi-model benchmarking.
    """

    @staticmethod
    def normalize_endpoint(api_base: str) -> str:
        """Ensure endpoint ends with /chat/completions for OpenAI-compatible providers."""
        base = api_base.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @classmethod
    async def run_single_slot(
        cls,
        slot_idx: int,
        slot: ModelSlotConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        global_hf_token: Optional[str] = None,
        timeout: float = 60.0
    ) -> CompareModelResult:
        """
        Execute an end-to-end benchmark run for a single model slot.
        """
        slot_id = slot.slot_id or f"slot_{slot_idx}"
        hf_token = slot.hf_token or global_hf_token

        # Step 1: Fetch architecture metadata
        try:
            metadata = await HFConfigParser.fetch_model_metadata(
                model_id=slot.model_id,
                hf_token=hf_token
            )
        except PermissionError as pe:
            metadata = ModelMetadata(
                model_id=slot.model_id,
                model_type="gated_or_private",
                is_fallback=True,
                notes=[str(pe)]
            )
        except Exception as exc:
            metadata = ModelMetadata(
                model_id=slot.model_id,
                model_type="unknown",
                is_fallback=True,
                notes=[f"Metadata fetch failed: {str(exc)}"]
            )

        # Step 2: Handle mock/simulated run if requested
        if slot.api_base.lower().startswith("mock") or slot.api_key == "demo-key" or not slot.api_key.strip():
            return await cls._run_simulated_slot(slot_idx, slot_id, slot.model_id, metadata)

        # Step 3: Live streaming inference
        endpoint = cls.normalize_endpoint(slot.api_base)
        effective_key = slot.api_key.strip() if slot.api_key else (hf_token or "")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}"
        }

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": slot.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }

        generated_chunks: List[str] = []
        t0 = time.perf_counter()
        t1: Optional[float] = None
        tn: Optional[float] = None
        error_msg: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        raw_err = err_body.decode('utf-8', errors='ignore')
                        diag = cls.classify_inference_error(
                            status_code=response.status_code,
                            raw_error=raw_err,
                            model_id=slot.model_id,
                            api_base=slot.api_base
                        )
                        error_msg = f"[{diag['badge']}] {diag['summary']} (Root cause: {diag['root_cause']}. Fix: {diag['fix_suggestion']})"
                    else:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                tn = time.perf_counter()
                                break

                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if t1 is None:
                                        t1 = time.perf_counter()
                                    generated_chunks.append(content)
                            except json.JSONDecodeError:
                                continue

            if tn is None:
                tn = time.perf_counter()
            if t1 is None:
                t1 = tn

        except Exception as exc:
            raw_err = str(exc)
            diag = cls.classify_inference_error(
                status_code=None,
                raw_error=raw_err,
                model_id=slot.model_id,
                api_base=slot.api_base
            )
            error_msg = f"[{diag['badge']}] {diag['summary']} (Root cause: {diag['root_cause']}. Fix: {diag['fix_suggestion']})"
            tn = time.perf_counter()
            if t1 is None:
                t1 = tn

        # Step 4: Metric Computations
        full_text = "".join(generated_chunks)
        token_count = max(len(generated_chunks), len(full_text.split()))
        if token_count == 0 and not error_msg:
            token_count = 1

        total_latency_sec = max(0.0001, tn - t0)
        ttft_sec = max(0.0001, t1 - t0)
        decode_duration_sec = max(0.0001, tn - t1)

        ttft_ms = round(ttft_sec * 1000.0, 2)
        if token_count > 1:
            tpot_ms = round((decode_duration_sec / (token_count - 1)) * 1000.0, 2)
            tps = round(token_count / decode_duration_sec, 2)
        else:
            tpot_ms = round(decode_duration_sec * 1000.0, 2)
            tps = round(token_count / total_latency_sec, 2)

        return CompareModelResult(
            slot_id=slot_id,
            model_id=slot.model_id,
            model_index=slot_idx,
            metadata=metadata,
            generated_text=full_text if not error_msg else f"[Inference Error]: {error_msg}",
            token_count=token_count,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            tps=tps,
            total_latency_ms=round(total_latency_sec * 1000.0, 2),
            error=error_msg
        )

    @staticmethod
    def classify_inference_error(status_code: Optional[int], raw_error: str, model_id: str, api_base: str) -> Dict[str, Any]:
        """
        Diagnose the root cause of an inference failure and distinguish
        API key / auth issues, provider routing errors, rate limits, and network issues.
        """
        raw_lower = raw_error.lower()
        
        # 1. Authentication / Key Issues (401)
        if status_code == 401 or "unauthorized" in raw_lower or "invalid token" in raw_lower or "invalid api key" in raw_lower:
            return {
                "error_type": "AUTH_ERROR",
                "badge": "API KEY / AUTH ERROR",
                "status_code": status_code or 401,
                "summary": "Invalid or missing API key for this provider endpoint.",
                "root_cause": "The provider rejected the request credentials (HTTP 401 Unauthorized).",
                "fix_suggestion": "Check the API Key entered in this slot (or the Global Hugging Face Token). For Hugging Face Inference API, ensure the token has 'Read' or 'Inference' permissions."
            }

        # 2. Gated Repository / License Agreement (403)
        if status_code == 403 or "forbidden" in raw_lower or "gated" in raw_lower:
            return {
                "error_type": "GATED_MODEL_FORBIDDEN",
                "badge": "GATED MODEL / PERMISSION DENIED",
                "status_code": status_code or 403,
                "summary": f"Access forbidden to repository '{model_id}'.",
                "root_cause": "Repository is gated by the model author or requires an explicit license acceptance on Hugging Face.",
                "fix_suggestion": f"Visit https://huggingface.co/{model_id} while logged into your HF account, accept the license/terms, and supply a valid HF access token."
            }

        # 3. Model Not Found / Unsupported on Provider (404)
        if status_code == 404 or "model not found" in raw_lower or "does not exist" in raw_lower:
            is_hf = "huggingface.co" in api_base.lower()
            return {
                "error_type": "MODEL_NOT_FOUND",
                "badge": "MODEL NOT HOSTED ON PROVIDER",
                "status_code": status_code or 404,
                "summary": f"Model '{model_id}' is not served at this endpoint.",
                "root_cause": (
                    "Hugging Face free serverless inference only hosts select small-to-medium models. Massive models (e.g. DeepSeek-V3 671B, Llama-3.3-70B) are not on the free tier."
                    if is_hf else
                    f"The endpoint '{api_base}' does not serve a model identified as '{model_id}'."
                ),
                "fix_suggestion": (
                    "For large frontier models (DeepSeek, Llama-70B), use a dedicated OpenAI-compatible provider URL (e.g., https://integrate.api.nvidia.com/v1, Together AI, Groq, or local vLLM/Ollama) with its respective API key."
                    if is_hf else
                    "Verify the exact model ID expected by this endpoint provider."
                )
            }

        # 4. Rate Limit (429)
        if status_code == 429 or "rate limit" in raw_lower or "too many requests" in raw_lower:
            return {
                "error_type": "RATE_LIMIT",
                "badge": "PROVIDER RATE LIMIT EXCEEDED",
                "status_code": status_code or 429,
                "summary": "Inference request rate limit exceeded.",
                "root_cause": "The upstream provider has throttled concurrent or per-minute requests on this token/tier.",
                "fix_suggestion": "Wait a moment before retrying, reduce concurrency, or upgrade to a provider tier with higher rate limits."
            }

        # 5. Model Loading / Cold Start (503)
        if status_code == 503 or "currently loading" in raw_lower or "loading" in raw_lower:
            return {
                "error_type": "MODEL_LOADING",
                "badge": "MODEL COLD START / LOADING",
                "status_code": status_code or 503,
                "summary": f"Model '{model_id}' is warming up on serverless infrastructure.",
                "root_cause": "The provider is spinning up model weights into GPU memory (HTTP 503 Service Unavailable).",
                "fix_suggestion": "Wait 20-40 seconds for the model to finish loading on the provider, then click 'Run Benchmark' again."
            }

        # 6. Network / Connection Errors
        if "connecterror" in raw_lower or "connection refused" in raw_lower or "getaddrinfo" in raw_lower:
            return {
                "error_type": "NETWORK_CONNECTION_ERROR",
                "badge": "CANNOT CONNECT TO ENDPOINT",
                "status_code": status_code,
                "summary": f"Could not reach endpoint '{api_base}'.",
                "root_cause": "DNS resolution failed or network connection was refused.",
                "fix_suggestion": "Verify that the provider base URL is reachable and correctly formatted (e.g., ensure protocol http:// or https:// is specified)."
            }

        if "timeout" in raw_lower:
            return {
                "error_type": "TIMEOUT_ERROR",
                "badge": "REQUEST TIMED OUT",
                "status_code": status_code or 504,
                "summary": "Request timed out waiting for inference response.",
                "root_cause": "The endpoint took longer than 60 seconds to stream tokens or begin generation.",
                "fix_suggestion": "The model may be experiencing heavy server load or cold startup. Try again or reduce max_tokens."
            }

        # Generic / Upstream Provider Error
        return {
            "error_type": "PROVIDER_ERROR",
            "badge": f"HTTP {status_code}" if status_code else "INFERENCE ERROR",
            "status_code": status_code,
            "summary": f"Inference failed with status {status_code or 'UNKNOWN'}.",
            "root_cause": raw_error,
            "fix_suggestion": "Inspect the raw response below and verify endpoint provider parameters."
        }

    @classmethod
    async def compare_models(
        cls,
        req: CompareBenchmarkRequest,
        timeout: float = 60.0
    ) -> CompareBenchmarkResponse:
        """
        Execute concurrent benchmarking across N arbitrary models using asyncio.gather.
        """
        tasks = [
            cls.run_single_slot(
                slot_idx=i,
                slot=slot,
                prompt=req.prompt,
                system_prompt=req.system_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                global_hf_token=req.global_hf_token,
                timeout=timeout
            )
            for i, slot in enumerate(req.models)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return CompareBenchmarkResponse(results=results)

    @classmethod
    async def stream_compare_events(
        cls,
        req: CompareBenchmarkRequest
    ) -> AsyncGenerator[str, None]:
        """
        Multiplex Server-Sent Events (SSE) across arbitrary N models concurrently.
        """
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        active_count = len(req.models)
        if active_count == 0:
            yield f"event: all_completed\ndata: {json.dumps({'count': 0})}\n\n"
            return

        async def worker(idx: int, slot: ModelSlotConfig):
            slot_id = slot.slot_id or f"slot_{idx}"
            hf_token = slot.hf_token or req.global_hf_token

            # Architecture metadata resolution
            try:
                metadata = await HFConfigParser.fetch_model_metadata(
                    model_id=slot.model_id,
                    hf_token=hf_token
                )
            except PermissionError as pe:
                metadata = ModelMetadata(
                    model_id=slot.model_id,
                    model_type="gated_or_private",
                    is_fallback=True,
                    notes=[str(pe)]
                )
            except Exception as exc:
                metadata = ModelMetadata(
                    model_id=slot.model_id,
                    model_type="unknown",
                    is_fallback=True,
                    notes=[f"Metadata: {str(exc)}"]
                )

            init_event = {
                "slot_id": slot_id,
                "model_index": idx,
                "model_id": slot.model_id,
                "metadata": metadata.model_dump()
            }
            await queue.put(f"event: metadata\ndata: {json.dumps(init_event)}\n\n")

            # Check simulated
            if slot.api_base.lower().startswith("mock") or slot.api_key == "demo-key" or not slot.api_key.strip():
                async for evt in cls._stream_simulated_slot(idx, slot_id, slot.model_id):
                    await queue.put(evt)
                await queue.put(None)
                return

            # Live streaming
            endpoint = cls.normalize_endpoint(slot.api_base)
            effective_key = slot.api_key.strip() if slot.api_key else (hf_token or "")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {effective_key}"
            }
            messages = []
            if req.system_prompt and req.system_prompt.strip():
                messages.append({"role": "system", "content": req.system_prompt.strip()})
            messages.append({"role": "user", "content": req.prompt})

            payload = {
                "model": slot.model_id,
                "messages": messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "stream": True
            }

            t0 = time.perf_counter()
            t1: Optional[float] = None
            token_count = 0
            generated_tokens: List[str] = []

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            err_b = await response.aread()
                            raw_err = err_b.decode("utf-8", errors="ignore")
                            diag = cls.classify_inference_error(
                                status_code=response.status_code,
                                raw_error=raw_err,
                                model_id=slot.model_id,
                                api_base=slot.api_base
                            )
                            err_event = {
                                "slot_id": slot_id,
                                "model_index": idx,
                                "model_id": slot.model_id,
                                "error": diag["summary"],
                                "error_diagnostic": diag,
                                "raw_error": raw_err,
                                "endpoint": endpoint
                            }
                            await queue.put(f"event: error\ndata: {json.dumps(err_event)}\n\n")
                            await queue.put(None)
                            return

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                content = ""
                                if choices:
                                    first_choice = choices[0]
                                    delta = first_choice.get("delta", {})
                                    content = delta.get("content") or delta.get("text") or first_choice.get("text") or ""
                                    # Some providers wrap content as dict or list
                                    if isinstance(content, list):
                                        content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
                                    elif isinstance(content, dict):
                                        content = content.get("text", "")
                                elif "token" in chunk_json and isinstance(chunk_json["token"], dict):
                                    content = chunk_json["token"].get("text", "")

                                if content:
                                    now = time.perf_counter()
                                    if t1 is None:
                                        t1 = now
                                        ttft_ms = round((t1 - t0) * 1000.0, 2)
                                        ttft_data = {
                                            "slot_id": slot_id,
                                            "model_index": idx,
                                            "model_id": slot.model_id,
                                            "ttft_ms": ttft_ms
                                        }
                                        await queue.put(f"event: ttft\ndata: {json.dumps(ttft_data)}\n\n")

                                    token_count += 1
                                    generated_tokens.append(content)
                                    elapsed = now - t0
                                    gen_duration = now - t1
                                    tpot_ms = round((gen_duration / max(1, token_count - 1)) * 1000.0, 2) if token_count > 1 else 20.0
                                    tps = round(token_count / max(0.001, gen_duration), 2) if gen_duration > 0 else 0.0

                                    chunk_event = {
                                        "slot_id": slot_id,
                                        "model_index": idx,
                                        "model_id": slot.model_id,
                                        "token": content,
                                        "token_count": token_count,
                                        "tpot_ms": tpot_ms,
                                        "tps": tps,
                                        "elapsed_ms": round(elapsed * 1000.0, 2)
                                    }
                                    await queue.put(f"event: token\ndata: {json.dumps(chunk_event)}\n\n")
                            except json.JSONDecodeError:
                                continue

            except Exception as exc:
                raw_exc = str(exc)
                diag = cls.classify_inference_error(
                    status_code=None,
                    raw_error=raw_exc,
                    model_id=slot.model_id,
                    api_base=slot.api_base
                )
                err_event = {
                    "slot_id": slot_id,
                    "model_index": idx,
                    "model_id": slot.model_id,
                    "error": diag["summary"],
                    "error_diagnostic": diag,
                    "raw_error": raw_exc,
                    "endpoint": endpoint
                }
                await queue.put(f"event: error\ndata: {json.dumps(err_event)}\n\n")
                await queue.put(None)
                return

            tn = time.perf_counter()
            gen_sec = tn - (t1 or tn)
            tps_final = round(token_count / max(0.001, gen_sec), 2) if gen_sec > 0 else 0.0
            full_text = "".join(generated_tokens)
            completed_event = {
                "slot_id": slot_id,
                "model_index": idx,
                "model_id": slot.model_id,
                "token_count": token_count,
                "generated_text": full_text,
                "ttft_ms": round(((t1 or tn) - t0) * 1000.0, 2),
                "tpot_ms": round((gen_sec / max(1, token_count - 1)) * 1000.0, 2) if token_count > 1 else 0.0,
                "tps": tps_final,
                "total_latency_ms": round((tn - t0) * 1000.0, 2)
            }
            await queue.put(f"event: completed\ndata: {json.dumps(completed_event)}\n\n")
            await queue.put(None)

        tasks = [
            asyncio.create_task(worker(i, slot))
            for i, slot in enumerate(req.models)
        ]

        sentinels_received = 0
        while sentinels_received < active_count:
            item = await queue.get()
            if item is None:
                sentinels_received += 1
            else:
                yield item

        await asyncio.gather(*tasks, return_exceptions=True)
        yield f"event: all_completed\ndata: {json.dumps({'count': active_count})}\n\n"

    @classmethod
    async def _run_simulated_slot(
        cls,
        slot_idx: int,
        slot_id: str,
        model_id: str,
        metadata: ModelMetadata
    ) -> CompareModelResult:
        """Simulate realistic inference for a slot."""
        text = "Quantum computing relies on quantum bits or qubits that exist in superpositions of 0 and 1. Through entanglement, quantum processors explore exponentially vast mathematical spaces concurrently. Scalable commercial utility depends on error-corrected fault-tolerant logical qubits."
        words = text.split(" ")
        await asyncio.sleep(0.15 + (slot_idx * 0.03))

        t0 = time.perf_counter()
        t1 = t0 + 0.16 + (slot_idx * 0.02)
        tn = t1 + (len(words) * 0.015)

        token_count = len(words)
        total_latency_sec = tn - t0
        gen_sec = tn - t1

        return CompareModelResult(
            slot_id=slot_id,
            model_id=model_id,
            model_index=slot_idx,
            metadata=metadata,
            generated_text=text,
            token_count=token_count,
            ttft_ms=round((t1 - t0) * 1000.0, 2),
            tpot_ms=15.0,
            tps=round(token_count / gen_sec, 2),
            total_latency_ms=round(total_latency_sec * 1000.0, 2),
            error=None
        )

    @classmethod
    async def _stream_simulated_slot(
        cls,
        slot_idx: int,
        slot_id: str,
        model_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream simulated tokens for interactive testing."""
        sample_texts = [
            "Quantum computing harnesses superposition and entanglement of qubits to evaluate complex computational states concurrently. Unlike classical binary transistors, quantum algorithms achieve polynomial and quadratic speedups across chemistry and cryptography.",
            "Multi-Head Latent Attention (MLA) decomposes Key and Value projections into low-rank latent vectors (d_c=512) and decoupled rotary embeddings (d_r=64). This reduces memory consumption by approximately 80% compared to standard MHA.",
            "Grouped-Query Attention (GQA) groups multiple query heads to share common key and value heads. With an 8:1 compression ratio, GQA drastically lowers high-bandwidth memory transfer volume while retaining multi-head expressivity.",
            "Sliding Window Attention (SWA) constrains the self-attention receptive field to a rolling window of tokens (W=4096). Cache eviction bounds memory footprint permanently, preventing out-of-memory crashes on long generation tasks."
        ]
        text = sample_texts[slot_idx % len(sample_texts)]
        words = text.split(" ")

        t0 = time.perf_counter()
        await asyncio.sleep(0.12 + (slot_idx * 0.04))
        t1 = time.perf_counter()
        ttft_ms = round((t1 - t0) * 1000.0, 2)
        ttft_data = {"slot_id": slot_id, "model_index": slot_idx, "model_id": model_id, "ttft_ms": ttft_ms}
        yield f"event: ttft\ndata: {json.dumps(ttft_data)}\n\n"

        token_count = 0
        for i, word in enumerate(words):
            token_str = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.015 + (slot_idx * 0.003))
            now = time.perf_counter()
            token_count += 1
            elapsed = now - t0
            gen_sec = now - t1
            tpot_ms = round((gen_sec / max(1, token_count - 1)) * 1000.0, 2) if token_count > 1 else 18.0
            tps = round(token_count / max(0.001, gen_sec), 2)

            chunk_event = {
                "slot_id": slot_id,
                "model_index": slot_idx,
                "model_id": model_id,
                "token": token_str,
                "token_count": token_count,
                "tpot_ms": tpot_ms,
                "tps": tps,
                "elapsed_ms": round(elapsed * 1000.0, 2)
            }
            yield f"event: token\ndata: {json.dumps(chunk_event)}\n\n"

        tn = time.perf_counter()
        gen_sec = tn - t1
        final_summary = {
            "slot_id": slot_id,
            "model_index": slot_idx,
            "model_id": model_id,
            "token_count": token_count,
            "generated_text": text,
            "ttft_ms": ttft_ms,
            "tpot_ms": round((gen_sec / max(1, token_count - 1)) * 1000.0, 2),
            "tps": round(token_count / max(0.001, gen_sec), 2),
            "total_latency_ms": round((tn - t0) * 1000.0, 2)
        }
        yield f"event: completed\ndata: {json.dumps(final_summary)}\n\n"
