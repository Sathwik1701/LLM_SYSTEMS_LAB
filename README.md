# LLM Architecture & Telemetry Dashboard

An open-source telemetry, benchmarking, and architectural analysis engine designed for side-by-side comparison of modern open-source language models.

## Project Overview
Navigating open-source LLM releases requires looking past marketing claims and raw benchmark scores. This dashboard provides a standardized, data-driven platform to dissect architectural trade-offs, KV cache memory footprint, parameter density, and real-time generation latency across frontier open-source models.

## Core Telemetry & Metrics
* **Architecture Breakdown:** Side-by-side analysis of attention variants (MHA, GQA, MLA) and layer topologies (Dense vs. MoE).
* **Memory & Footprint Dynamics:** Calculation of KV cache size per token across precision tiers (FP16, BF16, FP8, INT4) and total vs. active parameter ratios.
* **Real-Time Latency Tracking:** Live measurements of:
  * **TTFT (Time to First Token):** First-token latency and prefill efficiency.
  * **TPOT (Time Per Output Token):** Decode throughput and streaming stability.

## Technical Methodology
* **Dynamic Configuration Parsing:** Model parameters and layer configurations are parsed directly from open-source specifications.
* **Provider-Decoupled Benchmarking:** Telemetry runs across distributed API serving layers (e.g., Groq, NVIDIA NIM) using free-tier endpoints to assess real-world inference behavior without local hardware constraints.

## Repository Structure
* `/src` — Telemetry collection, API clients, and comparative UI logic.
* `/reports` — Detailed architectural breakdowns, layer diagrams, and side-by-side benchmark analyses.

## Quickstart & Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   ```
   Add your respective API keys (`HF_TOKEN`, `GROQ_API_KEY`, etc.) inside `.env`.

3. **Install Dependencies & Run:**
   ```bash
   # Follow stack-specific install instructions (e.g., pip install -r requirements.txt or npm install)
   ```

## Deployment
This project is configured for one-click deployment on platforms like Render and Vercel. Ensure all environment variables defined in `.env.example` are populated in the platform's Environment Variables dashboard prior to building.
