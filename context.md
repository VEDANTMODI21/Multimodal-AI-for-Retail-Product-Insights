# Project Context

## Overview

This repository contains the research artifacts, complete source code, training pipeline, and hyperparameter tuning system for the paper:

> **"Multimodal AI for Retail Product Insights: A Unified Framework Integrating Visual, Textual, and Structured Data"**

Published / Submitted by:
- **Krish Sati**
- **Vedant Modi**
- **Hema Sri Harsha Guggilam**
- **Pritham Mukesh Krishna**

All authors are B.Tech CSE (Big Data Analytics) students at **SRM University**.

---

## Problem Statement

Modern e-commerce analytics is fundamentally limited because it operates in **unimodal silos** — sales dashboards show numbers, image-tagging systems classify product photos, and NLP pipelines score review sentiment — but **none of these systems talk to each other**.

### The Concrete Scenario

A premium winter jacket launches on an e-commerce platform. The product photos look stunning, the price point is competitive, but within 3 weeks it hits a **40% return rate**. A standard analytics dashboard can only surface the symptom ("product is failing"), not the root cause.

A human analyst would need to:
1. Manually read dozens of customer reviews (finding complaints about a "cheap zipper")
2. Cross-reference those complaints with product photos
3. Correlate with structured data (return rates, ratings, pricing)

This process is **slow, expensive, and unscalable** across thousands of SKUs.

---

## Solution — The Unified Multimodal Pipeline

The paper proposes and validates a unified AI framework that:

| Stage | Component | Purpose |
|-------|-----------|---------|
| **Visual Encoding** | Vision Transformer (ViT-Base/16) | Extracts global visual features from product images |
| **Textual Encoding** | BERT (bert-base-uncased) | Captures contextual sentiment from customer reviews |
| **Structured Processing** | MLP + BatchNorm + Min-Max Scaling | Normalizes transactional data (price, ratings, return %) |
| **Fusion** | Concatenation + ReLU Projection + LayerNorm | Merges all modalities into a 512-dim joint representation |
| **Insight Generation** | DistilGPT-2 with LoRA (dev) / Llama-2 7B with LoRA (production) | Translates fused features into actionable business insights |
| **Hyperparameter Tuning** | Optuna (TPE Sampler + Median Pruner) | Automated search for optimal training configuration |

---

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | Can a late-fusion multimodal architecture identify root causes of product performance by aligning visual, textual, and structured features? |
| **RQ2** | Does a specialized fine-tuned pipeline beat massive generalist models like LLaVA v1.5 at generating business insights? |
| **RQ3** | What fusion strategy best balances insight quality vs. real-time latency requirements? |

---

## Key Results Summary

- **Insight Quality Score:** 0.83 (outperforming LLaVA v1.5's 0.79 and all unimodal baselines)
- **Statistical Significance:** Paired two-tailed t-test, p < 0.01
- **Inference Latency:** 115ms per product (42% faster than cross-attention alternatives)
- **Inter-Rater Reliability:** Fleiss' Kappa κ = 0.782 ("substantial agreement")

---

## Dataset

- **Source:** Amazon Review Dataset (2023), McAuley Lab (HuggingFace)
- **Categories:**
  - **Clothing, Shoes & Jewelry** — filtered to Winter Range (jacket, coat, parka, sweater, hoodie, fleece, thermal, wool, beanie, gloves, scarf, boots, puffer, insulated, warm, snow, etc.)
  - **Electronics** — filtered to consumer accessories (headphones, earbuds, speaker, charger, cable, smartwatch, mouse, keyboard, webcam, etc.)
- **Filtering Criteria:**
  - Title/description/features contain category-specific keywords
  - Minimum 3 customer reviews per product
  - Up to 1500 products per category
- **Split:** 70% Train / 15% Validation / 15% Test
- **Total:** ~3,000 products across both categories

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch 2.11+cu128 |
| Training Precision | Automatic Mixed Precision (AMP) |
| Hardware | NVIDIA RTX 5070 Laptop GPU (8GB VRAM, Blackwell SM_120) |
| CUDA | 12.8 (required for Blackwell architecture) |
| LLM (Dev) | DistilGPT-2 (~82M params) with LoRA |
| LLM (Production) | Llama-2 7B with LoRA (rank r=16, α=32) |
| Optimizer | AdamW, lr = 2×10⁻⁵ (tuned via Optuna) |
| Epochs | 25 (extended from 15 — loss was still decreasing) |
| Effective Batch Size | 32 (via gradient accumulation: batch=4 × accum=8) |
| LoRA Config | r=32, α=64, dropout=0.05, targets: c_attn + c_proj |
| Dropout | 0.1 (anti-modality-dominance) |
| Gradient Clipping | max_norm = 1.0 |
| LR Schedule | Cosine Annealing with Linear Warmup (10%) |
| Encoder Strategy | Frozen for 2 epochs, then unfrozen at 0.1× LR |
| HPT | Optuna (TPE Sampler, Median Pruner, 15 trials × 5 epochs) |

---

## Repository Structure

```
Multimodal-AI-for-Retail-Product-Insights/
├── README.md              # Project overview and quick-start
├── context.md             # This file — full project context
├── knowledge.md           # Deep technical knowledge base
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── src/                   # Source code
│   ├── __init__.py        # Package initializer
│   ├── config.py          # Centralized hyperparameters & paths
│   ├── model.py           # MultimodalRetailInsightModel
│   ├── dataset.py         # MultimodalRetailDataset + collate_fn
│   ├── train.py           # Training loop (AMP, validation, checkpointing)
│   ├── tune.py            # Optuna hyperparameter tuning
│   ├── evaluate.py        # Test set evaluation with quality metrics
│   ├── inference.py       # RetailInsightPredictor (real LLM generation)
│   ├── api.py             # Flask REST API
│   └── prepare_data.py    # Amazon Review Dataset download & preprocessing
├── data/                  # Generated by prepare_data.py
│   ├── train.csv / val.csv / test.csv
│   ├── images/            # Product images
│   └── scaler_params.json # Feature scaling parameters
└── checkpoints/           # Generated during training
    ├── best_model.pth     # Best validation loss checkpoint
    ├── epoch_XX.pth       # Per-epoch checkpoints
    ├── latest_model.pth   # Latest epoch checkpoint
    ├── training_log.csv   # Loss history
    └── optuna_tuning/     # HPT results
        ├── optuna_results.csv
        └── best_params.json
```

---

## Implementation Details

### Two-Tier LLM Strategy

Since Llama-2 7B requires ~14GB VRAM and a HuggingFace access token, we implement a two-tier approach:

| Tier | LLM | VRAM | Use Case |
|------|-----|------|----------|
| **Tier 1 (Dev)** | DistilGPT-2 (~82M) | ~2GB | Local development & training on consumer GPUs |
| **Tier 2 (Prod)** | Llama-2 7B (~7B) | ~14GB | Production deployment on A100/H100 GPUs |

Both tiers use identical:
- LoRA configuration (LoRA applied to attention layers)
- Fusion architecture (512-d multimodal representation)
- Training pipeline (AMP, gradient accumulation, cosine scheduler)

### Encoder Freezing Strategy

To stabilize training, we freeze the pretrained ViT and BERT encoders for the first 2 epochs, allowing the fusion layer and LLM adapter to learn meaningful representations before the encoder gradients begin flowing. After the freeze period, encoders are unfrozen with a reduced learning rate (0.1× base LR).

### Anti-Modality-Dominance

The paper identified that textual features (h_t) tend to dominate the fusion layer because customer reviews contain explicit signals. We address this with:
1. **Dropout (0.1)** in the fusion layer — forces the network to not over-rely on any single modality
2. **LayerNorm** after fusion — normalizes the contribution of each modality
3. **BatchNorm** in the structured MLP — prevents numerical feature scaling issues

### Actionable Insight Generation

All target insights follow a **3-sentence WHY → WHAT → HOW TO IMPROVE** format:
1. **Root cause analysis** — Why the rating is what it is
2. **What specifically drove satisfaction or complaints** — Grounded in actual review keywords
3. **Concrete actionable recommendations** — How the brand should respond

### Optuna Hyperparameter Tuning

The tuning pipeline (`src/tune.py`) searches over:
- Learning Rate: [5e-6, 1e-4] (log scale)
- LoRA Rank: [8, 16, 32, 64]
- LoRA Alpha: [16, 32, 64, 128]
- LoRA Dropout: [0.01, 0.15]
- Weight Decay: [0.001, 0.1] (log scale)
- Warmup Ratio: [0.05, 0.2]
- Gradient Accumulation Steps: [4, 8, 16]
- Freeze Epochs: [1, 2, 3]

Uses **TPE Sampler** with **Median Pruner** (3 startup trials, 2 warmup steps). After finding optimal params, auto-retrains with full 25 epochs.

---

## Status

| Milestone | Status |
|-----------|--------|
| Paper Draft | ✅ Complete |
| Experimental Validation | ✅ Complete |
| Human Expert Evaluation | ✅ Complete (5 domain experts, 1000 insights) |
| Repository Documentation | ✅ Complete |
| Source Code Implementation | ✅ Complete |
| Data Pipeline (Clothing + Electronics) | ✅ Complete |
| Training Pipeline (AMP + LoRA + Validation) | ✅ Complete |
| Per-Epoch Checkpointing | ✅ Complete |
| Actionable Insight Templates | ✅ Complete (WHY + WHAT + HOW) |
| Optuna Hyperparameter Tuning | ✅ Complete |
| Evaluation Pipeline | ✅ Complete |
| Inference Pipeline | ✅ Complete (Real LLM generation) |
| API Server | ✅ Complete (Flask REST API) |

---

## License & Citation

If you use this work, please cite:

```bibtex
@article{sati2025multimodal,
  title   = {Multimodal AI for Retail Product Insights: A Unified Framework
             Integrating Visual, Textual, and Structured Data},
  author  = {Sati, Krish and Modi, Vedant and Guggilam, Hema Sri Harsha
             and Krishna, Pritham Mukesh},
  year    = {2025},
  note    = {B.Tech CSE (Big Data Analytics), SRM University}
}
```
