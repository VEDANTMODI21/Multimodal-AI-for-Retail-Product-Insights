# Project Context

## Overview

This repository contains the research artifacts, documentation, and supplementary materials for the paper:

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
| **Structured Processing** | MLP + Min-Max Scaling | Normalizes transactional data (price, ratings, return %) |
| **Fusion** | Concatenation + ReLU Projection | Merges all modalities into a 512-dim joint representation |
| **Insight Generation** | Llama-2 7B (LoRA fine-tuned) | Translates fused features into human-readable business insights |

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

- **Source:** Amazon Review Dataset (2023), McAuley Lab
- **Categories:** Electronics & Fashion
- **Raw Entries:** 150,000+
- **Filtered Final Dataset:** 15,420 products
- **Filtering Criteria:**
  - At least one image > 800×800 pixels
  - Minimum 50 customer reviews per product
  - All structured fields fully populated
- **Split:** 70% Train / 15% Validation / 15% Test

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch 2.0 |
| Training Strategy | Distributed Data Parallel (DDP) |
| Precision | Automatic Mixed Precision (AMP) |
| Hardware | Dual NVIDIA A100 (80 GB) GPUs |
| Fine-Tuning | LoRA (rank r=16, α=32) |
| Optimizer | AdamW, lr = 2×10⁻⁵ |
| Epochs | 15 |
| Batch Size | 32 |
| Dropout | 0.1 |
| LLM Temperature | 0.3 |

---

## Repository Structure

```
Gen_ai/
├── README.md            # Project overview and quick-start
├── context.md           # This file — full project context
├── knowledge.md         # Deep technical knowledge base
└── .gitignore           # Git ignore rules
```

---

## Status

| Milestone | Status |
|-----------|--------|
| Paper Draft | ✅ Complete |
| Experimental Validation | ✅ Complete |
| Human Expert Evaluation | ✅ Complete (5 domain experts, 1000 insights) |
| Repository Documentation | ✅ Complete |
| GitHub Push | 🔄 In Progress |

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
