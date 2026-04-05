# Multimodal AI for Retail Product Insights

> **A Unified Framework Integrating Visual, Textual, and Structured Data**

[![Python](https://img.shields.io/badge/Framework-PyTorch%202.0-red)]()

---

## 📝 Overview

This repository contains the codebase and architecture for **Multimodal AI for Retail Product Insights**. We propose a unified pipeline that fuses visual features (ViT), textual features (BERT), and structured transactional data to generate human-readable business insights using a fine-tuned LLM.

### Key Contribution

Traditional retail analytics operates in unimodal silos — our framework bridges them:

```
Product Image → ViT → Visual Features ─┐
Customer Reviews → BERT → Text Features ─┼─→ Fusion → Llama-2 7B → Business Insight
Structured Data → MLP → Numerical Features ─┘
```

---



## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Insight Quality Score | **0.83** (beats LLaVA v1.5's 0.79) |
| Statistical Significance | p < 0.01 |
| Inference Latency | **115ms** per product |
| Inter-Rater Reliability | Fleiss' κ = 0.782 |

---

## 📂 Repository Structure

```
├── README.md        # Quick overview
├── src/             # Source code (Model, Dataset, Training, Inference)
├── requirements.txt # System dependencies
├── data/            # Local dataset folder (Manual Add)
├── context.md       # Full project context, motivation, and setup
└── knowledge.md     # Deep technical knowledge base
```

### Documentation Guide

| File | Purpose | When to Read |
|------|---------|--------------|
| `README.md` | Quick overview and entry point | First |
| `context.md` | Project background, problem statement, research questions, dataset, tech stack | Understanding the "what" and "why" |
| `knowledge.md` | Complete technical breakdown of every section + viva Q&A | Deep dive and exam preparation |

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| Visual Encoder | ViT-Base/16 (ImageNet-21k pretrained) |
| Text Encoder | BERT-base-uncased |
| LLM | Llama-2 7B (LoRA fine-tuned, r=16, α=32) |
| Framework | PyTorch 2.0 + DDP + AMP |
| Hardware | 2× NVIDIA A100 (80 GB) |
| Dataset | Amazon Review Dataset 2023 (15,420 products) |

---

## 📊 Comparative Results

| Model | Quality Score |
|-------|:---:|
| Structured-only MLP | 0.41 |
| Image-only ViT | 0.58 |
| Text-only BERT | 0.64 |
| CLIP + BERT + LLM | 0.76 |
| LLaVA v1.5 (zero-shot) | 0.79 |
| **Our Framework** | **0.83** |

---



## 📜 License

This project is open-source and intended for generalized use in retail AI platforms.
