# Knowledge Base — Multimodal AI for Retail Product Insights

This document is the **deep technical reference** for the entire paper. It covers every section in exhaustive detail — architecture decisions, mathematical formulations, experimental methodology, results analysis, failure modes, and future work.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Literature Review](#3-literature-review)
4. [Proposed Methodology](#4-proposed-methodology)
   - 4.1 [Formal Problem Definition](#41-formal-problem-definition)
   - 4.2 [Visual Feature Extraction (ViT)](#42-visual-feature-extraction-vit)
   - 4.3 [Textual Feature Extraction (BERT)](#43-textual-feature-extraction-bert)
   - 4.4 [Fusion — The Modality Dominance Problem](#44-fusion--the-modality-dominance-problem)
   - 4.5 [LLM Prompt Engineering & Instruction Tuning](#45-llm-prompt-engineering--instruction-tuning)
5. [Experimental Setup](#5-experimental-setup)
6. [Results](#6-results)
   - 6.1 [Insight Quality Score (Q)](#61-insight-quality-score-q)
   - 6.2 [Comparative Performance](#62-comparative-performance)
   - 6.3 [Fusion Ablation Study](#63-fusion-ablation-study)
   - 6.4 [Qualitative Case Study](#64-qualitative-case-study)
   - 6.5 [Failure Modes](#65-failure-modes)
7. [Conclusion & Future Work](#7-conclusion--future-work)
8. [Viva Preparation — Key Concepts](#8-viva-preparation--key-concepts)

---

## 1. Abstract

The paper proposes a **unified multimodal AI pipeline** for retail product analytics that moves beyond unimodal number-crunching. The core claim:

> Traditional e-commerce analytics ignores the rich information locked inside product images and customer reviews. Our framework fuses **visual features** (via ViT), **textual features** (via BERT), and **structured transactional data** through a late-fusion layer, then generates human-readable business insights using a **fine-tuned LLM (Llama-2 7B with LoRA)**.

### Key Claims
- Outperforms all unimodal baselines
- Beats LLaVA v1.5 (a large zero-shot vision-language model) in domain-specific insight generation
- Validated through blind human expert evaluation with statistical significance (p < 0.01)
- Tested on a curated subset of the **Amazon Review Dataset (2023)**

---

## 2. Introduction & Motivation

### The Winter Jacket Story (Opening Narrative)

A premium winter jacket is listed on an e-commerce platform. Product photos are professional, price is competitive, descriptions are well-crafted. Within **3 weeks**, the return rate hits **40%**.

**What a standard dashboard shows:** "Product is failing."
**What it can't show:** *Why* it's failing.

A human analyst would need to:
1. Read dozens of reviews → discover complaints about a "cheap zipper"
2. Cross-check product images → notice the zipper isn't prominently visible
3. Connect to structured data → correlate with high return rates

This process is **manual, slow, expensive, and unscalable** across catalogs with millions of SKUs.

### The Core Problem: Unimodal Silos

| Silo | What It Does | What It Misses |
|------|-------------|----------------|
| Sales Analytics | Tracks revenue, returns, ratings | Why returns happen |
| Computer Vision | Tags product colors, categories | What customers think |
| NLP / Sentiment | Scores review polarity | What the product looks like |

**Nobody connects them together.** This paper does.

### Research Questions

| RQ | Question | Answer (Spoiler) |
|----|----------|-------------------|
| **RQ1** | Can late-fusion multimodal architecture identify root causes? | ✅ Yes — the fused representation captures cross-modal patterns invisible to any single modality |
| **RQ2** | Does fine-tuned pipeline beat generalist VLMs? | ✅ Yes — 0.83 vs. LLaVA's 0.79 (p < 0.01) |
| **RQ3** | Best fusion strategy for quality vs. latency? | ✅ Concatenation — 0.83 quality at 115ms (vs. cross-attention's 0.85 at 198ms) |

---

## 3. Literature Review

### Phase 1 — Early Methods (Pre-Deep Learning)

- **Collaborative Filtering** — predict user preferences from historical behavior
- **Matrix Factorization** — decompose user-item interaction matrices (SVD, NMF)
- **Limitation:** Completely blind to product appearance and language

### Phase 2 — Unimodal Deep Learning

- **CNNs** (ResNet, VGG) for product image classification and tagging
- **BERT** for sentiment analysis of reviews
- **Critical Limitation — Context Dependence:** The phrase "lightweight design" is:
  - ✅ **Positive** for a laptop (portability)
  - ❌ **Negative** for heavy-duty winter outerwear (signals flimsiness)
  - Unimodal systems can't distinguish these because they lack cross-modal context

### Phase 3 — Multimodal Approaches (Recent)

| Author(s) | Modalities | Application |
|-----------|------------|-------------|
| Alabi (2025) | Text + Image | Product color accuracy verification |
| Xu et al. | Visual + Textual + Acoustic | Return prediction in live-streaming commerce |
| Chaube et al. | Multimodal cues | Cold-start problem for new product listings |
| Zhang & Guo | Fused modalities | Customer satisfaction score prediction |

### The Gap — The "Black-Box" Problem

All existing multimodal systems output a **probability score** (e.g., "0.89 chance of return"). This is:
- ✅ Useful for ML engineers
- ❌ Useless for a retail manager explaining a supply chain decision to stakeholders

**Business decisions require narrative, not raw probabilities.**

Large Vision-Language Models (LLaVA, CLIP) offer natural language output but are:
- Too slow for real-time processing
- Too large for deployment without massive GPU infrastructure
- Poor at incorporating domain-specific numerical data (return rates, pricing trends)

### This Paper's Contribution

Combine **lightweight specialized encoders** (ViT + BERT) with an **LLM's generative power** (Llama-2 7B) to produce **fast, accurate, human-readable business intelligence**.

---

## 4. Proposed Methodology

### Architecture Diagram (Conceptual)

```
┌─────────────┐    ┌────────────────┐    ┌──────────────────┐
│ Product Image│    │Customer Reviews │    │ Structured Data  │
│   (V)        │    │   (T)          │    │ (S: price, rating│
└──────┬───────┘    └───────┬────────┘    │  return rate)    │
       │                    │             └────────┬─────────┘
       ▼                    ▼                      ▼
┌──────────────┐   ┌────────────────┐    ┌─────────────────┐
│  ViT-Base/16 │   │ BERT-base-     │    │  MLP + Min-Max  │
│  (ImageNet)  │   │ uncased        │    │  Normalization  │
└──────┬───────┘   └───────┬────────┘    └────────┬────────┘
       │                   │                      │
       ▼                   ▼                      ▼
     hv (768-d)         ht (768-d)             hs (d_s)
       │                   │                      │
       └───────────┬───────┘──────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Concatenation +    │
        │  ReLU Projection    │
        │  → hf (512-d)       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Llama-2 7B (LoRA)  │
        │  + Prompt Template  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Human-Readable     │
        │  Business Insight   │
        └─────────────────────┘
```

---

### 4.1 Formal Problem Definition

A product is formally defined as a **tuple:**

```
P = (V, T, S)
```

Where:
- **V** = product image (visual modality)
- **T** = customer reviews corpus (textual modality)
- **S** = structured metrics (price, star ratings, return percentages)

**The goal is NOT binary classification** (good/bad product).

The goal is to learn a **generative mapping:**

```
f : (V, T, S) → I
```

Where **I** is a sequence of language tokens forming a **diagnostic business insight**.

This is fundamentally different from typical multimodal classification — the output is natural language, not a label or score.

---

### 4.2 Visual Feature Extraction (ViT)

#### Why ViT Over CNNs?

| Property | CNN (ResNet) | ViT |
|----------|-------------|-----|
| Receptive Field | Local (grows with depth) | **Global from Layer 1** |
| Positional Awareness | Implicit via convolutions | Explicit positional embeddings |
| Patch Relationships | Hierarchical (slow) | **Self-attention (direct)** |
| Product Understanding | Textures, edges | **Overall presentation, layout, branding** |

For retail product images, **global context matters more than local textures**. A ViT understands that a product is "premium-looking" by attending to the overall composition, not just pixel-level details.

#### How ViT Works (Step by Step)

1. **Patch Extraction:** The image is divided into **N non-overlapping patches** (for ViT-Base/16, each patch is 16×16 pixels on a 224×224 image → N = 196 patches)

2. **Linear Projection:** Each patch is flattened and projected into a latent dimension:
   ```
   z_i = W_patch · flatten(patch_i) + b_patch
   ```

3. **Class Token Prepend:** A learnable `[CLS]` token is prepended to the sequence:
   ```
   Z = [z_cls, z_1, z_2, ..., z_N]
   ```

4. **Positional Embeddings:** Added to preserve spatial information (since self-attention is order-agnostic):
   ```
   Z = Z + E_pos
   ```

5. **Transformer Encoder (12 layers):** Multi-head self-attention allows every patch to attend to every other patch:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) · V
   ```

6. **Output:** The final hidden state of the `[CLS]` token becomes **h_v** — a **768-dimensional dense visual summary vector**

#### Model Specification
- **Architecture:** ViT-Base/16
- **Pretraining:** ImageNet-21k
- **Output dimension:** 768
- **Parameters:** ~86M

---

### 4.3 Textual Feature Extraction (BERT)

#### Why BERT Over Simple Embeddings?

Simple word embeddings (Word2Vec, GloVe) assign **one fixed vector per word** regardless of context. This fails catastrophically for customer reviews:

| Phrase | Meaning | Word2Vec | BERT |
|--------|---------|----------|------|
| "sick design" | High praise (slang) | Negative ("sick") | ✅ Positive |
| "makes me sick" | Critical failure | Negative ("sick") | ✅ Negative |
| "works perfectly for 5 minutes" | Sarcasm — product fails | Positive ("perfectly") | ⚠️ Sometimes caught |

BERT's **bidirectional contextual attention** resolves these ambiguities.

#### How BERT Works (Step by Step)

1. **Tokenization (WordPiece):** Reviews are split into subword tokens. Max sequence length = 128 tokens.

2. **Special Tokens:**
   ```
   [CLS] review_token_1 review_token_2 ... review_token_n [SEP]
   ```

3. **Transformer Encoder (12 layers):** Bidirectional self-attention — each token attends to ALL other tokens (both left and right context).

4. **[CLS] Token Extraction:** The final hidden state of `[CLS]` is treated as the **aggregate semantic representation** of the entire review corpus.

5. **Output:** **h_t** — a **768-dimensional vector** capturing true sentiment, key complaints, and praise themes.

#### Model Specification
- **Architecture:** bert-base-uncased
- **Vocab Size:** 30,522 tokens
- **Output dimension:** 768
- **Parameters:** ~110M

---

### 4.4 Fusion — The Modality Dominance Problem

#### The Problem They Discovered During Development

During initial training, the fusion layer exhibited **Modality Dominance** — it learned a shortcut. Because customer reviews often contain explicit complaints (e.g., "zipper broke on day 1"), the textual vector h_t carried the strongest signal for predicting insight quality.

Result: The fusion layer **muted the visual vector h_v almost entirely**, effectively collapsing back to a text-only system.

#### Fusion Strategies Considered

| Strategy | Pros | Cons |
|----------|------|------|
| **Early Fusion** (raw concatenation before encoding) | Simple | Loses modality-specific patterns |
| **Cross-Attention** (every word attends to every patch) | Rich interactions | Computationally brutal (O(n²) over both sequences) |
| **Contrastive Learning** (CLIP-style) | Good alignment | Requires massive paired datasets |
| **Late Concatenation + Non-linear Projection** ✅ | Fast, balanced | Marginally lower quality than cross-attention |

#### Their Solution — Lightweight Non-linear Concatenation

```
h_f = ReLU(W_f [h_v ∥ h_t ∥ h_s] + b_f)
```

Where:
- `h_v` (768-d) = visual features from ViT
- `h_t` (768-d) = textual features from BERT
- `h_s` (d_s) = structured features from MLP normalization
- `∥` = concatenation operator
- `W_f` = learnable weight matrix projecting down to **512 dimensions**
- `b_f` = bias vector
- ReLU introduces non-linearity
- **Heavy dropout (0.1)** prevents modality dominance by forcing the network to not over-rely on any single modality

The result **h_f** is a **512-dimensional fused multimodal representation** of the product.

---

### 4.5 LLM Prompt Engineering & Instruction Tuning

#### From Vectors to Language

h_f is projected into **discrete language tokens** using a learned projection layer, then injected into a rigid prompt template.

#### The Prompt Template

```
System Directive: Act as a clinical retail data analyst.
Given the latent multimodal product representation [h_f tokens],
and the explicit numerical constraints [Return Rate: X%, Rating: Y],
generate a concise, objective 2-sentence insight explaining the
root cause of the product's market performance.
Do not invent visual features not present in the latent representation.
```

#### Why This Design?

| Design Choice | Rationale |
|---------------|-----------|
| "Clinical retail data analyst" | Prevents creative/marketing-style language |
| "Explicit numerical constraints" | Forces the LLM to ground insights in real data |
| "2-sentence" | Controls output length for operational use |
| "Do not invent visual features" | Anti-hallucination guardrail |

The LLM acts purely as a **translation layer** for the deterministic encoders — it doesn't create new information, it translates compressed representations into human language.

Inspired by **Chain-of-Thought (CoT) prompting** principles.

#### LoRA Fine-Tuning

Full fine-tuning of Llama-2 7B would require:
- ~28 GB just for model weights in FP32
- Optimizer states would double or triple memory
- Result: **Out of Memory (OOM)** errors even on A100 GPUs

**LoRA (Low-Rank Adaptation)** solution:
- Freezes all base model weights
- Injects small trainable matrices (rank r=16) into query and value projection layers
- **Reduces trainable parameters by over 98%** (from ~7B to ~16M)
- α = 32 (scaling factor for LoRA updates)

#### LLM Specification
- **Base Model:** Llama-2 7B
- **Fine-tuning:** LoRA (r=16, α=32)
- **Temperature:** 0.3 (low = more deterministic, less hallucination)
- **Trainable Parameters:** ~16M (vs. 7B total)

---

## 5. Experimental Setup

### Dataset Construction

| Stage | Count | Notes |
|-------|-------|-------|
| Raw entries | 150,000+ | Electronics & Fashion categories |
| After image filter (>800×800px) | ~50,000 | Removes low-quality product photos |
| After review filter (≥50 reviews) | ~25,000 | Ensures statistical relevance |
| After structured data completeness | **15,420** | All fields populated |

**Split:** 70% Train (10,794) / 15% Validation (2,313) / 15% Test (2,313)

### Preprocessing Pipeline

| Modality | Preprocessing |
|----------|---------------|
| **Images** | Resize to 224×224, center crop, normalize to ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| **Text** | WordPiece tokenization, max 128 tokens, padding/truncation |
| **Structured** | Min-max scaling to [0, 1] range |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| Epochs | 15 |
| Batch Size | 32 |
| Dropout | 0.1 |
| GPUs | 2 × NVIDIA A100 (80 GB) |
| Framework | PyTorch 2.0 |
| Parallelism | Distributed Data Parallel (DDP) |
| Precision | AMP (Automatic Mixed Precision) — ~40% memory reduction |

---

## 6. Results

### 6.1 Insight Quality Score (Q)

#### Why Not BLEU/ROUGE?

Standard NLP generation metrics (BLEU, ROUGE, METEOR) measure **lexical overlap** — how many words match between generated and reference text. This fails for business insights because:
- Two insights can use completely different words but convey the same actionable intelligence
- An insight can have high BLEU but be factually wrong
- Business value (usefulness) is not captured by word matching

#### Custom Metric Definition

```
Q = 0.4(R) + 0.4(C) + 0.2(U)
```

| Component | Weight | What It Measures | Scale |
|-----------|--------|------------------|-------|
| **R** (Relevance) | 0.4 | Is the insight relevant to this specific product category? | 1–5 Likert |
| **C** (Consistency) | 0.4 | Is the insight factually consistent with the input data? (No hallucinations?) | 1–5 Likert |
| **U** (Usefulness) | 0.2 | Would this insight actually help a manager make a decision? | 1–5 Likert |

**Why these weights?**
- R and C are equally weighted at 0.4 because an insight MUST be both relevant and factual
- U gets 0.2 because usefulness is somewhat subjective and harder to standardize

#### Human Evaluation Protocol

- **Evaluators:** 5 independent domain experts (retail analysts + supply chain managers)
- **Sample:** 1,000 randomly shuffled generated insights (blind — evaluators didn't know which model produced which insight)
- **Inter-rater reliability:** Fleiss' Kappa **κ = 0.782**
  - κ > 0.61 = "substantial agreement" (Landis & Koch benchmark)
  - This proves the metric is **scientifically defensible**, not just subjective opinion

---

### 6.2 Comparative Performance

| Model | Q Score | Notes |
|-------|---------|-------|
| Structured-only MLP | 0.41 | Numbers alone can't explain *why* |
| Image-only ViT | 0.58 | Can describe product but can't assess market performance |
| Text-only BERT | 0.64 | Reviews are informative but miss visual mismatches |
| CLIP + BERT + LLM (no fine-tuning) | 0.76 | Good but lacks domain-specific tuning |
| LLaVA v1.5 (zero-shot) | 0.79 | Excellent at image description, but **ignores structured data** |
| **Their Framework** | **0.83** | **Best overall — combines all modalities with domain fine-tuning** |

#### Key Insight on LLaVA v1.5

LLaVA v1.5 is a strong generalist vision-language model, but in **zero-shot settings** it:
- Describes product images well
- Generates fluent text
- **Completely ignores structured numerical data** (return rates, pricing) because it wasn't fine-tuned on retail-specific structured features
- Result: Misses critical business context

#### Statistical Validation

- **Test:** Paired two-tailed t-test
- **Result:** p < 0.01
- **Interpretation:** The 0.83 vs. 0.79 difference is **statistically significant**, not due to random chance

---

### 6.3 Fusion Ablation Study

| Fusion Method | Quality (Q) | Latency (ms/product) | Trade-off |
|--------------|------------|---------------------|-----------|
| CLIP-style Contrastive | 0.81 | 140ms | Good quality, moderate speed |
| **Concatenation (Theirs)** | **0.83** | **115ms** | **Best quality-speed balance** |
| Early Cross-Attention | 0.85 | 198ms | Highest quality, but 42% slower |

#### Why Not Cross-Attention?

Cross-attention scores marginally better (0.85 vs. 0.83, Δ = 0.02) but is **42% slower** (198ms vs. 115ms per product).

In a live retail environment processing **tens of thousands of products per minute**, the extra 83ms per product compounds into:
- ~83 seconds of extra delay per 1,000 products
- ~83 minutes of extra delay per 1,000,000 products

**Engineering trade-off:** The 0.02 quality improvement doesn't justify the latency penalty for production deployment.

---

### 6.4 Qualitative Case Study — Wireless Earbuds

**Product Profile:**
- Star Rating: 4.1 / 5.0
- Sales: Low (underperforming expectations)

**What Each Modality Contributed:**

| Modality | Finding |
|----------|---------|
| ViT (Visual) | Flagged "premium metallic aesthetic" in product images |
| BERT (Text) | Caught "feels like cheap plastic" buried in reviews |
| Structured | Confirmed acceptable audio performance metrics |

**Generated Insight:**

> *"High visual expectations set by metallic rendering are unmet by physical plastic build, causing conversion drop-off despite acceptable audio performance."*

**Analysis:** The system identified a **visual-tactile mismatch** — the photos promise premium quality, but the physical product disappoints. This is exactly the kind of cross-modal insight that no single modality could produce.

A human analyst might take **hours** to reach that conclusion. The model did it in **115ms**.

---

### 6.5 Failure Modes

| Failure Type | Example | Root Cause | Potential Fix |
|-------------|---------|------------|---------------|
| **Sarcasm Blindness** | "works perfectly if you only want to use it for five minutes" → classified as positive | BERT's contextual understanding has limits with subtle sarcasm | Sarcasm-specific fine-tuning or sentiment-aware preprocessing |
| **Image Clutter** | Promotional overlays, dimension arrows, chaotic backgrounds (common in dropshipping) | ViT treats overlays as part of the product, corrupting visual features | Image preprocessing to remove non-product elements |
| **Review Skew** | Near the 50-review threshold, one aggressive 1-star review disproportionately skews h_t | Small sample size amplifies outlier reviews | Increase minimum review threshold or add outlier detection |

---

## 7. Conclusion & Future Work

### Summary of Contributions

1. **Proved that multimodal fusion generates better business insights** than any single modality (RQ1 ✅)
2. **Demonstrated that specialized fine-tuning beats generalist VLMs** for domain-specific tasks (RQ2 ✅)
3. **Identified concatenation as the optimal fusion strategy** for quality-latency trade-off (RQ3 ✅)
4. **Moved AI from opaque predictions to human-readable business intelligence**

### Future Directions

| Direction | Description | Impact |
|-----------|-------------|--------|
| **Video + Audio Modalities** | Add support for live-streaming commerce analysis (unboxing videos, product demos) | Expands to a rapidly growing commerce channel |
| **Model Distillation** | Compress the pipeline for edge deployment without cloud GPUs | Enables real-time in-store analytics |
| **Causal Inference** | Move beyond correlation to counterfactual reasoning ("what if we fixed the zipper?") | Enables prescriptive (not just descriptive) analytics |

---

## 8. Viva Preparation — Key Concepts

### Concepts You Must Be Able to Explain

| Concept | One-Line Explanation |
|---------|---------------------|
| **Vision Transformer (ViT)** | Splits image into patches, uses self-attention to capture global visual context |
| **BERT** | Bidirectional transformer that understands word meaning from full context (left + right) |
| **Late Fusion** | Each modality is encoded independently, then combined at the representation level |
| **Modality Dominance** | When one modality's signal is so strong the model ignores others |
| **LoRA** | Fine-tuning technique that injects small trainable matrices instead of updating all 7B parameters |
| **Chain-of-Thought Prompting** | Structuring prompts to guide LLMs through step-by-step reasoning |
| **Fleiss' Kappa** | Statistical measure of inter-rater agreement for categorical ratings |
| **AMP (Mixed Precision)** | Using FP16 for forward pass and FP32 for gradients to save ~40% memory |
| **DDP (Distributed Data Parallel)** | Splits batches across GPUs and synchronizes gradients |
| **Ablation Study** | Systematic removal of components to measure individual contributions |

### Likely Viva Questions

1. **"Why ViT instead of ResNet?"**
   → ViT captures global image context from layer 1 via self-attention. CNNs need deep stacking to grow receptive fields, making them better for local textures but worse for overall product presentation understanding.

2. **"Why not just use GPT-4V or LLaVA for everything?"**
   → Large generalist models ignore structured numerical data in zero-shot settings (our results show LLaVA scores 0.79 vs. our 0.83). They're also too slow (198ms+ vs. 115ms) and too expensive for production retail environments processing millions of SKUs.

3. **"What is the modality dominance problem and how did you solve it?"**
   → During training, the fusion layer learned to mute visual features because textual reviews had stronger explicit signals. We solved it with heavy dropout (0.1) in the fusion layer, forcing the network to learn from all modalities.

4. **"Why is your quality metric better than BLEU/ROUGE?"**
   → BLEU/ROUGE measure word overlap, not business value. Our Q metric (0.4R + 0.4C + 0.2U) directly measures relevance, factual consistency, and decision-making usefulness, validated by 5 domain experts with κ = 0.782.

5. **"What does LoRA actually do?"**
   → Instead of updating all 7B parameters (which would cause OOM), LoRA freezes the base model and injects small rank-16 matrices into the attention layers' query and value projections. This reduces trainable parameters by 98% while preserving generation quality.

6. **"Why concatenation over cross-attention for fusion?"**
   → Cross-attention scored marginallyb better (0.85 vs. 0.83) but was 42% slower (198ms vs. 115ms). In production retail systems processing thousands of products per minute, latency matters more than a 0.02 quality improvement.

7. **"What are the limitations of your approach?"**
   → Three main failures: (1) BERT misreads heavy sarcasm, (2) ViT struggles with cluttered product images (overlays, watermarks), (3) Small review counts near the 50-review threshold amplify outlier effects.

8. **"How do you prevent the LLM from hallucinating?"**
   → Three mechanisms: (1) Rigid prompt template with explicit anti-hallucination instruction, (2) Low generation temperature (T=0.3), (3) Grounding in explicit numerical constraints injected into the prompt.

9. **"Can this system work in real-time?"**
   → Yes — 115ms per product. At scale: ~1,000 products analyzed in ~2 minutes. With batching and GPU parallelism, this supports real-time operational dashboards.

10. **"What would you do differently if you had more time/resources?"**
    → Add video and audio modalities for live-commerce, implement causal inference for prescriptive insights, and use model distillation for edge deployment.
