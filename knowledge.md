# Knowledge Base — Multimodal AI for Retail Product Insights

This document is the **deep technical reference** for the entire paper and implementation. It covers every section in exhaustive detail — architecture decisions, mathematical formulations, experimental methodology, results analysis, failure modes, implementation specifics, and future work.

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
7. [Implementation Details](#7-implementation-details)
8. [Conclusion & Future Work](#8-conclusion--future-work)
9. [Viva Preparation — Key Concepts](#9-viva-preparation--key-concepts)

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
│  ViT-Base/16 │   │ BERT-base-     │    │  MLP + BatchNorm│
│  (ImageNet)  │   │ uncased        │    │  + Min-Max Scale│
└──────┬───────┘   └───────┬────────┘    └────────┬────────┘
       │                   │                      │
       ▼                   ▼                      ▼
     hv (768-d)         ht (768-d)             hs (128-d)
       │                   │                      │
       └───────────┬───────┘──────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Concatenation +    │
        │  ReLU Projection +  │
        │  LayerNorm +        │
        │  Dropout(0.1)       │
        │  → hf (512-d)       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Projection Layer   │
        │  → 4 Virtual Tokens │
        │  (LLM embed space)  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  [Virtual] + [Prompt│
        │   Template] →       │
        │  LLM (LoRA)         │
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

#### Implementation (model.py)
```python
self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
h_v = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]  # [CLS] token
```

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

#### Implementation (model.py)
```python
self.bert = BertModel.from_pretrained("bert-base-uncased")
h_t = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
```

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
h_f = LayerNorm(Dropout(ReLU(W_f [h_v ∥ h_t ∥ h_s] + b_f)))
```

Where:
- `h_v` (768-d) = visual features from ViT
- `h_t` (768-d) = textual features from BERT
- `h_s` (128-d) = structured features from MLP (with BatchNorm)
- `∥` = concatenation operator
- `W_f` = learnable weight matrix projecting down to **512 dimensions**
- `b_f` = bias vector
- ReLU introduces non-linearity
- **Dropout (0.1)** prevents modality dominance
- **LayerNorm** normalizes modality contributions

The result **h_f** is a **512-dimensional fused multimodal representation** of the product.

#### Implementation (model.py)
```python
self.fusion_layer = nn.Sequential(
    nn.Linear(fused_input_dim, 512),  # 768+768+128 → 512
    nn.ReLU(),
    nn.Dropout(0.1),                   # Anti-modality-dominance
    nn.LayerNorm(512)                  # Normalize contributions
)
```

---

### 4.5 LLM Prompt Engineering & Instruction Tuning

#### From Vectors to Language

h_f is projected into **4 virtual tokens** in the LLM embedding space using a learned projection layer, then prepended to a system prompt.

#### The Prompt Template (Implementation)

```
Product analysis: Based on the multimodal product representation
including visual features, customer review sentiment, and structured data,
provide a concise 2-sentence business insight:
```

The virtual tokens carry the compressed multimodal information, while the text prompt instructs the LLM on output format.

#### Virtual Token Mechanism

```python
# Project h_f to 4 virtual tokens in LLM space
self.fusion_to_llm_proj = nn.Sequential(
    nn.Linear(512, llm_embed_dim * 4),  # 512 → 768*4 = 3072
    nn.ReLU(),
    nn.Dropout(0.1)
)

# During forward pass:
virtual_embeds = self.fusion_to_llm_proj(h_f)  # (B, 3072)
virtual_embeds = virtual_embeds.view(B, 4, 768)  # (B, 4, 768)

# Prepend to prompt: [virtual_tokens | prompt_tokens | target_tokens]
```

#### Why This Design?

| Design Choice | Rationale |
|---------------|-----------|
| 4 virtual tokens | Enough capacity to encode 512-d fusion without overwhelming the prompt |
| ReLU + Dropout in projection | Prevents mode collapse in embedding space |
| Fixed prompt template | Controls output format (2-sentence, clinical, objective) |
| Low temperature (T=0.3) | Reduces hallucination, increases consistency |

#### LoRA Fine-Tuning

Full fine-tuning of even DistilGPT-2 alongside ViT and BERT would consume excessive memory. **LoRA (Low-Rank Adaptation)** solution:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 (tuned via Optuna; increased from 16 for richer adaptation) |
| Alpha (α) | 64 (2× rank for stable scaling) |
| Dropout | 0.05 (reduced to allow deeper learning) |
| Target Modules | `c_attn` + `c_proj` (GPT-2) or `q_proj, v_proj` (Llama-2) |
| Trainable Params | ~2M (distilgpt2) or ~16M (Llama-2 7B) |

---

## 5. Experimental Setup

### Dataset Construction

| Stage | Count | Notes |
|-------|-------|-------|
| Raw entries | 571M+ | Full Amazon Review Dataset 2023 |
| Clothing category | ~30M | Clothing, Shoes & Jewelry |
| Electronics category | ~66M | Electronics |
| After winter keyword filter | ~50K | Winter/cold-weather clothing products |
| After electronics keyword filter | ~60K | Consumer electronics accessories |
| After review threshold (≥3 reviews) | ~3,000 | Final working set (max 1500/category) |

**Split:** 70% Train / 15% Validation / 15% Test

### Preprocessing Pipeline

| Modality | Preprocessing |
|----------|---------------|
| **Images** | Resize to 224×224, center crop, normalize to ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| **Text** | WordPiece tokenization (BERT), max 128 tokens, padding/truncation |
| **Structured** | Min-max scaling to [0, 1] range; BatchNorm in MLP |
| **Target Insights** | GPT-2 tokenization, max 150 tokens, padding/truncation (3-sentence actionable format) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (torch.optim) |
| Learning Rate | 2 × 10⁻⁵ (tunable via Optuna: [5e-6, 1e-4]) |
| Epochs | 25 (extended from 15 — loss continued decreasing) |
| Batch Size | 4 (actual) × 8 (gradient accumulation) = 32 (effective) |
| Dropout | 0.1 (fusion), 0.05 (LoRA) |
| Gradient Clipping | max_norm = 1.0 |
| LR Schedule | Cosine annealing with 10% linear warmup |
| GPU | NVIDIA RTX 5070 Laptop GPU (8GB VRAM, Blackwell SM_120) |
| Framework | PyTorch 2.11+cu128 |
| Precision | AMP (Automatic Mixed Precision) |
| Encoder Freezing | ViT + BERT frozen for 2 epochs |
| HPT | Optuna (TPE Sampler + Median Pruner, 15 trials × 5 epochs) |
| Checkpointing | Per-epoch (epoch_XX.pth) + best_model.pth + latest_model.pth |

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

In a live retail environment processing **tens of thousands of products per minute**, the extra 83ms per product compounds into ~83 minutes of extra delay per 1,000,000 products.

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

A human analyst might take **hours** to reach that conclusion. The model did it in **115ms**.

---

### 6.5 Failure Modes

| Failure Type | Example | Root Cause | Potential Fix |
|-------------|---------|------------|---------------|
| **Sarcasm Blindness** | "works perfectly if you only want to use it for five minutes" → classified as positive | BERT's contextual understanding has limits with subtle sarcasm | Sarcasm-specific fine-tuning or sentiment-aware preprocessing |
| **Image Clutter** | Promotional overlays, dimension arrows, chaotic backgrounds (common in dropshipping) | ViT treats overlays as part of the product, corrupting visual features | Image preprocessing to remove non-product elements |
| **Review Skew** | Near the 50-review threshold, one aggressive 1-star review disproportionately skews h_t | Small sample size amplifies outlier reviews | Increase minimum review threshold or add outlier detection |

---

## 7. Implementation Details

### 7.1 File-by-File Architecture

#### `src/config.py` — Centralized Configuration

All hyperparameters in one place. Key design decisions:

| Setting | Value | Rationale |
|---------|-------|-----------|
| BATCH_SIZE = 4 | Fits in 8GB VRAM alongside ViT + BERT + GPT2 |
| GRADIENT_ACCUMULATION_STEPS = 8 | Effective batch = 32 (matches paper) |
| FREEZE_ENCODERS_EPOCHS = 2 | Stabilizes fusion layer before encoder gradients flow |
| num_virtual_tokens = 4 | Sufficient to encode 512-d fusion; more would slow generation |

#### `src/model.py` — MultimodalRetailInsightModel

The core model class with:
1. **Three encoders** (ViT, BERT, MLP) — each independently pretrained
2. **Fusion layer** — concatenation + ReLU + Dropout + LayerNorm → 512-d
3. **Projection layer** — 512-d → 4 virtual tokens in LLM embedding space
4. **LLM with LoRA** — DistilGPT-2 (dev) or Llama-2 (prod)
5. **`generate_insight()` method** — autoregressive decoding with top-p sampling

#### `src/train.py` — Training Pipeline

Key features:
- **Real causal LM cross-entropy loss** (NOT dummy MSE)
- **AMP (Automatic Mixed Precision)** — ~40% memory savings
- **Gradient accumulation** — effective batch size 32 from actual batch size 4
- **Cosine annealing with warmup** — smooth learning rate decay
- **Gradient clipping** (max_norm=1.0) — prevents exploding gradients
- **Validation loop** — tracks val loss, saves best checkpoint
- **Encoder freeze/unfreeze** — epochs 0-1 frozen, then unfrozen at 0.1× LR

#### `src/dataset.py` — MultimodalRetailDataset

Handles all three modalities + target insight tokenization:
- Images: PIL → transforms → (3, 224, 224) tensor
- Reviews: BERT tokenizer → (128,) input_ids + attention_mask
- Structured: float tensor [price_scaled, rating_scaled, return_rate_scaled]
- Target: LLM tokenizer → (120,) labels_input_ids + labels_attention_mask

#### `src/inference.py` — RetailInsightPredictor

Encapsulates the full inference pipeline:
- Loads best checkpoint
- Processes raw inputs (unscaled price, rating, etc.)
- Runs multimodal fusion
- Generates insight via autoregressive decoding with top-p sampling

#### `src/prepare_data.py` — Data Pipeline

Downloads and processes the Amazon Review Dataset (2023):
1. Loads Clothing + Electronics categories from HuggingFace (McAuley Lab)
2. Filters for category-specific keywords in title + description + features
3. Aggregates reviews per product (concatenates top 10 reviews per product)
4. Downloads product images (fallback to placeholder if unavailable)
5. Min-max scales structured features (price, rating, return_rate)
6. Generates **actionable 3-sentence insights** following WHY → WHAT → HOW TO IMPROVE format
7. Creates 70/15/15 train/val/test splits
8. **Auto-starts training** upon completion

#### `src/tune.py` — Optuna Hyperparameter Tuning

Automated HPT pipeline with two phases:
1. **Phase 1 — Search:** 15 Optuna trials × 5 epochs each, TPE Sampler with Median Pruner
2. **Phase 2 — Retrain:** Automatically retrains with best params for 25 epochs

Search space:
- Learning Rate: [5e-6, 1e-4] (log scale)
- LoRA Rank: [8, 16, 32, 64]
- LoRA Alpha: [16, 32, 64, 128]
- LoRA Dropout: [0.01, 0.15]
- Weight Decay: [0.001, 0.1] (log scale)
- Warmup Ratio: [0.05, 0.2]
- Gradient Accumulation: [4, 8, 16]
- Freeze Epochs: [1, 2, 3]

#### `src/evaluate.py` — Evaluation Pipeline

Computes quality metrics on the test set:
- Word-overlap F1 between generated and reference insights
- Fluency, Specificity, Coherence, Relevance scores
- Generates sample insights for qualitative inspection

### 7.2 Training Loss — How It Actually Works

The training loss is computed as follows:

```
Input embeddings = [virtual_tokens (4) | prompt_tokens (~25) | target_tokens (≤120)]
                    ↓ ignored (-100)      ↓ ignored (-100)      ↓ cross-entropy loss

Labels =           [-100, ..., -100,     -100, ..., -100,      target_token_ids]
```

- Virtual tokens and prompt tokens have labels set to `-100` (ignored by PyTorch CrossEntropyLoss)
- Only the target insight tokens contribute to the loss
- This trains the LLM to generate the target insight conditioned on the multimodal virtual tokens + prompt

### 7.3 Memory Management (8GB VRAM Budget)

| Component | Approx. VRAM |
|-----------|-------------|
| ViT-Base (frozen, AMP) | ~650 MB |
| BERT-base (frozen, AMP) | ~900 MB |
| DistilGPT-2 + LoRA | ~400 MB |
| Fusion + Projection layers | ~50 MB |
| Batch data (B=4) | ~200 MB |
| Optimizer states | ~800 MB |
| Gradient buffers | ~500 MB |
| **Total** | **~3.5 GB** (leaves ~4.5GB headroom) |

When encoders are unfrozen (epoch 3+), memory usage increases to ~5-6GB, still well within 8GB.

---

## 8. Conclusion & Future Work

### Summary of Contributions

1. **Proved that multimodal fusion generates better business insights** than any single modality (RQ1 ✅)
2. **Demonstrated that specialized fine-tuning beats generalist VLMs** for domain-specific tasks (RQ2 ✅)
3. **Identified concatenation as the optimal fusion strategy** for quality-latency trade-off (RQ3 ✅)
4. **Moved AI from opaque predictions to human-readable business intelligence**
5. **Provided a complete, runnable implementation** with data pipeline, training, and inference

### Future Directions

| Direction | Description | Impact |
|-----------|-------------|--------|
| **Video + Audio Modalities** | Add support for live-streaming commerce analysis (unboxing videos, product demos) | Expands to a rapidly growing commerce channel |
| **Model Distillation** | Compress the pipeline for edge deployment without cloud GPUs | Enables real-time in-store analytics |
| **Causal Inference** | Move beyond correlation to counterfactual reasoning ("what if we fixed the zipper?") | Enables prescriptive (not just descriptive) analytics |
| **Llama-2 7B Production Deployment** | Swap DistilGPT-2 for Llama-2 7B for higher quality insights | 98% parameter reduction via LoRA makes this feasible on A100 |

---

## 9. Viva Preparation — Key Concepts

### Concepts You Must Be Able to Explain

| Concept | One-Line Explanation |
|---------|---------------------|
| **Vision Transformer (ViT)** | Splits image into patches, uses self-attention to capture global visual context |
| **BERT** | Bidirectional transformer that understands word meaning from full context (left + right) |
| **Late Fusion** | Each modality is encoded independently, then combined at the representation level |
| **Modality Dominance** | When one modality's signal is so strong the model ignores others |
| **LoRA** | Fine-tuning technique that injects small trainable matrices instead of updating all parameters |
| **Virtual Tokens** | Learned embeddings projected from fusion features, prepended to LLM prompt as "soft" inputs |
| **Chain-of-Thought Prompting** | Structuring prompts to guide LLMs through step-by-step reasoning |
| **Fleiss' Kappa** | Statistical measure of inter-rater agreement for categorical ratings |
| **AMP (Mixed Precision)** | Using FP16 for forward pass and FP32 for gradients to save ~40% memory |
| **Gradient Accumulation** | Accumulate gradients over N mini-batches before optimizer step; simulates larger batch sizes |
| **Cosine Annealing** | Learning rate schedule that smoothly decreases following a cosine curve |
| **Encoder Freezing** | Temporarily disabling gradient updates for pretrained encoders to stabilize training |
| **Ablation Study** | Systematic removal of components to measure individual contributions |

### Likely Viva Questions

1. **"Why ViT instead of ResNet?"**
   → ViT captures global image context from layer 1 via self-attention. CNNs need deep stacking to grow receptive fields, making them better for local textures but worse for overall product presentation understanding.

2. **"Why not just use GPT-4V or LLaVA for everything?"**
   → Large generalist models ignore structured numerical data in zero-shot settings (our results show LLaVA scores 0.79 vs. our 0.83). They're also too slow (198ms+ vs. 115ms) and too expensive for production retail environments processing millions of SKUs.

3. **"What is the modality dominance problem and how did you solve it?"**
   → During training, the fusion layer learned to mute visual features because textual reviews had stronger explicit signals. We solved it with: (1) Dropout (0.1) in the fusion layer, (2) LayerNorm to equalize modality contributions, (3) BatchNorm in the structured MLP, (4) Encoder freezing for early epochs so the fusion layer stabilizes first.

4. **"Why is your quality metric better than BLEU/ROUGE?"**
   → BLEU/ROUGE measure word overlap, not business value. Our Q metric (0.4R + 0.4C + 0.2U) directly measures relevance, factual consistency, and decision-making usefulness, validated by 5 domain experts with κ = 0.782.

5. **"What does LoRA actually do?"**
   → Instead of updating all model parameters (which would cause OOM), LoRA freezes the base model and injects small rank-32 matrices into the attention layers' projections (`c_attn` + `c_proj`). This reduces trainable parameters by 98%+ while preserving generation quality. Alpha is set to 2× rank (64) for stable scaling.

5b. **"Why did you use Optuna for hyperparameter tuning?"**
   → Manual hyperparameter selection is subjective and non-reproducible. Optuna's TPE (Tree-structured Parzen Estimator) sampler uses Bayesian optimization to efficiently search the hyperparameter space, while the Median Pruner terminates underperforming trials early, reducing total search time by ~40%. We search over 8 hyperparameters across 15 trials with 5-epoch warmup each, then retrain with the best configuration for the full 25 epochs.

5c. **"What hyperparameters did Optuna tune, and why those specifically?"**
   → We tuned learning rate, LoRA rank, LoRA alpha, LoRA dropout, weight decay, warmup ratio, gradient accumulation steps, and freeze epochs. These were selected because: (1) learning rate and weight decay directly control optimization dynamics, (2) LoRA parameters control the expressiveness of the LLM adaptation, and (3) warmup ratio and freeze epochs affect training stability during the critical early phases.

6. **"Why concatenation over cross-attention for fusion?"**
   → Cross-attention scored marginally better (0.85 vs. 0.83) but was 42% slower (198ms vs. 115ms). In production retail systems processing thousands of products per minute, latency matters more than a 0.02 quality improvement.

7. **"What are the limitations of your approach?"**
   → Three main failures: (1) BERT misreads heavy sarcasm, (2) ViT struggles with cluttered product images (overlays, watermarks), (3) Small review counts near the threshold amplify outlier effects.

8. **"How do you prevent the LLM from hallucinating?"**
   → Four mechanisms: (1) Structured prompt template with explicit instructions, (2) Low generation temperature (T=0.3), (3) Grounding in explicit numerical constraints via virtual tokens, (4) Top-p (nucleus) sampling with p=0.9.

9. **"Explain your virtual token mechanism."**
   → The 512-d fusion vector h_f is projected to 4 tokens in the LLM's embedding space via a learned linear layer. These tokens are prepended to the text prompt, acting as "soft" context that the LLM conditions its generation on. This is similar to prefix tuning but with multimodal inputs.

10. **"Why freeze encoders for 2 epochs?"**
    → If encoders update immediately, their gradients can overwhelm the randomly-initialized fusion layer, causing training instability. Freezing allows the fusion layer and LLM adapter to first learn meaningful representations from the pretrained encoder outputs, then unfreezing enables end-to-end fine-tuning with a reduced learning rate.

11. **"How does gradient accumulation help on limited VRAM?"**
    → With 8GB VRAM, we can only fit batch_size=4. But the paper specifies batch_size=32 for stable training. Gradient accumulation solves this: accumulate gradients across 8 mini-batches (4×8=32) before one optimizer step. Mathematically equivalent to batch=32, but only loads 4 samples at a time.

12. **"Can this system work in real-time?"**
    → Yes — 115ms per product with the production pipeline. At scale: ~1,000 products analyzed in ~2 minutes. With batching and GPU parallelism, this supports real-time operational dashboards.

13. **"What would you do differently if you had more time/resources?"**
    → Add video and audio modalities for live-commerce, implement causal inference for prescriptive insights, use model distillation for edge deployment, and swap DistilGPT-2 for Llama-2 7B on A100 infrastructure for higher quality generation.

14. **"Why did you train on both Clothing and Electronics?"**
    → To demonstrate cross-category generalization. A model trained only on winter clothing might learn clothing-specific heuristics. Adding electronics forces the model to learn generalizable multimodal reasoning patterns — visual quality assessment, sentiment extraction, and pricing analysis that work across product categories.

15. **"Explain your actionable insight format (WHY → WHAT → HOW TO IMPROVE)."**
    → Each insight follows a structured 3-sentence format: (1) WHY the product has its current rating — root cause analysis, (2) WHAT specific attributes drove satisfaction or complaints — grounded in actual review keywords, (3) HOW TO IMPROVE — concrete, actionable business recommendations. This format was chosen because traditional AI summaries describe symptoms without prescribing solutions, making them less useful for business decision-making.
