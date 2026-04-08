"""
Centralized Configuration for Multimodal AI for Retail Product Insights.
Supports lightweight (distilgpt2) and production (Llama-2) LLM backends.
Tuned for RTX 5070 (8GB VRAM).
"""
import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ─── Device (auto-detect CUDA → RTX 5070) ────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[Config] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[Config] WARNING: CUDA not available — running on CPU (slow!)")

# ─── Encoder Configuration ──────────────────────────────────────────────────
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
BERT_MODEL_NAME = "bert-base-uncased"

# ─── LLM Configuration (Lightweight for 8GB VRAM) ───────────────────────────
# Tier 1: distilgpt2 (~82M params) — fits comfortably in 8GB alongside ViT+BERT
LLM_MODEL_NAME = "distilgpt2"
LLM_MAX_NEW_TOKENS = 150       # 3-sentence actionable insight ≈ 80-150 tokens
LLM_TEMPERATURE = 0.3          # Low = deterministic, less hallucination
USE_LORA = True
LORA_R = 32                    # Increased from 16 for richer adaptation
LORA_ALPHA = 64                # 2x rank for stable LoRA learning
LORA_DROPOUT = 0.05            # Reduced to let model learn more
LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # Both attention layers

# ─── Structured Features ────────────────────────────────────────────────────
STRUCTURED_FEATURES = ["price_scaled", "rating_scaled", "return_rate_scaled"]
NUM_STRUCTURED_FEATURES = len(STRUCTURED_FEATURES)

# ─── Training Hyperparameters (Paper: AdamW, lr=2e-5, epochs=15, batch=32) ─
# Adjusted for 8GB VRAM: smaller batch, gradient accumulation to effective 32
BATCH_SIZE = 4                 # Fits in 8GB
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 4 * 8 = 32
EPOCHS = 25                    # Extended: loss was still decreasing at epoch 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0            # Gradient clipping
DROPOUT = 0.1
WARMUP_RATIO = 0.1             # 10% warmup

# ─── Text Processing ────────────────────────────────────────────────────────
BERT_MAX_LENGTH = 128          # Paper spec
LLM_MAX_LENGTH = 150           # Extended for diverse longer insights

# ─── Image Processing (Paper: 224x224, ImageNet normalization) ───────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Data Split Ratios (Paper: 70/15/15) ─────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── AMP (Automatic Mixed Precision) ─────────────────────────────────────────
USE_AMP = True

# ─── Freeze strategy (freeze encoders for first N epochs to stabilize) ───────
FREEZE_ENCODERS_EPOCHS = 2     # freeze ViT + BERT for 2 epochs, then unfreeze
