"""
Training Pipeline for Multimodal Retail Insight Model.

Implements:
  - Real Causal LM loss (cross-entropy) — NOT dummy MSE
  - Automatic Mixed Precision (AMP) — per paper
  - Validation loop with best-model checkpointing
  - Gradient clipping (max_norm=1.0) for stability
  - Cosine annealing learning rate scheduler
  - Encoder freezing strategy (freeze ViT+BERT for first N epochs)
  - Gradient accumulation (effective batch size 32 on 8GB VRAM)

Tuned for: RTX 5070 (8GB VRAM) with distilgpt2 as LLM backend.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalRetailInsightModel
from src.dataset import MultimodalRetailDataset, collate_fn
from src.config import (
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, MAX_GRAD_NORM, WARMUP_RATIO, USE_AMP,
    TRAIN_CSV, VAL_CSV, IMAGE_DIR, CHECKPOINT_DIR,
    LLM_MODEL_NAME, USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    NUM_STRUCTURED_FEATURES, BERT_MAX_LENGTH, LLM_MAX_LENGTH,
    FREEZE_ENCODERS_EPOCHS, DEVICE
)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    """Main training function."""
    print("=" * 70)
    print("MULTIMODAL RETAIL INSIGHT MODEL — TRAINING")
    print("=" * 70)
    print(f"  Device:        {DEVICE}")
    print(f"  LLM:           {LLM_MODEL_NAME}")
    print(f"  Batch Size:    {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  AMP:           {USE_AMP}")
    print(f"  LoRA:          {USE_LORA} (r={LORA_R}, α={LORA_ALPHA})")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. Check Data
    # ═══════════════════════════════════════════════════════════════════
    if not os.path.exists(TRAIN_CSV):
        print(f"\n✗ Dataset not found at {TRAIN_CSV}")
        print("  Run `python -m src.prepare_data` first to download and prepare the dataset.")
        return
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. Load Tokenizers
    # ═══════════════════════════════════════════════════════════════════
    print("\nLoading tokenizers...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. Load Data
    # ═══════════════════════════════════════════════════════════════════
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV) if os.path.exists(VAL_CSV) else None
    
    print(f"  Train: {len(train_df)} samples")
    if val_df is not None:
        print(f"  Val:   {len(val_df)} samples")
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. Initialize Model
    # ═══════════════════════════════════════════════════════════════════
    print("\nInitializing model...")
    model = MultimodalRetailInsightModel(
        num_structured_features=NUM_STRUCTURED_FEATURES,
        llm_model_name=LLM_MODEL_NAME,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        freeze_encoders=True  # Start frozen
    )
    model = model.to(DEVICE)
    
    # Get LLM tokenizer from model (shared instance)
    llm_tokenizer = model.llm_tokenizer
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. Create Datasets and DataLoaders
    # ═══════════════════════════════════════════════════════════════════
    train_dataset = MultimodalRetailDataset(
        data_df=train_df,
        bert_tokenizer=bert_tokenizer,
        llm_tokenizer=llm_tokenizer,
        image_dir=IMAGE_DIR,
        bert_max_length=BERT_MAX_LENGTH,
        llm_max_length=LLM_MAX_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = None
    if val_df is not None:
        val_dataset = MultimodalRetailDataset(
            data_df=val_df,
            bert_tokenizer=bert_tokenizer,
            llm_tokenizer=llm_tokenizer,
            image_dir=IMAGE_DIR,
            bert_max_length=BERT_MAX_LENGTH,
            llm_max_length=LLM_MAX_LENGTH
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. Optimizer + Scheduler
    # ═══════════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # AMP Scaler
    scaler = GradScaler(enabled=USE_AMP)
    
    # ═══════════════════════════════════════════════════════════════════
    # 7. Training Loop
    # ═══════════════════════════════════════════════════════════════════
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    training_log = []
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # ── Encoder Freeze/Unfreeze Strategy ──
        if epoch == FREEZE_ENCODERS_EPOCHS:
            model.unfreeze_encoders()
            # Rebuild optimizer to include encoder params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE * 0.1,  # Lower LR for pretrained encoders
                weight_decay=WEIGHT_DECAY
            )
            remaining_steps = len(train_loader) * (EPOCHS - epoch) // GRADIENT_ACCUMULATION_STEPS
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, remaining_steps)
        
        # ── TRAIN ──
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                        leave=True, ncols=100)
        
        for step, batch in enumerate(progress):
            # Move to device
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            structured_data = batch["structured_data"].to(DEVICE, non_blocking=True)
            labels_input_ids = batch["labels_input_ids"].to(DEVICE, non_blocking=True)
            labels_attention_mask = batch["labels_attention_mask"].to(DEVICE, non_blocking=True)
            
            # Forward + Loss with AMP
            with autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    structured_data=structured_data,
                    labels_input_ids=labels_input_ids,
                    labels_attention_mask=labels_attention_mask
                )
                loss = outputs["loss"] / GRADIENT_ACCUMULATION_STEPS
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            batch_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_train_loss += batch_loss
            num_train_batches += 1
            
            progress.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_train_loss = total_train_loss / max(1, num_train_batches)
        
        # ── VALIDATE ──
        avg_val_loss = float("inf")
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
                                  leave=False, ncols=100):
                    pixel_values = batch["pixel_values"].to(DEVICE)
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    structured_data = batch["structured_data"].to(DEVICE)
                    labels_input_ids = batch["labels_input_ids"].to(DEVICE)
                    labels_attention_mask = batch["labels_attention_mask"].to(DEVICE)
                    
                    with autocast(device_type="cuda", enabled=USE_AMP):
                        outputs = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            structured_data=structured_data,
                            labels_input_ids=labels_input_ids,
                            labels_attention_mask=labels_attention_mask
                        )
                    
                    total_val_loss += outputs["loss"].item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / max(1, num_val_batches)
        
        epoch_time = time.time() - epoch_start
        
        # ── Logging ──
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_s": epoch_time
        }
        training_log.append(log_entry)
        
        print(f"\n  Epoch {epoch+1}/{EPOCHS} │ "
              f"Train Loss: {avg_train_loss:.4f} │ "
              f"Val Loss: {avg_val_loss:.4f} │ "
              f"LR: {scheduler.get_last_lr()[0]:.2e} │ "
              f"Time: {epoch_time:.1f}s")
        
        # ── Checkpointing (save best) ──
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "training_log": training_log,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(checkpoint_data, ckpt_path)
            print(f"  ★ New Best! Saved checkpoint → {ckpt_path}")
        
        # Save every epoch checkpoint
        epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:02d}.pth")
        torch.save(checkpoint_data, epoch_path)
        print(f"  Saved epoch checkpoint → {epoch_path}")

        # Save latest checkpoint (always overwritten)
        latest_path = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
        torch.save(checkpoint_data, latest_path)
    
    # ═══════════════════════════════════════════════════════════════════
    # 8. Final Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"  Latest Checkpoint: {latest_path}")
    
    # Save training log
    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"  Training Log: {log_path}")
    
    # ── Generate sample insight from best model ──
    print("\n" + "=" * 70)
    print("GENERATING SAMPLE INSIGHTS FROM BEST MODEL")
    print("=" * 70)
    
    try:
        best_ckpt = torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            map_location=DEVICE, weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        
        # Get a few samples from validation set
        val_df_sample = pd.read_csv(VAL_CSV).head(3)
        sample_dataset = MultimodalRetailDataset(
            data_df=val_df_sample,
            bert_tokenizer=bert_tokenizer,
            llm_tokenizer=None,  # No target needed for inference
            image_dir=IMAGE_DIR,
            bert_max_length=BERT_MAX_LENGTH
        )
        
        for i in range(min(3, len(sample_dataset))):
            sample = sample_dataset[i]
            print(f"\n  Sample {i+1}:")
            print(f"  Title: {val_df_sample.iloc[i].get('title', 'N/A')}")
            
            with torch.no_grad():
                insight = model.generate_insight(
                    pixel_values=sample["pixel_values"].unsqueeze(0).to(DEVICE),
                    input_ids=sample["input_ids"].unsqueeze(0).to(DEVICE),
                    attention_mask=sample["attention_mask"].unsqueeze(0).to(DEVICE),
                    structured_data=sample["structured_data"].unsqueeze(0).to(DEVICE),
                    max_new_tokens=80,
                    temperature=0.3
                )
            print(f"  Generated Insight: {insight[0]}")
    except Exception as e:
        print(f"  Could not generate sample insights: {e}")
    
    return training_log


if __name__ == "__main__":
    train()
