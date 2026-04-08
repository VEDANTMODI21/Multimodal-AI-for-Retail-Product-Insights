"""
Optuna Hyperparameter Tuning for Multimodal Retail Insight Model.

Tunes: Learning Rate, LoRA Rank/Alpha/Dropout, Weight Decay, Warmup Ratio,
       Gradient Accumulation Steps, Fusion Dropout, and Freeze Epochs.

Uses 5-epoch trials with median pruning for efficiency.
After finding best params, automatically retrains with full 25 epochs.

Estimated time: ~15 trials × 8 min each ≈ 2 hours on RTX 5070.
"""
import os
import sys
import time
import math
import json
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalRetailInsightModel
from src.dataset import MultimodalRetailDataset, collate_fn
from src.config import (
    TRAIN_CSV, VAL_CSV, IMAGE_DIR, CHECKPOINT_DIR,
    LLM_MODEL_NAME, NUM_STRUCTURED_FEATURES,
    BERT_MAX_LENGTH, LLM_MAX_LENGTH, DEVICE, USE_AMP, MAX_GRAD_NORM
)

# ─── Tuning Constants ────────────────────────────────────────────────────────
TUNING_EPOCHS = 5           # Short trials for speed
N_TRIALS = 15               # Number of Optuna trials
FULL_RETRAIN_EPOCHS = 25    # Full training after best params found
STUDY_NAME = "multimodal_retail_hpt"
TUNING_DIR = os.path.join(CHECKPOINT_DIR, "optuna_tuning")


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


def run_trial_training(trial, train_loader, val_loader, hparams, epochs):
    """
    Run a single training trial with given hyperparameters.
    Returns the best validation loss achieved.
    """
    # ── Build Model ──
    model = MultimodalRetailInsightModel(
        num_structured_features=NUM_STRUCTURED_FEATURES,
        llm_model_name=LLM_MODEL_NAME,
        use_lora=True,
        lora_r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        lora_dropout=hparams["lora_dropout"],
        freeze_encoders=True
    )
    model = model.to(DEVICE)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"]
    )

    total_steps = len(train_loader) * epochs // hparams["grad_accum_steps"]
    warmup_steps = int(total_steps * hparams["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_loss = float("inf")
    freeze_epochs = hparams["freeze_epochs"]

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Encoder unfreeze ──
        if epoch == freeze_epochs:
            model.unfreeze_encoders()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=hparams["learning_rate"] * 0.1,
                weight_decay=hparams["weight_decay"]
            )
            remaining = len(train_loader) * (epochs - epoch) // hparams["grad_accum_steps"]
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, remaining)

        # ── TRAIN ──
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            structured_data = batch["structured_data"].to(DEVICE, non_blocking=True)
            labels_input_ids = batch["labels_input_ids"].to(DEVICE, non_blocking=True)
            labels_attention_mask = batch["labels_attention_mask"].to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    structured_data=structured_data,
                    labels_input_ids=labels_input_ids,
                    labels_attention_mask=labels_attention_mask
                )
                loss = outputs["loss"] / hparams["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % hparams["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * hparams["grad_accum_steps"]
            num_batches += 1

        avg_train_loss = total_train_loss / max(1, num_batches)

        # ── VALIDATE ──
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(f"    Epoch {epoch+1}/{epochs} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"Best: {best_val_loss:.4f} | Time: {epoch_time:.1f}s")

        # ── Optuna Pruning ──
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                # Clean up GPU memory before pruning
                del model, optimizer, scaler
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

    # Clean up
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_loss


def objective(trial):
    """Optuna objective function — minimize validation loss."""
    # ── Sample Hyperparameters ──
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
        "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32, 64]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64, 128]),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.15),
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
        "grad_accum_steps": trial.suggest_categorical("grad_accum_steps", [4, 8, 16]),
        "freeze_epochs": trial.suggest_int("freeze_epochs", 1, 3),
    }

    # Ensure alpha >= rank (standard LoRA practice)
    if hparams["lora_alpha"] < hparams["lora_r"]:
        hparams["lora_alpha"] = hparams["lora_r"] * 2

    print(f"\n{'='*60}")
    print(f"Trial {trial.number + 1}/{N_TRIALS}")
    print(f"{'='*60}")
    for k, v in hparams.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    try:
        best_val_loss = run_trial_training(
            trial, train_loader, val_loader, hparams, TUNING_EPOCHS
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  !! OOM with these params — pruning trial")
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        raise

    return best_val_loss


def full_retrain_with_best(best_params, train_loader, val_loader):
    """Retrain with the best hyperparameters for full epochs, saving all checkpoints."""
    print("\n" + "=" * 70)
    print("FULL RETRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  epochs: {FULL_RETRAIN_EPOCHS}")
    print("=" * 70)

    # Build model with best params
    model = MultimodalRetailInsightModel(
        num_structured_features=NUM_STRUCTURED_FEATURES,
        llm_model_name=LLM_MODEL_NAME,
        use_lora=True,
        lora_r=best_params["lora_r"],
        lora_alpha=best_params["lora_alpha"],
        lora_dropout=best_params["lora_dropout"],
        freeze_encoders=True
    )
    model = model.to(DEVICE)
    llm_tokenizer = model.llm_tokenizer

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    total_steps = len(train_loader) * FULL_RETRAIN_EPOCHS // best_params["grad_accum_steps"]
    warmup_steps = int(total_steps * best_params["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=USE_AMP)

    best_val_loss = float("inf")
    training_log = []

    for epoch in range(FULL_RETRAIN_EPOCHS):
        epoch_start = time.time()

        # Unfreeze encoders
        if epoch == best_params["freeze_epochs"]:
            model.unfreeze_encoders()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=best_params["learning_rate"] * 0.1,
                weight_decay=best_params["weight_decay"]
            )
            remaining = len(train_loader) * (FULL_RETRAIN_EPOCHS - epoch) // best_params["grad_accum_steps"]
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, remaining)

        # ── TRAIN ──
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            structured_data = batch["structured_data"].to(DEVICE, non_blocking=True)
            labels_input_ids = batch["labels_input_ids"].to(DEVICE, non_blocking=True)
            labels_attention_mask = batch["labels_attention_mask"].to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    structured_data=structured_data,
                    labels_input_ids=labels_input_ids,
                    labels_attention_mask=labels_attention_mask
                )
                loss = outputs["loss"] / best_params["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % best_params["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * best_params["grad_accum_steps"]
            num_batches += 1

        avg_train_loss = total_train_loss / max(1, num_batches)

        # ── VALIDATE ──
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
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

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_s": epoch_time
        })

        print(f"  Epoch {epoch+1}/{FULL_RETRAIN_EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.1f}s")

        # ── Save Checkpoints ──
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "best_params": best_params,
            "training_log": training_log,
        }

        # Every epoch
        epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:02d}.pth")
        torch.save(checkpoint_data, epoch_path)

        # Best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(checkpoint_data, best_path)
            print(f"  ★ New Best! Val Loss: {best_val_loss:.4f} → {best_path}")

        # Latest
        latest_path = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
        torch.save(checkpoint_data, latest_path)

    # Save training log
    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Training Log: {log_path}")

    # Generate sample insights
    try:
        model.eval()
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        val_df = pd.read_csv(VAL_CSV).head(3)
        sample_ds = MultimodalRetailDataset(
            data_df=val_df,
            bert_tokenizer=bert_tokenizer,
            llm_tokenizer=None,
            image_dir=IMAGE_DIR,
            bert_max_length=BERT_MAX_LENGTH
        )

        print(f"\n{'='*70}")
        print("SAMPLE INSIGHTS FROM BEST MODEL")
        print(f"{'='*70}")

        for i in range(min(3, len(sample_ds))):
            sample = sample_ds[i]
            with torch.no_grad():
                insight = model.generate_insight(
                    pixel_values=sample["pixel_values"].unsqueeze(0).to(DEVICE),
                    input_ids=sample["input_ids"].unsqueeze(0).to(DEVICE),
                    attention_mask=sample["attention_mask"].unsqueeze(0).to(DEVICE),
                    structured_data=sample["structured_data"].unsqueeze(0).to(DEVICE),
                    max_new_tokens=150,
                    temperature=0.3
                )
            print(f"\n  [{i+1}] {val_df.iloc[i].get('title', 'N/A')[:60]}")
            print(f"      Rating: {val_df.iloc[i].get('rating', 'N/A')}")
            print(f"      Insight: {insight[0][:200]}...")
    except Exception as e:
        print(f"  Could not generate samples: {e}")

    return best_val_loss


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print(f"  Trials: {N_TRIALS} | Epochs/trial: {TUNING_EPOCHS}")
    print(f"  Full retrain epochs: {FULL_RETRAIN_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # ── Check data ──
    if not os.path.exists(TRAIN_CSV):
        print(f"\nERROR: No training data at {TRAIN_CSV}")
        print("Run `python -m src.prepare_data` first.")
        sys.exit(1)

    os.makedirs(TUNING_DIR, exist_ok=True)

    # ── Load data once (shared across all trials) ──
    print("\nLoading tokenizers and data...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # We need the LLM tokenizer — load it once
    from transformers import GPT2Tokenizer
    llm_tokenizer = GPT2Tokenizer.from_pretrained(LLM_MODEL_NAME)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = MultimodalRetailDataset(
        data_df=train_df,
        bert_tokenizer=bert_tokenizer,
        llm_tokenizer=llm_tokenizer,
        image_dir=IMAGE_DIR,
        bert_max_length=BERT_MAX_LENGTH,
        llm_max_length=LLM_MAX_LENGTH
    )

    val_dataset = MultimodalRetailDataset(
        data_df=val_df,
        bert_tokenizer=bert_tokenizer,
        llm_tokenizer=llm_tokenizer,
        image_dir=IMAGE_DIR,
        bert_max_length=BERT_MAX_LENGTH,
        llm_max_length=LLM_MAX_LENGTH
    )

    # Use batch_size=4 (fixed to fit VRAM), accumulation steps are tuned
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Optuna Search
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: HYPERPARAMETER SEARCH")
    print(f"{'='*70}\n")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )

    study_start = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    study_time = time.time() - study_start

    # ── Results ──
    print(f"\n{'='*70}")
    print("OPTUNA SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {study_time/60:.1f} min")
    print(f"  Completed trials: {len(study.trials)}")
    print(f"  Best trial: #{study.best_trial.number + 1}")
    print(f"  Best val loss: {study.best_value:.4f}")
    print(f"\n  Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save study results
    results = []
    for t in study.trials:
        row = {"trial": t.number + 1, "val_loss": t.value, "state": str(t.state)}
        row.update(t.params)
        results.append(row)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(TUNING_DIR, "optuna_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved: {results_path}")

    # Save best params as JSON
    best_params = dict(study.best_params)
    # Ensure alpha >= rank
    if best_params["lora_alpha"] < best_params["lora_r"]:
        best_params["lora_alpha"] = best_params["lora_r"] * 2

    params_path = os.path.join(TUNING_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best params saved: {params_path}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Full Retrain with Best Params
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: FULL RETRAINING WITH OPTIMAL HYPERPARAMETERS")
    print(f"{'='*70}")

    final_val_loss = full_retrain_with_best(best_params, train_loader, val_loader)

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Search time:      {study_time/60:.1f} min ({N_TRIALS} trials)")
    print(f"  Best trial loss:  {study.best_value:.4f} (5 epochs)")
    print(f"  Final model loss: {final_val_loss:.4f} ({FULL_RETRAIN_EPOCHS} epochs)")
    print(f"\n  Outputs:")
    print(f"    Optuna results:  {results_path}")
    print(f"    Best params:     {params_path}")
    print(f"    Best model:      {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"    All checkpoints: {CHECKPOINT_DIR}/epoch_XX.pth")
    print(f"    Training log:    {os.path.join(CHECKPOINT_DIR, 'training_log.csv')}")
    print(f"{'='*70}")
