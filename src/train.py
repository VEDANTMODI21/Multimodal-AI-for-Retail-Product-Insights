import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from src.model import MultimodalRetailInsightModel
from src.dataset import MultimodalRetailDataset
import pandas as pd
from tqdm import tqdm
import os

def train():
    # 1. Hyperparameters & Configuration
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    DATASET_CSV = "data/train.csv" # Expected to be added manually
    IMAGE_DIR = "data/images" # Expected to be added manually
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Training on Device: {DEVICE}")
    
    # Check if data exists
    if not os.path.exists(DATASET_CSV):
        print(f"Warning: Dataset CSV '{DATASET_CSV}' not found.")
        print("Please load your data inside the 'data/' folder to proceed with training.")
        return

    # 2. Tokenizer Setup
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 3. Data Loading
    df = pd.read_csv(DATASET_CSV)
    
    dataset = MultimodalRetailDataset(
        data_df=df,
        tokenizer=tokenizer,
        image_dir=IMAGE_DIR,
        max_length=128
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 4. Model Setup
    # num_structured_features should match the ones expected in 'dataset.py' (price, rating, return_rate) = 3
    model = MultimodalRetailInsightModel(num_structured_features=3)
    model.to(DEVICE)
    model.train()

    # 5. Optimizer
    # Typically, in the paper, AdamW was used
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Assuming cross entropy loss for text generation / mapping constraints
    # (Simplified for demonstration; normally this would hook intimately into Llama's causal LM loss using `labels`)
    criterion = nn.MSELoss() # Placeholder: actual loss function depends on exact LLM hook

    # 6. Training Loop
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Move to device
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            structured_data = batch['structured_data'].to(DEVICE)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                structured_data=structured_data
            )
            
            # In a full generation setup, these `llm_virtual_tokens` 
            # are prepended as embeddings to the LLM token inputs, measuring causal LM log-likelihood.
            virtual_tokens = outputs['llm_virtual_tokens']
            
            # Simulated dummy loss calculation for script structural validation
            dummy_target = torch.randn_like(virtual_tokens) 
            loss = criterion(virtual_tokens, dummy_target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(dataloader):.4f}")
        
    print("Training finished. Saving model weights...")
    # NOTE: In production with LoRA, you would use `peft` model.save_pretrained()
    torch.save(model.state_dict(), "multimodal_model_weights.pth")
    print("Saved as multimodal_model_weights.pth")

if __name__ == "__main__":
    train()
