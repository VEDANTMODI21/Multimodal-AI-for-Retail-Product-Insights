"""
Dataset class for the Multimodal Retail Insight pipeline.
Handles text (BERT), image (ViT), structured data (MLP), and target insight (LLM).
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


class MultimodalRetailDataset(Dataset):
    """
    Dataset handling all three modalities + target insight for LLM training.
    
    Expected CSV columns:
      - image_path: filename of product image
      - review_text: concatenated customer reviews  
      - price_scaled, rating_scaled, return_rate_scaled: [0,1] scaled features
      - target_insight: ground truth business insight text
    """
    
    def __init__(self, data_df, bert_tokenizer, llm_tokenizer=None,
                 image_dir=None, bert_max_length=128, llm_max_length=120):
        """
        Args:
            data_df: DataFrame with required columns
            bert_tokenizer: BERT tokenizer for review text
            llm_tokenizer: LLM tokenizer for target insights (None for inference)
            image_dir: directory containing product images
            bert_max_length: max tokens for BERT input
            llm_max_length: max tokens for LLM target
        """
        self.data = data_df.reset_index(drop=True)
        self.bert_tokenizer = bert_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.image_dir = image_dir
        self.bert_max_length = bert_max_length
        self.llm_max_length = llm_max_length
        
        # Image transforms per paper spec:
        # Resize 224x224, center crop, normalize to ImageNet statistics
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # ═══════════════════════════════════════════════════════════════
        # 1. VISUAL MODALITY — Product Image
        # ═══════════════════════════════════════════════════════════════
        img_path = str(row["image_path"])
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.image_transforms(image)
        except Exception:
            # Fallback: zero tensor (model handles gracefully via dropout)
            pixel_values = torch.zeros(3, 224, 224)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. TEXTUAL MODALITY — Customer Reviews
        # ═══════════════════════════════════════════════════════════════
        review_text = str(row.get("review_text", ""))
        text_inputs = self.bert_tokenizer(
            review_text,
            max_length=self.bert_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        # ═══════════════════════════════════════════════════════════════
        # 3. STRUCTURED MODALITY — Scaled numerical features
        # ═══════════════════════════════════════════════════════════════
        structured_data = torch.tensor([
            float(row.get("price_scaled", 0.5)),
            float(row.get("rating_scaled", 0.5)),
            float(row.get("return_rate_scaled", 0.0)),
        ], dtype=torch.float32)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. TARGET INSIGHT — For LLM training
        # ═══════════════════════════════════════════════════════════════
        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structured_data": structured_data,
        }
        
        if self.llm_tokenizer is not None:
            target_insight = str(row.get("target_insight", ""))
            target_tokens = self.llm_tokenizer(
                target_insight,
                max_length=self.llm_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["labels_input_ids"] = target_tokens["input_ids"].squeeze(0)
            result["labels_attention_mask"] = target_tokens["attention_mask"].squeeze(0)
        
        return result


def collate_fn(batch):
    """Custom collate function to properly batch all modalities."""
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    
    return result
