"""
Inference Pipeline for Multimodal Retail Insight Model.
Generates real LLM business insights — NOT mock responses.
"""
import os
import sys
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalRetailInsightModel
from src.config import (
    DEVICE, CHECKPOINT_DIR, IMAGE_DIR, DATA_DIR,
    LLM_MODEL_NAME, USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    NUM_STRUCTURED_FEATURES, BERT_MAX_LENGTH, LLM_MAX_LENGTH,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
)


class RetailInsightPredictor:
    """
    End-to-end inference class for generating retail product insights.
    Loads trained model weights and processes raw inputs.
    """
    
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or DEVICE
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        
        # Initialize model
        print("Loading Multimodal Retail Insight Model...")
        self.model = MultimodalRetailInsightModel(
            num_structured_features=NUM_STRUCTURED_FEATURES,
            llm_model_name=LLM_MODEL_NAME,
            use_lora=USE_LORA,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
        )
        
        # Load trained weights
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
                  f"val_loss={ckpt.get('val_loss', '?'):.4f}")
        else:
            print(f"WARNING: No checkpoint found at {checkpoint_path}")
            print("  Running with untrained model weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Load scaler params for un-scaling
        scaler_path = os.path.join(DATA_DIR, "scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                self.scaler_params = json.load(f)
        else:
            self.scaler_params = {
                "price_min": 0, "price_max": 200,
                "rating_min": 1.0, "rating_max": 5.0,
                "return_rate_min": 0.0, "return_rate_max": 1.0,
            }
        
        print("Model ready for inference.\n")
    
    def scale_value(self, value, min_val, max_val):
        """Min-max scale a value to [0, 1]."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def predict(self, image_path, review_text, price, rating, return_rate,
                max_new_tokens=None, temperature=None):
        """
        Generate a business insight from raw product data.
        
        Args:
            image_path: path to product image
            review_text: customer review text (can be multiple reviews concatenated)
            price: product price (raw, unscaled)
            rating: product rating (1-5)
            return_rate: return rate (0-1 or 0-100%)
            max_new_tokens: max tokens for generation
            temperature: generation temperature
        
        Returns:
            dict with 'insight', 'fusion_features_shape'
        """
        if max_new_tokens is None:
            max_new_tokens = LLM_MAX_NEW_TOKENS
        if temperature is None:
            temperature = LLM_TEMPERATURE
        
        # ── Process Image ──
        try:
            pil_img = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transforms(pil_img).unsqueeze(0)
        except Exception as e:
            print(f"Warning: Could not load image ({e}). Using zero tensor.")
            pixel_values = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        # ── Process Text ──
        text_inputs = self.bert_tokenizer(
            review_text,
            max_length=BERT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # ── Process Structured Data (scale to [0,1]) ──
        price_scaled = self.scale_value(
            price, self.scaler_params["price_min"], self.scaler_params["price_max"]
        )
        rating_scaled = self.scale_value(
            rating, self.scaler_params["rating_min"], self.scaler_params["rating_max"]
        )
        # Normalize return_rate to 0-1 if given as percentage
        rr = return_rate / 100.0 if return_rate > 1 else return_rate
        rr_scaled = self.scale_value(
            rr, self.scaler_params["return_rate_min"], self.scaler_params["return_rate_max"]
        )
        
        structured_data = torch.tensor(
            [[price_scaled, rating_scaled, rr_scaled]], dtype=torch.float32
        )
        
        # ── Generate ──
        with torch.no_grad():
            insights = self.model.generate_insight(
                pixel_values=pixel_values.to(self.device),
                input_ids=text_inputs["input_ids"].to(self.device),
                attention_mask=text_inputs["attention_mask"].to(self.device),
                structured_data=structured_data.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        # Get fusion features for analysis
        with torch.no_grad():
            h_f = self.model.get_fusion_features(
                pixel_values.to(self.device),
                text_inputs["input_ids"].to(self.device),
                text_inputs["attention_mask"].to(self.device),
                structured_data.to(self.device)
            )
        
        insight = insights[0].strip() if insights else "Unable to generate insight."
        
        return {
            "insight": insight,
            "fusion_features_shape": list(h_f.shape),
            "input_summary": {
                "price": price,
                "rating": rating,
                "return_rate": return_rate,
                "review_length": len(review_text),
                "image": image_path
            }
        }


def inference(image_path, review_text, price, rating, return_rate):
    """
    Convenience function for backward compatibility and API usage.
    """
    predictor = RetailInsightPredictor()
    result = predictor.predict(image_path, review_text, price, rating, return_rate)
    
    print("\n[Generated Retail Insight]")
    print(result["insight"])
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Inference on a Retail Product")
    parser.add_argument("--image", type=str, default="data/images/sample.jpg",
                        help="Path to product image")
    parser.add_argument("--review", type=str,
                        default="Looks great but honestly feels very cheap. The zipper broke in 2 days.",
                        help="Customer review text")
    parser.add_argument("--price", type=float, default=89.99,
                        help="Product price (raw)")
    parser.add_argument("--rating", type=float, default=2.8,
                        help="Product rating (1-5)")
    parser.add_argument("--return_rate", type=float, default=0.4,
                        help="Return rate (0-1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    predictor = RetailInsightPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(
        image_path=args.image,
        review_text=args.review,
        price=args.price,
        rating=args.rating,
        return_rate=args.return_rate
    )
    
    print("\n" + "=" * 70)
    print("MULTIMODAL RETAIL INSIGHT — INFERENCE RESULT")
    print("=" * 70)
    print(f"  Product: {args.image}")
    print(f"  Price: ${args.price:.2f} | Rating: {args.rating}/5 | Return Rate: {args.return_rate*100:.0f}%")
    print(f"\n  Generated Insight:")
    print(f"  {result['insight']}")
    print(f"\n  Fusion Features Shape: {result['fusion_features_shape']}")
    print("=" * 70)
