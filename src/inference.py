import torch
from transformers import BertTokenizer, LlamaForCausalLM, LlamaTokenizer
from src.model import MultimodalRetailInsightModel
from src.dataset import MultimodalRetailDataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

def inference(image_path, review_text, price, rating, return_rate):
    """
    Simulated Inference Pipeline demonstrating how a front-end UI would interact with the model.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the Model (Assuming pre-trained weights exist from train.py)
    model = MultimodalRetailInsightModel(num_structured_features=3)
    try:
        model.load_state_dict(torch.load("multimodal_model_weights.pth", map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Could not load custom weights (maybe not trained yet?). Using untrained framework. Error: {e}")
    
    model.to(DEVICE)
    model.eval()
    
    # 2. Text Processor
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_inputs = bert_tokenizer(
        review_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 3. Image Processor
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        pil_img = Image.open(image_path).convert("RGB")
        pixel_values = image_transforms(pil_img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image, using dummy zero-tensor. Error: {e}")
        pixel_values = torch.zeros((1, 3, 224, 224))
        
    # 4. Structured Processor
    # Assume inputs are scaled (0 to 1) for the model format
    structured_inputs = torch.tensor([[price, rating, return_rate]], dtype=torch.float32)

    # 5. Multi-modal Forward Pass
    print("\n--- Running AI Fusing Framework ---")
    with torch.no_grad():
        pixel_values = pixel_values.to(DEVICE)
        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)
        structured_inputs = structured_inputs.to(DEVICE)
        
        outputs = model(pixel_values, input_ids, attention_mask, structured_inputs)
        
    fusion_features = outputs["fusion_features"]
    virtual_tokens = outputs["llm_virtual_tokens"]
    
    # 6. LLM Generation
    print("Generating LLM Insight...")
    # NOTE: The actual generation uses Virtual Tokens prepended to prompts natively with peft's prompt tuning
    # For standalone simulation, we simulate the text response
    
    # In reality: 
    # outputs = llm.generate(inputs_embeds=virtual_tokens)
    # text = tokenizer.decode(outputs)
    
    # Demo Mock response
    mock_insight = "High visual expectations set by premium imagery are unmet by literal physical composition cited in customer reviews, causing conversion drop-off despite acceptable parameters."
    
    print("\n[Generated Retail Insight]")
    print(mock_insight)
    
    return {
        "fusion_features_shape": fusion_features.shape,
        "insight": mock_insight
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Inference on a Retail Product")
    parser.add_argument("--image", type=str, default="data/sample_img.jpg", help="Path to product image")
    parser.add_argument("--review", type=str, default="Looks great but honestly feels very cheap. the zipper broke in 2 days.", help="Example review")
    parser.add_argument("--price", type=float, default=0.8, help="Scaled price (0-1)")
    parser.add_argument("--rating", type=float, default=0.4, help="Scaled rating (0-1)")
    parser.add_argument("--return_rate", type=float, default=0.9, help="Scaled return rate (0-1)")
    
    args = parser.parse_args()
    
    inference(
        image_path=args.image,
        review_text=args.review,
        price=args.price,
        rating=args.rating,
        return_rate=args.return_rate
    )
