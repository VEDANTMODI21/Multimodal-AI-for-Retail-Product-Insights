"""
Evaluation script for the Multimodal Retail Insight Model.
Generates insights on the test set and computes quality metrics.
"""
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DEVICE, TEST_CSV, CHECKPOINT_DIR, IMAGE_DIR,
    BERT_MAX_LENGTH, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
)
from src.model import MultimodalRetailInsightModel


def compute_text_metrics(generated, reference):
    """Compute overlap metrics between generated and reference text."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())

    if not gen_tokens or not ref_tokens:
        return {"precision": 0, "recall": 0, "f1": 0, "length": len(generated.split())}

    overlap = gen_tokens & ref_tokens
    precision = len(overlap) / len(gen_tokens) if gen_tokens else 0
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "length": len(generated.split())
    }


def evaluate_quality(generated_text):
    """Score the quality of a generated insight (0-1)."""
    scores = {}

    # Fluency: Does it form coherent sentences?
    sentences = [s.strip() for s in generated_text.split(".") if len(s.strip()) > 10]
    scores["fluency"] = min(1.0, len(sentences) / 2.0)

    # Specificity: Does it contain specific data points?
    specifics = ["rating", "star", "$", "price", "sentiment", "review", "quality",
                 "customer", "return", "risk", "premium", "budget", "mid-range",
                 "positive", "negative", "satisfaction", "performance"]
    spec_count = sum(1 for s in specifics if s in generated_text.lower())
    scores["specificity"] = min(1.0, spec_count / 5.0)

    # Coherence: Length and structure
    word_count = len(generated_text.split())
    scores["coherence"] = min(1.0, word_count / 30.0) if word_count > 5 else 0.0

    # Relevance: Contains business insight language
    biz_terms = ["analysis", "market", "segment", "competitive", "churn",
                 "conversion", "demand", "positioning", "value", "alignment",
                 "insight", "trend", "recommend", "forecast", "strategy"]
    biz_count = sum(1 for t in biz_terms if t in generated_text.lower())
    scores["relevance"] = min(1.0, biz_count / 3.0)

    scores["overall"] = round(
        0.3 * scores["fluency"] + 0.25 * scores["specificity"] +
        0.25 * scores["coherence"] + 0.2 * scores["relevance"], 4
    )
    return scores


def main():
    print("=" * 70)
    print("MULTIMODAL RETAIL INSIGHT MODEL — EVALUATION")
    print("=" * 70)

    # Load model
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: No checkpoint found at {model_path}")
        return

    print(f"\nLoading model from {model_path}...")
    model = MultimodalRetailInsightModel()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss: {checkpoint.get('val_loss', '?'):.4f})")

    # Load test data
    if not os.path.exists(TEST_CSV):
        print(f"ERROR: No test data at {TEST_CSV}")
        return

    test_df = pd.read_csv(TEST_CSV)
    print(f"  Test samples: {len(test_df)}")

    # Setup image transforms
    from torchvision import transforms
    from PIL import Image

    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Setup tokenizer
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Evaluate
    results = []
    n_samples = min(50, len(test_df))  # Evaluate first 50 for speed
    print(f"\nGenerating insights for {n_samples} test products...\n")

    for i in range(n_samples):
        row = test_df.iloc[i]

        # Load image
        img_path = os.path.join(IMAGE_DIR, str(row["image_path"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        pixel_values = img_transform(img).unsqueeze(0).to(DEVICE)

        # Tokenize review
        review = str(row.get("review_text", ""))[:512]
        encoded = bert_tokenizer(
            review, max_length=BERT_MAX_LENGTH,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # Structured features
        structured = torch.tensor([[
            float(row.get("price_scaled", 0.5)),
            float(row.get("rating_scaled", 0.5)),
            float(row.get("return_rate_scaled", 0.5))
        ]], dtype=torch.float32).to(DEVICE)

        # Generate
        with torch.no_grad():
            generated = model.generate_insight(
                pixel_values, input_ids, attention_mask, structured
            )

        # Compute metrics
        reference = str(row.get("target_insight", ""))
        metrics = compute_text_metrics(generated, reference)
        quality = evaluate_quality(generated)

        results.append({
            "idx": i,
            "title": str(row.get("title", ""))[:60],
            "category": row.get("category", "unknown"),
            "rating": row.get("rating", 0),
            "generated": generated,
            "reference": reference[:150],
            **metrics,
            **{f"q_{k}": v for k, v in quality.items()}
        })

        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{n_samples}] {row.get('title', '')[:50]}")
            print(f"    Rating: {row.get('rating', '?'):.1f} | Category: {row.get('category', '?')}")
            print(f"    Generated: {generated[:120]}...")
            print(f"    Quality:   F={quality['fluency']:.2f} S={quality['specificity']:.2f} "
                  f"C={quality['coherence']:.2f} R={quality['relevance']:.2f} → {quality['overall']:.2f}")
            print()

    # Summary
    results_df = pd.DataFrame(results)
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Samples evaluated: {len(results_df)}")
    print(f"  Avg word overlap F1: {results_df['f1'].mean():.4f}")
    print(f"  Avg generated length: {results_df['length'].mean():.1f} words")
    print(f"\n  Quality Scores (0-1):")
    print(f"    Fluency:      {results_df['q_fluency'].mean():.4f}")
    print(f"    Specificity:  {results_df['q_specificity'].mean():.4f}")
    print(f"    Coherence:    {results_df['q_coherence'].mean():.4f}")
    print(f"    Relevance:    {results_df['q_relevance'].mean():.4f}")
    print(f"    ─────────────────────────────")
    print(f"    OVERALL:      {results_df['q_overall'].mean():.4f}")

    # By category
    if "category" in results_df.columns:
        print(f"\n  By Category:")
        for cat in results_df["category"].unique():
            cat_df = results_df[results_df["category"] == cat]
            print(f"    {cat}: Overall={cat_df['q_overall'].mean():.4f} "
                  f"(n={len(cat_df)}) F1={cat_df['f1'].mean():.4f}")

    # Save results
    eval_path = os.path.join(CHECKPOINT_DIR, "eval_results.csv")
    results_df.to_csv(eval_path, index=False)
    print(f"\n  Results saved to: {eval_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
