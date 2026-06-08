"""
Lightweight free deployment inference.
Avoids importing torch, transformers, and large model dependencies.
This is used only for free Render deployment to keep memory low.
"""

import re

POSITIVE_KEYWORDS = [
    "good", "great", "excellent", "amazing", "perfect", "love", "loved", "best", "recommended", "happy", "satisfied", "nice"
]
NEGATIVE_KEYWORDS = [
    "bad", "poor", "terrible", "awful", "worst", "hate", "hated", "disappointed", "broken", "low quality", "refund", "return"
]

STRUCTURED_FEATURES = ["price_scaled", "rating_scaled", "return_rate_scaled"]


def sentiment_score(review_text: str) -> int:
    text = review_text.lower()
    score = 0
    for term in POSITIVE_KEYWORDS:
        if term in text:
            score += 1
    for term in NEGATIVE_KEYWORDS:
        if term in text:
            score -= 1
    return score


def format_insight(review_text: str, price: float, rating: float, return_rate: float) -> str:
    score = sentiment_score(review_text)
    if return_rate > 0.4:
        risk = "high return risk"
    elif return_rate > 0.2:
        risk = "moderate return risk"
    else:
        risk = "low return risk"

    if rating >= 4.0 and score >= 1:
        sentiment = "positive customer sentiment"
    elif rating >= 3.0 and score >= 0:
        sentiment = "mixed-to-positive sentiment"
    elif rating < 3.0 or score < 0:
        sentiment = "negative sentiment and quality concerns"
    else:
        sentiment = "neutral customer sentiment"

    if price > 150:
        price_comment = "a premium price point"
    elif price < 30:
        price_comment = "a budget-friendly price"
    else:
        price_comment = "a mid-range price"

    return (
        f"The product shows {sentiment} with {price_comment}, "
        f"and {risk}. The review text suggests practical feedback for merchandising and returns management."
    )


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


class FreeRetailInsightPredictor:
    """Simple rule-based predictor for free deployment."""

    def __init__(self):
        self.scaler_params = {
            "price_min": 0,
            "price_max": 200,
            "rating_min": 1.0,
            "rating_max": 5.0,
            "return_rate_min": 0.0,
            "return_rate_max": 1.0,
        }

    def predict(self, image_path, review_text, price, rating, return_rate, max_new_tokens=None, temperature=None):
        insight = format_insight(review_text, price, rating, return_rate)

        price_scaled = normalize_value(price, self.scaler_params["price_min"], self.scaler_params["price_max"])
        rating_scaled = normalize_value(rating, self.scaler_params["rating_min"], self.scaler_params["rating_max"])
        rr = return_rate / 100.0 if return_rate > 1 else return_rate
        return_rate_scaled = normalize_value(rr, self.scaler_params["return_rate_min"], self.scaler_params["return_rate_max"])

        return {
            "insight": insight,
            "fusion_features_shape": [1, len(STRUCTURED_FEATURES)],
            "input_summary": {
                "price": price,
                "rating": rating,
                "return_rate": return_rate,
                "review_length": len(review_text),
                "image": image_path,
                "price_scaled": price_scaled,
                "rating_scaled": rating_scaled,
                "return_rate_scaled": return_rate_scaled,
            },
        }
