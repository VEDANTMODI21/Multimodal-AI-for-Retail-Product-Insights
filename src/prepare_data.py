"""
Download and preprocess the Amazon Review Dataset (2023).
Categories: Clothing (Winter Range) + Electronics

Downloads from McAuley Lab (HuggingFace), filters for relevant products,
creates train/val/test splits, and auto-launches training when done.
"""
import os
import sys
import json
import random
import hashlib
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV,
    TRAIN_RATIO, VAL_RATIO
)

# ─── Category Keywords ───────────────────────────────────────────────────────
WINTER_KEYWORDS = [
    "winter", "jacket", "coat", "parka", "sweater", "hoodie", "fleece",
    "thermal", "wool", "beanie", "gloves", "scarf", "boots", "puffer",
    "down jacket", "windbreaker", "insulated", "warm", "cold weather",
    "snow", "flannel", "cardigan", "overcoat", "trench", "vest",
    "knit", "cashmere", "sherpa", "heated"
]

ELECTRONICS_KEYWORDS = [
    "headphones", "earbuds", "speaker", "bluetooth", "wireless",
    "charger", "cable", "battery", "power bank", "adapter",
    "smartwatch", "fitness tracker", "mouse", "keyboard", "webcam",
    "usb", "hdmi", "monitor", "tablet", "case", "stand", "hub",
    "microphone", "led", "light", "lamp", "portable", "gaming",
    "controller", "dock", "ring light", "tripod", "screen protector"
]


def generate_target_insight(row):
    """Generate an actionable 3-sentence business insight: WHY + WHAT + HOW TO IMPROVE."""
    rating = row.get("rating", 3.0)
    price = row.get("price", 50.0)
    category = row.get("category", "general")
    review_snippet = str(row.get("review_text", ""))[:800]

    negative_words = [
        "cheap", "broke", "poor", "bad", "terrible", "flimsy",
        "thin", "small", "tight", "uncomfortable", "disappointing",
        "returned", "defective", "ripped", "torn", "falling apart",
        "waste", "low quality", "not worth", "stopped working",
        "broken", "malfunction", "dead", "garbage", "refund",
        "awful", "horrible", "useless", "overpriced"
    ]
    positive_words = [
        "warm", "comfortable", "soft", "great", "love", "perfect",
        "quality", "excellent", "cozy", "durable", "thick", "nice",
        "amazing", "beautiful", "stylish", "fits", "recommend",
        "best", "worth", "fantastic", "reliable", "fast", "clear",
        "crisp", "powerful", "premium", "solid", "sturdy", "lightweight",
        "impressive", "superb", "outstanding", "value"
    ]

    review_lower = review_snippet.lower()
    neg_found = [w for w in negative_words if w in review_lower]
    pos_found = [w for w in positive_words if w in review_lower]
    neg_count = len(neg_found)
    pos_count = len(pos_found)

    top_pos = ", ".join(pos_found[:3]) if pos_found else "general satisfaction"
    top_neg = ", ".join(neg_found[:3]) if neg_found else "minor concerns"

    price_seg = "premium" if price > 100 else ("mid-range" if price > 40 else "budget")
    cat_label = "winter apparel" if category == "clothing" else "electronics accessory"
    ret_risk = "low" if rating >= 4.0 else ("moderate" if rating >= 3.0 else "high")

    # ── HIGH SATISFACTION ──
    if rating >= 4.0 and pos_count > neg_count:
        templates = [
            (f"This {price_seg} {cat_label} earns a strong {rating:.1f}-star rating primarily because "
             f"customers consistently experience {top_pos}, which directly aligns with the product's "
             f"marketed value proposition at ${price:.0f}. The positive sentiment is driven by accurate "
             f"product imagery that sets realistic expectations, resulting in low return rates and high "
             f"repeat purchase intent. To further capitalize on this momentum, the brand should leverage "
             f"these verified positive reviews in targeted ad campaigns and consider introducing a premium "
             f"variant at a 15-20% higher price point to capture additional margin."),

            (f"The {rating:.1f}-star success of this ${price:.0f} {cat_label} stems from delivering on "
             f"core customer needs around {top_pos}, creating a virtuous cycle of positive reviews and "
             f"organic discovery. Cross-modal analysis confirms the product photos faithfully represent "
             f"the physical item, which is the primary reason for the {ret_risk} return risk and strong "
             f"conversion rates. To sustain growth, the seller should expand the product line with "
             f"complementary items and implement a review solicitation strategy to maintain the high "
             f"rating as sales volume increases."),

            (f"Customer satisfaction for this {cat_label} at {rating:.1f} stars is rooted in verifiable "
             f"quality around {top_pos}, which {pos_count} reviewers independently confirmed. The "
             f"price-to-quality ratio at ${price:.0f} exceeds customer expectations, explaining the "
             f"strong organic recommendation rate visible in review language. To maximize long-term "
             f"value, the brand should invest in building a loyalty program around this hero product "
             f"and use it as an anchor to cross-sell related items in the catalog."),

            (f"Analysis reveals this {cat_label} achieves its {rating:.1f} rating because the physical "
             f"product surpasses the expectations set by imagery, with customers highlighting {top_pos} "
             f"as pleasant surprises at the ${price:.0f} price point. This under-promise-over-deliver "
             f"dynamic generates strong word-of-mouth, the most cost-effective acquisition channel. "
             f"The recommended improvement is to update product photography to better showcase actual "
             f"quality, which would increase click-through rates without risking the positive surprise."),

            (f"The {rating:.1f}-star performance of this {price_seg} {cat_label} is directly attributable "
             f"to solving a specific customer pain point around {top_pos} better than competing "
             f"alternatives in the same price range. Structured data shows the ${price:.0f} pricing "
             f"hits the sweet spot where perceived value exceeds cost. To improve further, the seller "
             f"should address the {neg_count} minor complaints found in reviews by iterating on product "
             f"design, then raise the price by 10% to reflect the enhanced value."),

            (f"Root cause analysis shows this {price_seg} {cat_label} earned its {rating:.1f} rating "
             f"because it addresses the top three customer priorities: {top_pos}. At ${price:.0f}, it "
             f"represents clear value against competitors, which is why {pos_count} positive sentiment "
             f"signals outweigh the {neg_count} negative mentions. The next improvement opportunity "
             f"lies in packaging and unboxing experience, which would further differentiate the product "
             f"and support a potential price increase."),

            (f"This ${price:.0f} {cat_label} achieves a {rating:.1f} rating because the manufacturer "
             f"invested in material quality that delivers on {top_pos}, creating a noticeable difference "
             f"from cheaper alternatives. The strong rating creates a compounding advantage: higher "
             f"search visibility leads to more sales, generating more positive reviews. To build on "
             f"this, the seller should introduce seasonal limited editions and invest in responding to "
             f"the few negative reviews to demonstrate active customer care."),

            (f"This {cat_label} succeeds at {rating:.1f} stars because it delivers authentic {top_pos} "
             f"that customers can verify upon receipt, creating trust at the ${price:.0f} price point. "
             f"What customers see in the listing is what they get, which is the single biggest driver "
             f"of the {ret_risk} return rate. For continued improvement, the brand should A/B test new "
             f"product images against conversion metrics, and consider bundling with complementary "
             f"accessories to increase average order value by 20-30%."),
        ]

    # ── MODERATE SATISFACTION ──
    elif rating >= 3.0 and neg_count <= pos_count:
        templates = [
            (f"This {price_seg} {cat_label} sits at a mediocre {rating:.1f} stars because while it "
             f"delivers on basic expectations around {top_pos}, it simultaneously frustrates customers "
             f"with {top_neg}, and these competing experiences cancel each other out. The ${price:.0f} "
             f"price point exacerbates this: customers paying {price_seg} prices expect consistency, "
             f"but recurring quality issues suggest manufacturing shortcuts. To improve, the manufacturer "
             f"should prioritize fixing {top_neg} through quality control investment, then solicit "
             f"updated reviews from satisfied customers to dilute negative sentiment."),

            (f"The {rating:.1f}-star rating reveals a product caught between potential and execution: "
             f"customers genuinely appreciate {top_pos}, proving the core concept works, but {top_neg} "
             f"issues indicate corners were cut during production. At ${price:.0f}, this {cat_label} "
             f"is priced correctly for what it delivers, but the brand is leaving revenue on the table. "
             f"The recommended strategy is to source higher-grade materials for components causing "
             f"{top_neg}, run a small batch test, and measure the impact on return rates before scaling."),

            (f"Mixed reviews at {rating:.1f} stars indicate this {cat_label} has a fundamental design "
             f"tension: the aspects customers praise ({top_pos}) and complain about ({top_neg}) exist "
             f"because of cost-optimization trade-offs at ${price:.0f}. First-time buyers appreciate "
             f"the positives but repeat customers increasingly notice negatives. The clear path forward "
             f"is to invest in the top complaint area ({top_neg}), which would shift the rating to 4.0+ "
             f"and unlock significantly higher conversion rates."),

            (f"This {cat_label} earns {rating:.1f} stars because it partially delivers: {top_pos} are "
             f"real strengths, but {top_neg} indicates optimization for first impressions rather than "
             f"long-term satisfaction. At ${price:.0f} in the {price_seg} tier, customers initially feel "
             f"fair value, but the {ret_risk} return risk grows as usage reveals weaknesses. To break "
             f"out of mediocre positioning, the brand needs to address {top_neg} through product revision "
             f"and update the listing to honestly represent current limitations while highlighting "
             f"genuine strengths."),

            (f"Analysis shows this {price_seg} {cat_label} earned {rating:.1f} stars because it executes "
             f"well on purchase-time features ({top_pos}) but underdelivers on durability factors "
             f"({top_neg}). This buy-now-regret-later pattern at ${price:.0f} creates a toxic review "
             f"trajectory where older reviews skew negative. The improvement strategy should involve: "
             f"(1) redesigning components causing {top_neg}, (2) extending the warranty to signal "
             f"confidence, and (3) proactively following up with buyers at the 30-day mark."),

            (f"The {rating:.1f}-star equilibrium for this ${price:.0f} {cat_label} exists because it "
             f"serves two segments differently: value-focused buyers appreciate {top_pos} and rate "
             f"highly, while quality-focused buyers encounter {top_neg} and rate poorly. The product "
             f"photography contributes by slightly overpromising on build quality. To improve, either "
             f"target the value segment by reducing price 15% with honest marketing, or invest in "
             f"eliminating {top_neg} to capture the premium segment."),

            (f"Root cause analysis reveals this {cat_label} at {rating:.1f} stars suffers from "
             f"inconsistent quality: some units deliver {top_pos}, while others exhibit {top_neg}, "
             f"suggesting batch-to-batch manufacturing variation at ${price:.0f}. This quality lottery "
             f"is the worst outcome for brand trust. The critical fix is implementing tighter quality "
             f"control at the factory level, even if it means raising price by 10-15%, because "
             f"consistent 4-star quality is vastly more profitable than inconsistent 3-star quality."),

            (f"This {price_seg} {cat_label} at {rating:.1f} stars represents a missed opportunity: "
             f"positive sentiment around {top_pos} proves the core concept is sound, but {top_neg} "
             f"prevents it from reaching the 4.0+ threshold needed for algorithmic search boosting. "
             f"At ${price:.0f}, the product breaks even on customer value but lacks competitive "
             f"differentiation. The recommended action is a focused product iteration on the top 2 "
             f"complaints ({top_neg}), followed by a relaunch with updated imagery."),
        ]

    # ── LOW SATISFACTION ──
    else:
        templates = [
            (f"This {price_seg} {cat_label} fails at {rating:.1f} stars primarily because the product "
             f"does not deliver on its core promise: customers report {top_neg} as fundamental flaws "
             f"rather than minor issues, indicating a design or manufacturing failure at ${price:.0f}. "
             f"The gap between product imagery and actual quality is the root cause of the {ret_risk} "
             f"return risk. To recover, the brand must either reformulate with better materials and "
             f"relaunch, or drastically reduce the price to match actual quality, then rebuild trust "
             f"through honest imagery and a satisfaction guarantee."),

            (f"The {rating:.1f}-star collapse of this ${price:.0f} {cat_label} happened because the "
             f"product failed on the exact attributes customers purchased it for, with {neg_count} "
             f"complaints citing {top_neg}. This is not a perception problem but a genuine product "
             f"deficiency. The path forward requires a complete product redesign focusing on eliminating "
             f"{top_neg}, a price reset reflecting actual quality, and a transparent marketing approach "
             f"that sets accurate expectations."),

            (f"Analysis shows this {cat_label} earned {rating:.1f} stars because of a fundamental "
             f"mismatch between manufacturing cost-cutting and customer expectations at ${price:.0f}: "
             f"the {top_neg} complaints reveal the manufacturer prioritized margin over quality. The "
             f"marketing imagery contradicts the physical product on arrival. To fix this: (1) audit "
             f"the supply chain for components causing {top_neg}, (2) invest in quality improvements "
             f"even at reduced margin, and (3) rewrite the listing to honestly represent the product."),

            (f"This {price_seg} {cat_label} at {rating:.1f} stars is a direct consequence of prioritizing "
             f"aesthetics in marketing over substance in manufacturing: it looks great in photos but "
             f"customers immediately discover {top_neg} upon use. At ${price:.0f}, this creates maximum "
             f"frustration because perceived deception amplifies actual shortcomings. The recovery plan "
             f"involves pulling the current listing, fixing the {neg_count} critical quality issues, then "
             f"relaunching with verified-purchase photos and a no-questions-asked return policy."),

            (f"Review mining reveals this ${price:.0f} {cat_label} fails at {rating:.1f} stars because "
             f"it is positioned as a {price_seg} item but delivers budget-tier quality, with {top_neg} "
             f"being problems that should not exist at this price point. Customers compare it unfavorably "
             f"to cheaper alternatives, indicating zero value proposition. The only viable strategy is "
             f"to either reposition at a lower price with honest marketing, or invest in genuine product "
             f"improvements and re-earn the {price_seg} positioning with verified quality claims."),

            (f"The critical {rating:.1f}-star rating stems from a cascading trust failure: {top_neg} "
             f"issues were not one-off defects but systemic problems affecting most buyers, destroying "
             f"brand credibility at the ${price:.0f} price point. The product imagery now actively "
             f"harms conversion because informed buyers recognize the gap through existing reviews. "
             f"To recover, the manufacturer must completely rethink the product, address every instance "
             f"of {top_neg} through engineering changes, then relaunch under updated branding."),

            (f"This {cat_label} at {rating:.1f} stars is a cautionary example of marketing outpacing "
             f"product development: customers attracted by professional imagery and {price_seg} "
             f"positioning at ${price:.0f} encounter {top_neg} within days of purchase. The {neg_count} "
             f"negative signals are not subjective complaints but verifiable product failures. The brand "
             f"must either discontinue this SKU and redirect resources to a properly engineered "
             f"replacement, or invest in immediate quality overhaul at the material level."),

            (f"Root cause analysis identifies this {price_seg} {cat_label} failure at {rating:.1f} stars "
             f"as a pricing strategy error compounded by quality issues: the ${price:.0f} price sets "
             f"expectations the product cannot meet, and {top_neg} confirms manufacturing shortcuts. "
             f"If priced 40% lower with honest imagery, the rating would likely stabilize at 3.5+. "
             f"The improvement roadmap should start with competitive pricing analysis, followed by "
             f"targeted quality improvements on {top_neg}, and conclude with a full listing refresh."),
        ]

    return random.choice(templates)


def download_image(url, save_path, timeout=10):
    """Download a product image from URL."""
    try:
        if not url or url == "nan" or not isinstance(url, str):
            return False
        response = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Research Project)"
        })
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(save_path, "JPEG", quality=85)
            return True
    except Exception:
        pass
    return False


def create_placeholder_image(save_path, text="Product"):
    """Create a placeholder product image when download fails."""
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    color = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
        short_text = text[:30]
        bbox = draw.textbbox((0, 0), short_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((256 - tw) // 2, (256 - th) // 2), short_text, fill="white", font=font)
    except Exception:
        pass
    img.save(save_path, "JPEG")


def load_category(category_name, review_config, meta_config):
    """Load a single category from Amazon Reviews 2023."""
    from datasets import load_dataset

    print(f"\n  Loading {category_name} reviews...")
    reviews_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        review_config,
        split="full",
        trust_remote_code=True
    )
    print(f"    -> {len(reviews_ds)} reviews loaded")

    print(f"  Loading {category_name} metadata...")
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        meta_config,
        split="full",
        trust_remote_code=True
    )
    print(f"    -> {len(meta_ds)} metadata entries loaded")

    return reviews_ds, meta_ds


def filter_products(reviews_ds, meta_ds, keywords, category_tag, max_products=1500):
    """Filter products by keywords and aggregate reviews."""
    print(f"\n  Filtering {category_tag} products...")

    meta_df = meta_ds.to_pandas()

    if len(reviews_ds) > 1000000:
        print(f"    Large dataset ({len(reviews_ds)}). Sampling 1M...")
        reviews_ds = reviews_ds.shuffle(seed=42).select(range(1000000))

    reviews_df = reviews_ds.to_pandas()
    print(f"    Reviews: {reviews_df.shape}, Metadata: {meta_df.shape}")

    asin_col_meta = "parent_asin" if "parent_asin" in meta_df.columns else "asin"
    asin_col_review = "parent_asin" if "parent_asin" in reviews_df.columns else "asin"

    def matches_keywords(row):
        text = ""
        title = row.get("title")
        if isinstance(title, str):
            text += title.lower() + " "
        desc = row.get("description")
        if isinstance(desc, list):
            text += " ".join([str(d).lower() for d in desc]) + " "
        elif isinstance(desc, str):
            text += desc.lower() + " "
        cats = row.get("categories", row.get("main_category", ""))
        if isinstance(cats, list):
            text += " ".join([str(c).lower() for c in cats]) + " "
        elif isinstance(cats, str):
            text += cats.lower() + " "
        feats = row.get("features")
        if isinstance(feats, list):
            text += " ".join([str(f).lower() for f in feats])
        return any(kw in text for kw in keywords)

    mask = meta_df.apply(matches_keywords, axis=1)
    filtered_meta = meta_df[mask].copy()
    print(f"    Keyword-matched metadata: {len(filtered_meta)}")

    filtered_asins = set(filtered_meta[asin_col_meta].values)
    filtered_reviews = reviews_df[reviews_df[asin_col_review].isin(filtered_asins)].copy()
    print(f"    Matched reviews: {len(filtered_reviews)}")

    text_col = next((c for c in ["text", "reviewText", "review_text", "body"]
                     if c in filtered_reviews.columns), None)
    rating_col = next((c for c in ["rating", "overall", "stars"]
                       if c in filtered_reviews.columns), None)

    if text_col is None:
        text_col = filtered_reviews.columns[2] if len(filtered_reviews.columns) > 2 else filtered_reviews.columns[0]
    if rating_col is None:
        rating_col = "rating"
        filtered_reviews[rating_col] = 3.0

    product_reviews = filtered_reviews.groupby(asin_col_review).agg(
        review_count=(text_col, "count"),
        avg_rating=(rating_col, "mean"),
        review_text=(text_col, lambda x: " [SEP] ".join(x.dropna().astype(str).head(10))),
    ).reset_index().rename(columns={asin_col_review: "parent_asin"})

    product_reviews = product_reviews[product_reviews["review_count"] >= 3]
    print(f"    Products with >= 3 reviews: {len(product_reviews)}")

    keep_cols = ["parent_asin", "title"]
    filtered_meta_r = filtered_meta.rename(columns={asin_col_meta: "parent_asin"})
    for col in ["images", "price", "average_rating"]:
        if col in filtered_meta_r.columns:
            keep_cols.append(col)

    merged = product_reviews.merge(
        filtered_meta_r[keep_cols].drop_duplicates(subset="parent_asin"),
        on="parent_asin", how="inner"
    ).drop_duplicates(subset="parent_asin")

    merged["category"] = category_tag

    if len(merged) > max_products:
        merged = merged.sample(n=max_products, random_state=42)

    print(f"    Final {category_tag}: {len(merged)} products")
    return merged


def process_and_save(merged_df):
    """Process the merged dataframe and save train/val/test splits."""
    print("\n" + "=" * 70)
    print("PROCESSING DATASET")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    records = []
    img_ok, img_fail = 0, 0

    for idx, row in merged_df.iterrows():
        asin = row["parent_asin"]
        category = row.get("category", "general")

        # Price
        price = None
        pv = row.get("price")
        if pv is not None:
            try:
                if isinstance(pv, str):
                    pv = pv.replace("$", "").replace(",", "").strip()
                    if " - " in pv:
                        pv = pv.split(" - ")[0]
                if pv and str(pv) != "None":
                    price = float(pv)
            except (ValueError, TypeError):
                pass
        if price is None or price <= 0 or price > 2000:
            price = random.uniform(15, 200)

        # Rating
        rating = row.get("avg_rating", row.get("average_rating", 3.0))
        try:
            rating = float(rating)
        except Exception:
            rating = 3.0
        rating = max(1.0, min(5.0, rating))

        # Return rate (simulated)
        base_return = max(0, (5.0 - rating) / 5.0)
        return_rate = min(1.0, max(0.0, base_return + random.uniform(-0.1, 0.1)))

        review_text = str(row.get("review_text", ""))[:2000]
        title = str(row.get("title", f"Product {asin}"))

        # Image
        img_filename = f"{asin}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)

        downloaded = False
        if not os.path.exists(img_path):
            images = row.get("images")
            if images and isinstance(images, list):
                for img_item in images[:2]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("large") or img_item.get("hi_res") or img_item.get("thumb")
                    elif isinstance(img_item, str):
                        url = img_item
                    if url and download_image(url, img_path):
                        downloaded = True
                        break
            if not downloaded:
                create_placeholder_image(img_path, title)
                img_fail += 1
            else:
                img_ok += 1
        else:
            img_ok += 1

        insight = generate_target_insight({
            "title": title, "rating": rating, "price": price,
            "review_text": review_text, "category": category
        })

        records.append({
            "asin": asin, "title": title, "image_path": img_filename,
            "review_text": review_text, "price": price, "rating": rating,
            "return_rate": return_rate, "target_insight": insight,
            "review_count": row.get("review_count", 0),
            "category": category
        })

        if len(records) % 200 == 0:
            print(f"  Processed {len(records)}/{len(merged_df)} "
                  f"(imgs: {img_ok} ok, {img_fail} placeholder)")

    df = pd.DataFrame(records)
    print(f"\nTotal: {len(df)} records | Images: {img_ok} downloaded, {img_fail} placeholders")
    print(f"  Clothing: {len(df[df['category'] == 'clothing'])} | "
          f"Electronics: {len(df[df['category'] == 'electronics'])}")

    # Scale structured features
    scaler = MinMaxScaler()
    df[["price_scaled", "rating_scaled", "return_rate_scaled"]] = scaler.fit_transform(
        df[["price", "rating", "return_rate"]]
    )

    with open(os.path.join(DATA_DIR, "scaler_params.json"), "w") as f:
        json.dump({
            "price_min": float(scaler.data_min_[0]),
            "price_max": float(scaler.data_max_[0]),
            "rating_min": float(scaler.data_min_[1]),
            "rating_max": float(scaler.data_max_[1]),
            "return_rate_min": float(scaler.data_min_[2]),
            "return_rate_max": float(scaler.data_max_[2]),
        }, f, indent=2)

    # Split 70/15/15
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    t_end = int(n * TRAIN_RATIO)
    v_end = t_end + int(n * VAL_RATIO)

    df[:t_end].to_csv(TRAIN_CSV, index=False)
    df[t_end:v_end].to_csv(VAL_CSV, index=False)
    df[v_end:].to_csv(TEST_CSV, index=False)

    print(f"\n{'='*70}")
    print(f"DATASET READY")
    print(f"{'='*70}")
    print(f"  Train: {t_end} -> {TRAIN_CSV}")
    print(f"  Val:   {v_end - t_end} -> {VAL_CSV}")
    print(f"  Test:  {n - v_end} -> {TEST_CSV}")
    return True


def main():
    """Full pipeline: download -> filter -> process -> save -> auto-train."""
    print("=" * 70)
    print("MULTIMODAL RETAIL AI - DATA PIPELINE")
    print("  Categories: Winter Clothing + Electronics")
    print("=" * 70)

    all_merged = []

    # Category 1: Winter Clothing
    print("\n" + "=" * 70)
    print("CATEGORY 1: WINTER CLOTHING")
    print("=" * 70)
    clothing_reviews, clothing_meta = load_category(
        "Clothing",
        "raw_review_Clothing_Shoes_and_Jewelry",
        "raw_meta_Clothing_Shoes_and_Jewelry"
    )
    clothing_merged = filter_products(
        clothing_reviews, clothing_meta,
        WINTER_KEYWORDS, "clothing", max_products=1500
    )
    all_merged.append(clothing_merged)

    del clothing_reviews, clothing_meta
    import gc; gc.collect()

    # Category 2: Electronics
    print("\n" + "=" * 70)
    print("CATEGORY 2: ELECTRONICS")
    print("=" * 70)
    electronics_reviews, electronics_meta = load_category(
        "Electronics",
        "raw_review_Electronics",
        "raw_meta_Electronics"
    )
    electronics_merged = filter_products(
        electronics_reviews, electronics_meta,
        ELECTRONICS_KEYWORDS, "electronics", max_products=1500
    )
    all_merged.append(electronics_merged)

    del electronics_reviews, electronics_meta
    import gc; gc.collect()

    # Combine
    combined = pd.concat(all_merged, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"COMBINED DATASET: {len(combined)} products")
    print(f"  Clothing: {len(combined[combined['category'] == 'clothing'])}")
    print(f"  Electronics: {len(combined[combined['category'] == 'electronics'])}")
    print(f"{'='*70}")

    # Process and save
    success = process_and_save(combined)

    # Auto-start training
    if success:
        print("\n" + "=" * 70)
        print("DATA PIPELINE COMPLETE - AUTO-STARTING TRAINING")
        print("=" * 70)
        from src.train import train
        train()


if __name__ == "__main__":
    main()
