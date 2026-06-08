"""
REST API for the Multimodal Retail Insight Model.
Allows frontend dashboards to submit product data and get AI-generated insights.
"""
import os
import sys
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize predictor once at startup
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        free_mode = os.environ.get("FREE_DEPLOYMENT", "false").lower() in ("1", "true", "yes")
        if free_mode:
            from src.free_inference import FreeRetailInsightPredictor
            predictor = FreeRetailInsightPredictor()
        else:
            from src.inference import RetailInsightPredictor
            predictor = RetailInsightPredictor()
    return predictor


@app.route('/api/analyze', methods=['POST'])
def analyze_product():
    """
    Analyze a product using the multimodal AI pipeline.
    
    Expects multipart form-data:
      - image: product image file (required)
      - review_text: customer review text
      - price: product price
      - rating: product rating (1-5)
      - return_rate: return rate (0-1)
    """
    try:
        # 1. Handle image upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save to temporary file (Windows-compatible)
        temp_dir = tempfile.mkdtemp()
        temp_img_path = os.path.join(temp_dir, file.filename)
        file.save(temp_img_path)
        
        # 2. Extract form data
        review_text = request.form.get('review_text', '')
        price = float(request.form.get('price', 50.0))
        rating = float(request.form.get('rating', 3.0))
        return_rate = float(request.form.get('return_rate', 0.0))
        
        # 3. Run inference
        pred = get_predictor()
        result = pred.predict(
            image_path=temp_img_path,
            review_text=review_text,
            price=price,
            rating=rating,
            return_rate=return_rate
        )
        
        # 4. Cleanup
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'status': 'success',
            'insight': result['insight'],
            'fusion_features_shape': result['fusion_features_shape'],
            'input_summary': result['input_summary']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


if __name__ == '__main__':
    print("Starting Retail Multimodal AI Backend...")
    print("Endpoints:")
    print("  POST /api/analyze  — Submit product for insight generation")
    print("  GET  /api/health   — Health check")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
