from flask import Flask, request, jsonify
from flask_cors import CORS
from src.inference import inference
import os

app = Flask(__name__)
# Enable CORS so any frontend dashboard (Next.js, React, Vanilla JS) can talk to this API
CORS(app) 

@app.route('/api/analyze', methods=['POST'])
def analyze_product():
    """
    Expects a mixed multipart form-data payload or JSON (if image is uploaded separately).
    For full multimodal, frontend sends image file + form data (reviews, price, rating).
    """
    try:
        # 1. Handle Visual Data (Image Upload)
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided in the request'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Temporarily save image for inference
        temp_img_path = os.path.join("/tmp", file.filename)
        file.save(temp_img_path)

        # 2. Handle Textual and Structured Data
        review_text = request.form.get('review_text', '')
        price = float(request.form.get('price', 0.5))
        rating = float(request.form.get('rating', 0.5))
        return_rate = float(request.form.get('return_rate', 0.0))

        # 3. Run Multimodal Inference
        result = inference(
            image_path=temp_img_path,
            review_text=review_text,
            price=price,
            rating=rating,
            return_rate=return_rate
        )
        
        # Cleanup temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        return jsonify({
            'status': 'success',
            'insight': result['insight'],
            'fusion_features_processed': True 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the backend API on port 5000
    print("Starting Retail Multimodal AI Backend...")
    app.run(host='0.0.0.0', port=5000, debug=True)
