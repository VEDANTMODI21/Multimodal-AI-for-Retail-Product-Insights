# Deployment Guide

This repository is prepared for free deployment on Render using the included `render.yaml` manifest.

## Recommended Free Deployment: Render

1. Sign in or create a free account at https://render.com
2. From the Render dashboard, choose **New** → **Web Service**
3. Connect your GitHub repository `VEDANTMODI21/Multimodal-AI-for-Retail-Product-Insights`
4. Select the `main` branch
5. Render should detect the `render.yaml` manifest and configure the service automatically.

If Render does not detect the manifest automatically, use these settings:

- **Build Command:** `python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt`
- **Start Command:** `gunicorn src.api:app --bind 0.0.0.0:$PORT`
- **Plan:** `free`

## Service endpoints

- `POST /api/analyze` — submit multipart form-data with `image`, `review_text`, `price`, `rating`, and `return_rate`
- `GET /api/health` — health check

## Notes

- The app uses `gunicorn` and `PORT` environment variables for cloud hosting.
- The model will still run if no checkpoint is present, but results will be untrained.
- Use `checkpoints/best_model.pth` if you want production-quality predictions.
