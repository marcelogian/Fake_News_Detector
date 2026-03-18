"""
Fake News Detector — Flask REST API
====================================
Endpoints:
  POST /predict          → classify a news article
  POST /predict/batch    → classify multiple articles
  GET  /health           → service health
  GET  /model/info       → loaded model metadata
"""

import os
import time
import logging
from functools import lru_cache
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Must be imported at module level so pickle can resolve custom classes
# (TextPreprocessor, etc.) when loading the saved .pkl file.
import ml.pipeline  # noqa: F401

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow requests from frontend demo

# ── Load model (lazy, cached) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_predictor():
    """Lazy-load the trained model (trains on first call if pkl missing)."""
    # These imports must happen before pickle.load so pickle can resolve
    # TextPreprocessor and any other custom classes stored in the .pkl file.
    import ml.pipeline  # noqa: F401 — registers the module in sys.modules
    from ml.pipeline import FakeNewsPredictor, MODELS_DIR, train

    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        logger.info("No saved model found — training now …")
        train()

    logger.info(f"Loading model from {model_path}")
    return FakeNewsPredictor(str(model_path))


# ── Helper ─────────────────────────────────────────────────────────────────────

def error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        predictor = get_predictor()
        clf_name = type(predictor.pipeline.named_steps["clf"]).__name__
        vec_name = type(predictor.pipeline.named_steps["vec"]).__name__
        return jsonify({
            "classifier": clf_name,
            "vectorizer": vec_name,
            "labels": list(predictor.le.classes_),
        })
    except Exception as e:
        return error_response(str(e), 500)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body:
      { "text": "news article content …" }

    Response:
      {
        "label": "FAKE" | "REAL",
        "is_fake": true | false,
        "confidence": 87.4,
        "processing_time_ms": 12
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return error_response("Request body must be JSON")

    text = data.get("text", "").strip()
    if not text:
        return error_response("'text' field is required and must not be empty")

    if len(text) > 50_000:
        return error_response("'text' exceeds 50,000 character limit")

    try:
        predictor = get_predictor()
        t0 = time.perf_counter()
        result = predictor.predict(text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["processing_time_ms"] = elapsed_ms

        logger.info(f"Predicted: {result['label']} ({result['confidence']}%) "
                    f"in {elapsed_ms}ms")
        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction error")
        return error_response(f"Prediction failed: {str(e)}", 500)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Request body:
      { "articles": ["text1", "text2", …] }   (max 50)

    Response:
      { "results": [ { …prediction… }, … ] }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return error_response("Request body must be JSON")

    articles = data.get("articles", [])
    if not isinstance(articles, list) or len(articles) == 0:
        return error_response("'articles' must be a non-empty list")

    if len(articles) > 50:
        return error_response("Batch limit is 50 articles per request")

    try:
        predictor = get_predictor()
        results = []
        for idx, text in enumerate(articles):
            if not isinstance(text, str) or not text.strip():
                results.append({"index": idx, "error": "Empty or invalid text"})
                continue
            r = predictor.predict(text)
            r["index"] = idx
            results.append(r)

        return jsonify({"results": results, "count": len(results)})

    except Exception as e:
        logger.exception("Batch prediction error")
        return error_response(f"Batch prediction failed: {str(e)}", 500)


# ── CLI entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Fake News Detector API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)