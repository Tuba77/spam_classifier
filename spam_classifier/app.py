"""
Flask API serving the classifier at POST /predict.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import load_model, predict_message

app = Flask(__name__)
CORS(app)

model, tfidf = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": 'Missing "text" field'}), 400

    text = data["text"]
    pred, proba = predict_message(text, tfidf, model)
    proba_rounded = {k: round(float(v), 4) for k, v in proba.items()}

    return jsonify({
        "text": text,
        "prediction": pred,
        "probabilities": proba_rounded,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
