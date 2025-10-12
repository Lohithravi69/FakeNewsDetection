from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "✅ Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        news_text = data.get("news", "").strip()
        if not news_text or len(news_text) < 10:
            return jsonify({"error": "News article must be at least 10 characters"}), 400
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = max(probabilities) * 100
        result = "REAL" if prediction[0] == 'REAL' else "FAKE"
        return jsonify({"prediction": result, "confidence": f"{confidence:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
