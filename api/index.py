from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "✅ Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        prediction = model.predict([text])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
