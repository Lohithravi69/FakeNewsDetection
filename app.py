# app.py

from flask import Flask, render_template, request, jsonify
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Request: {request.method} {request.path}")
    if request.method == 'POST':
        news_text = request.form.get('news', '').strip()
        if not news_text or len(news_text) < 10:
            error = "Please enter a valid news article (at least 10 characters)."
            return render_template("index.html", error=error)
        try:
            transformed_text = vectorizer.transform([news_text])
            prediction = model.predict(transformed_text)
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = max(probabilities) * 100  # Percentage
            result = "🟢 REAL NEWS" if prediction[0] == 'REAL' else "🔴 FAKE NEWS"
            logger.info(f"Prediction: {result}, Confidence: {confidence:.2f}%")
            return render_template("result.html", prediction=result, text=news_text, confidence=f"{confidence:.2f}%")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            error = "An error occurred while processing the article. Please try again."
            return render_template("index.html", error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    logger.info(f"API Request: {request.method} {request.path}")
    data = request.get_json()
    if not data or 'news' not in data:
        return jsonify({"error": "Missing 'news' field in JSON"}), 400
    news_text = data['news'].strip()
    if not news_text or len(news_text) < 10:
        return jsonify({"error": "News article must be at least 10 characters"}), 400
    try:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = max(probabilities) * 100
        result = "REAL" if prediction[0] == 'REAL' else "FAKE"
        return jsonify({"prediction": result, "confidence": f"{confidence:.2f}%"})
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run()
