<<<<<<< HEAD
# app.py

from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        result = "🟢 REAL NEWS" if prediction[0] == 'REAL' else "🔴 FAKE NEWS"
        return render_template("result.html", prediction=result, text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
=======
# app.py

from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        result = "🟢 REAL NEWS" if prediction[0] == 'REAL' else "🔴 FAKE NEWS"
        return render_template("result.html", prediction=result, text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 20e4694479615c77a937492bd46ae1f76a9e2a53
