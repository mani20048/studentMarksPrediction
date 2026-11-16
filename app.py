from flask import Flask,request,jsonify,send_from_directory
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "marksPrediction.pkl")

TEMPLATES = BASE_DIR / "templates"
app = Flask(__name__)
@app.route("/")
def home():
    return send_from_directory(TEMPLATES,"index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    hours = float(data["study_hours"])
    pred = model.predict(np.array([[hours]]))[0]
    return jsonify({
        "study_hours": hours,
        "predicted_marks": round(float(pred),2)
    })

if __name__ == "__main__":
    app.run(debug=True)
