from flask import Flask, request, jsonify
import os
import json
from dotenv import load_dotenv
from collections import OrderedDict
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from backend.ml.prediction.weather_prediction import WeatherPredictionService

app = Flask(__name__)

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Khởi tạo dịch vụ dự báo
model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'weather_models.joblib')
conditions_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'conditions.json')
prediction_service = WeatherPredictionService(model_path)

@app.route("/predict", methods=["GET"])
def predict_weather():
    api_key = request.args.get("api_key", WEATHER_API_KEY)
    location = request.args.get("location")
    hours = int(request.args.get("hours", 24)) # Mặc định dự báo 24h

    if not api_key or not location:
        return jsonify({"error": "Thiếu location"}), 400
    try:
        prediction = prediction_service.predict(api_key, location, prediction_hours=hours)

        ordered_prediction = OrderedDict([
            ("location", prediction["location"]),
            ("predictions", prediction["predictions"])
        ])

        return app.response_class(
            response=json.dumps(ordered_prediction, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
