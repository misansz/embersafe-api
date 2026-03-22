
import os
import joblib
import numpy as np
import pandas as pd
import requests
import datetime
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("embersafe_model_v3_compressed.pkl")
features = joblib.load("embersafe_features_v3_compressed.pkl")

def get_live_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.14,
        "longitude": -111.67,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_min,wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum,vapor_pressure_deficit_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Phoenix",
        "forecast_days": 7
    }
    r = requests.get(url, params=params)
    return r.json()

def build_features(data):
    d = data["daily"]
    temp_max          = d["temperature_2m_max"][0]
    temp_min          = d["temperature_2m_min"][0]
    humidity_min      = d["relative_humidity_2m_min"][0]
    wind_speed        = d["wind_speed_10m_max"][0]
    wind_gust         = d["wind_gusts_10m_max"][0]
    precip            = d["precipitation_sum"][0]
    vpd               = d["vapor_pressure_deficit_max"][0]
    precip_7day       = sum(d["precipitation_sum"][:7])
    precip_30day      = precip_7day * 4
    vpd_7day_avg      = sum(d["vapor_pressure_deficit_max"][:7]) / 7
    wind_7day_max     = max(d["wind_gusts_10m_max"][:7])
    temp_max_3day     = sum(d["temperature_2m_max"][:3]) / 3
    humidity_min_3day = sum(d["relative_humidity_2m_min"][:3]) / 3
    vpd_3day          = sum(d["vapor_pressure_deficit_max"][:3]) / 3
    wind_gust_3day    = max(d["wind_gusts_10m_max"][:3])
    kbdi              = min(800, max(0, (temp_max - 50) * 8 - precip_7day * 100))
    days_since_rain   = 0 if precip > 0.01 else 14
    month             = datetime.date.today().month

    return {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity_min": humidity_min,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "precip": precip,
        "vpd": vpd,
        "precip_7day": precip_7day,
        "precip_30day": precip_30day,
        "vpd_7day_avg": vpd_7day_avg,
        "wind_7day_max": wind_7day_max,
        "days_since_rain": days_since_rain,
        "month": month,
        "kbdi": kbdi,
        "temp_max_3day": temp_max_3day,
        "humidity_min_3day": humidity_min_3day,
        "vpd_3day": vpd_3day,
        "wind_gust_3day": wind_gust_3day,
    }

@app.route("/predict", methods=["GET"])
def predict():
    try:
        data = get_live_weather()
        feat = build_features(data)
        df   = pd.DataFrame([feat])[features]
        prob = model.predict_proba(df)[0][1]
        score = round(prob * 100, 1)

        c = data["current"]

        if score >= 75:   label = "EXTREME"
        elif score >= 55: label = "VERY HIGH"
        elif score >= 35: label = "HIGH"
        elif score >= 15: label = "MODERATE"
        else:             label = "LOW"

        return jsonify({
            "danger_score": score,
            "risk_label": label,
            "temperature": c["temperature_2m"],
            "humidity": c["relative_humidity_2m"],
            "wind_speed": c["wind_speed_10m"],
            "wind_gust": c["wind_gusts_10m"],
            "precipitation": c["precipitation"],
            "vpd": feat["vpd"],
            "kbdi": feat["kbdi"],
            "days_since_rain": feat["days_since_rain"],
            "model_version": "v3",
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "embersafe_v3"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
