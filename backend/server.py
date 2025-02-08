from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow React to communicate with Flask

# Load trained model, feature list, and label encoder
model = joblib.load("crop_yield_model.pkl")
selected_features = joblib.load("selected_features.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Load USDA soil data (State-wise N, P, K values)
soil_data = pd.read_csv("USDA_Soil_Data.csv")
soil_data["State"] = soil_data["State"].str.lower()  # Normalize state names

@app.route("/")
def home():
    return jsonify({"message": "Crop Yield Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict_yield():
    try:
        data = request.json  # Get JSON input from frontend

        # Ensure required inputs are present
        if "location" not in data or "crop_type" not in data or "soil_type" not in data:
            return jsonify({"error": "Missing 'location', 'crop_type', or 'soil_type'"}), 400

        # Get state-based soil data
        state_name = data["location"].strip().lower()
        state_soil = soil_data[soil_data["State"] == state_name]

        if state_soil.empty:
            return jsonify({"error": f"Unknown state: '{state_name}'. Please enter a valid U.S. state."}), 400

        # Extract soil properties from USDA data
        N_value = state_soil["N"].values[0]
        P_value = state_soil["P"].values[0]
        K_value = state_soil["K"].values[0]

        # Convert crop type to lowercase and check if it exists
        crop_name = data["crop_type"].strip().lower()
        if crop_name not in label_encoder.classes_:
            return jsonify({"error": f"Unknown crop type: '{crop_name}'. Please use one from the dataset: {list(label_encoder.classes_)}"}), 400

        # Encode crop type
        crop_encoded = label_encoder.transform([crop_name])[0]

        # Set input data with soil nutrients from USDA
        input_data = {
            "label": crop_encoded,
            "N": N_value,
            "P": P_value,
            "K": K_value,
            "temperature": data.get("temperature", 25),  # Default to 25°C
            "humidity": data.get("humidity", 60),  # Default to 60%
            "rainfall": data.get("rainfall", 100),  # Default to 100mm
            "sunlight_exposure": data.get("sunlight_exposure", 8),  # Default to 8 hrs/day
            "ph": data.get("ph", 6.5),  # Default to neutral pH
            "soil_moisture": data.get("soil_moisture", 30),  # Default to 30%
        }

        # Ensure all required features exist
        missing_features = [f for f in selected_features if f not in input_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert input to array format for prediction
        input_values = np.array([input_data[f] for f in selected_features]).reshape(1, -1)

        # Predict yield
        predicted_yield = model.predict(input_values)[0]

        # Return only the numeric result
        return jsonify({"predicted_yield": f"{predicted_yield:.2f}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
