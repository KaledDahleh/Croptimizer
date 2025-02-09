from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from prediction import predict_yield  
import dotenv
import os
import requests
from fetchWeather import getAvgs  

dotenv.load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

app = Flask(__name__)
CORS(app)


file_path = "crop_data.xlsx"
df = pd.read_excel(file_path)


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

scaler = StandardScaler()
X = df[['temperature', 'rainfall', 'wind_speed']].values
scaler.fit(X)

class CropClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

num_classes = len(label_encoder.classes_)
model = CropClassifier(input_size=3, num_classes=num_classes)
model.load_state_dict(torch.load("./saved_models/model.pth"))
model.eval()

def get_top3_crops(temperature, rainfall, wind_speed):
    """
    Process the input values, run them through the model,
    and return the top 3 predicted crops.
    """
    data = np.array([[temperature, rainfall, wind_speed]])
    data = scaler.transform(data)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(data_tensor)
        top3_indices = torch.topk(output, 6, dim=1).indices[0].tolist()
        top3_crops = label_encoder.inverse_transform(top3_indices)
    return top3_crops

@app.route("/")
def home():
    return jsonify({"message": "Crop Prediction API is Running!"})

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")
    soil_type = data.get("soilType")
    plant_type = data.get("plantType")

    result = {
        "message": "Data received successfully",
        "lat": lat,
        "lng": lng,
        "soilType": soil_type,
        "plantType": plant_type
    }

    return jsonify(result), 200

@app.route("/predict_crops", methods=["POST"])
def predict_crops():
    try:
        data = request.json
        required_fields = ["latitude", "longitude"]


        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing one or more required fields: latitude, longitude"
            }), 400
        
        all_data = getAvgs(data['latitude'], data['longitude'])
        temperature = all_data['avg_temp']
        rainfall = all_data['avg_precip']
        wind_speed = all_data['avg_wind_speed']

        top3_crops = get_top3_crops(temperature, rainfall, wind_speed)
        return jsonify({"top3_crops": list(top3_crops)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/predict_yield", methods=["POST"])
def predict_yield_route():
    try:
        data = request.json 
        required_fields = ["crop_type", "latitude", "longitude"]

        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing one or more required fields: crop_type, latitude, longitude"
            }), 400
        
        lat = data['latitude']
        lng = data['longitude']
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url)
        state = None
        if response.status_code == 200:
            geo_data = response.json()
            if geo_data.get('results'):
                for component in geo_data['results'][0]['address_components']:
                    if 'administrative_area_level_1' in component['types']:
                        state = component['long_name']
                        break

        if state is None:
            return jsonify({
                "error": "Could not determine state from latitude and longitude"
            }), 400

        data["state"] = state


        predicted_yield = predict_yield(data)
        

        return jsonify({
            "crop_type": data["crop_type"],
            "predicted_yield": predicted_yield
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
