from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

app = Flask(__name__)
CORS(app)


file_path = "/Users/kaleddahleh/Desktop/Croptimizer/backend/crop_data.xlsx"
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
model.load_state_dict(torch.load("/Users/kaleddahleh/Desktop/Croptimizer/backend/saved_models/model.pth"))
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
        top3_indices = torch.topk(output, 3, dim=1).indices[0].tolist()
        top3_crops = label_encoder.inverse_transform(top3_indices)
    return top3_crops

@app.route("/")
def home():
    return jsonify({"message": "Crop Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        required_fields = ["temperature", "rainfall", "wind_speed"]
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing one or more required fields: temperature, rainfall, wind_speed"
            }), 400

        temperature = float(data["temperature"])
        rainfall = float(data["rainfall"])
        wind_speed = float(data["wind_speed"])

        top3_crops = get_top3_crops(temperature, rainfall, wind_speed)
        return jsonify({"top3_crops": list(top3_crops)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
