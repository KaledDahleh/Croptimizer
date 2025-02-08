import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from fetchWeather import getAvgs  # Importing the function to get weather data

file_path = "./crop_data.xlsx"
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
    Returns the top 3 optimal crops based on the input conditions.
    """
    data = np.array([[temperature, rainfall, wind_speed]])
    data = scaler.transform(data)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(data_tensor)
        top3_indices = torch.topk(output, 3, dim=1).indices[0].tolist()
        top3_crops = label_encoder.inverse_transform(top3_indices)
    return top3_crops

if __name__ == "__main__":
    lat = float(input("Enter Latitude: "))
    lon = float(input("Enter Longitude: "))

    # Fetch weather data
    weather_data = getAvgs(lat, lon)

    temperature = weather_data["avg_temp"]
    rainfall = weather_data["avg_precip"]
    wind_speed = weather_data["avg_wind_speed"]

    print(f"Derived Weather Data:\nTemperature: {temperature:.2f}°C\nRainfall: {rainfall:.2f}mm\nWind Speed: {wind_speed:.2f}m/s")

    # Get top 3 crops based on derived weather data
    top3_crops = get_top3_crops(temperature, rainfall, wind_speed)
    print(f"Top 3 Optimal Crops for given conditions: {top3_crops}")
