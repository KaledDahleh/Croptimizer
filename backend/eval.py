import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

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

temperature = float(input("Enter Temperature: "))
rainfall = float(input("Enter Rainfall: "))
wind_speed = float(input("Enter Wind Speed: "))

data = np.array([[temperature, rainfall, wind_speed]])
data = scaler.transform(data)
data_tensor = torch.tensor(data, dtype=torch.float32)

with torch.no_grad():
    output = model(data_tensor)
    top3_indices = torch.topk(output, 3, dim=1).indices[0].tolist()
    top3_crops = label_encoder.inverse_transform(top3_indices)

print(f"Top 3 Optimal Crops for given conditions: {top3_crops}")
