import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

file_path = "./crop_data.xlsx"
df = pd.read_excel(file_path)

print("Dataset Loaded. First 5 rows:")
print(df.head())

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
print("Label Encoding Completed. Unique classes:", label_encoder.classes_)


X = df[['temperature', 'rainfall', 'wind_speed']].values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Feature Scaling Applied.")


X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.long)


batch_size = 32 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("DataLoader Created.")


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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  

print("Model Initialized.")


num_epochs = 30  
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs} - Training Start")

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

print("Training Completed!")


save_path = "./saved_models/model.pth"
torch.save(model.state_dict(), save_path)

print(f"Model Saved Successfully at: {save_path}")
