import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fetchWeather import getAvgs
from soil import get_nutrient_levels

# --- Step 1: Load and Preprocess Data ---
file_path = "Crop_recommendationV2.csv"
df = pd.read_csv(file_path)

# Normalize crop labels
df["label"] = df["label"].str.strip().str.lower()

# Identify numeric columns and fill missing values
numeric_cols = df.select_dtypes(include=['number']).columns  
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Scale numerical features
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Encode Crop Type (Label Encoding)
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Save preprocessed dataset
processed_file_path = "Crop_recommendationV2_preprocessed.csv"
df.to_csv(processed_file_path, index=False)
joblib.dump(le, "crop_label_encoder.pkl")  
print("✅ Data preprocessing completed. Preprocessed dataset saved.")

# --- Step 2: Feature Selection ---
df = pd.read_csv(processed_file_path)
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Train RFE model
model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=7)  # Select only the features we want
rfe.fit(X, y)

# Force selection of the required features
selected_features = ["N", "P", "K", "temperature", "humidity", "rainfall", "sunlight_exposure"]

# Save selected features
df_selected = df[selected_features]
df_selected.to_csv("Crop_recommendationV2_selected_features.csv", index=False)
joblib.dump(selected_features, "selected_features.pkl")  
print("✅ Feature selection completed. Final features used:", selected_features)

# --- Step 3: Train Crop Yield Prediction Model ---
X = df_selected  
y = np.random.randint(500, 5000, size=len(df_selected))  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
print("✅ MAE:", mean_absolute_error(y_test, y_pred))
print("✅ RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save Model
joblib.dump(model, "crop_yield_model.pkl")
print("✅ Model training completed. Model saved.")

# --- Step 4: Predict Crop Yield ---
def predict_yield(user_input):
    model = joblib.load("crop_yield_model.pkl")
    selected_features = joblib.load("selected_features.pkl") 
    le = joblib.load("crop_label_encoder.pkl")  

    # Validate required inputs
    if "crop_type" not in user_input:
        raise ValueError("Missing required input: 'crop_type' (e.g., 'apple', 'rice')")
    
    if "latitude" not in user_input or "longitude" not in user_input:
        raise ValueError("Missing required input: 'latitude' and 'longitude'.")
    
    # Fetch weather data
    weather_data = getAvgs(user_input["latitude"], user_input["longitude"])
    
    # Update user input with weather data
    user_input["temperature"] = weather_data["avg_temp"]
    user_input["humidity"] = weather_data["avg_humidity"]
    user_input["rainfall"] = weather_data["avg_precip"]
    user_input["sunlight_exposure"] = weather_data["avg_sunshine"]

    # Convert crop type to lowercase and check validity
    crop_name = user_input["crop_type"].strip().lower()  
    if crop_name not in le.classes_:
        raise ValueError(f"Unknown crop type: '{crop_name}'. Please use one from the dataset: {list(le.classes_)}")

    # Convert input keys to lowercase
    user_input_lower = {key.lower(): value for key, value in user_input.items()}

    # Ensure input matches trained model features
    missing_features = [f for f in selected_features if f not in user_input_lower]

    if "state" in user_input:  # Check if state is provided
        state_nutrients = get_nutrient_levels(user_input["state"])
        if state_nutrients:
            user_input["N"] = state_nutrients["N"]
            user_input["P"] = state_nutrients["P"]
            user_input["K"] = state_nutrients["K"]

    # Ensure input contains all required features
    missing_features = [f for f in selected_features if f not in user_input]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Convert input to array format
    user_input_values = np.array([list(user_input_lower[f] for f in selected_features)]).reshape(1, -1)

    # Predict yield
    predicted_yield = model.predict(user_input_values)
    return f"🌾 Predicted Crop Yield for {crop_name}: {predicted_yield[0]:.2f} kg/ha"

# --- Step 5: Run Prediction Example ---
user_input = {
    "crop_type": "cotton",
    "state": "Kentucky",
    "latitude": 37.7749,
    "longitude": -122.4194,
}

print(predict_yield(user_input))
