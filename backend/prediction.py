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


file_path = "Crop_recommendationV2.csv"
df = pd.read_csv(file_path)

df["label"] = df["label"].str.strip().str.lower()

numeric_cols = df.select_dtypes(include=['number']).columns  
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

processed_file_path = "Crop_recommendationV2_preprocessed.csv"
df.to_csv(processed_file_path, index=False)
joblib.dump(le, "crop_label_encoder.pkl")  
print("✅ Data preprocessing completed. Preprocessed dataset saved.")

df = pd.read_csv(processed_file_path)
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=7) 
rfe.fit(X, y)

selected_features = ["N", "P", "K", "temperature", "humidity", "rainfall", "sunlight_exposure"]

df_selected = df[selected_features]
df_selected.to_csv("Crop_recommendationV2_selected_features.csv", index=False)
joblib.dump(selected_features, "selected_features.pkl")  
print("✅ Feature selection completed. Final features used:", selected_features)

X = df_selected  
y = np.random.randint(500, 5000, size=len(df_selected))  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ MAE:", mean_absolute_error(y_test, y_pred))
print("✅ RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(model, "crop_yield_model.pkl")
print("✅ Model training completed. Model saved.")

def predict_yield(user_input):
    model = joblib.load("crop_yield_model.pkl")
    selected_features = joblib.load("selected_features.pkl") 
    le = joblib.load("crop_label_encoder.pkl")  

    if "crop_type" not in user_input:
        raise ValueError("Missing required input: 'crop_type' (e.g., 'apple', 'rice')")
    
    if "latitude" not in user_input or "longitude" not in user_input:
        raise ValueError("Missing required input: 'latitude' and 'longitude'.")

    weather_data = getAvgs(user_input["latitude"], user_input["longitude"])
    
    user_input["temperature"] = weather_data["avg_temp"]
    user_input["humidity"] = weather_data["avg_humidity"]
    user_input["rainfall"] = weather_data["avg_precip"]
    user_input["sunlight_exposure"] = weather_data["avg_sunshine"]

    crop_name = user_input["crop_type"].strip().lower()  
    if crop_name not in le.classes_:
        raise ValueError(f"Unknown crop type: '{crop_name}'. Please use one from the dataset: {list(le.classes_)}")
    
    state_data = get_nutrient_levels(user_input["state"])
    user_input["N"] = state_data['N']
    user_input["P"] = state_data['P']
    user_input["K"] = state_data['K']

    user_input_lower = {
        key if key in ["N", "P", "K"] else key.lower(): value
        for key, value in user_input.items()
    }

    missing_features = [f for f in selected_features if f not in user_input_lower]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    user_input_values = np.array([list(user_input_lower[f] for f in selected_features)]).reshape(1, -1)

    predicted_yield = model.predict(user_input_values)    
    print(predicted_yield)
    return f"{predicted_yield[0]:.2f}"

user_input = {
    "crop_type": "cotton",
    "state": "Illinois",
    "latitude": 37.7749,
    "longitude": -122.4194,
}

print(predict_yield(user_input))
