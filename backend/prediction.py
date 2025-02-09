import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fetchWeather import getAvgs
from soil import get_nutrient_levels


file_path = "Crop_recommendationV2_with_yield_fixed.csv"
df = pd.read_csv(file_path)
df["label"] = df["label"].str.strip().str.lower()


le_crop = LabelEncoder()
df["label_encoded"] = le_crop.fit_transform(df["label"])
joblib.dump(le_crop, "crop_label_encoder.pkl")

df = df.dropna(subset=["estimated_yield_kg_per_ha"]).copy()


scale_cols = df.select_dtypes(include=['number']).columns.drop("estimated_yield_kg_per_ha")

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
joblib.dump(scaler, "crop_yield_scaler.pkl")

X = df[scale_cols] 
y = df["estimated_yield_kg_per_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(model, "crop_yield_model.pkl")
print("Model training completed. Model saved.")


def predict_yield(user_input):
    model = joblib.load("crop_yield_model.pkl")
    scaler = joblib.load("crop_yield_scaler.pkl")
    le = joblib.load("crop_label_encoder.pkl")
    
    if "crop_type" not in user_input:
        raise ValueError("Missing required input: 'crop_type'")
    if "latitude" not in user_input or "longitude" not in user_input:
        raise ValueError("Missing required input: 'latitude' and 'longitude'.")
    

    weather_data = getAvgs(user_input["latitude"], user_input["longitude"])
    user_input.update({
        "temperature": weather_data["avg_temp"],
        "humidity": weather_data["avg_humidity"],
        "rainfall": weather_data["avg_precip"],
        "sunlight_exposure": weather_data["avg_sunshine"]
    })
    
    crop_name = user_input["crop_type"].strip().lower()
    print("Available crops in encoder:", list(le.classes_))
    if crop_name not in le.classes_:
        raise ValueError(f"Unknown crop type: '{crop_name}'")
    user_input["label_encoded"] = le.transform([crop_name])[0]
    
    state_data = get_nutrient_levels(user_input["state"])
    user_input.update({
        "N": state_data['N'],
        "P": state_data['P'],
        "K": state_data['K']
    })
    
    expected_features = list(scale_cols)  
    
    user_features = pd.DataFrame(
        [[user_input.get(col, 0) for col in expected_features]],
        columns=expected_features
    )
    
    user_features_scaled = scaler.transform(user_features)
    
    predicted_yield = model.predict(user_features_scaled)
    return f"Predicted Yield: {predicted_yield[0]:.2f} kg/ha"

user_input = {
    "crop_type": "orange",
    "state": "Minnesota",
    "latitude": 46.7296,
    "longitude": 94.6859,
}
print(predict_yield(user_input))
