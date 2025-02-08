import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Step 1: Load and Preprocess Data ---
file_path = "Crop_recommendationV2.csv"
df = pd.read_csv(file_path)

# Ensure all crop labels are lowercase (fix inconsistent labeling)
df["label"] = df["label"].str.strip().str.lower()  

# Identify numeric columns and handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns  
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  

# Scale numerical features
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  

# Encode Crop Type (Label Encoding)
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])  # Encode crop names into numbers

# Save preprocessed dataset
processed_file_path = "Crop_recommendationV2_preprocessed.csv"
df.to_csv(processed_file_path, index=False)
joblib.dump(le, "crop_label_encoder.pkl")  # Save label encoder
print("✅ Data preprocessing completed. Preprocessed dataset saved.")

# --- Step 2: Correlation Matrix (Exclude 'label') ---
df = pd.read_csv(processed_file_path)  # Reload after preprocessing
numeric_df = df.select_dtypes(include=['number'])  # Keep only numeric columns

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()
print("✅ Correlation matrix saved as 'correlation_matrix.png'.")

# --- Step 3: Feature Selection using RFE ---
X = df.drop(columns=["label", "label_encoded"])  # Features (excluding label)
y = df["label_encoded"]  # Crop labels (encoded)

# Train RFE model
model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=9)  # Select 9 features
rfe.fit(X, y)

# Print selected features
selected_features = X.columns[rfe.support_].tolist()
print("✅ Top Features Selected by RFE:", selected_features)

# Remove "urban_area_proximity" and Ensure "N", "P", "K" are Included
if "urban_area_proximity" in selected_features:
    selected_features.remove("urban_area_proximity")

for nutrient in ["N", "P", "K"]:
    if nutrient not in selected_features:
        selected_features.append(nutrient)

# --- Step 4: SHAP Feature Importance ---
explainer = shap.TreeExplainer(rfe.estimator_)
shap_values = explainer.shap_values(X)

# Save the SHAP summary plot instead of showing
plt.figure()
shap.summary_plot(shap_values, X, show=False)  
plt.savefig("shap_summary_plot.png")
plt.close()
print("✅ SHAP summary plot saved as 'shap_summary_plot.png'.")

# --- Step 5: Save Selected Features ---
df_selected = df[selected_features]  
df_selected.to_csv("Crop_recommendationV2_selected_features.csv", index=False)

joblib.dump(selected_features, "selected_features.pkl")  # Save feature list
print("✅ Feature selection completed. Selected features saved.")

# --- Step 6: Train Crop Yield Prediction Model ---
X = df_selected  # Only selected features
y = np.random.randint(500, 5000, size=len(df_selected))  # Simulated Crop Yield Data

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

# --- Step 7: Predict Yield for a User Input ---
def predict_yield(user_input):
    model = joblib.load("crop_yield_model.pkl")
    selected_features = joblib.load("selected_features.pkl")  # Load feature list
    le = joblib.load("crop_label_encoder.pkl")  # Load label encoder

    # Ensure crop type is provided
    if "crop_type" not in user_input:
        raise ValueError("Missing required input: 'crop_type' (e.g., 'wheat', 'rice')")

    # Convert crop type to lowercase and check if it exists
    crop_name = user_input["crop_type"].strip().lower()  
    if crop_name not in le.classes_:
        raise ValueError(f"Unknown crop type: '{crop_name}'. Please use one from the dataset: {list(le.classes_)}")

    # Convert input keys to lowercase for consistency
    user_input_lower = {key.lower(): value for key, value in user_input.items()}

    # Ensure all required features exist
    missing_features = [f for f in selected_features if f not in user_input_lower]

    # Assign default values to missing features (or raise an error)
    default_values = {"N": 40, "P": 20, "K": 30}  

    for feature in missing_features:
        if feature in default_values:
            user_input_lower[feature] = default_values[feature]  # Assign default
        else:
            raise ValueError(f"Missing feature: {feature}. Please provide it.")

    # Convert input to array format
    user_input_values = np.array([list(user_input_lower[f] for f in selected_features)]).reshape(1, -1)

    # Predict yield
    predicted_yield = model.predict(user_input_values)
    return f"🌾 Predicted Crop Yield for {crop_name}: {predicted_yield[0]:.2f} kg/ha"

# Example User Input (Including Crop Type and Soil NPK)
user_input = {
    "crop_type": "cotton",  
    "temperature": 30,  
    "humidity": 50,  
    "rainfall": 100,  
    "sunlight_exposure": 8,  
    "ph": 6.5,
    "soil_moisture": 40,  
    "N": 50,
    "P": 30,  
    "K": 20,
}

# Run Prediction
print(predict_yield(user_input))
