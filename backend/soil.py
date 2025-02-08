import pandas as pd

file_path = "soil_nutrients.csv"
df = pd.read_csv(file_path)

# Normalize column names
df.columns = df.columns.str.strip()
df["State"] = df["State"].astype(str).str.lower().str.strip()  # Normalize state names

def get_nutrient_levels(state):
    """
    Returns the N, P, and K values for the given state, handling duplicates and missing data.
    """
    state = state.lower().strip()  # Normalize input state
    result = df[df["State"] == state]  # Filter by state

    if result.empty:
        print(f"⚠️ No data found for state: {state}")
        return None

    # Fill missing values with the column mean (or set a default)
    result = result.fillna({"Soil Nitrogen (ppm)": 1000, "Phosphorus (ppm)_y": 40, "Potassium (ppm)_y": 30})

    # Compute the average if multiple entries exist for the same state
    avg_values = result[["Soil Nitrogen (ppm)", "Phosphorus (ppm)_y", "Potassium (ppm)_y"]].mean()

    return {
        'N': avg_values["Soil Nitrogen (ppm)"],
        'P': avg_values["Phosphorus (ppm)_y"],
        'K': avg_values["Potassium (ppm)_y"]
    }