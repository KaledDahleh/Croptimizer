import pandas as pd

file_path = "soil_nutrients.csv"
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip()
df["State"] = df["State"].astype(str).str.lower().str.strip()  

def get_nutrient_levels(state):
    """
    Returns the N, P, and K values for the given state, handling duplicates and missing data.
    """
    state = state.lower().strip() 
    result = df[df["State"] == state]
    if result.empty:
        print(f"No data found for state: {state}")
        return None

    result = result.fillna({"Soil Nitrogen (ppm)": 1000, "Phosphorus (ppm)_y": 40, "Potassium (ppm)_y": 30})

    avg_values = result[["Soil Nitrogen (ppm)", "Phosphorus (ppm)_y", "Potassium (ppm)_y"]].mean()

    return {
        'N': avg_values["Soil Nitrogen (ppm)"],
        'P': avg_values["Phosphorus (ppm)_y"],
        'K': avg_values["Potassium (ppm)_y"]
    }