import pandas as pd

file_path = "soil_nutrients.csv"
df = pd.read_csv(file_path)

def get_nutrient_levels(state):
    """
    Returns the N, P, and K values for the given state.
    """
    result = df[df["State"].str.lower() == state.lower()]
    if result.empty:
        return None
    
    nutrients = result.iloc[0][["Soil Nitrogen (ppm)", "Phosphorus (ppm)_y", "Potassium (ppm)_y"]]
    return {"N": nutrients[0], "P": nutrients[1], "K": nutrients[2]}
