import pandas as pd

file_path = "companion_plants.csv"

df = pd.read_csv(file_path, header=None)

#convert rows to a set and store in a list
crop_groups = [set(row.dropna()) for _, in df.iterrows()]

#replace with user input
given_crops = [tomato, corn, basil]

def find_compatible_crops(given_crops, crop_groups):
    comp_crops = set()

    for group in crop_groups:
        if given_crops.issubset(group):
            comp_crops.update(group)

    return comp_crops - given_crops


print(find_compatible_crops(given_crops, crop_groups))


if __name__ == "__main__":
    print(df.head())
    main()