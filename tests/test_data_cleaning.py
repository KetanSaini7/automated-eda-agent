import pandas as pd

from src.data_cleaning import standardize_categorical_values


data = {
    "City": [
        "Delhi",
        "delhi",
        "delHi",
        " DELHI ",
        "Mumbai",
        "mumbai",
        " MUMBAI "
    ],
    "Sales": [
        100,
        200,
        150,
        300,
        250,
        180,
        220
    ]
}

df = pd.DataFrame(data)

cleaned_df = standardize_categorical_values(df)

print("\nOriginal Data:")
print(df)

print("\nCleaned Data:")
print(cleaned_df)

print("\nUnique Cities:")
print(cleaned_df["City"].unique())

