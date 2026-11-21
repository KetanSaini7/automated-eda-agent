import pandas as pd

from src.eda import perform_eda


data = {
    "Sales": [100, 200, 300, 400, 500],
    "Profit": [10, 20, 30, 40, 50],
    "Quantity": [2, 4, 6, 8, 10],
    "Region": [
        "North",
        "South",
        "North",
        "West",
        "East"
    ],
    "Order_Date": [
        "01-06-2022",
        "15-06-2022",
        "01-07-2022",
        "15-07-2022",
        "01-08-2022"
    ]
}

df = pd.DataFrame(data)

# Convert date column
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    format="%d-%m-%Y"
)

datetime_columns = ["Order_Date"]

result = perform_eda(
    df,
    datetime_columns
)

for key, value in result.items():

    print("\n" + "=" * 50)
    print(key.upper())
    print("=" * 50)

    print(value)

