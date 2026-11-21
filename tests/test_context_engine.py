import pandas as pd

from src.context_engine import (
    analyze_dataset_context
)


data = {
    "Transaction_ID": [1, 2, 3, 4, 5],
    "Amount": [100, 200, 50, 500, 1000],
    "Fraud_Actual": [0, 0, 0, 1, 0],
    "Fraud_Prediction": [0, 0, 0, 1, 1],
    "Fraud_Probability": [
        0.01,
        0.02,
        0.03,
        0.91,
        0.78
    ]
}

df = pd.DataFrame(data)

context = analyze_dataset_context(df)

print(context)