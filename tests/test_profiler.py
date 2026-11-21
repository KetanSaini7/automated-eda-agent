import pandas as pd

from src.profiler import profile_dataset


data = {
    "Sales": [100, 200, 300, 400],
    "Profit": [10, 20, 30, 40],
    "Region": ["North", "South", "North", "West"]
}

df = pd.DataFrame(data)

result = profile_dataset(df)

print(result)