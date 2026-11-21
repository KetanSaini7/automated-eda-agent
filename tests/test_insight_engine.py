import pandas as pd

from src.profiler import profile_dataset
from src.data_quality import check_data_quality
from src.data_cleaning import standardize_categorical_values
from src.eda import perform_eda
from src.anomaly import detect_anomalies
from src.insight_engine import generate_insights


# -----------------------------------------
# Load dataset
# -----------------------------------------

df = pd.read_csv(
    r"D:\download\ann_full_predictions.csv"
)


# -----------------------------------------
# Run existing pipeline
# -----------------------------------------

profile = profile_dataset(df)

quality = check_data_quality(df)

cleaned_df = standardize_categorical_values(df)

eda_results = perform_eda(
    cleaned_df,
    profile["datetime_columns"]
)

anomaly_results = detect_anomalies(
    cleaned_df
)


# -----------------------------------------
# Generate insights
# -----------------------------------------

insights = generate_insights(
    profile,
    quality,
    eda_results,
    anomaly_results
)


# -----------------------------------------
# Display results
# -----------------------------------------

print("\n")
print("=" * 60)
print("AUTOMATED INSIGHTS")
print("=" * 60)


for category, category_insights in insights.items():

    print(
        f"\n--- {category.upper()} ---"
    )

    if not category_insights:

        print("No significant insights detected.")

    else:

        for insight in category_insights:

            print(f"• {insight}")

