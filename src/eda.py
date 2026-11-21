import pandas as pd
import numpy as np


def numerical_analysis(df):
    """
    Generate statistical analysis for numerical columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {}

    analysis = {}

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if len(series) == 0:
            continue

        analysis[column] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q1": round(float(series.quantile(0.25)), 2),
            "q3": round(float(series.quantile(0.75)), 2),
            "skewness": round(float(series.skew()), 2)
        }

    return analysis


def categorical_analysis(df):
    """
    Generate frequency analysis for categorical columns.
    """

    categorical_df = df.select_dtypes(
        include=["object", "category"]
    )

    if categorical_df.empty:
        return {}

    analysis = {}

    for column in categorical_df.columns:

        series = categorical_df[column].dropna()

        if len(series) == 0:
            continue

        value_counts = series.value_counts()

        analysis[column] = {
            "unique_values": int(series.nunique()),
            "top_value": str(value_counts.index[0]),
            "top_value_count": int(value_counts.iloc[0]),
            "top_value_percentage": round(
                float(
                    value_counts.iloc[0] / len(series) * 100
                ),
                2
            ),
            "top_10_values": value_counts.head(10).to_dict()
        }

    return analysis


def correlation_analysis(df):
    """
    Find meaningful correlations between numerical columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return []

    correlation_matrix = numeric_df.corr()

    correlations = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            column_1 = columns[i]
            column_2 = columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2
            ]

            if pd.notna(correlation):

                correlations.append({
                    "column_1": column_1,
                    "column_2": column_2,
                    "correlation": round(
                        float(correlation),
                        3
                    )
                })

    # Sort by absolute correlation
    correlations.sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )

    return correlations


def datetime_analysis(df, datetime_columns):
    """
    Analyze datetime columns and identify date ranges.
    """

    analysis = {}

    for column in datetime_columns:

        dates = pd.to_datetime(
            df[column],
            errors="coerce"
        ).dropna()

        if dates.empty:
            continue

        analysis[column] = {
            "start_date": str(dates.min().date()),
            "end_date": str(dates.max().date()),
            "days": int(
                (dates.max() - dates.min()).days
            )
        }

    return analysis


def perform_eda(df, datetime_columns=None):
    """
    Run the complete automated EDA pipeline.
    """

    if datetime_columns is None:
        datetime_columns = []

    results = {}

    results["numerical"] = numerical_analysis(df)

    results["categorical"] = categorical_analysis(df)

    results["correlations"] = correlation_analysis(df)

    results["datetime"] = datetime_analysis(
        df,
        datetime_columns
    )

    return results

