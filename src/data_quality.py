
import pandas as pd


def check_data_quality(df):

    quality = {}

    # ==========================================
    # 1. Missing Values
    # ==========================================

    missing_count = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    quality["missing_values"] = {
        column: int(count)
        for column, count in missing_count.items()
        if count > 0
    }

    quality["missing_percentage"] = {
        column: float(percentage)
        for column, percentage in missing_percentage.items()
        if percentage > 0
    }


    # ==========================================
    # 2. Duplicate Rows
    # ==========================================

    quality["duplicate_rows"] = int(
        df.duplicated().sum()
    )


    # ==========================================
    # 3. Constant Columns
    # ==========================================

    constant_columns = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    quality["constant_columns"] = constant_columns


    # ==========================================
    # 4. High Cardinality Columns
    # ==========================================

    high_cardinality_columns = []

    for column in df.select_dtypes(
        include=["object", "category"]
    ).columns:

        n_total = len(df)
        n_unique = df[column].nunique(dropna=True)

        # Require at least 50 rows and 10 unique values before applying the
        # ratio rule — prevents false positives on tiny DataFrames.
        if n_total < 50 or n_unique < 10:
            continue

        unique_ratio = n_unique / n_total

        if unique_ratio > 0.5:
            high_cardinality_columns.append(column)

    quality["high_cardinality_columns"] = (
        high_cardinality_columns
    )


    # ==========================================
    # 5. Numeric Columns with Negative Values
    # ==========================================

    negative_values = {}

    for column in df.select_dtypes(
        include="number"
    ).columns:

        count = (df[column] < 0).sum()

        if count > 0:
            negative_values[column] = int(count)

    quality["negative_values"] = negative_values


    # ==========================================
    # 6. Zero Values
    # ==========================================

    zero_values = {}

    for column in df.select_dtypes(
        include="number"
    ).columns:

        count = (df[column] == 0).sum()

        if count > 0:
            zero_values[column] = int(count)

    quality["zero_values"] = zero_values


    # ==========================================
    # 7. Unique Values
    # ==========================================

    unique_values = {}

    for column in df.columns:

        unique_values[column] = int(
            df[column].nunique(dropna=True)
        )

    quality["unique_values"] = unique_values


    # ==========================================
    # 8. Data Types
    # ==========================================

    quality["data_types"] = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }


    # ==========================================
    # 9. Empty Columns (all values are NaN)
    # ==========================================

    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    quality["empty_columns"] = empty_columns


    # ==========================================
    # 10. Mixed-Type Columns
    #     Object columns where values contain
    #     a mix of numeric and non-numeric data.
    # ==========================================

    mixed_type_columns = []

    for column in df.select_dtypes(include=["object"]).columns:

        non_null = df[column].dropna()

        if len(non_null) == 0:
            continue

        # Try to coerce to numeric; count how many succeed
        numeric_count = pd.to_numeric(
            non_null, errors="coerce"
        ).notna().sum()

        numeric_ratio = numeric_count / len(non_null)

        # Flag as mixed-type if between 5% and 95% of values parse as numeric
        if 0.05 < numeric_ratio < 0.95:
            mixed_type_columns.append({
                "column": column,
                "numeric_ratio": round(float(numeric_ratio), 3),
                "non_numeric_ratio": round(1.0 - float(numeric_ratio), 3),
            })

    quality["mixed_type_columns"] = mixed_type_columns


    return quality

