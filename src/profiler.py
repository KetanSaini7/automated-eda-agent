import pandas as pd


def detect_datetime_columns(df):
    """
    Detect columns containing date/time values.

    Handles dates stored as strings, including:
    DD-MM-YYYY
    DD/MM/YYYY
    YYYY-MM-DD
    """

    datetime_columns = []

    for column in df.columns:

        # Already recognized as datetime
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)
            continue

        # Only try parsing object/string columns
        if df[column].dtype == "object":

            # Remove missing values and convert to string
            values = df[column].dropna().astype(str).str.strip()

            if len(values) == 0:
                continue

            # Try DD-MM-YYYY / DD/MM/YYYY first
            converted = pd.to_datetime(
                values,
                format="%d-%m-%Y",
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            # If that didn't work, try DD/MM/YYYY
            if valid_ratio < 0.8:

                converted = pd.to_datetime(
                    values,
                    format="%d/%m/%Y",
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

            # Finally try general date parsing
            if valid_ratio < 0.8:

                converted = pd.to_datetime(
                    values,
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

            # Consider column a datetime column
            # if at least 80% of its values are valid dates
            if valid_ratio >= 0.8:
                datetime_columns.append(column)

    return datetime_columns


def profile_dataset(df):

    profile = {}

    # --------------------------------
    # Basic Dataset Information
    # --------------------------------

    profile["rows"] = df.shape[0]

    profile["columns"] = df.shape[1]

    profile["column_names"] = df.columns.tolist()


    # --------------------------------
    # Detect Datetime Columns
    # --------------------------------

    datetime_columns = detect_datetime_columns(df)

    profile["datetime_columns"] = datetime_columns


    # --------------------------------
    # Numeric Columns
    # --------------------------------

    profile["numeric_columns"] = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )


    # --------------------------------
    # Categorical Columns
    # --------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    # Remove datetime columns
    categorical_columns = [
        column
        for column in categorical_columns
        if column not in datetime_columns
    ]

    profile["categorical_columns"] = categorical_columns


    # --------------------------------
    # Missing Values
    # --------------------------------

    missing_values = df.isnull().sum()

    profile["missing_values"] = (
        missing_values[missing_values > 0]
        .to_dict()
    )


    # --------------------------------
    # Duplicate Rows
    # --------------------------------

    profile["duplicate_rows"] = int(
        df.duplicated().sum()
    )


    return profile

