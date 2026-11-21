import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Minimum observations required before using Isolation Forest
MIN_ISOLATION_SAMPLES = 50

# Maximum number of rows used by Isolation Forest
# for large datasets
MAX_ISOLATION_SAMPLES = 200_000

# Percentage of observations that Isolation Forest
# will classify as anomalies
ANOMALY_CONTAMINATION = 0.05

# Number of trees
ISOLATION_N_ESTIMATORS = 100


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

def detect_iqr_anomalies(df):
    """
    Detect statistical outliers in suitable numerical columns
    using the IQR method.

    Excludes:
    - ID / identifier columns
    - Target / actual / label columns
    - Prediction / probability / score columns

    Returns:
    - anomaly count
    - anomaly percentage
    - lower and upper bounds
    - sample anomalous values

    IQR anomaly indices are NOT retained because they are not
    required for target validation.
    """

    results = {}

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        column_lower = column.lower()

        # ----------------------------------------------------
        # Exclude ID / identifier columns
        # ----------------------------------------------------

        if (
            "id" in column_lower
            or "identifier" in column_lower
        ):
            continue

        # ----------------------------------------------------
        # Exclude target / actual / label columns
        # ----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in [
                "target",
                "label",
                "actual",
                "outcome"
            ]
        ):
            continue

        # ----------------------------------------------------
        # Exclude prediction / model output columns
        # ----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in [
                "prediction",
                "predicted",
                "probability",
                "prob",
                "score"
            ]
        ):
            continue

        # ----------------------------------------------------
        # Get non-null values
        # ----------------------------------------------------

        series = df[column].dropna()

        if len(series) < 5:
            continue

        # ----------------------------------------------------
        # Calculate quartiles
        # ----------------------------------------------------

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        # ----------------------------------------------------
        # Calculate boundaries
        # ----------------------------------------------------

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # ----------------------------------------------------
        # Identify anomalies
        # ----------------------------------------------------

        anomaly_mask = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        anomaly_count = int(
            anomaly_mask.sum()
        )

        # ----------------------------------------------------
        # Keep only a small sample of anomaly values
        # ----------------------------------------------------

        anomalous_values = (
            series[anomaly_mask]
            .head(10)
            .tolist()
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results[column] = {

            "anomaly_count":
                anomaly_count,

            "anomaly_percentage":
                round(
                    anomaly_count / len(series) * 100,
                    2
                ),

            "lower_bound":
                round(
                    float(lower_bound),
                    2
                ),

            "upper_bound":
                round(
                    float(upper_bound),
                    2
                ),

            "anomalous_values":
                [
                    float(value)
                    for value in anomalous_values
                ]
        }

    return results


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_anomaly_features(df):
    """
    Select suitable numerical features for multivariate
    anomaly detection using Isolation Forest.

    Excludes:
    - ID-like columns
    - Target/label columns
    - Prediction/output columns
    - Highly unique identifier-like columns

    Returns:
    - selected numerical features
    - excluded features with reasons
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    selected_columns = []
    excluded_columns = {}

    for column in numeric_columns:

        column_lower = column.lower()

        # ----------------------------------------------------
        # Exclude ID / identifier columns
        # ----------------------------------------------------

        if (
            "id" in column_lower
            or "identifier" in column_lower
        ):

            excluded_columns[column] = (
                "Identifier column"
            )

            continue

        # ----------------------------------------------------
        # Exclude target / label columns
        # ----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in [
                "target",
                "label",
                "actual",
                "outcome"
            ]
        ):

            excluded_columns[column] = (
                "Target/label column"
            )

            continue

        # ----------------------------------------------------
        # Exclude prediction / model output columns
        # ----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in [
                "prediction",
                "predicted",
                "probability",
                "prob",
                "score"
            ]
        ):

            excluded_columns[column] = (
                "Model output column"
            )

            continue

        # ----------------------------------------------------
        # Exclude extremely high-cardinality columns
        # ----------------------------------------------------

        unique_ratio = (
            df[column].nunique(
                dropna=True
            )
            / len(df)
        )

        if unique_ratio > 0.95:

            excluded_columns[column] = (
                "Likely identifier/high-cardinality column"
            )

            continue

        selected_columns.append(column)

    return {
        "selected_columns":
            selected_columns,

        "excluded_columns":
            excluded_columns
    }


# ============================================================
# ISOLATION FOREST
# ============================================================

def detect_multivariate_isolation_forest(df):
    """
    Detect row-level anomalies using multiple numerical
    features with Isolation Forest.

    For large datasets, a representative sample is used
    to reduce memory usage and processing time.

    Original DataFrame indices are preserved so detected
    rows
    can still be validated against the target column.
    """

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    feature_selection = (
        select_anomaly_features(df)
    )

    selected_columns = (
        feature_selection[
            "selected_columns"
        ]
    )

    excluded_columns = (
        feature_selection[
            "excluded_columns"
        ]
    )

    # --------------------------------------------------------
    # Need at least two suitable features
    # --------------------------------------------------------

    if len(selected_columns) < 2:

        return {
            "status": "skipped",

            "reason": (
                "At least 2 suitable numerical "
                "features are required."
            ),

            "features_used":
                selected_columns,

            "excluded_features":
                excluded_columns
        }

    # --------------------------------------------------------
    # Select numerical features
    # --------------------------------------------------------

    numeric_df = df[
        selected_columns
    ]

    # --------------------------------------------------------
    # Remove rows containing missing values
    # --------------------------------------------------------

    clean_numeric_df = numeric_df.dropna()

    # --------------------------------------------------------
    # Check minimum number of observations
    # --------------------------------------------------------

    if len(clean_numeric_df) < MIN_ISOLATION_SAMPLES:

        return {
            "status": "skipped",

            "reason": (
                f"Not enough observations. "
                f"Minimum required: "
                f"{MIN_ISOLATION_SAMPLES}."
            ),

            "features_used":
                selected_columns,

            "excluded_features":
                excluded_columns
        }

    # --------------------------------------------------------
    # Sample large datasets
    # --------------------------------------------------------

    rows_before_sampling = len(
        clean_numeric_df
    )

    if rows_before_sampling > MAX_ISOLATION_SAMPLES:

        clean_numeric_df = (
            clean_numeric_df.sample(
                n=MAX_ISOLATION_SAMPLES,
                random_state=42
            )
        )

    rows_analyzed = len(
        clean_numeric_df
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=ISOLATION_N_ESTIMATORS,
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    predictions = model.fit_predict(
        clean_numeric_df
    )

    # -1 = anomaly
    #  1 = normal

    anomaly_mask = (
        predictions == -1
    )

    anomaly_indices = (
        clean_numeric_df.index[
            anomaly_mask
        ]
    )

    anomaly_count = int(
        anomaly_mask.sum()
    )

    anomaly_percentage = round(
        anomaly_count
        / rows_analyzed
        * 100,
        2
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "status":
            "completed",

        "features_used":
            selected_columns,

        "excluded_features":
            excluded_columns,

        "rows_available":
            rows_before_sampling,

        "rows_analyzed":
            rows_analyzed,

        "sampling_applied":
            rows_before_sampling > MAX_ISOLATION_SAMPLES,

        "anomaly_count":
            anomaly_count,

        "anomaly_percentage":
            anomaly_percentage,

        # Required for target validation
        "anomaly_indices":
            anomaly_indices.tolist()
    }


# ============================================================
# COMPLETE ANOMALY PIPELINE
# ============================================================

def detect_anomalies(df):
    """
    Run the complete anomaly detection pipeline.

    Includes:
    1. IQR statistical outlier detection
    2. Multivariate Isolation Forest
    """

    results = {}

    # --------------------------------------------------------
    # Statistical outliers
    # --------------------------------------------------------

    results["iqr"] = (
        detect_iqr_anomalies(df)
    )

    # --------------------------------------------------------
    # Multivariate ML-based anomalies
    # --------------------------------------------------------

    results[
        "multivariate_isolation_forest"
    ] = (
        detect_multivariate_isolation_forest(
            df
        )
    )

    return results