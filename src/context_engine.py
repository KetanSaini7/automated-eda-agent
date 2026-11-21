def detect_target_columns(df):
    """
    Detect columns that are likely to represent
    ground-truth target/label variables.
    """

    target_columns = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in [
                "target",
                "label",
                "actual",
                "outcome"
            ]
        ):
            target_columns.append(column)

    return target_columns


def detect_prediction_columns(df):
    """
    Detect columns that appear to contain
    model predictions or prediction probabilities.
    """

    prediction_columns = []

    for column in df.columns:

        column_lower = column.lower()

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
            prediction_columns.append(column)

    return prediction_columns


def detect_problem_type(df, target_column):
    """
    Determine whether the target appears to represent
    classification or regression.
    """

    if target_column not in df.columns:
        return "unknown"

    unique_values = df[target_column].nunique(
        dropna=True
    )

    if (
        df[target_column].dtype == "object"
        or unique_values <= 20
    ):
        return "classification"

    return "regression"


def analyze_target_distribution(df, target_column):
    """
    Analyze the distribution of values in the target column.
    """

    if target_column not in df.columns:
        return None

    value_counts = df[target_column].value_counts(
        dropna=False
    )

    total = len(df)

    distribution = {}

    for value, count in value_counts.items():

        distribution[str(value)] = {
            "count": int(count),
            "percentage": round(
                count / total * 100,
                2
            )
        }

    return distribution


def detect_class_imbalance(target_distribution):
    """
    Detect whether a classification target is highly imbalanced.
    """

    if not target_distribution:
        return {
            "is_imbalanced": False,
            "minority_percentage": None,
            "majority_percentage": None
        }

    percentages = [
        value["percentage"]
        for value in target_distribution.values()
    ]

    majority_percentage = max(percentages)
    minority_percentage = min(percentages)

    is_imbalanced = (
        minority_percentage < 20
    )

    return {
        "is_imbalanced": is_imbalanced,
        "minority_percentage": minority_percentage,
        "majority_percentage": majority_percentage
    }


def analyze_dataset_context(df):

    target_columns = detect_target_columns(df)

    prediction_columns = detect_prediction_columns(df)

    context = {
        "target_columns": target_columns,
        "prediction_columns": prediction_columns,
        "target_column": None,
        "problem_type": "unknown",
        "target_distribution": None,
        "class_imbalance": None
    }

    if target_columns:

        target_column = target_columns[0]

        context["target_column"] = target_column

        problem_type = detect_problem_type(
            df,
            target_column
        )

        context["problem_type"] = problem_type

        if problem_type == "classification":

            target_distribution = (
                analyze_target_distribution(
                    df,
                    target_column
                )
            )

            context["target_distribution"] = (
                target_distribution
            )

            context["class_imbalance"] = (
                detect_class_imbalance(
                    target_distribution
                )
            )

    return context

def validate_anomalies_against_target(
    df,
    anomaly_indices,
    target_column
):
    """
    Validate detected anomalies against a ground-truth
    target column.
    """

    if target_column not in df.columns:
        return None

    if not anomaly_indices:
        return {
            "anomaly_count": 0,
            "confirmed_positive": 0,
            "false_positive": 0,
            "precision": None,
            "recall": None
        }

    anomaly_rows = df.loc[anomaly_indices]

    actual_anomalies = (
        anomaly_rows[target_column] == 1
    )

    confirmed_positive = int(
        actual_anomalies.sum()
    )

    anomaly_count = len(anomaly_rows)

    false_positive = (
        anomaly_count - confirmed_positive
    )

    total_actual_positive = int(
        (df[target_column] == 1).sum()
    )

    precision = (
        confirmed_positive / anomaly_count
        if anomaly_count > 0
        else 0
    )

    recall = (
        confirmed_positive / total_actual_positive
        if total_actual_positive > 0
        else 0
    )

    return {
        "anomaly_count": anomaly_count,
        "confirmed_positive": confirmed_positive,
        "false_positive": false_positive,
        "total_actual_positive": total_actual_positive,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2)
    }