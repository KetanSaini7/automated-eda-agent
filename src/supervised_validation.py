import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def validate_supervised_model(
    df,
    target_column,
    prediction_column,
    probability_column=None
):
    """
    Evaluate a supervised classification model.

    Calculates:
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC
    - Confusion Matrix
    """

    required_columns = [
        target_column,
        prediction_column
    ]

    if probability_column is not None:
        required_columns.append(
            probability_column
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "status": "skipped",
            "reason": (
                "Required columns are missing: "
                + ", ".join(missing_columns)
            )
        }

    evaluation_df = df[required_columns].dropna()

    if evaluation_df.empty:
        return {
            "status": "skipped",
            "reason": "No valid rows available for evaluation."
        }

    y_true = evaluation_df[target_column]
    y_pred = evaluation_df[prediction_column]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    results = {
        "status": "completed",
        "rows_evaluated": len(evaluation_df),

        "precision": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall * 100,
            2
        ),

        "f1_score": round(
            f1 * 100,
            2
        ),

        "confusion_matrix": matrix.tolist()
    }

    # ROC-AUC and PR-AUC require prediction probabilities
    if probability_column is not None:

        y_probability = evaluation_df[
            probability_column
        ]

        try:

            roc_auc = roc_auc_score(
                y_true,
                y_probability
            )

            pr_auc = average_precision_score(
                y_true,
                y_probability
            )

            results["roc_auc"] = round(
                roc_auc * 100,
                2
            )

            results["pr_auc"] = round(
                pr_auc * 100,
                2
            )

        except ValueError:

            results["roc_auc"] = None
            results["pr_auc"] = None

    else:

        results["roc_auc"] = None
        results["pr_auc"] = None

    return results

