def generate_dataset_insights(profile):
    """
    Generate basic insights about the dataset.
    """

    insights = []

    rows = profile["rows"]
    columns = profile["columns"]

    insights.append(
        f"The dataset contains {rows:,} rows "
        f"and {columns:,} columns."
    )

    if rows < 1000:
        insights.append(
            "The dataset is relatively small, "
            "so statistical and ML-based findings "
            "should be interpreted carefully."
        )

    elif rows >= 100000:
        insights.append(
            "The dataset is large, so automated "
            "analysis is useful for quickly identifying "
            "important patterns and potential issues."
        )

    if profile["datetime_columns"]:
        insights.append(
            "The dataset contains time-based information "
            f"in: {', '.join(profile['datetime_columns'])}."
        )

    return insights

def generate_context_insights(context):
    """
    Generate insights based on dataset context,
    target distribution, problem type, and class imbalance.
    """

    insights = []

    problem_type = context["problem_type"]
    target_column = context["target_column"]
    class_imbalance = context["class_imbalance"]
    target_distribution = context["target_distribution"]

    # -------------------------
    # Problem Type
    # -------------------------

    if problem_type == "classification":

        insights.append(
            f"The dataset represents a classification "
            f"problem with **{target_column}** as the target."
        )

    elif problem_type == "regression":

        insights.append(
            f"The dataset represents a regression "
            f"problem with **{target_column}** as the target."
        )

    # -------------------------
    # Class Distribution
    # -------------------------

    if (
        problem_type == "classification"
        and target_distribution
    ):

        distribution_text = []

        for value, data in target_distribution.items():

            distribution_text.append(
                f"{value}: {data['percentage']}%"
            )

        insights.append(
            "Target distribution: "
            + ", ".join(distribution_text)
            + "."
        )

    # -------------------------
    # Class Imbalance
    # -------------------------

    if (
        problem_type == "classification"
        and class_imbalance
    ):

        if class_imbalance["is_imbalanced"]:

            minority = class_imbalance[
                "minority_percentage"
            ]

            majority = class_imbalance[
                "majority_percentage"
            ]

            insights.append(
                f"The target is highly imbalanced, "
                f"with the majority class representing "
                f"{majority}% of observations and the "
                f"minority class representing only "
                f"{minority}%."
            )

            insights.append(
                "For this imbalanced classification "
                "problem, model performance should be "
                "evaluated using metrics such as "
                "precision, recall, F1 score, and PR-AUC "
                "rather than accuracy alone."
            )

        else:

            insights.append(
                "The target classes do not show "
                "severe class imbalance."
            )

    return insights


def generate_anomaly_validation_insights(
    anomaly_validation
):
    """
    Generate insights by comparing detected anomalies
    against the ground-truth target.
    """

    insights = []

    if anomaly_validation is None:
        return insights

    anomaly_count = anomaly_validation[
        "anomaly_count"
    ]

    confirmed_positive = anomaly_validation[
        "confirmed_positive"
    ]

    false_positive = anomaly_validation[
        "false_positive"
    ]

    precision = anomaly_validation[
        "precision"
    ]

    recall = anomaly_validation[
        "recall"
    ]

    # -------------------------
    # No anomalies
    # -------------------------

    if anomaly_count == 0:

        insights.append(
            "No anomalies were detected for "
            "ground-truth validation."
        )

        return insights

    # -------------------------
    # Detection summary
    # -------------------------

    insights.append(
        f"Isolation Forest detected "
        f"{anomaly_count:,} potentially unusual rows, "
        f"of which {confirmed_positive:,} were confirmed "
        f"as positive cases by the ground-truth target."
    )

    # -------------------------
    # Precision interpretation
    # -------------------------

    if precision is not None:

        if precision >= 80:

            insights.append(
                f"The anomaly detector showed high "
                f"precision ({precision}%), meaning most "
                f"detected anomalies corresponded to "
                f"positive target cases."
            )

        elif precision >= 40:

            insights.append(
                f"The anomaly detector showed moderate "
                f"precision ({precision}%), meaning a "
                f"substantial portion of detected anomalies "
                f"were not positive target cases."
            )

        else:

            insights.append(
                f"The anomaly detector showed low "
                f"precision ({precision}%), indicating that "
                f"many detected anomalies were not positive "
                f"target cases."
            )

    # -------------------------
    # Recall interpretation
    # -------------------------

    if recall is not None:

        if recall >= 80:

            insights.append(
                f"The anomaly detector captured most "
                f"of the positive target cases "
                f"(recall = {recall}%)."
            )

        elif recall >= 40:

            insights.append(
                f"The anomaly detector captured a "
                f"moderate proportion of positive target "
                f"cases (recall = {recall}%)."
            )

        else:

            insights.append(
                f"The anomaly detector missed a large "
                f"proportion of positive target cases "
                f"(recall = {recall}%)."
            )

    # -------------------------
    # False positives
    # -------------------------

    if false_positive > 0:

        insights.append(
            f"{false_positive:,} detected anomalies did "
            f"not correspond to positive target cases. "
            f"Therefore, Isolation Forest anomalies should "
            f"not be treated as confirmed positive cases "
            f"without further validation."
        )

    return insights


def generate_supervised_model_insights(
    supervised_validation
):
    """
    Generate insights from supervised model
    evaluation metrics.
    """

    insights = []

    if supervised_validation is None:
        return insights

    if supervised_validation.get("status") != "completed":
        return insights

    precision = supervised_validation.get("precision")
    recall = supervised_validation.get("recall")
    f1 = supervised_validation.get("f1_score")
    roc_auc = supervised_validation.get("roc_auc")
    pr_auc = supervised_validation.get("pr_auc")

    # Precision
    if precision is not None:
        if precision >= 80:
            insights.append(
                f"The model has high precision ({precision}%), "
                "meaning most predicted positive cases were "
                "actually positive."
            )
        elif precision >= 50:
            insights.append(
                f"The model has moderate precision ({precision}%), "
                "meaning a noticeable portion of predicted "
                "positive cases were incorrect."
            )
        else:
            insights.append(
                f"The model has low precision ({precision}%), "
                "indicating a relatively high number of "
                "false positive predictions."
            )

    # Recall
    if recall is not None:
        if recall >= 80:
            insights.append(
                f"The model has high recall ({recall}%), "
                "meaning it successfully identified most "
                "of the actual positive cases."
            )
        elif recall >= 50:
            insights.append(
                f"The model has moderate recall ({recall}%), "
                "meaning it identified a reasonable portion "
                "of the actual positive cases."
            )
        else:
            insights.append(
                f"The model has low recall ({recall}%), "
                "meaning many actual positive cases were missed."
            )

    # F1 Score
    if f1 is not None:
        if f1 >= 80:
            insights.append(
                f"The F1-score is strong ({f1}%), showing a good "
                "balance between precision and recall."
            )
        elif f1 >= 50:
            insights.append(
                f"The F1-score is moderate ({f1}%), indicating "
                "a reasonable but improvable balance between "
                "precision and recall."
            )
        else:
            insights.append(
                f"The F1-score is relatively low ({f1}%), "
                "suggesting that the balance between precision "
                "and recall could be improved."
            )

    # ROC-AUC
    if roc_auc is not None:
        if roc_auc >= 80:
            insights.append(
                f"The ROC-AUC score is strong ({roc_auc}%), "
                "indicating good overall class discrimination."
            )
        elif roc_auc >= 60:
            insights.append(
                f"The ROC-AUC score is moderate ({roc_auc}%), "
                "indicating some ability to distinguish between "
                "the classes."
            )
        else:
            insights.append(
                f"The ROC-AUC score is low ({roc_auc}%), "
                "indicating weak class discrimination."
            )

    # PR-AUC
    if pr_auc is not None:
        if pr_auc >= 80:
            insights.append(
                f"The PR-AUC is strong ({pr_auc}%), indicating "
                "good performance in identifying the positive "
                "class."
            )
        elif pr_auc >= 50:
            insights.append(
                f"The PR-AUC is moderate ({pr_auc}%), suggesting "
                "that positive-class detection is reasonable "
                "but can be improved."
            )
        else:
            insights.append(
                f"The PR-AUC is relatively low ({pr_auc}%), "
                "which may indicate difficulty identifying "
                "positive cases, especially in an imbalanced "
                "dataset."
            )

    return insights



def generate_quality_insights(quality):
    """
    Generate insights from data-quality results.
    """

    insights = []

    missing_values = quality["missing_values"]

    if missing_values:

        for column, count in missing_values.items():

            percentage = quality[
                "missing_percentage"
            ][column]

            insights.append(
                f"{column} contains {count:,} missing "
                f"values ({percentage}%)."
            )

    else:
        insights.append(
            "No missing values were detected."
        )

    duplicate_rows = quality["duplicate_rows"]

    if duplicate_rows > 0:

        insights.append(
            f"The dataset contains "
            f"{duplicate_rows:,} duplicate rows."
        )

    else:

        insights.append(
            "No duplicate rows were detected."
        )

    if quality["constant_columns"]:

        insights.append(
            "Constant-value columns were detected: "
            + ", ".join(
                quality["constant_columns"]
            )
        )

    if quality["high_cardinality_columns"]:

        insights.append(
            "Potential high-cardinality categorical "
            "columns were detected: "
            + ", ".join(
                quality["high_cardinality_columns"]
            )
        )

    return insights


def generate_numerical_insights(eda_results):
    """
    Generate insights from numerical analysis.
    """

    insights = []

    numerical_results = eda_results["numerical"]

    for column, stats in numerical_results.items():

        mean = stats["mean"]
        median = stats["median"]
        skewness = stats["skewness"]

        # Mean vs median
        if mean > median * 1.2:

            insights.append(
                f"{column} has a mean ({mean}) "
                f"considerably higher than its median "
                f"({median}), suggesting a right-skewed "
                "distribution."
            )

        elif median > mean * 1.2:

            insights.append(
                f"{column} has a median ({median}) "
                f"considerably higher than its mean "
                f"({mean}), suggesting a left-skewed "
                "distribution."
            )

        # Skewness
        if abs(skewness) >= 1:

            direction = (
                "right"
                if skewness > 0
                else "left"
            )

            insights.append(
                f"{column} is strongly {direction}-skewed "
                f"(skewness = {skewness})."
            )

        elif abs(skewness) >= 0.5:

            direction = (
                "right"
                if skewness > 0
                else "left"
            )

            insights.append(
                f"{column} shows moderate {direction} "
                f"skewness (skewness = {skewness})."
            )

    return insights


def generate_categorical_insights(eda_results):
    """
    Generate insights from categorical analysis.
    """

    insights = []

    categorical_results = (
        eda_results["categorical"]
    )

    for column, stats in categorical_results.items():

        top_value = stats["top_value"]
        top_percentage = (
            stats["top_value_percentage"]
        )

        if top_percentage >= 70:

            insights.append(
                f"{column} is highly concentrated: "
                f"'{top_value}' represents "
                f"{top_percentage}% of the values."
            )

        elif top_percentage >= 40:

            insights.append(
                f"'{top_value}' is the most common "
                f"value in {column}, representing "
                f"{top_percentage}% of observations."
            )

    return insights


def generate_correlation_insights(eda_results):
    """
    Generate insights from correlation analysis.
    """

    insights = []

    correlations = eda_results[
        "correlations"
    ]

    for item in correlations[:5]:

        correlation = item["correlation"]

        if abs(correlation) >= 0.8:

            strength = "very strong"

        elif abs(correlation) >= 0.6:

            strength = "strong"

        elif abs(correlation) >= 0.4:

            strength = "moderate"

        else:

            continue

        direction = (
            "positive"
            if correlation > 0
            else "negative"
        )

        insights.append(
            f"{item['column_1']} and "
            f"{item['column_2']} have a "
            f"{strength} {direction} correlation "
            f"({correlation})."
        )

    return insights


def generate_anomaly_insights(anomaly_results):
    """
    Generate insights from anomaly detection.
    """

    insights = []

    # IQR insights
    iqr_results = anomaly_results["iqr"]

    for column, result in iqr_results.items():

        count = result["anomaly_count"]
        percentage = result["anomaly_percentage"]

        if count > 0:

            insights.append(
                f"{column} contains "
                f"{count:,} potential statistical "
                f"outlier(s), representing "
                f"{percentage}% of observations."
            )

    # Multivariate Isolation Forest
    ml_result = anomaly_results[
        "multivariate_isolation_forest"
    ]

    if ml_result["status"] == "completed":

        count = ml_result["anomaly_count"]
        percentage = ml_result[
            "anomaly_percentage"
        ]

        insights.append(
            f"Multivariate Isolation Forest identified "
            f"{count:,} potentially unusual rows "
            f"({percentage}% of analyzed observations)."
        )

    return insights

def generate_insights(
    profile,
    quality,
    eda_results,
    anomaly_results,
    context=None,
    anomaly_validation=None,
    supervised_validation=None
):
    insights = {}

    insights["dataset"] = generate_dataset_insights(profile)
    insights["quality"] = generate_quality_insights(quality)
    insights["numerical"] = generate_numerical_insights(eda_results)
    insights["categorical"] = generate_categorical_insights(eda_results)
    insights["correlations"] = generate_correlation_insights(eda_results)
    insights["anomalies"] = generate_anomaly_insights(anomaly_results)

    if context is not None:
        insights["context"] = generate_context_insights(context)
    else:
        insights["context"] = []

    if anomaly_validation is not None:
        insights["anomaly_validation"] = (
            generate_anomaly_validation_insights(anomaly_validation)
        )
    else:
        insights["anomaly_validation"] = []

    if supervised_validation is not None:
        insights["supervised_model"] = (
            generate_supervised_model_insights(supervised_validation)
        )
    else:
        insights["supervised_model"] = []

    return insights