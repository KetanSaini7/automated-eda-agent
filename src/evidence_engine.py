def limit_items(data, max_items=20):
    """
    Limit large dictionaries or lists so that the
    evidence remains compact for LLM processing.
    """

    if isinstance(data, dict):
        items = list(data.items())

        if len(items) <= max_items:
            return data

        return dict(items[:max_items])

    if isinstance(data, list):
        return data[:max_items]

    return data

def build_evidence(
    profile,
    context,
    quality,
    eda_results,
    anomaly_results,
    anomaly_validation=None,
    supervised_validation=None
):
    """
    Combine the outputs of the analytical pipeline
    into one structured evidence object.

    This evidence will later be provided to
    the Hugging Face LLM.
    """

    evidence = {}

    # -------------------------
    # Dataset Context
    # -------------------------

    evidence["dataset"] = {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "datetime_columns": profile[
            "datetime_columns"
        ],
        "numeric_columns": profile[
            "numeric_columns"
        ],
        "categorical_columns": profile[
            "categorical_columns"
        ]
    }

    # -------------------------
    # Problem Context
    # -------------------------

    evidence["context"] = {
        "target_column": context[
            "target_column"
        ],
        "target_columns": context[
            "target_columns"
        ],
        "prediction_columns": context[
            "prediction_columns"
        ],
        "problem_type": context[
            "problem_type"
        ],
        "target_distribution": context[
            "target_distribution"
        ],
        "class_imbalance": context[
            "class_imbalance"
        ]
    }

    # -------------------------
    # Data Quality
    # -------------------------

    evidence["quality"] = {
        "missing_values": quality[
            "missing_values"
        ],
        "missing_percentage": quality[
            "missing_percentage"
        ],
        "duplicate_rows": quality[
            "duplicate_rows"
        ],
        "constant_columns": quality[
            "constant_columns"
        ],
        "high_cardinality_columns": quality[
            "high_cardinality_columns"
        ],
        "negative_values": quality[
            "negative_values"
        ],
        "zero_values": quality[
            "zero_values"
        ]
    }

    # -------------------------
    # EDA
    # -------------------------

    evidence["eda"] = {
        "numerical": limit_items(
            eda_results["numerical"]
        ),
        "categorical": limit_items(
            eda_results["categorical"]
        ),
        "correlations": limit_items(
            eda_results["correlations"]
        ),
        "datetime": limit_items(
            eda_results["datetime"]
        )
    }

    # -------------------------
    # Anomaly Detection
    # -------------------------

    evidence["anomalies"] = {
        "iqr": summarize_iqr_evidence(
            anomaly_results["iqr"]
        ),
        "isolation_forest": {
            "status": anomaly_results[
                "multivariate_isolation_forest"
            ]["status"],
            "anomaly_count": anomaly_results[
                "multivariate_isolation_forest"
            ].get("anomaly_count"),
            "anomaly_percentage": anomaly_results[
                "multivariate_isolation_forest"
            ].get("anomaly_percentage"),
            "features_used": anomaly_results[
                "multivariate_isolation_forest"
            ].get("features_used", [])
        }
    }

    # -------------------------
    # Anomaly Validation
    # -------------------------

    evidence["anomaly_validation"] = (
        anomaly_validation
    )

    # -------------------------
    # Supervised Model
    # Validation
    # -------------------------

    evidence["supervised_model"] = (
        supervised_validation
    )

    return evidence

def create_llm_evidence(
    evidence,
    insights=None
):
    """
    Create a compact evidence package designed for
    consumption by an LLM.

    Python performs the analysis and calculations.
    The LLM interprets the resulting evidence.
    """

    llm_evidence = {}

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    llm_evidence["dataset"] = evidence.get(
        "dataset",
        {}
    )

    # ------------------------------------------------
    # Dataset context
    # ------------------------------------------------

    llm_evidence["context"] = evidence.get(
        "context",
        {}
    )

    # ------------------------------------------------
    # Data quality
    # ------------------------------------------------

    llm_evidence["quality"] = evidence.get(
        "quality",
        {}
    )

    # ------------------------------------------------
    # EDA
    # ------------------------------------------------

    llm_evidence["eda"] = evidence.get(
        "eda",
        {}
    )

    # ------------------------------------------------
    # Anomalies
    # ------------------------------------------------

    llm_evidence["anomalies"] = evidence.get(
        "anomalies",
        {}
    )

    # ------------------------------------------------
    # Anomaly validation
    # ------------------------------------------------

    llm_evidence["anomaly_validation"] = evidence.get(
        "anomaly_validation"
    )

    # ------------------------------------------------
    # Supervised model
    # ------------------------------------------------

    llm_evidence["supervised_model"] = evidence.get(
        "supervised_model"
    )

    # ------------------------------------------------
    # Deterministic insights
    # ------------------------------------------------

    llm_evidence["insights"] = insights or {}

    return llm_evidence

def summarize_iqr_evidence(iqr_results):
    """
    Create a compact representation of IQR anomaly results
    for the Evidence Layer and LLM.
    """

    summary = {}

    for column, result in iqr_results.items():

        summary[column] = {
            "anomaly_count": result.get(
                "anomaly_count"
            ),

            "anomaly_percentage": result.get(
                "anomaly_percentage"
            ),

            "lower_bound": result.get(
                "lower_bound"
            ),

            "upper_bound": result.get(
                "upper_bound"
            ),

            # Keep only the sample values
            "anomalous_values": result.get(
                "anomalous_values",
                []
            )
        }

    return summary  