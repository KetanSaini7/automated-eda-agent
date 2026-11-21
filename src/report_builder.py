"""
report_builder.py
=================
Assembles a readable Markdown report from all analysis outputs.

The report contains:
  1. Dataset summary (profile + context)
  2. Data quality results
  3. EDA statistics (numerical, categorical, correlations, datetime)
  4. Anomaly detection (IQR + Isolation Forest + validation)
  5. Supervised model metrics + confusion matrix
  6. Automated insights
  7. AI-generated analysis (optional — included if already generated)

The function returns a plain string so callers can:
  - Offer a Streamlit download_button (as .md or .txt)
  - Write it to outputs/

No external dependencies beyond Python stdlib — avoids any import that
might not be available in the user's environment.
"""

from __future__ import annotations

import datetime


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _h1(text: str) -> str:
    return f"\n# {text}\n"


def _h2(text: str) -> str:
    return f"\n## {text}\n"


def _h3(text: str) -> str:
    return f"\n### {text}\n"


def _hr() -> str:
    return "\n---\n"


def _bullet(items: list, indent: int = 0) -> str:
    prefix = "  " * indent
    return "\n".join(f"{prefix}- {item}" for item in items) + "\n"


def _kv_table(rows: list[tuple[str, str]], headers=("Metric", "Value")) -> str:
    """Render a simple two-column Markdown table."""
    col1 = max(len(headers[0]), max((len(str(r[0])) for r in rows), default=0))
    col2 = max(len(headers[1]), max((len(str(r[1])) for r in rows), default=0))
    sep = f"| {'-' * col1} | {'-' * col2} |"
    head = f"| {headers[0].ljust(col1)} | {headers[1].ljust(col2)} |"
    lines = [head, sep]
    for k, v in rows:
        lines.append(f"| {str(k).ljust(col1)} | {str(v).ljust(col2)} |")
    return "\n".join(lines) + "\n"


def _safe(value, default="N/A") -> str:
    """Return str(value) or default when value is None / empty."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


# ============================================================
# SECTION BUILDERS
# ============================================================

def _section_header(profile: dict, context: dict) -> str:
    lines = [_h1("AI Data Analyst — EDA Report")]
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {generated_at}\n")

    lines.append(_h2("1. Dataset Summary"))

    rows = [
        ("Rows", f"{profile.get('rows', 'N/A'):,}"),
        ("Columns", f"{profile.get('columns', 'N/A'):,}"),
        ("Numeric columns", ", ".join(profile.get("numeric_columns", [])) or "None"),
        ("Categorical columns", ", ".join(profile.get("categorical_columns", [])) or "None"),
        ("Datetime columns", ", ".join(profile.get("datetime_columns", [])) or "None"),
        ("Duplicate rows", f"{profile.get('duplicate_rows', 0):,}"),
    ]

    if context:
        rows += [
            ("Problem type", _safe(context.get("problem_type"))),
            ("Target column", _safe(context.get("target_column"))),
        ]

    lines.append(_kv_table(rows))
    return "".join(lines)


def _section_quality(quality: dict) -> str:
    lines = [_h2("2. Data Quality")]

    # Missing values
    missing = quality.get("missing_values", {})
    missing_pct = quality.get("missing_percentage", {})

    lines.append(_h3("Missing Values"))
    if missing:
        rows = [
            (col, f"{cnt:,}  ({missing_pct.get(col, 0):.1f}%)")
            for col, cnt in missing.items()
        ]
        lines.append(_kv_table(rows, ("Column", "Missing (count / %)")))
    else:
        lines.append("No missing values found.\n")

    # Empty columns
    empty_cols = quality.get("empty_columns", [])
    lines.append(_h3("Empty Columns (100% missing)"))
    lines.append(
        _bullet(empty_cols) if empty_cols else "No empty columns.\n"
    )

    # Constant columns
    const_cols = quality.get("constant_columns", [])
    lines.append(_h3("Constant Columns"))
    lines.append(
        _bullet(const_cols) if const_cols else "No constant columns.\n"
    )

    # High cardinality
    hc_cols = quality.get("high_cardinality_columns", [])
    lines.append(_h3("High-Cardinality Columns"))
    lines.append(
        _bullet(hc_cols) if hc_cols else "No high-cardinality columns.\n"
    )

    # Mixed-type columns
    mixed = quality.get("mixed_type_columns", [])
    lines.append(_h3("Mixed-Type Columns"))
    if mixed:
        rows = [
            (m["column"],
             f"numeric {m['numeric_ratio']*100:.1f}% / "
             f"non-numeric {m['non_numeric_ratio']*100:.1f}%")
            for m in mixed
        ]
        lines.append(_kv_table(rows, ("Column", "Type Mix")))
    else:
        lines.append("No mixed-type columns detected.\n")

    # Negative / zero values
    neg = quality.get("negative_values", {})
    lines.append(_h3("Negative Values"))
    if neg:
        lines.append(_kv_table(
            [(col, f"{cnt:,}") for col, cnt in neg.items()],
            ("Column", "Count")
        ))
    else:
        lines.append("No negative values.\n")

    zero = quality.get("zero_values", {})
    lines.append(_h3("Zero Values"))
    if zero:
        lines.append(_kv_table(
            [(col, f"{cnt:,}") for col, cnt in zero.items()],
            ("Column", "Count")
        ))
    else:
        lines.append("No zero values.\n")

    return "".join(lines)


def _section_eda(eda_results: dict) -> str:
    lines = [_h2("3. Exploratory Data Analysis")]

    # Numerical
    lines.append(_h3("Numerical Statistics"))
    numerical = eda_results.get("numerical", {})
    if numerical:
        for col, stats in numerical.items():
            lines.append(f"**{col}**\n")
            rows = [(k, _safe(v)) for k, v in stats.items()]
            lines.append(_kv_table(rows))
    else:
        lines.append("No numerical columns.\n")

    # Categorical
    lines.append(_h3("Categorical Summary"))
    categorical = eda_results.get("categorical", {})
    if categorical:
        for col, stats in categorical.items():
            lines.append(f"**{col}**\n")
            rows = [(k, _safe(v)) for k, v in stats.items()]
            lines.append(_kv_table(rows))
    else:
        lines.append("No categorical columns.\n")

    # Correlations
    lines.append(_h3("Top Correlations"))
    correlations = eda_results.get("correlations", [])
    if correlations:
        rows = [
            (f"{c['column_1']} ↔ {c['column_2']}",
             f"{c['correlation']:.4f}")
            for c in correlations[:15]
        ]
        lines.append(_kv_table(rows, ("Column Pair", "Pearson r")))
    else:
        lines.append("No correlations computed.\n")

    # Datetime
    dt = eda_results.get("datetime", {})
    if dt:
        lines.append(_h3("Datetime Analysis"))
        for col, stats in dt.items():
            lines.append(f"**{col}**\n")
            rows = [(k, _safe(v)) for k, v in stats.items()]
            lines.append(_kv_table(rows))

    return "".join(lines)


def _section_anomalies(
    anomaly_results: dict,
    anomaly_validation: dict | None,
) -> str:
    lines = [_h2("4. Anomaly Detection")]

    # IQR
    lines.append(_h3("Statistical Outliers (IQR Method)"))
    iqr = anomaly_results.get("iqr", {})
    if iqr:
        for col, res in iqr.items():
            lines.append(f"**{col}**\n")
            rows = [
                ("Anomaly count", f"{res.get('anomaly_count', 0):,}"),
                ("Anomaly %", f"{res.get('anomaly_percentage', 0):.2f}%"),
                ("Normal range",
                 f"{res.get('lower_bound', 'N/A')} → {res.get('upper_bound', 'N/A')}"),
            ]
            lines.append(_kv_table(rows))
    else:
        lines.append("No IQR results available.\n")

    # Isolation Forest
    lines.append(_h3("ML-Based Anomalies (Isolation Forest)"))
    ml = anomaly_results.get("multivariate_isolation_forest", {})
    if ml.get("status") == "skipped":
        lines.append(f"Skipped: {ml.get('reason', 'unknown reason')}\n")
    elif ml.get("status") == "completed":
        rows = [
            ("Anomaly count",   f"{ml.get('anomaly_count', 0):,}"),
            ("Anomaly %",       f"{ml.get('anomaly_percentage', 0):.2f}%"),
            ("Rows analyzed",   f"{ml.get('rows_analyzed', 0):,}"),
            ("Features used",   ", ".join(ml.get("features_used", []))),
        ]
        lines.append(_kv_table(rows))
    else:
        lines.append("Isolation Forest results not available.\n")

    # Anomaly validation against target
    if anomaly_validation:
        lines.append(_h3("Anomaly Validation Against Target"))
        rows = [
            ("Detected anomalies",   f"{anomaly_validation.get('anomaly_count', 0):,}"),
            ("Confirmed positives",  f"{anomaly_validation.get('confirmed_positive', 0):,}"),
            ("Precision",            f"{anomaly_validation.get('precision', 'N/A')}%"),
            ("Recall",               f"{anomaly_validation.get('recall', 'N/A')}%"),
        ]
        lines.append(_kv_table(rows))

    return "".join(lines)


def _section_supervised(supervised_validation: dict | None) -> str:
    if not supervised_validation:
        return ""

    lines = [_h2("5. Supervised Model Validation")]

    if supervised_validation.get("status") != "completed":
        reason = supervised_validation.get("reason", "Validation did not complete.")
        lines.append(f"{reason}\n")
        return "".join(lines)

    rows = [
        ("Precision",  f"{supervised_validation.get('precision', 'N/A')}%"),
        ("Recall",     f"{supervised_validation.get('recall', 'N/A')}%"),
        ("F1 Score",   f"{supervised_validation.get('f1_score', 'N/A')}%"),
        ("ROC-AUC",    f"{supervised_validation.get('roc_auc', 'N/A')}%"
                       if supervised_validation.get('roc_auc') is not None else "N/A"),
        ("PR-AUC",     f"{supervised_validation.get('pr_auc', 'N/A')}%"
                       if supervised_validation.get('pr_auc') is not None else "N/A"),
    ]
    lines.append(_kv_table(rows))

    # Confusion matrix
    cm = supervised_validation.get("confusion_matrix")
    if cm and len(cm) == 2:
        lines.append(_h3("Confusion Matrix"))
        lines.append("```\n")
        lines.append(
            f"                Predicted 0   Predicted 1\n"
            f"  Actual 0      {cm[0][0]:<12}  {cm[0][1]:<12}\n"
            f"  Actual 1      {cm[1][0]:<12}  {cm[1][1]:<12}\n"
        )
        lines.append("```\n")

    return "".join(lines)


def _section_insights(insights: dict) -> str:
    lines = [_h2("6. Automated Insights")]

    category_titles = {
        "dataset":      "Dataset",
        "quality":      "Data Quality",
        "numerical":    "Numerical",
        "categorical":  "Categorical",
        "correlations": "Correlations",
        "anomalies":    "Anomalies",
    }

    for key, title in category_titles.items():
        items = insights.get(key, [])
        if items:
            lines.append(_h3(title))
            lines.append(_bullet(items))

    if not any(insights.get(k) for k in category_titles):
        lines.append("No significant insights generated.\n")

    return "".join(lines)


def _section_ai_analysis(analysis: str | None) -> str:
    lines = [_h2("7. AI Data Analyst Report")]
    if analysis and not analysis.startswith("⚠️"):
        lines.append(analysis)
        lines.append("\n")
    else:
        lines.append(
            "_AI analysis not yet generated or was unavailable. "
            "Click 'Generate AI Analysis' in the app to include it here._\n"
        )
    return "".join(lines)


# ============================================================
# PUBLIC API
# ============================================================

def build_report(
    profile: dict,
    context: dict,
    quality: dict,
    eda_results: dict,
    anomaly_results: dict,
    insights: dict,
    anomaly_validation: dict | None = None,
    supervised_validation: dict | None = None,
    ai_analysis: str | None = None,
) -> str:
    """
    Assemble a complete Markdown EDA report from all analysis outputs.

    Args:
        profile:               Output of profile_dataset().
        context:               Output of analyze_dataset_context().
        quality:               Output of check_data_quality().
        eda_results:           Output of perform_eda().
        anomaly_results:       Output of detect_anomalies().
        insights:              Output of generate_insights().
        anomaly_validation:    Output of validate_anomalies_against_target()
                               (optional, may be None).
        supervised_validation: Output of validate_supervised_model()
                               (optional, may be None).
        ai_analysis:           The raw text from generate_analysis()
                               (optional, may be None if not yet generated).

    Returns:
        A Markdown-formatted string ready for download or display.
    """
    sections = [
        _section_header(profile, context),
        _hr(),
        _section_quality(quality),
        _hr(),
        _section_eda(eda_results),
        _hr(),
        _section_anomalies(anomaly_results, anomaly_validation),
    ]

    sv = _section_supervised(supervised_validation)
    if sv:
        sections += [_hr(), sv]

    sections += [
        _hr(),
        _section_insights(insights),
        _hr(),
        _section_ai_analysis(ai_analysis),
    ]

    return "\n".join(sections)
