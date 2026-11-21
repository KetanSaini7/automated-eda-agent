import streamlit as st
import pandas as pd

# Page config must be the first Streamlit call in the script.
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.profiler import profile_dataset
from src.data_quality import check_data_quality
from src.eda import perform_eda
from src.visualization import (
    create_numeric_histogram,
    create_categorical_bar,
    create_correlation_heatmap,
    create_time_series,
)
from src.data_cleaning import standardize_categorical_values
from src.anomaly import detect_anomalies
from src.insight_engine import generate_insights
from src.context_engine import (
    analyze_dataset_context,
    validate_anomalies_against_target
)
from src.supervised_validation import (
    validate_supervised_model
)
from src.evidence_engine import (
    build_evidence,
    create_llm_evidence
)
from src.llm_agent import (
    build_analysis_prompt,
    generate_analysis
)
from src.report_builder import build_report
from src.ask_your_data import answer_question


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

MAX_PLOT_ROWS = 50_000
MAX_ISOLATION_ROWS = 200_000


# ============================================================
# CACHED ANALYSIS FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def get_profile(df):
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def get_quality(df):
    return check_data_quality(df)


@st.cache_data(show_spinner=False)
def get_context(df):
    return analyze_dataset_context(df)


@st.cache_data(show_spinner=False)
def clean_data(df):
    return standardize_categorical_values(df)


@st.cache_data(show_spinner=False)
def get_eda(cleaned_df, datetime_columns):
    return perform_eda(
        cleaned_df,
        datetime_columns
    )


@st.cache_data(show_spinner=False)
def get_anomalies(anomaly_df):
    return detect_anomalies(anomaly_df)


@st.cache_data(show_spinner=False)
def get_supervised_validation(
    df,
    target_column,
    prediction_column,
    probability_column
):
    return validate_supervised_model(
        df,
        target_column,
        prediction_column,
        probability_column
    )


@st.cache_data(show_spinner=False)
def get_aqd_schema(cleaned_df, profile_tuple, eda_results_json):
    """
    Build the compact AYD schema once per (cleaned_df, profile, eda_results)
    combination and cache it.  Avoids rebuilding on every Streamlit rerun
    while the same file is loaded.

    profile_tuple and eda_results_json are passed as hashable types so
    st.cache_data can detect changes.
    """
    from src.query_engine import build_schema
    import json
    profile = dict(profile_tuple)
    eda_results = json.loads(eda_results_json) if eda_results_json else None
    return build_schema(cleaned_df, profile, eda_results)


# ============================================================
# APP
# ============================================================

st.title("🤖 AI Data Analyst")

st.write(
    "Upload a CSV file and let the AI Data Analyst inspect it."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # ========================================================
    # LOAD DATA
    # ========================================================

    with st.spinner("Loading dataset..."):
        # ============================================================
        # LOAD CSV WITH ENCODING FALLBACK
        # ============================================================

        def load_csv(uploaded_file):
            encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=encoding)
                except UnicodeDecodeError:
                    continue

            raise ValueError(
                "Unable to decode this CSV file. "
                "Please save the file as UTF-8 CSV and try again."
            )

        try:
            df = load_csv(uploaded_file)
        except ValueError as _csv_err:
            st.error(f"⚠️ Could not load file: {_csv_err}")
            st.stop()
        except Exception as _csv_err:
            st.error(
                f"⚠️ Unexpected error while reading the CSV: {_csv_err}"
            )
            st.stop()

    # Guard: empty file
    if df.empty or len(df.columns) == 0:
        st.error(
            "⚠️ The uploaded file appears to be empty or has no columns. "
            "Please upload a valid CSV file."
        )
        st.stop()

    # Guard: header-only file (0 data rows)
    if len(df) == 0:
        st.warning(
            "⚠️ The uploaded file has column headers but no data rows."
        )
        st.stop()

    st.success(
        f"Dataset uploaded successfully! "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )


    # ========================================================
    # LARGE DATASET HANDLING
    # ========================================================

    is_large_dataset = len(df) > MAX_PLOT_ROWS

    if is_large_dataset:

        plot_df = df.sample(
            n=MAX_PLOT_ROWS,
            random_state=42
        )

        st.info(
            f"Large dataset detected ({len(df):,} rows). "
            f"Visualizations will use a representative sample "
            f"of {MAX_PLOT_ROWS:,} rows to improve performance."
        )

    else:

        plot_df = df


    # ========================================================
    # DATASET CONTEXT
    # ========================================================

    with st.spinner("Detecting dataset context..."):
        context = get_context(df)


    # ========================================================
    # DATASET PROFILE
    # ========================================================

    with st.spinner("Profiling dataset..."):
        profile = get_profile(df)


    st.header("📊 Dataset Overview")

    st.write(context)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            f"{profile['rows']:,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{profile['columns']:,}"
        )

    st.subheader("Column Types")

    # Build a compact, readable column-type table from profiling results
    _col_type_rows = []
    for _col in df.columns:
        if _col in profile["numeric_columns"]:
            _kind = "Numeric"
        elif _col in profile["datetime_columns"]:
            _kind = "Datetime"
        elif _col in profile["categorical_columns"]:
            _kind = "Categorical"
        else:
            _kind = "Other"
        _col_type_rows.append({
            "Column": _col,
            "Classification": _kind,
            "dtype": str(df[_col].dtype),
        })
    st.dataframe(
        pd.DataFrame(_col_type_rows),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    with st.spinner("Checking data quality..."):
        quality = get_quality(df)


    st.header("🔍 Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Duplicate Rows",
            f"{quality['duplicate_rows']:,}"
        )

    with col2:
        st.metric(
            "Columns with Missing Values",
            len(quality["missing_values"])
        )


    # ----------------------------------------------------------
    # Missing Values
    # ----------------------------------------------------------
    st.subheader("Missing Values")

    if quality["missing_values"]:

        missing_df = pd.DataFrame({
            "Column": list(quality["missing_values"].keys()),
            "Missing Count": list(quality["missing_values"].values()),
            "Missing %": [
                quality["missing_percentage"].get(col, 0.0)
                for col in quality["missing_values"]
            ],
        })

        st.dataframe(missing_df, use_container_width=True)

    else:

        st.success("No missing values found.")


    # ----------------------------------------------------------
    # Empty Columns (all NaN)
    # ----------------------------------------------------------
    st.subheader("Empty Columns")

    if quality.get("empty_columns"):

        st.error(
            f"The following columns contain **no data at all** "
            f"(100% missing): "
            + ", ".join(f"`{c}`" for c in quality["empty_columns"])
        )

    else:

        st.success("No empty columns found.")


    # ----------------------------------------------------------
    # Constant Columns
    # ----------------------------------------------------------
    st.subheader("Constant Columns")

    if quality["constant_columns"]:

        st.warning(
            "The following columns have only one unique value and carry "
            "no analytical information: "
            + ", ".join(f"`{c}`" for c in quality["constant_columns"])
        )

    else:

        st.success("No constant columns found.")


    # ----------------------------------------------------------
    # High Cardinality Columns
    # ----------------------------------------------------------
    st.subheader("High Cardinality Columns")

    if quality["high_cardinality_columns"]:

        st.warning(
            "The following categorical columns have more than 50% unique "
            "values and may be identifier-like fields: "
            + ", ".join(
                f"`{c}`" for c in quality["high_cardinality_columns"]
            )
        )

    else:

        st.success("No high-cardinality columns found.")


    # ----------------------------------------------------------
    # Mixed-Type Columns
    # ----------------------------------------------------------
    st.subheader("Mixed-Type Columns")

    mixed = quality.get("mixed_type_columns", [])

    if mixed:

        mixed_df = pd.DataFrame(mixed).rename(columns={
            "column": "Column",
            "numeric_ratio": "Numeric %",
            "non_numeric_ratio": "Non-Numeric %",
        })

        mixed_df["Numeric %"] = (
            mixed_df["Numeric %"] * 100
        ).round(1).astype(str) + "%"

        mixed_df["Non-Numeric %"] = (
            mixed_df["Non-Numeric %"] * 100
        ).round(1).astype(str) + "%"

        st.warning(
            "The following object columns contain a mix of numeric and "
            "non-numeric values. This may indicate data entry errors or "
            "encoding issues."
        )

        st.dataframe(mixed_df, use_container_width=True)

    else:

        st.success("No mixed-type columns detected.")


    # ----------------------------------------------------------
    # Negative Values
    # ----------------------------------------------------------
    st.subheader("Negative Values")

    if quality["negative_values"]:

        neg_df = pd.DataFrame({
            "Column": list(quality["negative_values"].keys()),
            "Negative Count": list(quality["negative_values"].values()),
        })

        st.warning(
            "The following numeric columns contain negative values "
            "(may be expected or may indicate data issues):"
        )

        st.dataframe(neg_df, use_container_width=True)

    else:

        st.success("No negative values found.")


    # ----------------------------------------------------------
    # Zero Values
    # ----------------------------------------------------------
    st.subheader("Zero Values")

    if quality["zero_values"]:

        zero_df = pd.DataFrame({
            "Column": list(quality["zero_values"].keys()),
            "Zero Count": list(quality["zero_values"].values()),
        })

        st.dataframe(zero_df, use_container_width=True)

    else:

        st.success("No zero values found.")


    # ========================================================
    # STANDARDIZE CATEGORICAL VALUES
    # ========================================================

    with st.spinner(
        "Standardizing categorical values..."
    ):

        cleaned_df = clean_data(df)


    # ========================================================
    # AUTOMATED EDA
    # ========================================================

    with st.spinner(
        "Running automated EDA..."
    ):

        eda_results = get_eda(
            cleaned_df,
            profile["datetime_columns"]
        )


    st.header("📊 Automated EDA")


    # ========================================================
    # NUMERICAL ANALYSIS
    # ========================================================

    st.subheader("📈 Numerical Analysis")

    if eda_results["numerical"]:

        numerical_df = pd.DataFrame(
            eda_results["numerical"]
        ).T

        st.dataframe(
            numerical_df,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


    # ========================================================
    # CATEGORICAL ANALYSIS
    # ========================================================

    st.subheader("📊 Categorical Analysis")

    if eda_results["categorical"]:

        categorical_df = pd.DataFrame(
            eda_results["categorical"]
        ).T

        st.dataframe(
            categorical_df,
            use_container_width=True
        )

    else:

        st.info(
            "No categorical columns found."
        )


    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    st.subheader("🔗 Correlation Analysis")

    if eda_results["correlations"]:

        correlation_df = pd.DataFrame(
            eda_results["correlations"]
        )

        st.dataframe(
            correlation_df,
            use_container_width=True
        )

    else:

        st.info(
            "Not enough numerical columns "
            "for correlation analysis."
        )


    # ========================================================
    # DATETIME ANALYSIS
    # ========================================================

    st.subheader("📅 Date Analysis")

    if eda_results["datetime"]:

        datetime_df = pd.DataFrame(
            eda_results["datetime"]
        ).T

        st.dataframe(
            datetime_df,
            use_container_width=True
        )

    else:

        st.info(
            "No datetime columns found."
        )


    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    st.header("⚠️ Anomaly Detection")


    # --------------------------------------------------------
    # Decide dataset for Isolation Forest
    # --------------------------------------------------------

    if len(cleaned_df) > MAX_ISOLATION_ROWS:

        anomaly_df = cleaned_df.sample(
            n=MAX_ISOLATION_ROWS,
            random_state=42
        )

        st.info(
            f"Large dataset detected. "
            f"Isolation Forest will analyze a representative "
            f"sample of {MAX_ISOLATION_ROWS:,} rows instead of "
            f"all {len(cleaned_df):,} rows."
        )

    else:

        anomaly_df = cleaned_df


    with st.spinner(
        "Running anomaly detection..."
    ):

        anomaly_results = get_anomalies(
            anomaly_df
        )


    iqr_results = anomaly_results["iqr"]

    ml_result = anomaly_results[
        "multivariate_isolation_forest"
    ]


    # ========================================================
    # IQR RESULTS
    # ========================================================

    st.subheader(
        "📊 Statistical Outliers (IQR)"
    )

    if not iqr_results:

        st.info(
            "No numerical columns available "
            "for anomaly detection."
        )

    else:

        for column, result in iqr_results.items():

            st.markdown(
                f"### {column}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Anomalies",
                    f"{result['anomaly_count']:,}"
                )

            with col2:

                st.metric(
                    "Anomaly %",
                    f"{result['anomaly_percentage']}%"
                )

            with col3:

                st.write(
                    f"Normal Range: "
                    f"{result['lower_bound']} "
                    f"→ "
                    f"{result['upper_bound']}"
                )


            if result["anomaly_count"] > 0:

                st.warning(
                    f"{result['anomaly_count']:,} "
                    f"potentially unusual value(s) "
                    f"detected in **{column}**."
                )


                with st.expander(
                    "🔍 View Anomaly Details"
                ):

                    anomalous_values = (
                        result["anomalous_values"]
                    )


                    if anomalous_values:

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Minimum Anomaly",
                                f"{min(anomalous_values):,.2f}"
                            )

                        with col2:

                            st.metric(
                                "Maximum Anomaly",
                                f"{max(anomalous_values):,.2f}"
                            )

                        with col3:

                            st.metric(
                                "Average Anomaly",
                                f"{sum(anomalous_values) / len(anomalous_values):,.2f}"
                            )


                        PREVIEW_SIZE = 10

                        st.write(
                            "### Anomaly Preview"
                        )


                        preview_df = pd.DataFrame(
                            {
                                column:
                                anomalous_values[
                                    :PREVIEW_SIZE
                                ]
                            }
                        )


                        st.dataframe(
                            preview_df,
                            use_container_width=True
                        )


                        if len(anomalous_values) > PREVIEW_SIZE:

                            st.caption(
                                f"Showing {PREVIEW_SIZE:,} "
                                f"of {len(anomalous_values):,} "
                                f"anomalies."
                            )

            else:

                st.success(
                    f"No unusual values detected "
                    f"in **{column}**."
                )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    st.subheader(
        "🤖 ML-Based Anomalies"
    )


    anomaly_validation = None


    if (
        ml_result["status"] == "completed"
        and context["target_column"] is not None
    ):

        anomaly_validation = (
            validate_anomalies_against_target(
                anomaly_df,
                ml_result["anomaly_indices"],
                context["target_column"]
            )
        )


    if ml_result["status"] == "skipped":

        st.info(
            f"Isolation Forest skipped: "
            f"{ml_result['reason']}"
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Anomalies",
                f"{ml_result['anomaly_count']:,}"
            )

        with col2:

            st.metric(
                "Anomaly %",
                f"{ml_result['anomaly_percentage']}%"
            )

        with col3:

            st.metric(
                "Rows Analyzed",
                f"{ml_result['rows_analyzed']:,}"
            )


        if ml_result["anomaly_count"] > 0:

            st.warning(
                f"{ml_result['anomaly_count']:,} "
                f"potentially unusual row(s) detected."
            )


            with st.expander(
                "🔍 View ML Anomalies"
            ):

                anomaly_indices = (
                    ml_result["anomaly_indices"]
                )


                anomaly_rows = anomaly_df.loc[
                    anomaly_indices
                ]


                PREVIEW_SIZE = 10

                preview_anomaly_rows = (
                    anomaly_rows.head(
                        PREVIEW_SIZE
                    )
                )


                st.dataframe(
                    preview_anomaly_rows,
                    use_container_width=True
                )


                if len(anomaly_rows) > PREVIEW_SIZE:

                    st.caption(
                        f"Showing {PREVIEW_SIZE:,} "
                        f"of {len(anomaly_rows):,} "
                        f"detected anomalies."
                    )

        else:

            st.success(
                "No unusual rows detected by "
                "Isolation Forest."
            )


        # ====================================================
        # ANOMALY VALIDATION
        # ====================================================

        if anomaly_validation is not None:

            st.subheader(
                "🎯 Anomaly Validation"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Detected Anomalies",
                    f"{anomaly_validation['anomaly_count']:,}"
                )

            with col2:

                st.metric(
                    "Confirmed Fraud",
                    f"{anomaly_validation['confirmed_positive']:,}"
                )

            with col3:

                st.metric(
                    "Anomaly Precision",
                    f"{anomaly_validation['precision']}%"
                )

            with col4:

                st.metric(
                    "Anomaly Recall",
                    f"{anomaly_validation['recall']}%"
                )


    # ========================================================
    # SUPERVISED VALIDATION
    # ========================================================

    supervised_validation = None

    target_column = context[
        "target_column"
    ]

    prediction_columns = context[
        "prediction_columns"
    ]


    prediction_column = None
    probability_column = None


    for column in prediction_columns:

        column_lower = column.lower()

        if (
            "probability" in column_lower
            or "prob" in column_lower
        ):

            probability_column = column

        elif (
            "prediction" in column_lower
            or "predicted" in column_lower
        ):

            prediction_column = column


    if (
        target_column is not None
        and prediction_column is not None
    ):

        with st.spinner(
            "Validating supervised model..."
        ):

            supervised_validation = (
                get_supervised_validation(
                    df,
                    target_column,
                    prediction_column,
                    probability_column
                )
            )


    # ========================================================
    # MODEL VALIDATION UI
    # ========================================================

    if supervised_validation is not None:

        if supervised_validation["status"] == "completed":

            st.subheader(
                "🤖 Supervised Model Validation"
            )

            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )


            with col1:

                st.metric(
                    "Precision",
                    f"{supervised_validation['precision']}%"
                )


            with col2:

                st.metric(
                    "Recall",
                    f"{supervised_validation['recall']}%"
                )


            with col3:

                st.metric(
                    "F1 Score",
                    f"{supervised_validation['f1_score']}%"
                )


            with col4:

                roc_auc = (
                    supervised_validation[
                        "roc_auc"
                    ]
                )

                if roc_auc is not None:

                    st.metric(
                        "ROC-AUC",
                        f"{roc_auc}%"
                    )

                else:

                    st.metric(
                        "ROC-AUC",
                        "N/A"
                    )


            with col5:

                pr_auc = (
                    supervised_validation[
                        "pr_auc"
                    ]
                )

                if pr_auc is not None:

                    st.metric(
                        "PR-AUC",
                        f"{pr_auc}%"
                    )

                else:

                    st.metric(
                        "PR-AUC",
                        "N/A"
                    )


            st.write(
                "### Confusion Matrix"
            )


            confusion_matrix_data = (
                supervised_validation[
                    "confusion_matrix"
                ]
            )


            confusion_df = pd.DataFrame(
                confusion_matrix_data,
                index=[
                    "Actual Normal",
                    "Actual Fraud"
                ],
                columns=[
                    "Predicted Normal",
                    "Predicted Fraud"
                ]
            )


            st.dataframe(
                confusion_df,
                use_container_width=True
            )


        else:

            st.warning(
                supervised_validation["reason"]
            )


    # ========================================================
    # AUTOMATED INSIGHTS
    # ========================================================

    insights = generate_insights(
        profile=profile,
        quality=quality,
        eda_results=eda_results,
        anomaly_results=anomaly_results,
        context=context,
        anomaly_validation=anomaly_validation,
        supervised_validation=supervised_validation
    )


    st.header(
        "💡 Automated Insights"
    )


    insight_titles = {

        "dataset":
            "📋 Dataset Insights",

        "quality":
            "🔎 Data Quality Insights",

        "numerical":
            "📈 Numerical Insights",

        "categorical":
            "📊 Categorical Insights",

        "correlations":
            "🔗 Correlation Insights",

        "anomalies":
            "⚠️ Anomaly Insights"
    }


    for category, title in insight_titles.items():

        category_insights = insights.get(
            category,
            []
        )


        with st.expander(title):

            if not category_insights:

                st.info(
                    "No significant insights detected."
                )

            else:

                for insight in category_insights:

                    st.write(
                        f"• {insight}"
                    )


    # ========================================================
    # EVIDENCE LAYER
    # ========================================================

    evidence = build_evidence(
        profile=profile,
        context=context,
        quality=quality,
        eda_results=eda_results,
        anomaly_results=anomaly_results,
        anomaly_validation=anomaly_validation,
        supervised_validation=supervised_validation
    )


    llm_evidence = create_llm_evidence(
        evidence=evidence,
        insights=insights
    )


    analysis_prompt = build_analysis_prompt(
        llm_evidence
    )


    # ========================================================
    # AI DATA ANALYST
    # ========================================================

    st.header("🤖 AI Data Analyst")

    if st.button("Generate AI Analysis"):

        with st.spinner(
            "AI Data Analyst is analyzing the evidence..."
        ):
            analysis = generate_analysis(analysis_prompt)

        # Persist across reruns so the download button can include it
        st.session_state["ai_analysis"] = analysis

    # Retrieve persisted analysis (None on first load)
    ai_analysis = st.session_state.get("ai_analysis")

    if ai_analysis:
        st.markdown(ai_analysis)


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.header("📥 Download Report")

    report_md = build_report(
        profile=profile,
        context=context,
        quality=quality,
        eda_results=eda_results,
        anomaly_results=anomaly_results,
        insights=insights,
        anomaly_validation=anomaly_validation,
        supervised_validation=supervised_validation,
        ai_analysis=ai_analysis,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report_md.encode("utf-8"),
            file_name="eda_report.md",
            mime="text/markdown",
            help=(
                "Markdown format — opens in GitHub, VS Code, "
                "Obsidian, Notion, and most text editors."
            ),
        )

    with col2:
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report_md.encode("utf-8"),
            file_name="eda_report.txt",
            mime="text/plain",
            help="Plain-text format — opens in any text editor.",
        )

    if not ai_analysis:
        st.caption(
            "💡 Tip: Click **Generate AI Analysis** above first to include "
            "the full AI narrative in your downloaded report."
        )


    # ========================================================
    # LLM PROMPT PREVIEW
    # ========================================================

    st.header(
        "🤖 LLM Prompt Preview"
    )


    with st.expander(
        "View Generated Prompt"
    ):

        st.text(
            analysis_prompt
        )


    # ========================================================
    # EVIDENCE DEBUG VIEW
    # ========================================================

    st.header(
        "🧠 Evidence Layer"
    )


    with st.expander(
        "🔍 View Evidence"
    ):

        st.json(
            evidence
        )


    # ========================================================
    # AUTOMATED VISUALIZATIONS
    # ========================================================

    st.header("📊 Automated Visualizations")

    # Sampling context passed to every chart function
    _is_sample = is_large_dataset
    _total_rows = len(df)


    # ========================================================
    # NUMERICAL VISUALIZATIONS
    # ========================================================

    st.subheader("📈 Numerical Distributions")

    numeric_columns = profile["numeric_columns"]

    if not numeric_columns:
        st.info("No numerical columns found.")
    else:
        for column in numeric_columns:

            # Skip constant columns — a 1-bar histogram is meaningless
            if column in quality["constant_columns"]:
                st.caption(
                    f"⏭ Skipping **{column}** — constant column (single value)."
                )
                continue

            fig = create_numeric_histogram(
                plot_df,
                column,
                is_sample=_is_sample,
                total_rows=_total_rows
            )

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(
                    f"⏭ Skipping **{column}** — no plottable values."
                )


    # ========================================================
    # CATEGORICAL VISUALIZATIONS
    # ========================================================

    st.subheader("📊 Categorical Distributions")

    categorical_columns = profile["categorical_columns"]

    if not categorical_columns:
        st.info("No categorical columns found.")
    else:
        plotted_any_cat = False

        for column in categorical_columns:

            if cleaned_df[column].nunique() <= 50:

                fig = create_categorical_bar(
                    plot_df,
                    column,
                    is_sample=_is_sample,
                    total_rows=_total_rows
                )

                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    plotted_any_cat = True
                else:
                    st.caption(
                        f"⏭ Skipping **{column}** — no non-null values."
                    )

        if not plotted_any_cat:
            st.info(
                "No categorical columns with ≤ 50 unique values to display."
            )


    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    st.subheader("🔗 Correlation Heatmap")

    heatmap = create_correlation_heatmap(
        plot_df,
        is_sample=_is_sample,
        total_rows=_total_rows
    )

    if heatmap is not None:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info(
            "Not enough numerical columns "
            "for correlation analysis."
        )


    # ========================================================
    # TIME SERIES
    # ========================================================

    if profile["datetime_columns"]:

        st.subheader("📅 Time-Based Analysis")

        date_column = profile["datetime_columns"][0]

        for column in numeric_columns[:5]:

            fig = create_time_series(
                plot_df,
                date_column,
                column,
                is_sample=_is_sample,
                total_rows=_total_rows
            )

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(
                    f"⏭ Skipping **{column}** over time — no valid rows."
                )



    # ========================================================
    # ASK YOUR DATA
    # ========================================================

    st.divider()
    st.header("💬 Ask Your Data")
    st.caption(
        "Ask natural-language questions about your dataset. "
        "Python/Pandas computes every result — Gemini only explains."
    )

    # ---- Reset chat history when a new file is uploaded ----------------
    _current_file = uploaded_file.name if uploaded_file else None
    if st.session_state.get("_aqd_file") != _current_file:
        st.session_state["_aqd_file"] = _current_file
        st.session_state["chat_history"] = []

    # ---- Render existing conversation history --------------------------
    for _turn in st.session_state.get("chat_history", []):
        with st.chat_message(_turn["role"]):
            st.markdown(_turn["content"])
            if _turn.get("data_df") is not None:
                st.dataframe(_turn["data_df"], use_container_width=True)
            if _turn.get("chart") is not None:
                st.plotly_chart(_turn["chart"], use_container_width=True)

    # ---- Input + clear button -----------------------------------------
    _col_input, _col_clear = st.columns([5, 1])
    with _col_clear:
        if st.button("🗑️ Clear", help="Clear conversation history"):
            st.session_state["chat_history"] = []
            st.rerun()

    _question = st.chat_input("Ask a question about your data…")

    if _question:
        # Add user turn to history
        st.session_state.setdefault("chat_history", []).append(
            {"role": "user", "content": _question, "data_df": None, "chart": None}
        )
        with st.chat_message("user"):
            st.markdown(_question)

        # Retrieve optional results safely from session_state
        _anomaly_results = st.session_state.get("anomaly_results")
        _supervised_val = st.session_state.get("supervised_validation")

        # Run the pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                _response = answer_question(
                    question=_question,
                    df=cleaned_df,
                    profile=profile,
                    eda_results=eda_results,
                    anomaly_results=_anomaly_results,
                    supervised_validation=_supervised_val,
                    conversation_history=st.session_state.get("chat_history", []),
                )

            if _response.get("error") and not _response.get("answer"):
                st.error(_response["error"])
            else:
                st.markdown(_response["answer"])
                if _response.get("data_df") is not None:
                    st.dataframe(_response["data_df"], use_container_width=True)
                if _response.get("chart") is not None:
                    st.plotly_chart(_response["chart"], use_container_width=True)

        # Add assistant turn to history
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": _response.get("answer", ""),
            "data_df": _response.get("data_df"),
            "chart": _response.get("chart"),
        })
