"""
pytest test suite for src/report_builder.py

Run from project root:
    python -m pytest tests/test_report_builder.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.report_builder import (
    build_report,
    _section_header,
    _section_quality,
    _section_eda,
    _section_anomalies,
    _section_supervised,
    _section_insights,
    _section_ai_analysis,
    _kv_table,
)


# ============================================================
# Shared minimal fixtures
# ============================================================

@pytest.fixture
def profile():
    return {
        "rows": 1000,
        "columns": 5,
        "numeric_columns": ["Sales", "Profit"],
        "categorical_columns": ["Region"],
        "datetime_columns": [],
        "duplicate_rows": 2,
    }


@pytest.fixture
def context():
    return {
        "problem_type": "binary_classification",
        "target_column": "is_fraud",
        "prediction_columns": [],
    }


@pytest.fixture
def quality():
    return {
        "missing_values": {"Profit": 12},
        "missing_percentage": {"Profit": 1.2},
        "empty_columns": [],
        "constant_columns": [],
        "high_cardinality_columns": [],
        "mixed_type_columns": [],
        "negative_values": {"Profit": 3},
        "zero_values": {"Sales": 10},
        "unique_values": {"Sales": 980, "Profit": 950, "Region": 4},
        "data_types": {"Sales": "float64", "Profit": "float64", "Region": "object"},
        "duplicate_rows": 2,
    }


@pytest.fixture
def eda_results():
    return {
        "numerical": {
            "Sales": {
                "mean": 250.5, "median": 220.0, "std": 95.3,
                "min": 10.0, "max": 1200.0, "skewness": 1.4
            }
        },
        "categorical": {
            "Region": {
                "unique_values": 4, "top_value": "North",
                "top_value_percentage": 42.0
            }
        },
        "correlations": [
            {"column_1": "Sales", "column_2": "Profit", "correlation": 0.87}
        ],
        "datetime": {},
    }


@pytest.fixture
def anomaly_results():
    return {
        "iqr": {
            "Sales": {
                "anomaly_count": 23,
                "anomaly_percentage": 2.3,
                "lower_bound": -72.0,
                "upper_bound": 542.0,
                "anomalous_values": [600.0, 700.0],
            }
        },
        "multivariate_isolation_forest": {
            "status": "completed",
            "anomaly_count": 50,
            "anomaly_percentage": 5.0,
            "rows_analyzed": 1000,
            "features_used": ["Sales", "Profit"],
            "anomaly_indices": [1, 2, 3],
        },
    }


@pytest.fixture
def insights():
    return {
        "dataset": ["The dataset contains 1,000 rows."],
        "quality": ["Profit has 12 missing values."],
        "numerical": ["Sales is right-skewed."],
        "categorical": ["Region has 4 unique values."],
        "correlations": ["Sales and Profit are strongly correlated (0.87)."],
        "anomalies": ["50 anomalous rows detected by Isolation Forest."],
    }


@pytest.fixture
def supervised_validation():
    return {
        "status": "completed",
        "precision": 91.5,
        "recall": 88.3,
        "f1_score": 89.9,
        "roc_auc": 95.2,
        "pr_auc": 93.1,
        "confusion_matrix": [[900, 50], [30, 20]],
    }


# ============================================================
# _kv_table helper
# ============================================================

class TestKvTable:

    def test_outputs_markdown_table(self):
        result = _kv_table([("Rows", "1,000"), ("Cols", "5")])
        assert "|" in result
        assert "Rows" in result
        assert "1,000" in result

    def test_custom_headers(self):
        result = _kv_table([("A", "B")], headers=("Left", "Right"))
        assert "Left" in result
        assert "Right" in result


# ============================================================
# _section_header
# ============================================================

class TestSectionHeader:

    def test_contains_title(self, profile, context):
        result = _section_header(profile, context)
        assert "AI Data Analyst" in result

    def test_contains_row_count(self, profile, context):
        result = _section_header(profile, context)
        assert "1,000" in result

    def test_contains_column_names(self, profile, context):
        result = _section_header(profile, context)
        assert "Sales" in result
        assert "Region" in result

    def test_contains_problem_type(self, profile, context):
        result = _section_header(profile, context)
        assert "binary_classification" in result

    def test_contains_target_column(self, profile, context):
        result = _section_header(profile, context)
        assert "is_fraud" in result

    def test_generated_at_timestamp(self, profile, context):
        result = _section_header(profile, context)
        assert "Generated" in result


# ============================================================
# _section_quality
# ============================================================

class TestSectionQuality:

    def test_contains_missing_values(self, quality):
        result = _section_quality(quality)
        assert "Profit" in result
        assert "12" in result

    def test_missing_percentage_shown(self, quality):
        result = _section_quality(quality)
        assert "1.2" in result

    def test_negative_values_shown(self, quality):
        result = _section_quality(quality)
        assert "Negative" in result

    def test_zero_values_shown(self, quality):
        result = _section_quality(quality)
        assert "Zero" in result

    def test_no_missing_message(self):
        q = {
            "missing_values": {}, "missing_percentage": {},
            "empty_columns": [], "constant_columns": [],
            "high_cardinality_columns": [], "mixed_type_columns": [],
            "negative_values": {}, "zero_values": {},
        }
        result = _section_quality(q)
        assert "No missing values" in result

    def test_mixed_type_section_present(self, quality):
        result = _section_quality(quality)
        assert "Mixed-Type" in result

    def test_empty_columns_section_present(self, quality):
        result = _section_quality(quality)
        assert "Empty Columns" in result


# ============================================================
# _section_eda
# ============================================================

class TestSectionEda:

    def test_contains_numerical_stats(self, eda_results):
        result = _section_eda(eda_results)
        assert "Sales" in result
        assert "250.5" in result

    def test_contains_categorical_summary(self, eda_results):
        result = _section_eda(eda_results)
        assert "Region" in result
        assert "North" in result

    def test_contains_correlations(self, eda_results):
        result = _section_eda(eda_results)
        assert "0.8700" in result or "0.87" in result

    def test_no_numerical_message(self):
        result = _section_eda({"numerical": {}, "categorical": {}, "correlations": [], "datetime": {}})
        assert "No numerical" in result

    def test_no_categorical_message(self):
        result = _section_eda({"numerical": {}, "categorical": {}, "correlations": [], "datetime": {}})
        assert "No categorical" in result


# ============================================================
# _section_anomalies
# ============================================================

class TestSectionAnomalies:

    def test_contains_iqr_results(self, anomaly_results):
        result = _section_anomalies(anomaly_results, None)
        assert "Sales" in result
        assert "23" in result

    def test_contains_isolation_forest_results(self, anomaly_results):
        result = _section_anomalies(anomaly_results, None)
        assert "50" in result
        assert "Isolation Forest" in result

    def test_skipped_isolation_forest(self):
        ar = {
            "iqr": {},
            "multivariate_isolation_forest": {
                "status": "skipped",
                "reason": "Not enough numeric columns",
            },
        }
        result = _section_anomalies(ar, None)
        assert "Skipped" in result

    def test_validation_included_when_provided(self, anomaly_results):
        validation = {
            "anomaly_count": 50,
            "confirmed_positive": 20,
            "precision": 40.0,
            "recall": 66.7,
        }
        result = _section_anomalies(anomaly_results, validation)
        assert "Validation" in result
        assert "40.0" in result


# ============================================================
# _section_supervised
# ============================================================

class TestSectionSupervised:

    def test_returns_empty_string_when_none(self):
        result = _section_supervised(None)
        assert result == ""

    def test_contains_metrics(self, supervised_validation):
        result = _section_supervised(supervised_validation)
        assert "91.5" in result
        assert "88.3" in result
        assert "95.2" in result

    def test_contains_confusion_matrix(self, supervised_validation):
        result = _section_supervised(supervised_validation)
        assert "Confusion Matrix" in result
        assert "900" in result

    def test_incomplete_status_message(self):
        sv = {"status": "failed", "reason": "Target column not found."}
        result = _section_supervised(sv)
        assert "Target column not found" in result


# ============================================================
# _section_insights
# ============================================================

class TestSectionInsights:

    def test_contains_all_categories(self, insights):
        result = _section_insights(insights)
        assert "Dataset" in result
        assert "Quality" in result
        assert "Numerical" in result
        assert "Categorical" in result
        assert "Correlations" in result
        assert "Anomalies" in result

    def test_contains_insight_text(self, insights):
        result = _section_insights(insights)
        assert "right-skewed" in result
        assert "1,000 rows" in result

    def test_empty_insights_message(self):
        result = _section_insights({})
        assert "No significant" in result


# ============================================================
# _section_ai_analysis
# ============================================================

class TestSectionAiAnalysis:

    def test_includes_analysis_text(self):
        result = _section_ai_analysis("## Findings\nSales is high.")
        assert "Sales is high" in result

    def test_placeholder_when_none(self):
        result = _section_ai_analysis(None)
        assert "not yet generated" in result.lower() or "unavailable" in result.lower()

    def test_placeholder_when_error_message(self):
        result = _section_ai_analysis("⚠️ API error occurred.")
        assert "not yet generated" in result.lower() or "unavailable" in result.lower()


# ============================================================
# build_report (integration)
# ============================================================

class TestBuildReport:

    def test_returns_string(
        self, profile, context, quality, eda_results,
        anomaly_results, insights
    ):
        result = build_report(
            profile=profile,
            context=context,
            quality=quality,
            eda_results=eda_results,
            anomaly_results=anomaly_results,
            insights=insights,
        )
        assert isinstance(result, str)
        assert len(result) > 500

    def test_contains_all_sections(
        self, profile, context, quality, eda_results,
        anomaly_results, insights, supervised_validation
    ):
        result = build_report(
            profile=profile,
            context=context,
            quality=quality,
            eda_results=eda_results,
            anomaly_results=anomaly_results,
            insights=insights,
            supervised_validation=supervised_validation,
            ai_analysis="Great dataset!",
        )
        assert "Dataset Summary" in result
        assert "Data Quality" in result
        assert "Exploratory Data Analysis" in result
        assert "Anomaly Detection" in result
        assert "Supervised Model Validation" in result
        assert "Automated Insights" in result
        assert "AI Data Analyst Report" in result
        assert "Great dataset!" in result

    def test_works_without_optional_sections(
        self, profile, context, quality, eda_results,
        anomaly_results, insights
    ):
        # No supervised_validation, no anomaly_validation, no ai_analysis
        result = build_report(
            profile=profile,
            context=context,
            quality=quality,
            eda_results=eda_results,
            anomaly_results=anomaly_results,
            insights=insights,
        )
        assert "Dataset Summary" in result
        # Supervised section should be absent
        assert "Supervised Model Validation" not in result

    def test_report_is_valid_markdown(
        self, profile, context, quality, eda_results,
        anomaly_results, insights
    ):
        result = build_report(
            profile=profile,
            context=context,
            quality=quality,
            eda_results=eda_results,
            anomaly_results=anomaly_results,
            insights=insights,
        )
        # Should have markdown headers
        assert "# " in result
        assert "## " in result
        # Should have markdown table pipes
        assert "|" in result

    def test_report_encodable_as_utf8(
        self, profile, context, quality, eda_results,
        anomaly_results, insights
    ):
        result = build_report(
            profile=profile,
            context=context,
            quality=quality,
            eda_results=eda_results,
            anomaly_results=anomaly_results,
            insights=insights,
            ai_analysis="Analysis with unicode: café, naïve, résumé",
        )
        # Must encode to UTF-8 without errors (matches download_button usage)
        encoded = result.encode("utf-8")
        assert len(encoded) > 0
        decoded = encoded.decode("utf-8")
        assert "café" in decoded
