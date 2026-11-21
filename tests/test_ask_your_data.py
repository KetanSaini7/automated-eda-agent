"""
tests/test_ask_your_data.py
===========================
Unit tests for the Ask Your Data pipeline.
All Gemini / API calls are mocked — no network access required.
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, "d:/Projects/Automated_eda_agent")

from src.query_interpreter import interpret_question, _extract_json, _compact_history
from src.ask_your_data import (
    answer_question,
    _truncate_result_for_explain,
    _result_to_dataframe,
    _build_chart,
    _format_local_answer,
    _get_overview_explanation,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "date":     dates,
        "sales":    rng.uniform(100, 5000, 60).round(2),
        "profit":   rng.uniform(-100, 800, 60).round(2),
        "region":   rng.choice(["North", "South", "East"], 60),
        "category": rng.choice(["A", "B"], 60),
    })


@pytest.fixture
def profile(sample_df):
    return {
        "rows": len(sample_df),
        "columns": len(sample_df.columns),
        "numeric_columns": ["sales", "profit"],
        "categorical_columns": ["region", "category"],
        "datetime_columns": ["date"],
        "duplicate_rows": 0,
    }


@pytest.fixture
def schema(sample_df, profile):
    from src.query_engine import build_schema
    return build_schema(sample_df, profile)


@pytest.fixture
def anomaly_results():
    return {
        "iqr": {
            "sales": {
                "anomaly_count": 3,
                "anomaly_percentage": 5.0,
                "lower_bound": 150.0,
                "upper_bound": 4800.0,
            }
        },
        "multivariate_isolation_forest": {
            "status": "completed",
            "anomaly_count": 4,
            "anomaly_percentage": 6.67,
            "rows_analyzed": 60,
            "features_used": ["sales", "profit"],
        },
    }


@pytest.fixture
def supervised_validation():
    return {
        "model": "RandomForest",
        "accuracy": 0.88,
        "precision": 0.85,
        "recall": 0.87,
        "f1": 0.86,
    }


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["sales", "region"])


# ============================================================
# HELPER: mock interpret_question to return a fixed plan
# ============================================================

def _mock_interpreter(plan_dict):
    """
    Patch interpret_question at its source module.
    answer_question imports it inside the function via deferred import,
    so we must patch the source, not the caller namespace.
    """
    return patch(
        "src.query_interpreter.interpret_question",
        return_value=plan_dict,
    )


def _mock_explain(text="Results are as expected."):
    """
    Mock _get_explanation which is called inside answer_question.
    Also mock _build_client so no real API initialization happens.
    """
    from contextlib import ExitStack
    from unittest.mock import patch, MagicMock

    class _DoubleCtx:
        """Combines two context managers so callers can use a single `with` block."""
        def __init__(self):
            self._stack = ExitStack()
        def __enter__(self):
            self._stack.__enter__()
            self._stack.enter_context(
                patch("src.ask_your_data._get_explanation", return_value=text)
            )
            self._stack.enter_context(
                patch("src.llm_agent._build_client", return_value=MagicMock())
            )
            return self
        def __exit__(self, *args):
            return self._stack.__exit__(*args)

    return _DoubleCtx()


# ============================================================
# 1. _extract_json
# ============================================================

class TestExtractJson:
    def test_plain_json(self):
        text = '{"operation": "row_count"}'
        assert _extract_json(text) == '{"operation": "row_count"}'

    def test_fenced_json(self):
        text = '```json\n{"operation": "basic_stats"}\n```'
        result = _extract_json(text)
        assert '"operation"' in result

    def test_json_with_prose(self):
        text = 'Here is the plan: {"operation": "schema_info"} done.'
        assert '"operation"' in _extract_json(text)

    def test_no_json_returns_empty(self):
        assert _extract_json("No JSON here.") == ""

    def test_empty_string(self):
        assert _extract_json("") == ""

    def test_nested_json(self):
        text = '{"operation": "group_by_aggregate", "group_by": "region"}'
        parsed = json.loads(_extract_json(text))
        assert parsed["operation"] == "group_by_aggregate"


# ============================================================
# 2. _compact_history
# ============================================================

class TestCompactHistory:
    def test_empty_history(self):
        result = _compact_history([])
        assert "no prior" in result.lower()

    def test_single_turn(self):
        history = [{"role": "user", "content": "What is the average sales?"}]
        result = _compact_history(history)
        assert "average sales" in result

    def test_max_6_turns(self):
        history = [{"role": "user", "content": f"Q{i}"} for i in range(20)]
        result = _compact_history(history)
        # Only last 6 → Q14 through Q19
        assert "Q19" in result
        assert "Q0" not in result

    def test_long_content_truncated(self):
        history = [{"role": "user", "content": "x" * 2000}]
        result = _compact_history(history)
        assert len(result) < 2000


# ============================================================
# 3. _truncate_result_for_explain
# ============================================================

class TestTruncateResultForExplain:
    def test_short_list_not_truncated(self):
        result = {
            "operation": "value_counts",
            "approximate": False,
            "row_count_used": 60,
            "data": [{"value": "A", "count": 10}] * 5,
            "error": None,
        }
        compact = _truncate_result_for_explain(result)
        assert compact["truncated_to"] is None
        assert len(compact["data"]) == 5

    def test_long_list_truncated(self):
        result = {
            "operation": "top_n",
            "approximate": False,
            "row_count_used": 60,
            "data": [{"row": i} for i in range(50)],
            "error": None,
        }
        compact = _truncate_result_for_explain(result)
        assert len(compact["data"]) == 30
        assert compact["truncated_to"] == 30

    def test_dict_data_preserved(self):
        result = {
            "operation": "basic_stats",
            "approximate": False,
            "row_count_used": 60,
            "data": {"mean": 42.0, "std": 5.0},
            "error": None,
        }
        compact = _truncate_result_for_explain(result)
        assert compact["data"]["mean"] == 42.0


# ============================================================
# 4. _result_to_dataframe
# ============================================================

class TestResultToDataframe:
    def test_list_of_dicts(self):
        result = {
            "data": [
                {"region": "North", "sum": 1000},
                {"region": "South", "sum": 800},
            ]
        }
        df = _result_to_dataframe(result)
        assert df is not None
        assert len(df) == 2
        assert "region" in df.columns

    def test_single_message_dict_returns_none(self):
        result = {"data": [{"message": "No missing values found."}]}
        df = _result_to_dataframe(result)
        assert df is None

    def test_simple_dict_flattened(self):
        result = {"data": {"mean": 42.0, "std": 5.0, "count": 100}}
        df = _result_to_dataframe(result)
        assert df is not None
        assert "Metric" in df.columns

    def test_empty_data_returns_none(self):
        assert _result_to_dataframe({"data": None}) is None
        assert _result_to_dataframe({"data": []}) is None


# ============================================================
# 5. _build_chart
# ============================================================

class TestBuildChart:
    def test_group_by_aggregate_chart(self):
        result = {
            "operation": "group_by_aggregate",
            "data": [
                {"region": "North", "sum": 1000},
                {"region": "South", "sum": 800},
                {"region": "East",  "sum": 600},
            ],
        }
        fig = _build_chart(result)
        assert fig is not None

    def test_date_group_aggregate_chart(self):
        result = {
            "operation": "date_group_aggregate",
            "data": [
                {"period": "2023-01", "sum": 5000},
                {"period": "2023-02", "sum": 6000},
                {"period": "2023-03", "sum": 4500},
            ],
        }
        fig = _build_chart(result)
        assert fig is not None

    def test_too_few_rows_no_chart(self):
        result = {
            "operation": "group_by_aggregate",
            "data": [{"region": "North", "sum": 1000}],
        }
        fig = _build_chart(result)
        assert fig is None

    def test_non_chartable_operation(self):
        result = {
            "operation": "basic_stats",
            "data": {"mean": 42.0},
        }
        fig = _build_chart(result)
        assert fig is None

    def test_value_counts_chart(self):
        result = {
            "operation": "value_counts",
            "data": [
                {"value": "A", "count": 30, "percentage": 50},
                {"value": "B", "count": 30, "percentage": 50},
            ],
        }
        fig = _build_chart(result)
        assert fig is not None


# ============================================================
# 6. answer_question — numerical
# ============================================================

class TestAnswerQuestionNumerical:
    def test_average_sales(self, sample_df, profile):
        plan = {"operation": "basic_stats", "column": "sales"}
        with _mock_interpreter(plan), _mock_explain("The average sales is ..."):
            resp = answer_question("What is the average sales?", sample_df, profile)
        assert resp["error"] is None
        assert "average sales" in resp["answer"].lower() or resp["answer"]

    def test_maximum_profit(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "profit",
            "aggregation": "max",
            "sort": "descending",
            "limit": 1,
        }
        with _mock_interpreter(plan), _mock_explain("Max profit is ..."):
            resp = answer_question("What is the maximum profit?", sample_df, profile)
        assert resp["error"] is None

    def test_basic_stats_returns_dataframe(self, sample_df, profile):
        plan = {"operation": "basic_stats", "column": "sales"}
        with _mock_interpreter(plan), _mock_explain("stats"):
            resp = answer_question("Describe sales column", sample_df, profile)
        # basic_stats returns a dict → flattened to df
        assert resp["data_df"] is not None


# ============================================================
# 7. answer_question — categorical
# ============================================================

class TestAnswerQuestionCategorical:
    def test_top_regions_by_sales(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "sort": "descending",
            "limit": 3,
        }
        with _mock_interpreter(plan), _mock_explain("Top regions are ..."):
            resp = answer_question("Show top 3 regions by sales", sample_df, profile)
        assert resp["error"] is None
        assert resp["data_df"] is not None
        assert len(resp["data_df"]) == 3

    def test_value_counts_category(self, sample_df, profile):
        plan = {"operation": "value_counts", "column": "category", "limit": 10}
        with _mock_interpreter(plan), _mock_explain("Distribution of categories ..."):
            resp = answer_question("What is the distribution of categories?", sample_df, profile)
        assert resp["error"] is None
        assert resp["chart"] is not None


# ============================================================
# 8. answer_question — date-based
# ============================================================

class TestAnswerQuestionDateBased:
    def test_monthly_revenue(self, sample_df, profile):
        plan = {
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "month",
        }
        with _mock_interpreter(plan), _mock_explain("Monthly revenue shows ..."):
            resp = answer_question("Show monthly revenue", sample_df, profile)
        assert resp["error"] is None
        assert resp["chart"] is not None


# ============================================================
# 9. answer_question — filtering
# ============================================================

class TestAnswerQuestionFiltering:
    def test_filtered_sales_gt(self, sample_df, profile):
        plan = {
            "operation": "filtered_aggregate",
            "filter_column": "sales",
            "filter_operator": ">",
            "filter_value": 2000,
            "measure_column": "profit",
            "aggregation": "mean",
        }
        with _mock_interpreter(plan), _mock_explain("Mean profit for high sales ..."):
            resp = answer_question(
                "What is the average profit for sales > 2000?", sample_df, profile
            )
        assert resp["error"] is None

    def test_filtered_region_eq(self, sample_df, profile):
        plan = {
            "operation": "filtered_aggregate",
            "filter_column": "region",
            "filter_operator": "==",
            "filter_value": "North",
            "measure_column": "sales",
            "aggregation": "sum",
        }
        with _mock_interpreter(plan), _mock_explain("North region total sales ..."):
            resp = answer_question(
                "Total sales in North region", sample_df, profile
            )
        assert resp["error"] is None


# ============================================================
# 10. answer_question — data quality
# ============================================================

class TestAnswerQuestionQuality:
    def test_missing_values(self, sample_df, profile):
        plan = {"operation": "missing_values"}
        with _mock_interpreter(plan), _mock_explain("No missing values."):
            resp = answer_question("How many missing values?", sample_df, profile)
        assert resp["error"] is None

    def test_data_quality_summary(self, sample_df, profile):
        plan = {"operation": "data_quality_summary"}
        with _mock_interpreter(plan), _mock_explain("Data quality summary ..."):
            resp = answer_question("Summarize data quality", sample_df, profile)
        assert resp["error"] is None


# ============================================================
# 11. answer_question — anomaly & model
# ============================================================

class TestAnswerQuestionAnomalyModel:
    def test_anomaly_summary(self, sample_df, profile, anomaly_results):
        plan = {"operation": "anomaly_summary"}
        with _mock_interpreter(plan), _mock_explain("7 anomalies detected."):
            resp = answer_question(
                "How many anomalies were detected?",
                sample_df, profile,
                anomaly_results=anomaly_results,
            )
        assert resp["error"] is None
        assert resp["raw_result"]["data"]["isolation_forest"]["anomaly_count"] == 4

    def test_model_summary(self, sample_df, profile, supervised_validation):
        plan = {"operation": "model_summary"}
        with _mock_interpreter(plan), _mock_explain("Model accuracy is 0.88."):
            resp = answer_question(
                "What is the model accuracy?",
                sample_df, profile,
                supervised_validation=supervised_validation,
            )
        assert resp["error"] is None
        assert resp["raw_result"]["data"]["accuracy"] == 0.88

    def test_anomaly_no_results(self, sample_df, profile):
        plan = {"operation": "anomaly_summary"}
        with _mock_interpreter(plan), _mock_explain("No anomaly data."):
            resp = answer_question(
                "Show anomalies", sample_df, profile,
                anomaly_results=None,
            )
        assert resp["error"] is None
        assert "message" in resp["raw_result"]["data"]


# ============================================================
# 12. Edge cases
# ============================================================

class TestAnswerQuestionEdgeCases:
    def test_empty_dataframe(self, empty_df, profile):
        profile2 = {**profile, "rows": 0, "columns": 2}
        resp = answer_question("Any question", empty_df, profile2)
        assert resp["error"] is not None or "No dataset" in resp["answer"]

    def test_empty_question(self, sample_df, profile):
        resp = answer_question("", sample_df, profile)
        assert resp["error"] is not None

    def test_whitespace_question(self, sample_df, profile):
        resp = answer_question("   ", sample_df, profile)
        assert resp["error"] is not None

    def test_clarification_needed(self, sample_df, profile):
        clarification = {"clarification_needed": "Which column did you mean?"}
        with _mock_interpreter(clarification):
            resp = answer_question("Show me the data", sample_df, profile)
        assert resp["error"] is None
        assert "clarification" in resp["answer"].lower() or "Which column" in resp["answer"]

    def test_unsupported_question(self, sample_df, profile):
        unsupported = {"unsupported": "Cannot create visualizations of that type."}
        with _mock_interpreter(unsupported):
            resp = answer_question("Draw a 3D scatter plot", sample_df, profile)
        assert resp["error"] is None
        assert "cannot" in resp["answer"].lower() or "not supported" in resp["answer"].lower()

    def test_invalid_column_in_plan(self, sample_df, profile):
        # Plan references a non-existent column → validation fails
        bad_plan = {"operation": "basic_stats", "column": "nonexistent_col"}
        with _mock_interpreter(bad_plan), _mock_explain():
            resp = answer_question("Average of xyz", sample_df, profile)
        assert resp["error"] is not None
        assert "not exist" in resp["error"]

    def test_non_numeric_for_numeric_op(self, sample_df, profile):
        bad_plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "category",
            "aggregation": "sum",
        }
        with _mock_interpreter(bad_plan), _mock_explain():
            resp = answer_question("Sum of categories by region", sample_df, profile)
        assert resp["error"] is not None

    def test_gemini_interpreter_error(self, sample_df, profile):
        error_plan = {"error": "Gemini API unavailable"}
        with _mock_interpreter(error_plan):
            resp = answer_question("What is the average?", sample_df, profile)
        assert resp["error"] is not None or "Could not interpret" in resp["answer"]

    def test_answer_always_has_answer_key(self, sample_df, profile):
        plan = {"operation": "row_count"}
        with _mock_interpreter(plan), _mock_explain("60 rows, 5 columns."):
            resp = answer_question("How many rows?", sample_df, profile)
        assert "answer" in resp
        assert "data_df" in resp
        assert "chart" in resp
        assert "error" in resp


# ============================================================
# 13. interpret_question — unit tests (mocked Gemini)
# ============================================================

class TestInterpretQuestion:
    def _mock_llm(self, response_text):
        """
        Mock _build_client, _call_with_retry, and the text-extraction helpers at
        their source (src.llm_agent).  interpret_question imports all of them via
        deferred 'from src.llm_agent import ...' inside the function body, so we
        patch the originals in src.llm_agent.

        _call_with_retry is mocked to return response_text directly.
        _extract_text_from_interaction is mocked to return response_text so the
        interactions-API path works in tests without a real API object.
        _extract_text_from_response is left as a no-op (fallback path not needed).
        """
        mock_client = MagicMock()
        return (
            patch("src.llm_agent._build_client", return_value=mock_client),
            patch("src.llm_agent._call_with_retry", return_value=response_text),
            patch(
                "src.llm_agent._extract_text_from_interaction",
                return_value=response_text,
            ),
        )

    def test_valid_plan_returned(self, schema):
        plan_json = '{"operation": "basic_stats", "column": "sales"}'
        p1, p2, p3 = self._mock_llm(plan_json)
        with p1, p2, p3:
            result = interpret_question("Average sales?", schema)
        assert result.get("operation") == "basic_stats"

    def test_clarification_returned(self, schema):
        clarify_json = '{"clarification_needed": "Which column?"}'
        p1, p2, p3 = self._mock_llm(clarify_json)
        with p1, p2, p3:
            result = interpret_question("Show me data", schema)
        assert "clarification_needed" in result

    def test_unsupported_returned(self, schema):
        unsupported_json = '{"unsupported": "Cannot do 3D plots."}'
        p1, p2, p3 = self._mock_llm(unsupported_json)
        with p1, p2, p3:
            result = interpret_question("3D scatter", schema)
        assert "unsupported" in result

    def test_empty_response_returns_error(self, schema):
        p1, p2, p3 = self._mock_llm("")
        with p1, p2, p3:
            result = interpret_question("Any question", schema)
        assert "error" in result

    def test_non_json_response_returns_error(self, schema):
        p1, p2, p3 = self._mock_llm("I cannot answer that.")
        with p1, p2, p3:
            result = interpret_question("What is the meaning of life?", schema)
        assert "error" in result

    def test_missing_operation_returns_error(self, schema):
        p1, p2, p3 = self._mock_llm('{"group_by": "region"}')
        with p1, p2, p3:
            result = interpret_question("Group by region", schema)
        assert "error" in result

    def test_empty_question_returns_error(self, schema):
        result = interpret_question("", schema)
        assert "error" in result

    def test_no_schema_returns_error(self):
        result = interpret_question("How many rows?", schema={})
        assert "error" in result

    def test_fenced_json_extracted(self, schema):
        fenced = '```json\n{"operation": "row_count"}\n```'
        p1, p2, p3 = self._mock_llm(fenced)
        with p1, p2, p3:
            result = interpret_question("How many rows?", schema)
        assert result.get("operation") == "row_count"

    def test_api_exception_returns_error(self, schema):
        mock_client = MagicMock()
        with patch("src.llm_agent._build_client", return_value=mock_client), \
             patch("src.llm_agent._call_with_retry", side_effect=Exception("timeout")):
            result = interpret_question("Any question", schema)
        assert "error" in result


# ============================================================
# 14. dataset_overview
# ============================================================

class TestDatasetOverview:
    """Tests for the dataset_overview intent end-to-end."""

    @pytest.fixture
    def overview_result(self, sample_df, profile):
        """A realistic dataset_overview execute_plan result."""
        from src.query_engine import execute_plan
        return execute_plan({"operation": "dataset_overview"}, sample_df, profile)

    def test_format_local_answer_returns_none_for_overview(self, overview_result):
        """dataset_overview is NOT a local-answer op — Gemini explanation is required."""
        result = _format_local_answer(overview_result)
        assert result is None, (
            "dataset_overview must NOT be in _LOCAL_ANSWER_OPS; "
            "Gemini narrative is essential for a meaningful description."
        )

    def test_result_to_dataframe_extracts_column_table(self, overview_result):
        """_result_to_dataframe should return a column-level summary table."""
        df = _result_to_dataframe(overview_result)
        assert df is not None
        # sample_df in this file has 5 columns: date, sales, profit, region, category
        assert len(df) == 5
        assert "Column" in df.columns
        assert "Type" in df.columns
        assert "Missing" in df.columns
        assert "Unique" in df.columns

    def test_answer_question_dataset_overview_pipeline(self, sample_df, profile):
        """Full pipeline: interpreter returns dataset_overview plan, overview
        explanation is called (mocked), answer and DataFrame are returned."""
        plan = {"operation": "dataset_overview"}
        narrative = (
            "**Dataset Overview**\nThis appears to be a sales dataset with 120 records.\n"
            "**Column Summary**\n- sales: numeric\n"
            "**Data Quality**\nNo missing values.\n"
            "**Key Observations**\nSales range from 100 to 5000."
        )
        with _mock_interpreter(plan), \
             patch("src.ask_your_data._get_overview_explanation", return_value=narrative), \
             patch("src.llm_agent._build_client", return_value=MagicMock()):
            resp = answer_question("What is this dataset about?", sample_df, profile)

        assert resp["error"] is None
        assert resp["answer"] == narrative
        # Should include a column-summary DataFrame
        assert resp["data_df"] is not None
        assert "Column" in resp["data_df"].columns
        # All required keys present
        for key in ("answer", "data_df", "chart", "error", "raw_result"):
            assert key in resp
