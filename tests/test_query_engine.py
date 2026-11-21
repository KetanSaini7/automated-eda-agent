"""
tests/test_query_engine.py
==========================
Unit tests for src/query_engine.py – the safe validated query execution engine.
No live API calls; uses synthetic DataFrames only.
"""

import sys
import math
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "d:/Projects/Automated_eda_agent")

from src.query_engine import (
    build_schema,
    validate_plan,
    execute_plan,
    ALLOWED_OPERATIONS,
    ALLOWED_AGGREGATIONS,
    ALLOWED_FILTER_OPERATORS,
    ALLOWED_DATE_PERIODS,
    MAX_RESULT_ROWS,
    MAX_PLAN_LIMIT,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    """Standard multi-type DataFrame used by most tests."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date":     dates,
        "sales":    rng.uniform(100, 5000, 120).round(2),
        "profit":   rng.uniform(-200, 1000, 120).round(2),
        "region":   rng.choice(["North", "South", "East", "West"], 120),
        "category": rng.choice(["A", "B", "C"], 120),
        "quantity": rng.integers(1, 50, 120),
    })


@pytest.fixture
def profile(sample_df):
    return {
        "rows": len(sample_df),
        "columns": len(sample_df.columns),
        "numeric_columns": ["sales", "profit", "quantity"],
        "categorical_columns": ["region", "category"],
        "datetime_columns": ["date"],
        "duplicate_rows": 0,
    }


@pytest.fixture
def eda_results():
    return {
        "correlations": [
            {"col1": "sales", "col2": "profit", "r": 0.72},
            {"col1": "sales", "col2": "quantity", "r": 0.31},
        ]
    }


@pytest.fixture
def anomaly_results():
    return {
        "iqr": {
            "sales": {
                "anomaly_count": 5,
                "anomaly_percentage": 4.17,
                "lower_bound": 150.0,
                "upper_bound": 4800.0,
            }
        },
        "multivariate_isolation_forest": {
            "status": "completed",
            "anomaly_count": 7,
            "anomaly_percentage": 5.83,
            "rows_analyzed": 120,
            "features_used": ["sales", "profit", "quantity"],
        },
    }


@pytest.fixture
def supervised_validation():
    return {
        "model": "RandomForest",
        "accuracy": 0.91,
        "precision": 0.88,
        "recall": 0.90,
        "f1": 0.89,
    }


@pytest.fixture
def missing_df(sample_df):
    """DataFrame with some NaN values."""
    df = sample_df.copy()
    df.loc[[0, 5, 10], "sales"] = np.nan
    df.loc[[2, 7], "region"] = np.nan
    return df


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["sales", "region", "date"])


@pytest.fixture
def constant_df():
    return pd.DataFrame({
        "const_col": [42] * 10,
        "value": range(10),
    })


# ============================================================
# 1. build_schema
# ============================================================

class TestBuildSchema:
    def test_returns_dict(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        assert isinstance(schema, dict)

    def test_row_col_counts(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        assert schema["row_count"] == 120
        assert schema["column_count"] == 6

    def test_columns_list_length(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        assert len(schema["columns"]) == len(sample_df.columns)

    def test_column_fields(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        col = schema["columns"][0]
        assert "name" in col
        assert "dtype" in col
        assert "role" in col
        assert "missing_pct" in col
        assert "sample_values" in col

    def test_role_assignments(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        col_map = {c["name"]: c["role"] for c in schema["columns"]}
        assert col_map["sales"] == "numeric"
        assert col_map["region"] == "categorical"
        assert col_map["date"] == "datetime"

    def test_sample_values_max_5(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        for col in schema["columns"]:
            assert len(col["sample_values"]) <= 5

    def test_missing_pct_calculated(self, missing_df, profile):
        profile2 = profile.copy()
        profile2["rows"] = len(missing_df)
        schema = build_schema(missing_df, profile2)
        col_map = {c["name"]: c for c in schema["columns"]}
        assert col_map["sales"]["missing_pct"] > 0

    def test_top_correlations_included(self, sample_df, profile, eda_results):
        schema = build_schema(sample_df, profile, eda_results)
        assert "top_correlations" in schema
        assert len(schema["top_correlations"]) <= 5

    def test_no_correlations_without_eda(self, sample_df, profile):
        schema = build_schema(sample_df, profile, eda_results=None)
        assert "top_correlations" not in schema

    def test_numeric_categorical_datetime_lists(self, sample_df, profile):
        schema = build_schema(sample_df, profile)
        assert "sales" in schema["numeric_columns"]
        assert "region" in schema["categorical_columns"]
        assert "date" in schema["datetime_columns"]


# ============================================================
# 2. validate_plan – general
# ============================================================

class TestValidatePlanGeneral:
    def test_non_dict_rejected(self, sample_df, profile):
        ok, err = validate_plan("bad input", sample_df, profile)
        assert not ok
        assert "JSON object" in err

    def test_missing_operation_rejected(self, sample_df, profile):
        ok, err = validate_plan({}, sample_df, profile)
        assert not ok
        assert "operation" in err

    def test_unknown_operation_rejected(self, sample_df, profile):
        ok, err = validate_plan({"operation": "hack_db"}, sample_df, profile)
        assert not ok
        assert "not supported" in err

    def test_valid_simple_ops_pass(self, sample_df, profile):
        for op in ("schema_info", "row_count", "missing_values",
                   "anomaly_summary", "model_summary", "data_quality_summary"):
            ok, err = validate_plan({"operation": op}, sample_df, profile)
            assert ok, f"Op '{op}' failed: {err}"


# ============================================================
# 3. validate_plan – operation-specific
# ============================================================

class TestValidatePlanOps:
    def test_basic_stats_requires_column(self, sample_df, profile):
        ok, err = validate_plan({"operation": "basic_stats"}, sample_df, profile)
        assert not ok and "column" in err

    def test_basic_stats_col_not_exist(self, sample_df, profile):
        ok, err = validate_plan({"operation": "basic_stats", "column": "nonexistent"}, sample_df, profile)
        assert not ok and "not exist" in err

    def test_basic_stats_non_numeric_rejected(self, sample_df, profile):
        ok, err = validate_plan({"operation": "basic_stats", "column": "region"}, sample_df, profile)
        assert not ok and "not numeric" in err

    def test_basic_stats_valid(self, sample_df, profile):
        ok, err = validate_plan({"operation": "basic_stats", "column": "sales"}, sample_df, profile)
        assert ok, err

    def test_distinct_count_requires_column(self, sample_df, profile):
        ok, err = validate_plan({"operation": "distinct_count"}, sample_df, profile)
        assert not ok

    def test_value_counts_col_not_exist(self, sample_df, profile):
        ok, err = validate_plan({"operation": "value_counts", "column": "xyz"}, sample_df, profile)
        assert not ok and "not exist" in err

    def test_value_counts_valid(self, sample_df, profile):
        ok, err = validate_plan({"operation": "value_counts", "column": "region"}, sample_df, profile)
        assert ok, err

    def test_group_by_missing_field(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            # missing aggregation
        }, sample_df, profile)
        assert not ok and "aggregation" in err

    def test_group_by_invalid_aggregation(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "variance",
        }, sample_df, profile)
        assert not ok and "not allowed" in err

    def test_group_by_non_numeric_measure(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "category",
            "aggregation": "sum",
        }, sample_df, profile)
        assert not ok and "numeric" in err

    def test_group_by_count_accepts_non_numeric(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "category",
            "aggregation": "count",
        }, sample_df, profile)
        assert ok, err

    def test_group_by_invalid_sort(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "sort": "sideways",
        }, sample_df, profile)
        assert not ok and "sort" in err

    def test_group_by_limit_too_large(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "limit": 9999,
        }, sample_df, profile)
        assert not ok and "limit" in err

    def test_top_n_missing_sort_column(self, sample_df, profile):
        ok, err = validate_plan({"operation": "top_n"}, sample_df, profile)
        assert not ok and "sort_column" in err

    def test_top_n_valid(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "top_n",
            "sort_column": "sales",
            "limit": 5,
        }, sample_df, profile)
        assert ok, err

    def test_filtered_aggregate_missing_field(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "filtered_aggregate",
            "filter_column": "region",
            "filter_operator": "==",
            "filter_value": "North",
            # missing measure_column, aggregation
        }, sample_df, profile)
        assert not ok

    def test_filtered_aggregate_invalid_operator(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "filtered_aggregate",
            "filter_column": "sales",
            "filter_operator": "LIKE",
            "filter_value": 1000,
            "measure_column": "profit",
            "aggregation": "mean",
        }, sample_df, profile)
        assert not ok and "not allowed" in err

    def test_filtered_aggregate_valid(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "filtered_aggregate",
            "filter_column": "sales",
            "filter_operator": ">",
            "filter_value": 1000,
            "measure_column": "profit",
            "aggregation": "mean",
        }, sample_df, profile)
        assert ok, err

    def test_date_group_missing_period(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "sum",
            # missing period
        }, sample_df, profile)
        assert not ok and "period" in err

    def test_date_group_invalid_period(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "decade",
        }, sample_df, profile)
        assert not ok and "period" in err

    def test_date_group_non_date_col(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "date_group_aggregate",
            "date_column": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "month",
        }, sample_df, profile)
        assert not ok and "dates" in err.lower()

    def test_date_group_valid(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "month",
        }, sample_df, profile)
        assert ok, err

    def test_correlation_requires_two_columns(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "correlation",
            "column_1": "sales",
        }, sample_df, profile)
        assert not ok

    def test_correlation_non_numeric(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "correlation",
            "column_1": "sales",
            "column_2": "region",
        }, sample_df, profile)
        assert not ok and "numeric" in err

    def test_correlation_valid(self, sample_df, profile):
        ok, err = validate_plan({
            "operation": "correlation",
            "column_1": "sales",
            "column_2": "profit",
        }, sample_df, profile)
        assert ok, err


# ============================================================
# 4. execute_plan
# ============================================================

class TestExecutePlanSchemaInfo:
    def test_schema_info_structure(self, sample_df, profile):
        result = execute_plan({"operation": "schema_info"}, sample_df, profile)
        assert result["error"] is None
        assert result["approximate"] is False
        assert "column_details" in result["data"]
        assert len(result["data"]["column_details"]) == len(sample_df.columns)

    def test_schema_info_has_rows(self, sample_df, profile):
        result = execute_plan({"operation": "schema_info"}, sample_df, profile)
        assert result["data"]["rows"] == 120


class TestExecutePlanRowCount:
    def test_row_count_correct(self, sample_df, profile):
        result = execute_plan({"operation": "row_count"}, sample_df, profile)
        assert result["error"] is None
        assert result["data"]["rows"] == 120
        assert result["data"]["columns"] == 6

    def test_row_count_column_names(self, sample_df, profile):
        result = execute_plan({"operation": "row_count"}, sample_df, profile)
        assert "column_names" in result["data"]
        assert "sales" in result["data"]["column_names"]


class TestExecutePlanMissingValues:
    def test_no_missing_all_clear(self, sample_df, profile):
        result = execute_plan({"operation": "missing_values"}, sample_df, profile)
        assert result["error"] is None
        # sample_df has no NaN → message indicates none found
        data = result["data"]
        assert isinstance(data, list)
        if data:
            assert "message" in data[0] or "missing_count" in data[0]

    def test_missing_values_detected(self, missing_df, profile):
        profile2 = {**profile, "rows": len(missing_df)}
        result = execute_plan({"operation": "missing_values"}, missing_df, profile2)
        assert result["error"] is None
        cols_with_missing = {r["column"] for r in result["data"] if "column" in r}
        assert "sales" in cols_with_missing

    def test_specific_column_missing(self, missing_df, profile):
        profile2 = {**profile, "rows": len(missing_df)}
        result = execute_plan(
            {"operation": "missing_values", "column": "sales"},
            missing_df, profile2,
        )
        assert result["data"][0]["missing_count"] == 3

    def test_no_missing_in_whole_df(self, sample_df, profile):
        result = execute_plan({"operation": "missing_values"}, sample_df, profile)
        # Should return a message about no missing
        data = result["data"]
        assert isinstance(data, list)
        if len(data) == 1 and "message" in data[0]:
            assert "No missing" in data[0]["message"]


class TestExecutePlanDistinctCount:
    def test_distinct_count_region(self, sample_df, profile):
        result = execute_plan(
            {"operation": "distinct_count", "column": "region"},
            sample_df, profile,
        )
        assert result["error"] is None
        assert result["data"]["distinct_count"] == 4

    def test_distinct_count_category(self, sample_df, profile):
        result = execute_plan(
            {"operation": "distinct_count", "column": "category"},
            sample_df, profile,
        )
        assert result["data"]["distinct_count"] == 3


class TestExecutePlanBasicStats:
    def test_basic_stats_fields(self, sample_df, profile):
        result = execute_plan({"operation": "basic_stats", "column": "sales"}, sample_df, profile)
        assert result["error"] is None
        d = result["data"]
        for key in ("count", "sum", "mean", "median", "std", "min", "max", "q1", "q3"):
            assert key in d, f"Missing key: {key}"

    def test_basic_stats_mean_in_range(self, sample_df, profile):
        result = execute_plan({"operation": "basic_stats", "column": "sales"}, sample_df, profile)
        mean = result["data"]["mean"]
        assert 100 <= mean <= 5000

    def test_basic_stats_all_null_col(self, sample_df, profile):
        df2 = sample_df.copy()
        df2["sales"] = np.nan
        result = execute_plan({"operation": "basic_stats", "column": "sales"}, df2, profile)
        # Should not crash; mean etc. will be None
        assert result["error"] is None


class TestExecutePlanValueCounts:
    def test_value_counts_structure(self, sample_df, profile):
        result = execute_plan(
            {"operation": "value_counts", "column": "region"},
            sample_df, profile,
        )
        assert result["error"] is None
        assert isinstance(result["data"], list)
        row = result["data"][0]
        assert "value" in row and "count" in row and "percentage" in row

    def test_value_counts_limit(self, sample_df, profile):
        result = execute_plan(
            {"operation": "value_counts", "column": "region", "limit": 2},
            sample_df, profile,
        )
        assert len(result["data"]) <= 2

    def test_value_counts_percentages_sum_100(self, sample_df, profile):
        result = execute_plan(
            {"operation": "value_counts", "column": "region"},
            sample_df, profile,
        )
        total_pct = sum(r["percentage"] for r in result["data"])
        assert abs(total_pct - 100.0) < 1.0  # allow rounding


class TestExecutePlanGroupByAggregate:
    def test_group_by_sum(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "sort": "descending",
            "limit": 4,
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        assert len(result["data"]) <= 4
        assert "sum" in result["data"][0]

    def test_group_by_descending_order(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "sort": "descending",
            "limit": 10,
        }
        result = execute_plan(plan, sample_df, profile)
        values = [r["sum"] for r in result["data"] if r["sum"] is not None]
        assert values == sorted(values, reverse=True)

    def test_group_by_ascending_order(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "sales",
            "aggregation": "sum",
            "sort": "ascending",
            "limit": 10,
        }
        result = execute_plan(plan, sample_df, profile)
        values = [r["sum"] for r in result["data"] if r["sum"] is not None]
        assert values == sorted(values)

    def test_group_by_count(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "region",
            "measure_column": "category",
            "aggregation": "count",
            "limit": 10,
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        total = sum(r["count"] for r in result["data"])
        assert total == 120

    def test_group_by_mean(self, sample_df, profile):
        plan = {
            "operation": "group_by_aggregate",
            "group_by": "category",
            "measure_column": "profit",
            "aggregation": "mean",
            "limit": 10,
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        for row in result["data"]:
            assert "mean" in row


class TestExecutePlanTopN:
    def test_top_n_returns_correct_count(self, sample_df, profile):
        result = execute_plan(
            {"operation": "top_n", "sort_column": "sales", "limit": 5},
            sample_df, profile,
        )
        assert result["error"] is None
        assert len(result["data"]) == 5

    def test_top_n_sorted_descending(self, sample_df, profile):
        result = execute_plan(
            {"operation": "top_n", "sort_column": "sales", "limit": 5},
            sample_df, profile,
        )
        vals = [float(r["sales"]) for r in result["data"]]
        assert vals == sorted(vals, reverse=True)

    def test_bottom_n_sorted_ascending(self, sample_df, profile):
        result = execute_plan(
            {"operation": "bottom_n", "sort_column": "sales", "limit": 5},
            sample_df, profile,
        )
        vals = [float(r["sales"]) for r in result["data"]]
        assert vals == sorted(vals)

    def test_top_n_display_columns(self, sample_df, profile):
        result = execute_plan(
            {"operation": "top_n", "sort_column": "sales",
             "display_columns": ["region", "category"], "limit": 3},
            sample_df, profile,
        )
        # sales should be auto-prepended
        assert "sales" in result["data"][0]


class TestExecutePlanFilteredAggregate:
    def test_filtered_aggregate_gt(self, sample_df, profile):
        plan = {
            "operation": "filtered_aggregate",
            "filter_column": "sales",
            "filter_operator": ">",
            "filter_value": 2000,
            "measure_column": "profit",
            "aggregation": "mean",
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        assert result["data"]["matching_rows"] > 0
        expected_mean = sample_df[sample_df["sales"] > 2000]["profit"].mean()
        assert abs(result["data"]["result"] - expected_mean) < 0.01

    def test_filtered_aggregate_eq_string(self, sample_df, profile):
        plan = {
            "operation": "filtered_aggregate",
            "filter_column": "region",
            "filter_operator": "==",
            "filter_value": "North",
            "measure_column": "sales",
            "aggregation": "sum",
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        expected = sample_df[sample_df["region"] == "North"]["sales"].sum()
        assert abs(result["data"]["result"] - expected) < 0.01

    def test_filtered_aggregate_no_match(self, sample_df, profile):
        plan = {
            "operation": "filtered_aggregate",
            "filter_column": "sales",
            "filter_operator": ">",
            "filter_value": 999999,
            "measure_column": "profit",
            "aggregation": "mean",
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        assert result["data"]["matching_rows"] == 0
        assert result["data"]["result"] is None


class TestExecutePlanDateGroupAggregate:
    def test_monthly_sum(self, sample_df, profile):
        plan = {
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "month",
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        assert isinstance(result["data"], list)
        assert len(result["data"]) > 0
        # 120 days ~ 4 months
        assert 3 <= len(result["data"]) <= 5

    def test_weekly_count(self, sample_df, profile):
        plan = {
            "operation": "date_group_aggregate",
            "date_column": "date",
            "measure_column": "sales",
            "aggregation": "count",
            "period": "week",
        }
        result = execute_plan(plan, sample_df, profile)
        assert result["error"] is None
        # sum of counts must equal 120
        total = sum(r["count"] for r in result["data"])
        assert total == 120

    def test_string_date_column(self, profile):
        df = pd.DataFrame({
            "date_str": ["2023-01-15", "2023-02-20", "2023-02-28", "2023-03-10"],
            "amount": [100, 200, 150, 300],
        })
        p = {**profile, "rows": 4, "columns": 2}
        plan = {
            "operation": "date_group_aggregate",
            "date_column": "date_str",
            "measure_column": "amount",
            "aggregation": "sum",
            "period": "month",
        }
        result = execute_plan(plan, df, p)
        assert result["error"] is None
        assert len(result["data"]) == 3  # Jan, Feb, Mar

    def test_all_invalid_dates_returns_error(self, profile):
        df = pd.DataFrame({
            "bad_date": ["not_a_date"] * 10,
            "sales": range(10),
        })
        p = {**profile, "rows": 10, "columns": 2}
        plan = {
            "operation": "date_group_aggregate",
            "date_column": "bad_date",
            "measure_column": "sales",
            "aggregation": "sum",
            "period": "month",
        }
        result = execute_plan(plan, df, p)
        # Should return an error (no valid dates)
        assert result["error"] is not None or result["data"] == []


class TestExecutePlanCorrelation:
    def test_correlation_value_range(self, sample_df, profile):
        result = execute_plan(
            {"operation": "correlation", "column_1": "sales", "column_2": "profit"},
            sample_df, profile,
        )
        assert result["error"] is None
        r = result["data"]["pearson_r"]
        assert r is None or -1.0 <= r <= 1.0

    def test_correlation_perfect_positive(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        profile = {"rows": 5, "columns": 2, "numeric_columns": ["x", "y"],
                   "categorical_columns": [], "datetime_columns": []}
        result = execute_plan(
            {"operation": "correlation", "column_1": "x", "column_2": "y"},
            df, profile,
        )
        assert abs(result["data"]["pearson_r"] - 1.0) < 0.001

    def test_correlation_insufficient_data(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        profile = {"rows": 1, "columns": 2, "numeric_columns": ["x", "y"],
                   "categorical_columns": [], "datetime_columns": []}
        result = execute_plan(
            {"operation": "correlation", "column_1": "x", "column_2": "y"},
            df, profile,
        )
        assert result["error"] is not None


class TestExecutePlanAnomalySummary:
    def test_no_anomaly_results(self, sample_df, profile):
        result = execute_plan(
            {"operation": "anomaly_summary"},
            sample_df, profile,
            anomaly_results=None,
        )
        assert result["error"] is None
        assert "message" in result["data"]

    def test_anomaly_summary_data(self, sample_df, profile, anomaly_results):
        result = execute_plan(
            {"operation": "anomaly_summary"},
            sample_df, profile,
            anomaly_results=anomaly_results,
        )
        assert result["error"] is None
        d = result["data"]
        assert "isolation_forest" in d
        assert d["isolation_forest"]["anomaly_count"] == 7

    def test_iqr_in_summary(self, sample_df, profile, anomaly_results):
        result = execute_plan(
            {"operation": "anomaly_summary"},
            sample_df, profile,
            anomaly_results=anomaly_results,
        )
        assert "iqr_outliers" in result["data"]
        assert "sales" in result["data"]["iqr_outliers"]


class TestExecutePlanModelSummary:
    def test_no_model_results(self, sample_df, profile):
        result = execute_plan(
            {"operation": "model_summary"},
            sample_df, profile,
            supervised_validation=None,
        )
        assert result["error"] is None
        assert "message" in result["data"]

    def test_model_summary_data(self, sample_df, profile, supervised_validation):
        result = execute_plan(
            {"operation": "model_summary"},
            sample_df, profile,
            supervised_validation=supervised_validation,
        )
        assert result["error"] is None
        assert result["data"]["accuracy"] == 0.91


class TestExecutePlanDataQualitySummary:
    def test_quality_summary_fields(self, sample_df, profile):
        result = execute_plan(
            {"operation": "data_quality_summary"},
            sample_df, profile,
        )
        assert result["error"] is None
        d = result["data"]
        assert "rows" in d
        assert "duplicate_rows" in d
        assert "columns_with_missing" in d
        assert "constant_columns" in d

    def test_constant_col_detected(self, constant_df):
        profile = {"rows": 10, "columns": 2, "numeric_columns": ["const_col", "value"],
                   "categorical_columns": [], "datetime_columns": [], "duplicate_rows": 0}
        result = execute_plan(
            {"operation": "data_quality_summary"},
            constant_df, profile,
        )
        assert "const_col" in result["data"]["constant_columns"]


# ============================================================
# 5. Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_df_row_count(self, empty_df):
        profile = {"rows": 0, "columns": 3, "numeric_columns": [],
                   "categorical_columns": [], "datetime_columns": [], "duplicate_rows": 0}
        result = execute_plan({"operation": "row_count"}, empty_df, profile)
        assert result["data"]["rows"] == 0

    def test_empty_df_missing_values(self, empty_df):
        profile = {"rows": 0, "columns": 3, "numeric_columns": [],
                   "categorical_columns": [], "datetime_columns": [], "duplicate_rows": 0}
        result = execute_plan({"operation": "missing_values"}, empty_df, profile)
        assert result["error"] is None

    def test_max_result_rows_limit(self, sample_df, profile):
        # value_counts with limit > MAX_RESULT_ROWS should be capped
        big_plan = {"operation": "value_counts", "column": "region", "limit": 9999}
        result = execute_plan(big_plan, sample_df, profile)
        assert len(result["data"]) <= MAX_RESULT_ROWS

    def test_approximate_flag_false_for_full_operations(self, sample_df, profile):
        result = execute_plan({"operation": "basic_stats", "column": "sales"}, sample_df, profile)
        assert result["approximate"] is False

    def test_safe_float_handles_nan(self):
        from src.query_engine import _safe_float
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(42.5) == 42.5

    def test_allowlists_are_non_empty(self):
        assert len(ALLOWED_OPERATIONS) > 0
        assert len(ALLOWED_AGGREGATIONS) > 0
        assert len(ALLOWED_FILTER_OPERATORS) > 0
        assert len(ALLOWED_DATE_PERIODS) > 0

    def test_max_plan_limit_is_reasonable(self):
        assert 10 <= MAX_PLAN_LIMIT <= 1000

    def test_max_result_rows_is_reasonable(self):
        assert 10 <= MAX_RESULT_ROWS <= 500


# ============================================================
# 6. dataset_overview
# ============================================================

class TestExecutePlanDatasetOverview:
    """Tests for the dataset_overview operation executor."""

    def test_dataset_overview_in_allowlist(self):
        assert "dataset_overview" in ALLOWED_OPERATIONS

    def test_validate_plan_accepts_dataset_overview(self, sample_df, profile):
        ok, err = validate_plan({"operation": "dataset_overview"}, sample_df, profile)
        assert ok, f"validate_plan rejected dataset_overview: {err}"

    def test_result_structure(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        assert result["error"] is None
        assert result["operation"] == "dataset_overview"
        assert result["approximate"] is False
        assert isinstance(result["data"], dict)

    def test_top_level_counts(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        data = result["data"]
        assert data["rows"] == 120
        assert data["columns"] == 6
        assert isinstance(data["column_names"], list)
        assert len(data["column_names"]) == 6

    def test_column_lists_match_profile(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        data = result["data"]
        assert "sales" in data["numeric_columns"]
        assert "region" in data["categorical_columns"]
        assert "date" in data["datetime_columns"]

    def test_column_details_count(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        assert len(result["data"]["column_details"]) == 6

    def test_column_detail_fields(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        for d in result["data"]["column_details"]:
            for field in ("column", "dtype", "role", "non_null", "missing_count",
                          "missing_pct", "unique_count"):
                assert field in d, f"Missing field '{field}' in column detail for '{d.get('column')}'"

    def test_numeric_col_has_stats(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        sales_detail = next(d for d in result["data"]["column_details"] if d["column"] == "sales")
        assert sales_detail["role"] == "numeric"
        assert "stats" in sales_detail
        stats = sales_detail["stats"]
        for key in ("mean", "median", "min", "max", "std"):
            assert key in stats

    def test_numeric_stats_values_in_range(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        sales_detail = next(d for d in result["data"]["column_details"] if d["column"] == "sales")
        stats = sales_detail["stats"]
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_categorical_col_has_top_values(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        region_detail = next(d for d in result["data"]["column_details"] if d["column"] == "region")
        assert region_detail["role"] == "categorical"
        assert "top_values" in region_detail
        assert len(region_detail["top_values"]) <= 5
        for tv in region_detail["top_values"]:
            assert "value" in tv and "count" in tv

    def test_missing_values_detected(self, missing_df, profile):
        profile2 = {**profile, "rows": len(missing_df)}
        result = execute_plan({"operation": "dataset_overview"}, missing_df, profile2)
        assert result["error"] is None
        # sales has 3 NaNs
        sales_d = next(d for d in result["data"]["column_details"] if d["column"] == "sales")
        assert sales_d["missing_count"] == 3
        assert sales_d["missing_pct"] > 0
        # missing_summary should report sales
        assert "sales" in result["data"]["missing_summary"]

    def test_no_missing_summary_when_complete(self, sample_df, profile):
        result = execute_plan({"operation": "dataset_overview"}, sample_df, profile)
        assert result["data"]["missing_summary"] == {}

    def test_empty_dataframe_does_not_crash(self, empty_df):
        profile = {"rows": 0, "columns": 3, "numeric_columns": [],
                   "categorical_columns": [], "datetime_columns": [], "duplicate_rows": 0}
        result = execute_plan({"operation": "dataset_overview"}, empty_df, profile)
        assert result["error"] is None
        assert result["data"]["rows"] == 0
