"""
pytest test suite for src/visualization.py

Run from project root:
    python -m pytest tests/test_visualization.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.visualization import (
    create_numeric_histogram,
    create_categorical_bar,
    create_scatter_plot,
    create_correlation_heatmap,
    create_time_series,
    _sample_label,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def base_df():
    return pd.DataFrame({
        "Sales":      [100, 200, 300, 400, 500],
        "Profit":     [10,  20,  30,  40,  50],
        "Quantity":   [2,   4,   6,   8,   10],
        "Region":     ["North", "South", "North", "West", "East"],
        "Order_Date": pd.to_datetime([
            "2022-06-01", "2022-06-15", "2022-07-01",
            "2022-07-15", "2022-08-01"
        ]),
    })


@pytest.fixture
def constant_df():
    return pd.DataFrame({
        "Constant": [5, 5, 5, 5, 5],
        "Region":   ["North", "North", "North", "North", "North"],
    })


@pytest.fixture
def all_null_df():
    return pd.DataFrame({
        "NullNum": [None, None, None],
        "Sales":   [1.0, 2.0, 3.0],
    })


# ============================================================
# _sample_label
# ============================================================

class TestSampleLabel:

    def test_format(self):
        label = _sample_label(50_000, 200_000)
        assert "50,000" in label
        assert "200,000" in label

    def test_contains_slash(self):
        label = _sample_label(1000, 5000)
        assert "/" in label


# ============================================================
# create_numeric_histogram
# ============================================================

class TestNumericHistogram:

    def test_returns_figure_for_valid_column(self, base_df):
        fig = create_numeric_histogram(base_df, "Sales")
        assert fig is not None

    def test_returns_none_for_constant_column(self, constant_df):
        fig = create_numeric_histogram(constant_df, "Constant")
        assert fig is None

    def test_returns_none_for_all_null_column(self, all_null_df):
        fig = create_numeric_histogram(all_null_df, "NullNum")
        assert fig is None

    def test_title_contains_column_name(self, base_df):
        fig = create_numeric_histogram(base_df, "Sales")
        assert "Sales" in fig.layout.title.text

    def test_sample_label_appears_in_title(self, base_df):
        fig = create_numeric_histogram(
            base_df, "Sales", is_sample=True, total_rows=200_000
        )
        assert "200,000" in fig.layout.title.text

    def test_no_sample_label_when_not_sample(self, base_df):
        fig = create_numeric_histogram(base_df, "Sales", is_sample=False)
        assert "sample" not in fig.layout.title.text.lower()

    def test_has_marginal_box(self, base_df):
        fig = create_numeric_histogram(base_df, "Sales")
        # Plotly adds a box trace when marginal="box"
        trace_types = [t.type for t in fig.data]
        assert "box" in trace_types


# ============================================================
# create_categorical_bar
# ============================================================

class TestCategoricalBar:

    def test_returns_figure_for_valid_column(self, base_df):
        fig = create_categorical_bar(base_df, "Region")
        assert fig is not None

    def test_returns_none_for_all_null_column(self):
        df = pd.DataFrame({"Cat": [None, None, None]})
        fig = create_categorical_bar(df, "Cat")
        assert fig is None

    def test_title_contains_column_name(self, base_df):
        fig = create_categorical_bar(base_df, "Region")
        assert "Region" in fig.layout.title.text

    def test_sample_label_appears_in_title(self, base_df):
        fig = create_categorical_bar(
            base_df, "Region", is_sample=True, total_rows=100_000
        )
        assert "100,000" in fig.layout.title.text

    def test_top_n_respected(self, base_df):
        # 4 unique values, top_n=2 → only 2 bars
        fig = create_categorical_bar(base_df, "Region", top_n=2)
        # The bar chart has one trace with 2 x-values
        assert len(fig.data[0].x) == 2

    def test_constant_categorical_still_renders(self, constant_df):
        # A column with one value is not "empty" — it has data to show
        fig = create_categorical_bar(constant_df, "Region")
        assert fig is not None


# ============================================================
# create_scatter_plot
# ============================================================

class TestScatterPlot:

    def test_returns_figure(self, base_df):
        fig = create_scatter_plot(base_df, "Sales", "Profit")
        assert fig is not None

    def test_title_contains_column_names(self, base_df):
        fig = create_scatter_plot(base_df, "Sales", "Profit")
        assert "Sales" in fig.layout.title.text
        assert "Profit" in fig.layout.title.text

    def test_sample_label_appears(self, base_df):
        fig = create_scatter_plot(
            base_df, "Sales", "Profit", is_sample=True, total_rows=50_000
        )
        assert "50,000" in fig.layout.title.text


# ============================================================
# create_correlation_heatmap
# ============================================================

class TestCorrelationHeatmap:

    def test_returns_figure_for_multiple_numeric_columns(self, base_df):
        fig = create_correlation_heatmap(base_df)
        assert fig is not None

    def test_returns_none_for_single_numeric_column(self):
        df = pd.DataFrame({"Sales": [1, 2, 3]})
        fig = create_correlation_heatmap(df)
        assert fig is None

    def test_returns_none_for_all_null_numeric(self):
        df = pd.DataFrame({
            "A": [None, None, None],
            "B": [None, None, None],
        })
        fig = create_correlation_heatmap(df)
        assert fig is None

    def test_returns_none_when_all_columns_constant(self):
        df = pd.DataFrame({
            "A": [1, 1, 1],
            "B": [2, 2, 2],
        })
        fig = create_correlation_heatmap(df)
        assert fig is None

    def test_title_contains_heatmap(self, base_df):
        fig = create_correlation_heatmap(base_df)
        assert "Heatmap" in fig.layout.title.text

    def test_sample_label_appears(self, base_df):
        fig = create_correlation_heatmap(
            base_df, is_sample=True, total_rows=75_000
        )
        assert "75,000" in fig.layout.title.text

    def test_mixed_null_and_valid_columns(self):
        # One all-null column should be dropped; heatmap drawn from the rest
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
            "C": [None, None, None],
        })
        fig = create_correlation_heatmap(df)
        assert fig is not None


# ============================================================
# create_time_series
# ============================================================

class TestTimeSeries:

    def test_returns_figure_for_valid_data(self, base_df):
        fig = create_time_series(base_df, "Order_Date", "Sales")
        assert fig is not None

    def test_returns_none_when_no_valid_rows(self):
        df = pd.DataFrame({
            "Date":  ["not-a-date", "also-not"],
            "Value": [1.0, 2.0],
        })
        fig = create_time_series(df, "Date", "Value")
        assert fig is None

    def test_title_contains_column_name(self, base_df):
        fig = create_time_series(base_df, "Order_Date", "Sales")
        assert "Sales" in fig.layout.title.text

    def test_sample_label_appears(self, base_df):
        fig = create_time_series(
            base_df, "Order_Date", "Sales", is_sample=True, total_rows=90_000
        )
        assert "90,000" in fig.layout.title.text

    def test_data_is_sorted_by_date(self, base_df):
        # Shuffle rows; output should still be chronologically ordered
        shuffled = base_df.sample(frac=1, random_state=99).reset_index(drop=True)
        fig = create_time_series(shuffled, "Order_Date", "Sales")
        assert fig is not None
        # The x-axis values in the line trace should be in ascending order
        x_values = list(fig.data[0].x)
        assert x_values == sorted(x_values)
