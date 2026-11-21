"""
pytest test suite for src/data_quality.py and src/data_cleaning.py

Run from project root:
    python -m pytest tests/test_data_quality.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.data_quality import check_data_quality
from src.data_cleaning import standardize_categorical_values, _is_id_like_column


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df():
    """Standard multi-type DataFrame covering all DQ checks."""
    return pd.DataFrame({
        "Sales":    [100, 200, 300, 400, 500],
        "Profit":   [10,  20,  None, 40, 50],
        "Region":   ["North", "South", "North", "West", "East"],
        "Constant": [1, 1, 1, 1, 1],
        "ID":       ["A001", "A002", "A003", "A004", "A005"],
    })


@pytest.fixture
def all_missing_df():
    return pd.DataFrame({
        "empty_col": [None, None, None],
        "Sales":     [1, 2, 3],
    })


@pytest.fixture
def mixed_type_df():
    """Column where some values are numeric strings and some are text."""
    return pd.DataFrame({
        "value": ["100", "200", "three hundred", "400", "five"],
    })


@pytest.fixture
def duplicate_df():
    return pd.DataFrame({
        "A": [1, 1, 2, 3, 3],
        "B": ["x", "x", "y", "z", "z"],
    })


# ============================================================
# check_data_quality — missing values
# ============================================================

class TestMissingValues:

    def test_detects_missing_values(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Profit" in result["missing_values"]
        assert result["missing_values"]["Profit"] == 1

    def test_missing_percentage_correct(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Profit" in result["missing_percentage"]
        assert result["missing_percentage"]["Profit"] == pytest.approx(20.0, abs=0.1)

    def test_no_false_positives_for_complete_columns(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Sales" not in result["missing_values"]
        assert "Region" not in result["missing_values"]

    def test_no_missing_values_returns_empty_dict(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        result = check_data_quality(df)
        assert result["missing_values"] == {}
        assert result["missing_percentage"] == {}


# ============================================================
# check_data_quality — duplicates
# ============================================================

class TestDuplicates:

    def test_detects_duplicate_rows(self, duplicate_df):
        result = check_data_quality(duplicate_df)
        assert result["duplicate_rows"] == 2

    def test_no_duplicates(self, sample_df):
        result = check_data_quality(sample_df)
        assert result["duplicate_rows"] == 0


# ============================================================
# check_data_quality — constant columns
# ============================================================

class TestConstantColumns:

    def test_detects_constant_column(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Constant" in result["constant_columns"]

    def test_non_constant_not_flagged(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Sales" not in result["constant_columns"]
        assert "Region" not in result["constant_columns"]

    def test_all_null_column_is_constant(self):
        df = pd.DataFrame({"NullCol": [None, None, None], "A": [1, 2, 3]})
        result = check_data_quality(df)
        assert "NullCol" in result["constant_columns"]


# ============================================================
# check_data_quality — empty columns (DQ-7, new feature)
# ============================================================

class TestEmptyColumns:

    def test_detects_empty_column(self, all_missing_df):
        result = check_data_quality(all_missing_df)
        assert "empty_col" in result["empty_columns"]

    def test_non_empty_not_flagged(self, all_missing_df):
        result = check_data_quality(all_missing_df)
        assert "Sales" not in result["empty_columns"]

    def test_no_empty_columns(self, sample_df):
        result = check_data_quality(sample_df)
        assert result["empty_columns"] == []


# ============================================================
# check_data_quality — mixed-type columns (DQ-6, new feature)
# ============================================================

class TestMixedTypeColumns:

    def test_detects_mixed_type_column(self, mixed_type_df):
        result = check_data_quality(mixed_type_df)
        mixed_cols = [m["column"] for m in result["mixed_type_columns"]]
        assert "value" in mixed_cols

    def test_pure_numeric_string_not_flagged(self):
        df = pd.DataFrame({"nums": ["1", "2", "3", "4", "5"]})
        result = check_data_quality(df)
        assert result["mixed_type_columns"] == []

    def test_pure_text_not_flagged(self, sample_df):
        result = check_data_quality(sample_df)
        mixed_cols = [m["column"] for m in result["mixed_type_columns"]]
        assert "Region" not in mixed_cols


# ============================================================
# check_data_quality — high cardinality
# ============================================================

class TestHighCardinality:

    def test_detects_high_cardinality(self):
        # 100 unique values in 100 rows → ratio = 1.0 > 0.5, n_total=100>=50,
        # n_unique=100>=10 → should be flagged
        df = pd.DataFrame({"uid": [str(i) for i in range(100)]})
        result = check_data_quality(df)
        assert "uid" in result["high_cardinality_columns"]

    def test_low_cardinality_not_flagged(self, sample_df):
        # sample_df has only 5 rows — the min-row guard (50) prevents flagging
        result = check_data_quality(sample_df)
        assert "Region" not in result["high_cardinality_columns"]

    def test_small_dataset_not_flagged_even_at_high_ratio(self):
        # 10 rows, 10 unique values → ratio=1.0 but n_total<50 → not flagged
        df = pd.DataFrame({"cat": [str(i) for i in range(10)]})
        result = check_data_quality(df)
        assert "cat" not in result["high_cardinality_columns"]


# ============================================================
# check_data_quality — negative / zero values
# ============================================================

class TestNegativeAndZero:

    def test_detects_negative_values(self):
        df = pd.DataFrame({"Temp": [10, -5, 20, -3, 0]})
        result = check_data_quality(df)
        assert "Temp" in result["negative_values"]
        assert result["negative_values"]["Temp"] == 2

    def test_detects_zero_values(self):
        df = pd.DataFrame({"Count": [0, 1, 2, 0, 5]})
        result = check_data_quality(df)
        assert "Count" in result["zero_values"]
        assert result["zero_values"]["Count"] == 2

    def test_no_negatives(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Sales" not in result["negative_values"]


# ============================================================
# check_data_quality — data types
# ============================================================

class TestDataTypes:

    def test_data_types_present(self, sample_df):
        result = check_data_quality(sample_df)
        assert "Sales" in result["data_types"]
        assert "Region" in result["data_types"]

    def test_correct_dtype_string(self, sample_df):
        result = check_data_quality(sample_df)
        assert "float" in result["data_types"]["Sales"] or \
               "int" in result["data_types"]["Sales"]
        assert "object" in result["data_types"]["Region"]


# ============================================================
# standardize_categorical_values — data_cleaning.py
# ============================================================

class TestDataCleaning:

    def test_strips_whitespace(self):
        df = pd.DataFrame({"City": ["  London  ", " Paris", "Berlin "]})
        result = standardize_categorical_values(df)
        assert result["City"].tolist() == ["London", "Paris", "Berlin"]

    def test_collapses_internal_spaces(self):
        df = pd.DataFrame({"City": ["New  York", "Los   Angeles"]})
        result = standardize_categorical_values(df)
        assert result["City"].tolist() == ["New York", "Los Angeles"]

    def test_title_cases_label_column(self):
        df = pd.DataFrame({
            "Region": ["north", "SOUTH", "East", "west"]
        })
        result = standardize_categorical_values(df)
        assert result["Region"].tolist() == ["North", "South", "East", "West"]

    def test_skips_title_case_for_id_column(self):
        df = pd.DataFrame({
            "customer_id": ["ABC123", "DEF456", "GHI789"]
        })
        result = standardize_categorical_values(df)
        # Must NOT apply title-casing (would change "ABC123" to "Abc123")
        assert result["customer_id"].tolist() == ["ABC123", "DEF456", "GHI789"]

    def test_skips_title_case_for_email_column(self):
        df = pd.DataFrame({
            "email": ["user@example.com", "admin@test.org"]
        })
        result = standardize_categorical_values(df)
        assert result["email"].tolist() == ["user@example.com", "admin@test.org"]

    def test_skips_title_case_for_high_cardinality(self):
        # 100 unique values → ratio = 1.0 > 0.5 → skip title-casing
        values = [f"item_{i}" for i in range(100)]
        df = pd.DataFrame({"description": values})
        result = standardize_categorical_values(df)
        assert result["description"].tolist() == values  # unchanged

    def test_preserves_null_values(self):
        df = pd.DataFrame({"Region": ["north", None, "south"]})
        result = standardize_categorical_values(df)
        assert result["Region"][1] is None or pd.isna(result["Region"][1])
        assert result["Region"][0] == "North"

    def test_numeric_columns_untouched(self):
        df = pd.DataFrame({
            "Sales": [100, 200, 300],
            "Region": ["north", "south", "east"],
        })
        result = standardize_categorical_values(df)
        assert result["Sales"].tolist() == [100, 200, 300]


# ============================================================
# _is_id_like_column helper
# ============================================================

class TestIsIdLikeColumn:

    def test_id_in_name(self):
        s = pd.Series(["001", "002", "003"])
        assert _is_id_like_column("customer_id", s) is True

    def test_email_values(self):
        s = pd.Series(["a@b.com", "c@d.org"])
        assert _is_id_like_column("contact", s) is True

    def test_url_values(self):
        s = pd.Series(["https://example.com", "http://test.org"])
        assert _is_id_like_column("link", s) is True

    def test_regular_label_column_small(self):
        # 4 unique values in 4 rows — small dataset, should NOT be flagged
        s = pd.Series(["North", "South", "East", "West"])
        assert _is_id_like_column("Region", s) is False

    def test_high_cardinality_large_dataset(self):
        # 100 unique values in 100 rows, name has no ID keyword → flagged
        s = pd.Series([f"desc_{i}" for i in range(100)])
        assert _is_id_like_column("description", s) is True

    def test_short_all_caps_code_flagged(self):
        # Short all-caps tokens like "USD", "GB" should be flagged
        s = pd.Series(["USD", "EUR", "GBP"])
        assert _is_id_like_column("currency", s) is True

    def test_long_all_caps_not_flagged_as_code(self):
        # "SOUTH", "NORTH" etc. are too long to be short codes
        s = pd.Series(["NORTH", "SOUTH", "EAST", "WEST"])
        assert _is_id_like_column("Region", s) is False


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"A": [1], "B": ["x"]})
        result = check_data_quality(df)
        assert result["duplicate_rows"] == 0
        assert result["unique_values"]["A"] == 1

    def test_all_null_dataframe(self):
        df = pd.DataFrame({"X": [None, None], "Y": [None, None]})
        result = check_data_quality(df)
        assert "X" in result["empty_columns"]
        assert "Y" in result["empty_columns"]
        assert result["missing_values"]["X"] == 2

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "City": ["NYC", "LA"]})
        result = check_data_quality(df)
        assert result["negative_values"] == {}
        assert result["zero_values"] == {}
