"""
query_engine.py
===============
Safe, validated query execution engine for the "Ask Your Data" feature.

Design principles (from prompt.md §8):
  - NEVER execute arbitrary code, eval(), exec(), or generated Python.
  - All operations are on a fixed allowlist.
  - Every column reference is validated against the actual DataFrame.
  - Every data-type requirement is checked before execution.
  - Results are compact dicts/lists — never raw DataFrames sent anywhere.
  - Approximate results (from sampling) are explicitly flagged.

Public API
----------
  build_schema(df, profile, eda_results)         → compact schema dict
  validate_plan(plan, df, profile)               → (ok: bool, error: str)
  execute_plan(plan, df, profile, anomaly_results, supervised_validation)
                                                 → result dict
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np
from typing import Any


# ============================================================
# CONSTANTS / ALLOWLISTS
# ============================================================

# Every operation name Gemini is allowed to request.
ALLOWED_OPERATIONS = frozenset({
    "dataset_overview",      # comprehensive dataset description with per-column stats
    "schema_info",           # column names, dtypes, sample values
    "row_count",             # total rows and columns
    "missing_values",        # per-column missing counts/percentages
    "distinct_count",        # unique value count for a column
    "basic_stats",           # mean/median/std/min/max/sum for a column
    "value_counts",          # frequency table for a categorical column
    "group_by_aggregate",    # group-by + aggregation
    "top_n",                 # top N records by a column
    "bottom_n",              # bottom N records by a column
    "filtered_aggregate",    # filter rows then aggregate
    "date_group_aggregate",  # group by time period then aggregate
    "correlation",           # correlation between two columns
    "anomaly_summary",       # summary of anomaly detection results
    "model_summary",         # supervised validation summary
    "data_quality_summary",  # quality checks summary
})

ALLOWED_AGGREGATIONS = frozenset({
    "sum", "mean", "median", "min", "max", "std", "count",
})

ALLOWED_FILTER_OPERATORS = frozenset({
    "==", "!=", ">", ">=", "<", "<=",
})

ALLOWED_DATE_PERIODS = frozenset({
    "day", "week", "month", "quarter", "year",
})

ALLOWED_SORT_DIRECTIONS = frozenset({"ascending", "descending"})

# Maximum rows returned in any result table (protects LLM context window)
MAX_RESULT_ROWS = 50

# Maximum allowed limit value from a query plan
MAX_PLAN_LIMIT = 100


# ============================================================
# SCHEMA BUILDER
# ============================================================

def build_schema(
    df: pd.DataFrame,
    profile: dict,
    eda_results: dict | None = None,
) -> dict:
    """
    Build a compact schema description of the DataFrame suitable for
    sending to Gemini as context.  Never includes raw row data.

    Returns a dict with:
      - columns: list of {name, dtype, role, sample_values, missing_pct}
      - row_count / column_count
      - numeric_columns / categorical_columns / datetime_columns
      - top correlations (from eda_results, if provided)
    """
    schema: dict[str, Any] = {
        "row_count": int(profile.get("rows", len(df))),
        "column_count": int(profile.get("columns", len(df.columns))),
        "numeric_columns": profile.get("numeric_columns", []),
        "categorical_columns": profile.get("categorical_columns", []),
        "datetime_columns": profile.get("datetime_columns", []),
        "columns": [],
    }

    numeric_set = set(profile.get("numeric_columns", []))
    categorical_set = set(profile.get("categorical_columns", []))
    datetime_set = set(profile.get("datetime_columns", []))

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        missing_pct = round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0.0

        # Sample up to 5 non-null values as examples (never send bulk data)
        sample_vals = (
            series.dropna()
            .head(5)
            .astype(str)
            .tolist()
        )

        if col in numeric_set:
            role = "numeric"
        elif col in datetime_set:
            role = "datetime"
        elif col in categorical_set:
            role = "categorical"
        else:
            role = "other"

        schema["columns"].append({
            "name": col,
            "dtype": str(series.dtype),
            "role": role,
            "missing_pct": missing_pct,
            "sample_values": sample_vals,
        })

    # Include top-5 correlations as context if available
    if eda_results and eda_results.get("correlations"):
        schema["top_correlations"] = eda_results["correlations"][:5]

    return schema


# ============================================================
# VALIDATION
# ============================================================

def _col_exists(col: str, df: pd.DataFrame) -> bool:
    return col in df.columns


def _is_numeric(col: str, df: pd.DataFrame) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _is_categorical(col: str, df: pd.DataFrame) -> bool:
    return df[col].dtype == "object" or hasattr(df[col], "cat")


def _is_datetime_compatible(col: str, df: pd.DataFrame) -> bool:
    """True if column is datetime or can be parsed as dates."""
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    if df[col].dtype == "object":
        sample = df[col].dropna().head(20)
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8
    return False


def validate_plan(
    plan: dict,
    df: pd.DataFrame,
    profile: dict,
) -> tuple[bool, str]:
    """
    Validate a query plan against the DataFrame schema and allowlists.

    Returns (True, "") on success, or (False, "human-readable error") on failure.
    Never raises; always returns a tuple.
    """
    if not isinstance(plan, dict):
        return False, "Query plan must be a JSON object."

    op = plan.get("operation")
    if not op:
        return False, "Query plan is missing the 'operation' field."
    if op not in ALLOWED_OPERATIONS:
        allowed = ", ".join(sorted(ALLOWED_OPERATIONS))
        return False, (
            f"Operation '{op}' is not supported. "
            f"Supported operations: {allowed}."
        )

    # ---- operation-specific validation ----

    if op in ("basic_stats", "distinct_count", "value_counts"):
        col = plan.get("column")
        if not col:
            return False, f"'{op}' requires a 'column' field."
        if not _col_exists(col, df):
            return False, f"Column '{col}' does not exist in this dataset."

    if op == "basic_stats":
        col = plan["column"]
        if not _is_numeric(col, df):
            return False, (
                f"Column '{col}' is not numeric. "
                "basic_stats requires a numeric column."
            )

    if op == "group_by_aggregate":
        for field in ("group_by", "measure_column", "aggregation"):
            if not plan.get(field):
                return False, f"'group_by_aggregate' requires '{field}'."
        if not _col_exists(plan["group_by"], df):
            return False, f"group_by column '{plan['group_by']}' does not exist."
        if not _col_exists(plan["measure_column"], df):
            return False, f"measure_column '{plan['measure_column']}' does not exist."
        agg = plan["aggregation"]
        if agg not in ALLOWED_AGGREGATIONS:
            return False, f"Aggregation '{agg}' is not allowed. Use: {', '.join(sorted(ALLOWED_AGGREGATIONS))}."
        if agg not in ("count",) and not _is_numeric(plan["measure_column"], df):
            return False, (
                f"Aggregation '{agg}' requires a numeric measure_column. "
                f"'{plan['measure_column']}' is not numeric."
            )
        sort = plan.get("sort", "descending")
        if sort not in ALLOWED_SORT_DIRECTIONS:
            return False, f"'sort' must be 'ascending' or 'descending'."
        limit = plan.get("limit", 10)
        if not isinstance(limit, int) or limit < 1 or limit > MAX_PLAN_LIMIT:
            return False, f"'limit' must be an integer between 1 and {MAX_PLAN_LIMIT}."

    if op in ("top_n", "bottom_n"):
        for field in ("sort_column",):
            if not plan.get(field):
                return False, f"'{op}' requires '{field}'."
        if not _col_exists(plan["sort_column"], df):
            return False, f"sort_column '{plan['sort_column']}' does not exist."
        limit = plan.get("limit", 10)
        if not isinstance(limit, int) or limit < 1 or limit > MAX_PLAN_LIMIT:
            return False, f"'limit' must be between 1 and {MAX_PLAN_LIMIT}."

    if op == "filtered_aggregate":
        for field in ("filter_column", "filter_operator", "filter_value",
                      "measure_column", "aggregation"):
            if plan.get(field) is None:
                return False, f"'filtered_aggregate' requires '{field}'."
        if not _col_exists(plan["filter_column"], df):
            return False, f"filter_column '{plan['filter_column']}' does not exist."
        if not _col_exists(plan["measure_column"], df):
            return False, f"measure_column '{plan['measure_column']}' does not exist."
        if plan["filter_operator"] not in ALLOWED_FILTER_OPERATORS:
            return False, (
                f"filter_operator '{plan['filter_operator']}' is not allowed. "
                f"Use: {', '.join(sorted(ALLOWED_FILTER_OPERATORS))}."
            )
        agg = plan["aggregation"]
        if agg not in ALLOWED_AGGREGATIONS:
            return False, f"Aggregation '{agg}' is not allowed."
        if agg not in ("count",) and not _is_numeric(plan["measure_column"], df):
            return False, f"Aggregation '{agg}' requires a numeric measure_column."

    if op == "date_group_aggregate":
        for field in ("date_column", "measure_column", "aggregation", "period"):
            if not plan.get(field):
                return False, f"'date_group_aggregate' requires '{field}'."
        if not _col_exists(plan["date_column"], df):
            return False, f"date_column '{plan['date_column']}' does not exist."
        if not _col_exists(plan["measure_column"], df):
            return False, f"measure_column '{plan['measure_column']}' does not exist."
        if not _is_datetime_compatible(plan["date_column"], df):
            return False, f"Column '{plan['date_column']}' cannot be parsed as dates."
        if plan["period"] not in ALLOWED_DATE_PERIODS:
            return False, (
                f"period '{plan['period']}' is not supported. "
                f"Use: {', '.join(sorted(ALLOWED_DATE_PERIODS))}."
            )
        agg = plan["aggregation"]
        if agg not in ALLOWED_AGGREGATIONS:
            return False, f"Aggregation '{agg}' is not allowed."
        if agg not in ("count",) and not _is_numeric(plan["measure_column"], df):
            return False, f"Aggregation '{agg}' requires a numeric measure_column."

    if op == "correlation":
        for field in ("column_1", "column_2"):
            if not plan.get(field):
                return False, f"'correlation' requires '{field}'."
        for field in ("column_1", "column_2"):
            col = plan[field]
            if not _col_exists(col, df):
                return False, f"Column '{col}' does not exist."
            if not _is_numeric(col, df):
                return False, f"Column '{col}' is not numeric. Correlation requires numeric columns."

    return True, ""


# ============================================================
# HELPERS
# ============================================================

def _safe_float(val) -> Any:
    """Convert to float; return None if NaN/Inf."""
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _apply_aggregation(series: pd.Series, agg: str):
    """Apply a named aggregation to a numeric series."""
    series = series.dropna()
    if len(series) == 0:
        return None
    ops = {
        "sum":    series.sum,
        "mean":   series.mean,
        "median": series.median,
        "min":    series.min,
        "max":    series.max,
        "std":    series.std,
        "count":  series.count,
    }
    return _safe_float(ops[agg]())


def _period_key(period: str) -> str:
    """Map plan period name to pandas Period freq alias.

    Note: dt.to_period() uses the legacy aliases (M, Q, Y),
    NOT the newer offset aliases (ME, QE, YE) that date_range accepts.
    """
    return {
        "day": "D", "week": "W", "month": "M",
        "quarter": "Q", "year": "A",
    }[period]


# ============================================================
# EXECUTION
# ============================================================

def execute_plan(
    plan: dict,
    df: pd.DataFrame,
    profile: dict,
    anomaly_results: dict | None = None,
    supervised_validation: dict | None = None,
) -> dict:
    """
    Execute a validated query plan against the DataFrame.

    Returns a result dict always containing:
      - operation: the executed operation name
      - approximate: bool — True if result is from a sample
      - data: the computed result (varies by operation)
      - row_count_used: how many rows were analyzed
      - error: None on success, str on failure

    NEVER executes arbitrary code.  All operations are explicitly coded.
    """
    op = plan.get("operation")
    n_rows = len(df)

    # ---- dataset_overview ----------------------------------
    if op == "dataset_overview":
        numeric_set  = set(profile.get("numeric_columns", []))
        cat_set      = set(profile.get("categorical_columns", []))
        datetime_set = set(profile.get("datetime_columns", []))

        # Vectorised missing counts in one pass
        missing_counts = df.isna().sum()

        col_details = []
        for col in df.columns:
            series = df[col]
            n_missing = int(missing_counts[col])
            missing_pct = round(n_missing / n_rows * 100, 2) if n_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            n_non_null = n_rows - n_missing

            if col in numeric_set:
                role = "numeric"
            elif col in datetime_set:
                role = "datetime"
            elif col in cat_set:
                role = "categorical"
            else:
                role = "other"

            detail: dict = {
                "column": col,
                "dtype": str(series.dtype),
                "role": role,
                "non_null": n_non_null,
                "missing_count": n_missing,
                "missing_pct": missing_pct,
                "unique_count": n_unique,
            }

            # Numeric stats — compact, no raw rows
            if role == "numeric":
                num = series.dropna()
                if len(num) > 0:
                    detail["stats"] = {
                        "mean":   _safe_float(num.mean()),
                        "median": _safe_float(num.median()),
                        "min":    _safe_float(num.min()),
                        "max":    _safe_float(num.max()),
                        "std":    _safe_float(num.std()),
                    }

            # Categorical top-5 values — only value labels, no raw record indices
            elif role in ("categorical", "other"):
                top = series.dropna().value_counts().head(5)
                detail["top_values"] = [
                    {"value": str(k), "count": int(v)}
                    for k, v in top.items()
                ]

            col_details.append(detail)

        # Columns with any missing data (compact summary for Gemini)
        missing_summary = {
            col: {"missing_count": int(cnt),
                  "missing_pct": round(cnt / n_rows * 100, 2) if n_rows > 0 else 0.0}
            for col, cnt in missing_counts.items()
            if cnt > 0
        }

        return {
            "operation": op,
            "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "rows": n_rows,
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "numeric_columns": profile.get("numeric_columns", []),
                "categorical_columns": profile.get("categorical_columns", []),
                "datetime_columns": profile.get("datetime_columns", []),
                "missing_summary": missing_summary,
                "column_details": col_details,
            },
            "error": None,
        }

    # ---- schema_info ----------------------------------------
    if op == "schema_info":
        cols = []
        for col in df.columns:
            series = df[col]
            cols.append({
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique()),
                "sample": series.dropna().head(3).astype(str).tolist(),
            })
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "rows": n_rows,
                "columns": len(df.columns),
                "column_details": cols,
            },
            "error": None,
        }

    # ---- row_count ------------------------------------------
    if op == "row_count":
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "rows": n_rows,
                "columns": len(df.columns),
                "column_names": list(df.columns),
            },
            "error": None,
        }

    # ---- missing_values -------------------------------------
    if op == "missing_values":
        col = plan.get("column")
        if col:
            series = df[col]
            count = int(series.isna().sum())
            pct = round(count / n_rows * 100, 2) if n_rows > 0 else 0.0
            data = [{"column": col, "missing_count": count, "missing_pct": pct}]
        else:
            # Vectorised: compute all missing counts in one pass.
            missing_counts = df.isna().sum()
            data = [
                {
                    "column": c,
                    "missing_count": int(cnt),
                    "missing_pct": round(cnt / n_rows * 100, 2) if n_rows > 0 else 0.0,
                }
                for c, cnt in missing_counts.items()
                if cnt > 0
            ]
            if not data:
                data = [{"message": "No missing values found in any column."}]
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # ---- distinct_count -------------------------------------
    if op == "distinct_count":
        col = plan["column"]
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {"column": col, "distinct_count": int(df[col].nunique())},
            "error": None,
        }

    # ---- basic_stats ----------------------------------------
    if op == "basic_stats":
        col = plan["column"]
        series = df[col].dropna()
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "column": col,
                "count": int(len(series)),
                "sum": _safe_float(series.sum()),
                "mean": _safe_float(series.mean()),
                "median": _safe_float(series.median()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "q1": _safe_float(series.quantile(0.25)),
                "q3": _safe_float(series.quantile(0.75)),
                "skewness": _safe_float(series.skew()),
            },
            "error": None,
        }

    # ---- value_counts ---------------------------------------
    if op == "value_counts":
        col = plan["column"]
        limit = min(int(plan.get("limit", 10)), MAX_RESULT_ROWS)
        # Drop nulls once; reuse for both frequency table and total count.
        non_null = df[col].dropna()
        vc = non_null.value_counts().head(limit)
        total = len(non_null)
        data = [
            {
                "value": str(k),
                "count": int(v),
                "percentage": round(v / total * 100, 2) if total > 0 else 0,
            }
            for k, v in vc.items()
        ]
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # ---- group_by_aggregate ---------------------------------
    if op == "group_by_aggregate":
        group_col = plan["group_by"]
        measure_col = plan["measure_column"]
        agg = plan["aggregation"]
        sort = plan.get("sort", "descending")
        limit = min(int(plan.get("limit", 10)), MAX_RESULT_ROWS)

        if agg == "count":
            grouped = (
                df.groupby(group_col, observed=True)
                .size()
                .reset_index(name="result")
            )
        else:
            grouped = (
                df.groupby(group_col, observed=True)[measure_col]
                .agg(agg)
                .reset_index()
            )
            grouped.columns = [group_col, "result"]

        grouped["result"] = grouped["result"].apply(_safe_float)
        ascending = (sort == "ascending")
        grouped = grouped.sort_values("result", ascending=ascending).head(limit)

        data = grouped.rename(columns={"result": agg}).to_dict("records")
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # ---- top_n / bottom_n -----------------------------------
    if op in ("top_n", "bottom_n"):
        sort_col = plan["sort_column"]
        display_cols = plan.get("display_columns") or list(df.columns[:6])
        # Ensure sort_col is included
        if sort_col not in display_cols:
            display_cols = [sort_col] + display_cols
        display_cols = [c for c in display_cols if c in df.columns][:8]

        limit = min(int(plan.get("limit", 10)), MAX_RESULT_ROWS)
        ascending = (op == "bottom_n")
        result_df = (
            df[display_cols]
            .sort_values(sort_col, ascending=ascending)
            .head(limit)
        )
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": result_df.astype(str).to_dict("records"),
            "error": None,
        }

    # ---- filtered_aggregate ---------------------------------
    if op == "filtered_aggregate":
        fcol = plan["filter_column"]
        fop = plan["filter_operator"]
        fval = plan["filter_value"]
        measure_col = plan["measure_column"]
        agg = plan["aggregation"]

        # Build mask safely (no eval)
        try:
            series = df[fcol]
            # Coerce value to column's dtype for numeric comparisons
            if pd.api.types.is_numeric_dtype(series):
                fval_typed = float(fval)
            else:
                fval_typed = str(fval)

            mask_map = {
                "==": series == fval_typed,
                "!=": series != fval_typed,
                ">":  series > fval_typed,
                ">=": series >= fval_typed,
                "<":  series < fval_typed,
                "<=": series <= fval_typed,
            }
            mask = mask_map[fop]
            filtered = df[mask]
        except Exception as exc:
            return {
                "operation": op, "approximate": False,
                "row_count_used": n_rows,
                "data": None,
                "error": f"Filter error: {exc}",
            }

        n_filtered = len(filtered)
        result_val = _apply_aggregation(filtered[measure_col], agg)
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "filter": f"{fcol} {fop} {fval}",
                "matching_rows": n_filtered,
                "measure_column": measure_col,
                "aggregation": agg,
                "result": result_val,
            },
            "error": None,
        }

    # ---- date_group_aggregate -------------------------------
    if op == "date_group_aggregate":
        date_col = plan["date_column"]
        measure_col = plan["measure_column"]
        agg = plan["aggregation"]
        period = plan["period"]
        limit = min(int(plan.get("limit", 50)), MAX_RESULT_ROWS)

        # Avoid a full .copy() when the column is already datetime-typed.
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            parsed_dates = df[date_col]
        else:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

        # Work only on the two relevant columns via a lightweight assign.
        temp = df[[date_col, measure_col]].assign(**{date_col: parsed_dates})
        temp = temp.dropna(subset=[date_col])

        if temp.empty:
            return {
                "operation": op, "approximate": False,
                "row_count_used": n_rows,
                "data": [], "error": "No valid date values found after parsing.",
            }

        freq = _period_key(period)
        temp["_period"] = temp[date_col].dt.to_period(freq).astype(str)

        if agg == "count":
            grouped = temp.groupby("_period").size().reset_index(name="result")
        else:
            grouped = temp.groupby("_period")[measure_col].agg(agg).reset_index()
            grouped.columns = ["_period", "result"]

        grouped["result"] = grouped["result"].apply(_safe_float)
        grouped = grouped.sort_values("_period").tail(limit)
        data = [
            {"period": r["_period"], agg: r["result"]}
            for _, r in grouped.iterrows()
        ]
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # ---- correlation ----------------------------------------
    if op == "correlation":
        col1 = plan["column_1"]
        col2 = plan["column_2"]
        pair = df[[col1, col2]].dropna()
        if len(pair) < 2:
            return {
                "operation": op, "approximate": False,
                "row_count_used": n_rows,
                "data": None,
                "error": "Not enough non-null rows to compute correlation.",
            }
        corr_val = _safe_float(pair[col1].corr(pair[col2]))
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": {
                "column_1": col1,
                "column_2": col2,
                "pearson_r": corr_val,
                "pairs_used": len(pair),
            },
            "error": None,
        }

    # ---- anomaly_summary ------------------------------------
    if op == "anomaly_summary":
        if not anomaly_results:
            return {
                "operation": op, "approximate": False,
                "row_count_used": n_rows,
                "data": {"message": "No anomaly results available."},
                "error": None,
            }
        ml = anomaly_results.get("multivariate_isolation_forest", {})
        iqr = anomaly_results.get("iqr", {})
        iqr_summary = {
            col: {
                "anomaly_count": r.get("anomaly_count"),
                "anomaly_pct": r.get("anomaly_percentage"),
                "bounds": f"{r.get('lower_bound')} to {r.get('upper_bound')}",
            }
            for col, r in iqr.items()
        }
        data = {
            "iqr_outliers": iqr_summary,
            "isolation_forest": {
                "status": ml.get("status"),
                "anomaly_count": ml.get("anomaly_count"),
                "anomaly_pct": ml.get("anomaly_percentage"),
                "rows_analyzed": ml.get("rows_analyzed"),
                "features_used": ml.get("features_used", []),
            },
        }
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # ---- model_summary --------------------------------------
    if op == "model_summary":
        if not supervised_validation:
            return {
                "operation": op, "approximate": False,
                "row_count_used": n_rows,
                "data": {"message": "No supervised model validation results available."},
                "error": None,
            }
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": supervised_validation,
            "error": None,
        }

    # ---- data_quality_summary --------------------------------
    if op == "data_quality_summary":
        # Pull quality-related fields from profile
        data = {
            "rows": profile.get("rows", n_rows),
            "columns": profile.get("columns", len(df.columns)),
            "duplicate_rows": profile.get("duplicate_rows", 0),
            "columns_with_missing": {
                c: int(df[c].isna().sum())
                for c in df.columns
                if df[c].isna().any()
            },
            "constant_columns": [
                c for c in df.columns
                if df[c].nunique(dropna=False) <= 1
            ],
        }
        return {
            "operation": op, "approximate": False,
            "row_count_used": n_rows,
            "data": data, "error": None,
        }

    # Should not reach here if validate_plan ran first
    return {
        "operation": op, "approximate": False,
        "row_count_used": n_rows,
        "data": None,
        "error": f"Operation '{op}' has no executor. This is a bug.",
    }
