"""
ask_your_data.py
================
Orchestrates the full "Ask Your Data" pipeline:

  1. Build compact schema (no raw data).
  2. Send schema + question to Gemini → structured query plan.
  3. Validate plan against DataFrame schema + allowlists.
  4. Execute plan in Python/Pandas (never eval/exec).
  5. Send compact results to Gemini for plain-language explanation.
  6. Optionally generate a Plotly chart.
  7. Return structured response dict.

Security rules (prompt.md §8):
  - NEVER use eval(), exec(), or generated Python.
  - NEVER send raw dataset rows to Gemini.
  - Gemini only receives schema + compact result summary.
  - All calculations are performed by Python/Pandas.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# Maximum rows to include in Gemini's explanation prompt
_MAX_EXPLAIN_ROWS = 30

# Operations whose results are self-explanatory scalars/tables — skip Gemini explanation.
# Gemini is still used for analytical operations where narrative adds real value.
_LOCAL_ANSWER_OPS = frozenset({
    "row_count",        # just numbers — no narrative needed
    "basic_stats",      # structured dict of well-named stats
    "distinct_count",   # single number
    "missing_values",   # structured table, self-describing
    "schema_info",      # structured table
})

# Explanation prompt template
_EXPLAIN_PROMPT = """You are a data analyst assistant explaining Python-computed results.

QUESTION: {question}

OPERATION: {operation}

PYTHON-COMPUTED RESULTS:
{results_json}

INSTRUCTIONS:
- Explain these results clearly and concisely in plain English.
- Do NOT recalculate, verify, or second-guess the numbers above.
- Do NOT fabricate any statistics not present in the results.
- Do NOT claim correlation implies causation.
- If results show approximate values, mention that.
- If results are empty or show no data, explain that clearly.
- Keep the explanation under 200 words.
- Do not use markdown headers.
"""


# ============================================================
# LOCAL ANSWER FORMATTER
# ============================================================

def _format_local_answer(result: dict) -> str | None:
    """
    For self-explanatory operations, format a clear plain-English answer
    directly from the computed result dict — without calling Gemini.

    Returns a formatted string, or None if the operation requires Gemini.
    """
    op = result.get("operation")
    data = result.get("data")
    n = result.get("row_count_used", "?")

    if op == "row_count" and isinstance(data, dict):
        rows = data.get("rows", "?")
        cols = data.get("columns", "?")
        return (
            f"The dataset contains **{rows:,}** rows and **{cols}** columns."
            if isinstance(rows, int) else
            f"The dataset contains {rows} rows and {cols} columns."
        )

    if op == "distinct_count" and isinstance(data, dict):
        col = data.get("column", "?")
        count = data.get("distinct_count", "?")
        return (
            f"Column **{col}** has **{count:,}** distinct (unique) values out of {n:,} rows."
            if isinstance(count, int) and isinstance(n, int) else
            f"Column '{col}' has {count} distinct values."
        )

    if op == "basic_stats" and isinstance(data, dict):
        col = data.get("column", "?")
        mean = data.get("mean")
        median = data.get("median")
        mn = data.get("min")
        mx = data.get("max")
        total = data.get("sum")
        std = data.get("std")
        parts = [f"Summary statistics for **{col}**:"]
        if mean is not None:   parts.append(f"- Mean: **{mean:,.4g}**")
        if median is not None: parts.append(f"- Median: **{median:,.4g}**")
        if mn is not None:     parts.append(f"- Min: **{mn:,.4g}**")
        if mx is not None:     parts.append(f"- Max: **{mx:,.4g}**")
        if total is not None:  parts.append(f"- Sum: **{total:,.4g}**")
        if std is not None:    parts.append(f"- Std Dev: **{std:,.4g}**")
        return "\n".join(parts)

    if op == "missing_values":
        if isinstance(data, list) and len(data) == 1 and "message" in data[0]:
            return "No missing values were found in any column of this dataset."
        if isinstance(data, list):
            total_missing = sum(r.get("missing_count", 0) for r in data)
            cols_with_missing = len(data)
            return (
                f"Found **{total_missing:,}** missing values across "
                f"**{cols_with_missing}** column(s). See the table below for details."
            )

    if op == "schema_info" and isinstance(data, dict):
        rows = data.get("rows", "?")
        cols = data.get("columns", "?")
        return (
            f"This dataset has **{rows:,}** rows and **{cols}** columns. "
            "The table below shows each column's data type, non-null count, "
            "number of missing values, and unique value count."
        )

    return None  # Operation needs Gemini explanation


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _truncate_result_for_explain(result: dict) -> dict:
    """
    Return a compact version of the result dict for Gemini's explanation prompt.
    Lists are truncated to _MAX_EXPLAIN_ROWS items.  Never includes raw data.
    """
    data = result.get("data")
    if isinstance(data, list) and len(data) > _MAX_EXPLAIN_ROWS:
        data = data[:_MAX_EXPLAIN_ROWS]
        truncated = True
    else:
        truncated = False

    return {
        "operation": result.get("operation"),
        "approximate": result.get("approximate", False),
        "row_count_used": result.get("row_count_used"),
        "data": data,
        "truncated_to": _MAX_EXPLAIN_ROWS if truncated else None,
        "error": result.get("error"),
    }


def _get_explanation(
    question: str,
    result: dict,
    client,
    _call_with_retry,
) -> str:
    """Call Gemini to explain the computed result in plain English."""
    from src.llm_agent import (
        _extract_text_from_interaction,
        _extract_text_from_response,
        MODEL_NAME,
    )
    compact = _truncate_result_for_explain(result)
    results_json = json.dumps(compact, ensure_ascii=False, indent=2)
    prompt = _EXPLAIN_PROMPT.format(
        question=question.strip(),
        operation=result.get("operation", "unknown"),
        results_json=results_json,
    )
    try:
        # Try Interactions API first (native gemini-3.8-flash path with thinking)
        explanation = None
        try:
            interaction = _call_with_retry(
                client.interactions.create,
                model=MODEL_NAME,
                input=prompt,
                generation_config={"thinking_level": "low"},
            )
            explanation = _extract_text_from_interaction(interaction)
        except Exception:
            explanation = None

        # Fallback to generate_content if interactions returned nothing
        if not explanation:
            response = _call_with_retry(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
            )
            explanation = _extract_text_from_response(response)

        return explanation.strip() if explanation else "No explanation returned."
    except Exception as exc:
        logger.warning("Explanation API call failed: %s", exc)
        return f"Results computed successfully. (Explanation unavailable: {exc})"


def _get_overview_explanation(
    result: dict,
    client,
    _call_with_retry,
) -> str:
    """
    Specialised Gemini call for the dataset_overview operation.
    Builds a richer prompt focused on describing what the dataset represents.
    Returns a structured markdown narrative grounded entirely in computed stats.
    """
    from src.llm_agent import (
        _extract_text_from_interaction,
        _extract_text_from_response,
        MODEL_NAME,
    )

    data = result.get("data", {})
    rows = data.get("rows", "?")
    cols = data.get("columns", "?")
    col_details = data.get("column_details", [])
    missing_summary = data.get("missing_summary", {})
    numeric_cols = data.get("numeric_columns", [])
    cat_cols = data.get("categorical_columns", [])
    dt_cols = data.get("datetime_columns", [])

    # Build a compact, human-readable stats block for Gemini
    # Only include what Gemini needs — never raw dataset rows
    stats_lines = [
        f"Rows: {rows:,}" if isinstance(rows, int) else f"Rows: {rows}",
        f"Columns: {cols}",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) or 'none'}",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) or 'none'}",
        f"Datetime columns ({len(dt_cols)}): {', '.join(dt_cols) or 'none'}",
    ]

    if missing_summary:
        missing_parts = [
            f"{c}: {v['missing_count']:,} ({v['missing_pct']}%)"
            for c, v in missing_summary.items()
        ]
        stats_lines.append("Columns with missing values: " + "; ".join(missing_parts))
    else:
        stats_lines.append("Missing values: none")

    stats_lines.append("")
    stats_lines.append("Per-column details:")
    for d in col_details:
        line = f"  {d['column']} ({d['dtype']}, {d['role']}): {d['non_null']:,} non-null, {d['unique_count']:,} unique"
        if "stats" in d:
            s = d["stats"]
            line += f" | min={s.get('min')}, mean={s.get('mean')}, max={s.get('max')}"
        elif "top_values" in d:
            top_vals = ", ".join(f"{t['value']} ({t['count']:,})" for t in d["top_values"][:5])
            line += f" | top values: {top_vals}"
        stats_lines.append(line)

    stats_text = "\n".join(stats_lines)

    prompt = f"""You are a senior data analyst describing a dataset to a business stakeholder.

Python has computed the following statistics about the dataset.
Use ONLY these facts. Do NOT invent column meanings, values, or patterns that are not in the statistics below.

--- DATASET STATISTICS ---
{stats_text}
--- END STATISTICS ---

Write a structured dataset overview in markdown. Use exactly these four sections:

**Dataset Overview**
A 2-3 sentence description of what this dataset likely represents, inferred from the column names and data types.
Clearly label any inference as "appears to" or "likely".
State the number of records and columns.

**Column Summary**
A concise description of each column: its name, data type/role, and what it appears to contain.
For categorical columns, mention the most common values.
For numeric columns, mention the range (min to max).

**Data Quality**
Note any missing values or data quality observations. If no missing values, state that explicitly.

**Key Observations**
2-3 specific, evidence-grounded observations about patterns, distributions, or notable features.
Every observation must cite specific numbers from the statistics above.

Write in a natural, professional tone. Do not use phrases like "based on the data provided" or "as computed"."""

    try:
        explanation = None
        try:
            interaction = _call_with_retry(
                client.interactions.create,
                model=MODEL_NAME,
                input=prompt,
                generation_config={"thinking_level": "low"},
            )
            explanation = _extract_text_from_interaction(interaction)
        except Exception:
            explanation = None

        if not explanation:
            response = _call_with_retry(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
            )
            explanation = _extract_text_from_response(response)

        return explanation.strip() if explanation else "Dataset overview computed. (Narrative unavailable.)"
    except Exception as exc:
        logger.warning("Overview explanation API call failed: %s", exc)
        return f"Dataset overview computed. (Narrative unavailable: {exc})"


def _result_to_dataframe(result: dict) -> pd.DataFrame | None:
    """Convert a result dict's data into a DataFrame for display, if applicable."""
    op = result.get("operation")
    data = result.get("data")
    if not data:
        return None

    # dataset_overview → extract column_details as a clean summary table
    if op == "dataset_overview" and isinstance(data, dict):
        col_details = data.get("column_details", [])
        if col_details:
            rows = []
            for d in col_details:
                rows.append({
                    "Column":    d.get("column", ""),
                    "Type":      d.get("dtype", ""),
                    "Role":      d.get("role", "").capitalize(),
                    "Non-Null":  f"{d.get('non_null', 0):,}",
                    "Missing":   f"{d.get('missing_count', 0):,} ({d.get('missing_pct', 0.0):.1f}%)",
                    "Unique":    f"{d.get('unique_count', 0):,}",
                })
            return pd.DataFrame(rows)
        return None

    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Skip generic single-message dicts
        if len(data) == 1 and "message" in data[0]:
            return None
        try:
            return pd.DataFrame(data)
        except Exception:
            return None
    if isinstance(data, dict):
        # Flatten simple key→value dicts into a 2-column table
        rows = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if rows:
            return pd.DataFrame(list(rows.items()), columns=["Metric", "Value"])
    return None


def _build_chart(result: dict) -> go.Figure | None:
    """
    Generate a Plotly chart from the result if appropriate.
    Returns None when no meaningful chart can be drawn.
    """
    op = result.get("operation")
    data = result.get("data")

    if not data or not isinstance(data, list) or len(data) < 2:
        return None

    try:
        df = pd.DataFrame(data)

        # group_by_aggregate / value_counts → bar chart
        if op in ("group_by_aggregate", "value_counts"):
            x_col = df.columns[0]
            y_col = df.columns[1]
            # Only plot if y is numeric-ish
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            df = df.dropna(subset=[y_col])
            if df.empty:
                return None
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"{y_col} by {x_col}",
                labels={x_col: x_col, y_col: y_col},
                template="plotly_white",
            )
            fig.update_layout(xaxis_tickangle=-30)
            return fig

        # date_group_aggregate → line chart
        if op == "date_group_aggregate":
            x_col = df.columns[0]   # period
            y_col = df.columns[1]   # aggregated value
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            df = df.dropna(subset=[y_col])
            if df.empty:
                return None
            fig = px.line(
                df, x=x_col, y=y_col,
                title=f"{y_col} over time",
                labels={x_col: "Period", y_col: y_col},
                markers=True,
                template="plotly_white",
            )
            return fig

        # top_n / bottom_n → horizontal bar chart if sort_column is numeric
        if op in ("top_n", "bottom_n") and len(df.columns) >= 2:
            # Use last numeric column as y, first text column as x
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
                x_col = df.columns[0]
                fig = px.bar(
                    df, x=x_col, y=y_col,
                    orientation="v",
                    title=f"{'Top' if op == 'top_n' else 'Bottom'} records by {y_col}",
                    template="plotly_white",
                )
                return fig

    except Exception as exc:
        logger.debug("Chart generation skipped: %s", exc)

    return None


# ============================================================
# PUBLIC API
# ============================================================

def answer_question(
    question: str,
    df: pd.DataFrame,
    profile: dict,
    eda_results: dict | None = None,
    anomaly_results: dict | None = None,
    supervised_validation: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Full Ask Your Data pipeline.

    Parameters
    ----------
    question : str
        User's natural-language question.
    df : pd.DataFrame
        The uploaded (cleaned) DataFrame.
    profile : dict
        Dataset profile from src/data_profiler.py.
    eda_results : dict, optional
        EDA results (correlations, etc.).
    anomaly_results : dict, optional
        Anomaly detection results.
    supervised_validation : dict, optional
        Supervised model validation results.
    conversation_history : list[dict], optional
        Prior chat turns [{role, content}, ...].

    Returns
    -------
    dict:
        answer   : str    — plain-language explanation
        data_df  : pd.DataFrame | None — tabular result for display
        chart    : plotly Figure | None — chart for display
        error    : str | None — error message (None on success)
        raw_result : dict  — raw execute_plan output (for debugging)
    """
    # Deferred imports to avoid circular dependencies
    from src.query_engine import build_schema, validate_plan, execute_plan
    from src.query_interpreter import interpret_question
    from src.llm_agent import _build_client, _call_with_retry

    # ---- Guard: empty inputs ----------------------------------------
    if df is None or df.empty:
        return {
            "answer": "No dataset is available. Please upload a CSV file first.",
            "data_df": None, "chart": None,
            "error": "No dataset.", "raw_result": {},
        }
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "data_df": None, "chart": None,
            "error": "Empty question.", "raw_result": {},
        }

    history = conversation_history or []

    # ---- Step 1: Build compact schema (no raw data) -----------------
    _t0 = time.perf_counter()
    try:
        schema = build_schema(df, profile, eda_results)
    except Exception as exc:
        logger.error("build_schema failed: %s", exc)
        return {
            "answer": "Failed to build dataset schema.",
            "data_df": None, "chart": None,
            "error": str(exc), "raw_result": {},
        }
    logger.debug("build_schema: %.1f ms", (time.perf_counter() - _t0) * 1000)

    # ---- Step 2: Gemini interprets question → query plan ------------
    _t1 = time.perf_counter()
    plan = interpret_question(question, schema, history)
    logger.debug("interpret_question: %.2f s", time.perf_counter() - _t1)

    # Handle special responses from interpreter
    if "clarification_needed" in plan:
        return {
            "answer": f"I need clarification: {plan['clarification_needed']}",
            "data_df": None, "chart": None,
            "error": None, "raw_result": plan,
        }
    if "unsupported" in plan:
        return {
            "answer": f"I cannot answer this type of question. {plan['unsupported']}",
            "data_df": None, "chart": None,
            "error": None, "raw_result": plan,
        }
    if "error" in plan:
        return {
            "answer": f"Could not interpret your question. {plan['error']}",
            "data_df": None, "chart": None,
            "error": plan["error"], "raw_result": plan,
        }

    # ---- Step 3: Validate plan --------------------------------------
    ok, validation_error = validate_plan(plan, df, profile)
    if not ok:
        return {
            "answer": (
                f"Your question could not be answered because the generated query "
                f"plan is invalid: {validation_error}"
            ),
            "data_df": None, "chart": None,
            "error": validation_error, "raw_result": plan,
        }

    # ---- Step 4: Execute plan in Python/Pandas ----------------------
    _t2 = time.perf_counter()
    try:
        result = execute_plan(
            plan, df, profile,
            anomaly_results=anomaly_results,
            supervised_validation=supervised_validation,
        )
    except Exception as exc:
        logger.error("execute_plan crashed: %s", exc)
        return {
            "answer": f"Query execution failed: {exc}",
            "data_df": None, "chart": None,
            "error": str(exc), "raw_result": plan,
        }
    logger.debug("execute_plan: %.1f ms", (time.perf_counter() - _t2) * 1000)

    # Execution-level error (e.g. filter type mismatch)
    if result.get("error"):
        return {
            "answer": f"Query execution error: {result['error']}",
            "data_df": None, "chart": None,
            "error": result["error"], "raw_result": result,
        }

    # ---- Step 5: Explain results ------------------------------------
    # For simple self-explanatory operations, format the answer locally
    # without a second Gemini call.  This saves 3-8 s per question for
    # the most common question types (counts, averages, missing values).
    op = plan.get("operation", "")
    _t3 = time.perf_counter()

    # dataset_overview uses a dedicated, richer explanation prompt
    if op == "dataset_overview":
        try:
            client = _build_client()
            explanation = _get_overview_explanation(result, client, _call_with_retry)
        except Exception as exc:
            logger.warning("Could not initialize Gemini for overview: %s", exc)
            explanation = "Dataset overview computed. (AI narrative unavailable.)"
        logger.debug("explain dataset_overview (Gemini): %.2f s", time.perf_counter() - _t3)

    elif op in _LOCAL_ANSWER_OPS:
        local_answer = _format_local_answer(result)
        if local_answer is not None:
            explanation = local_answer
            logger.debug("explain (local, no Gemini): %.1f ms", (time.perf_counter() - _t3) * 1000)
        else:
            # Fallback if formatter returned None unexpectedly
            try:
                client = _build_client()
                explanation = _get_explanation(question, result, client, _call_with_retry)
            except Exception as exc:
                logger.warning("Could not initialize Gemini for explanation: %s", exc)
                explanation = "Results computed. (AI explanation unavailable.)"
            logger.debug("explain (Gemini fallback): %.2f s", time.perf_counter() - _t3)

    else:
        # Gemini explains analytical results (group-by, date trends, correlations...)
        try:
            client = _build_client()
            explanation = _get_explanation(question, result, client, _call_with_retry)
        except Exception as exc:
            logger.warning("Could not initialize Gemini for explanation: %s", exc)
            explanation = "Results computed. (AI explanation unavailable.)"
        logger.debug("explain (Gemini): %.2f s", time.perf_counter() - _t3)
    # ---- Step 6: Convert result data to DataFrame for display -------
    data_df = _result_to_dataframe(result)

    # ---- Step 7: Optionally generate chart --------------------------
    chart = _build_chart(result)

    return {
        "answer": explanation,
        "data_df": data_df,
        "chart": chart,
        "error": None,
        "raw_result": result,
    }
