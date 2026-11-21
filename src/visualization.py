import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# HELPERS
# ============================================================

def _sample_label(n_sample: int, n_total: int) -> str:
    """Return a subtitle string when a chart is drawn from a sample."""
    return f"(sample of {n_sample:,} / {n_total:,} rows)"


# ============================================================
# NUMERICAL HISTOGRAM
# ============================================================

def create_numeric_histogram(df, column, is_sample=False, total_rows=None):
    """
    Create a histogram for a numerical column.

    Args:
        df: DataFrame to plot from (may be a sample).
        column: Name of the numeric column.
        is_sample: True when df is a down-sampled subset of a larger dataset.
        total_rows: Total row count of the original dataset (used for label).

    Returns:
        Plotly figure, or None if the column is constant or all-NaN.
    """
    series = df[column].dropna()

    if series.empty:
        return None

    # Constant column → a histogram with one bar conveys nothing useful
    if series.nunique() <= 1:
        return None

    title = f"Distribution of {column}"
    if is_sample and total_rows:
        title += f"<br><sup>{_sample_label(len(df), total_rows)}</sup>"

    fig = px.histogram(
        df,
        x=column,
        title=title,
        marginal="box"
    )

    return fig


# ============================================================
# CATEGORICAL BAR CHART
# ============================================================

def create_categorical_bar(df, column, top_n=10, is_sample=False, total_rows=None):
    """
    Create a bar chart for the most frequent categorical values.

    Args:
        df: DataFrame to plot from (may be a sample).
        column: Name of the categorical column.
        top_n: Maximum number of categories to show.
        is_sample: True when df is a down-sampled subset of a larger dataset.
        total_rows: Total row count of the original dataset (used for label).

    Returns:
        Plotly figure, or None if the column has no non-null values.
    """
    value_counts = (
        df[column]
        .dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    if value_counts.empty:
        return None

    value_counts.columns = [column, "Count"]

    title = f"Top {top_n} {column} Values"
    if is_sample and total_rows:
        title += f"<br><sup>{_sample_label(len(df), total_rows)}</sup>"

    fig = px.bar(
        value_counts,
        x=column,
        y="Count",
        title=title
    )

    return fig


# ============================================================
# SCATTER PLOT
# ============================================================

def create_scatter_plot(df, x_column, y_column, is_sample=False, total_rows=None):
    """
    Create a scatter plot between two numerical columns.
    """
    title = f"{x_column} vs {y_column}"
    if is_sample and total_rows:
        title += f"<br><sup>{_sample_label(len(df), total_rows)}</sup>"

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=title
    )

    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def create_correlation_heatmap(df, is_sample=False, total_rows=None):
    """
    Create a correlation heatmap for numerical columns.

    Returns:
        Plotly figure, or None if fewer than 2 usable numeric columns exist.
    """
    numeric_df = df.select_dtypes(include="number")

    # Drop columns that are entirely NaN (corr() would return NaN rows/cols)
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Drop constant columns (zero variance → NaN correlation)
    numeric_df = numeric_df.loc[
        :, numeric_df.nunique() > 1
    ]

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr()

    title = "Correlation Heatmap"
    if is_sample and total_rows:
        title += f"<br><sup>{_sample_label(len(df), total_rows)}</sup>"

    fig = px.imshow(
        correlation,
        text_auto=True,
        title=title
    )

    return fig


# ============================================================
# TIME SERIES
# ============================================================

def create_time_series(df, date_column, value_column, is_sample=False, total_rows=None):
    """
    Create a time-series chart using a datetime column and a numerical column.

    Returns:
        Plotly figure, or None if no valid rows remain after parsing/dropna.
    """
    temp_df = df[[date_column, value_column]].copy()

    temp_df[date_column] = pd.to_datetime(
        temp_df[date_column],
        errors="coerce"
    )

    temp_df = temp_df.dropna()

    if temp_df.empty:
        return None

    temp_df = temp_df.sort_values(date_column)

    title = f"{value_column} over Time"
    if is_sample and total_rows:
        title += f"<br><sup>{_sample_label(len(df), total_rows)}</sup>"

    fig = px.line(
        temp_df,
        x=date_column,
        y=value_column,
        title=title
    )

    return fig
