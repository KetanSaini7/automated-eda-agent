import pandas as pd


# Columns are skipped from title-casing if they look like identifiers,
# codes, emails, or have too many unique values relative to the dataset.

_ID_PATTERNS = (
    "id", "uuid", "guid", "key", "code", "hash",
    "email", "url", "phone", "ip", "address", "token",
    "ref", "number", "num", "no",
)

# If a column's unique-value ratio exceeds this threshold it is treated as
# a high-cardinality / identifier column and skipped.
_HIGH_CARDINALITY_RATIO = 0.5

# Minimum number of rows before the cardinality ratio rule activates.
# Below this count, almost any column looks "high cardinality" by ratio alone,
# so we require an absolute unique-count minimum instead.
_MIN_ROWS_FOR_RATIO = 50

# Absolute minimum unique-value count for the ratio rule to apply.
# Columns with fewer unique values than this are never flagged by ratio alone.
_MIN_UNIQUE_FOR_RATIO = 10

# Maximum length of an all-caps token to be treated as a short code (e.g. "USD",
# "OK", "GB"). Tokens of 4+ chars like "EAST", "SOUTH" are regular label values.
_MAX_CODE_LENGTH = 3


def _is_id_like_column(column_name: str, series: pd.Series) -> bool:
    """
    Return True if the column should be excluded from title-casing because
    it looks like an identifier, code, email, URL, or high-cardinality field.

    Rules (any one is sufficient):
    1. Column name contains a known ID-like keyword (case-insensitive).
    2. High-cardinality ratio > 0.5, but only when the dataset has at least
       50 rows AND the column has at least 10 unique values (guards against
       false positives on tiny DataFrames).
    3. Any sampled value contains '@' (email) or '://' (URL).
    4. Any sampled value is a SHORT all-caps alpha token (<= 4 chars), e.g.
       "USD", "OK", "GB" — but NOT "SOUTH" or "NORTH" (too long).
    """
    name_lower = column_name.lower()

    # Rule 1 — keyword in column name
    if any(pat in name_lower for pat in _ID_PATTERNS):
        return True

    # Rule 2 — high cardinality (only on sufficiently large datasets)
    n_total = len(series)
    n_unique = series.nunique(dropna=True)
    if (
        n_total >= _MIN_ROWS_FOR_RATIO
        and n_unique >= _MIN_UNIQUE_FOR_RATIO
        and n_unique / n_total > _HIGH_CARDINALITY_RATIO
    ):
        return True

    # Rule 3 & 4 — value-pattern heuristic (sample up to 50 non-null values)
    sample = series.dropna().astype(str).head(50)
    for val in sample:
        stripped = val.strip()
        # Email or URL
        if "@" in stripped or "://" in stripped:
            return True
        # Short all-caps alpha token (currency code, status code, country code)
        if (
            stripped
            and len(stripped) <= _MAX_CODE_LENGTH
            and stripped == stripped.upper()
            and stripped.isalpha()
        ):
            return True

    return False


def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical columns by:
    - Removing leading / trailing whitespace
    - Collapsing multiple internal spaces to one
    - Applying title-case ONLY to columns that look like genuine label
      columns (city, region, category, etc.) — NOT to ID columns, email
      addresses, URLs, codes, or high-cardinality fields.

    Example of what IS changed:
        "  north  " → "North"
        "new  york" → "New York"

    Example of what is intentionally NOT changed:
        "USD"       → "USD"   (all-caps code)
        "user@x.com"→ "user@x.com"   (email)
        customer_id column → unchanged entirely
    """
    cleaned_df = df.copy()

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        series = cleaned_df[column]

        # --- Step 1: strip whitespace and collapse spaces (always safe) ---
        cleaned_df[column] = series.apply(
            lambda x: " ".join(x.split()) if isinstance(x, str) else x
        )

        # --- Step 2: title-case only for genuine label columns -----------
        if _is_id_like_column(column, cleaned_df[column]):
            # Skip title-casing for this column
            continue

        cleaned_df[column] = cleaned_df[column].apply(
            lambda x: x.title() if isinstance(x, str) else x
        )

    return cleaned_df
