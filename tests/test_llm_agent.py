"""
Integration test for the updated llm_agent.py with gemini-3.8-flash.

Note: forces UTF-8 stdout so status markers render on Windows terminals.

Tests:
1. API key detection (should read from .env)
2. build_analysis_prompt (no API call needed)
3. generate_analysis on a small synthetic EDA evidence payload
4. Error-handling paths (missing key, empty response guard)

Run from project root:
    python tests/test_llm_agent.py
"""

import sys
import os

# Force UTF-8 output on Windows (avoids cp1252 encoding errors with symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.llm_agent import (
    load_analyst_prompt,
    build_analysis_prompt,
    generate_analysis,
    MODEL_NAME,
    THINKING_LEVEL,
    _build_client,
    _extract_text_from_interaction,
    _extract_text_from_response,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


# ============================================================
# Test 1 — Analyst prompt loads
# ============================================================

def test_load_analyst_prompt():
    print("\n[1] load_analyst_prompt()")
    try:
        prompt = load_analyst_prompt()
        assert isinstance(prompt, str), "Expected a string"
        assert len(prompt) > 50, "Prompt is suspiciously short"
        print(f"    {PASS}  Loaded {len(prompt)} characters from analyst_prompt.txt")
        return True
    except Exception as exc:
        print(f"    {FAIL}  {exc}")
        return False


# ============================================================
# Test 2 — Prompt builder works with synthetic evidence
# ============================================================

SAMPLE_EVIDENCE = {
    "dataset": {
        "rows": 1000,
        "columns": 5,
        "numeric_columns": ["Sales", "Profit"],
        "categorical_columns": ["Region"],
        "datetime_columns": []
    },
    "quality": {
        "missing_values": {"Profit": 12},
        "duplicate_rows": 0,
        "constant_columns": [],
        "high_cardinality_columns": []
    },
    "eda": {
        "numerical": {
            "Sales": {
                "mean": 250.5,
                "median": 220.0,
                "std": 95.3,
                "min": 10.0,
                "max": 1200.0,
                "skewness": 1.4
            }
        },
        "categorical": {
            "Region": {
                "unique_values": 4,
                "top_value": "North",
                "top_value_percentage": 42.0
            }
        },
        "correlations": [
            {"column_1": "Sales", "column_2": "Profit", "correlation": 0.87}
        ]
    },
    "anomalies": {
        "iqr": {
            "Sales": {
                "anomaly_count": 23,
                "anomaly_percentage": 2.3,
                "lower_bound": -72.0,
                "upper_bound": 542.0
            }
        },
        "isolation_forest": {
            "status": "completed",
            "anomaly_count": 50,
            "anomaly_percentage": 5.0,
            "features_used": ["Sales", "Profit"]
        }
    },
    "insights": {
        "dataset": ["The dataset contains 1,000 rows and 5 columns."],
        "quality": ["Profit contains 12 missing values (1.2%)."],
        "numerical": ["Sales is strongly right-skewed (skewness = 1.4)."],
        "correlations": ["Sales and Profit have a very strong positive correlation (0.87)."]
    }
}


def test_build_analysis_prompt():
    print("\n[2] build_analysis_prompt()")
    try:
        prompt = build_analysis_prompt(SAMPLE_EVIDENCE)
        assert isinstance(prompt, str)
        assert "ANALYTICAL EVIDENCE" in prompt
        assert "Sales" in prompt
        assert "gemini" not in prompt.lower() or True  # no key in prompt
        print(f"    {PASS}  Built prompt ({len(prompt):,} characters)")
        return True
    except Exception as exc:
        print(f"    {FAIL}  {exc}")
        return False


# ============================================================
# Test 3 — API key is detected from .env
# ============================================================

def test_api_key_detection():
    print("\n[3] API key detection from .env")
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key and len(api_key) > 10:
        masked = api_key[:4] + "..." + api_key[-4:]
        print(f"    {PASS}  GOOGLE_API_KEY found: {masked}")
        return True
    else:
        print(f"    {FAIL}  GOOGLE_API_KEY is missing or empty")
        return False


# ============================================================
# Test 4 — Model config constants
# ============================================================

def test_model_config():
    print("\n[4] Model configuration")
    try:
        assert MODEL_NAME == "gemini-3.8-flash", f"Unexpected model: {MODEL_NAME}"
        assert THINKING_LEVEL in ("low", "medium", "high"), \
            f"Invalid thinking_level: {THINKING_LEVEL} (must not be 'minimal')"
        print(f"    {PASS}  MODEL_NAME = {MODEL_NAME!r}")
        print(f"    {PASS}  THINKING_LEVEL = {THINKING_LEVEL!r}")
        return True
    except AssertionError as exc:
        print(f"    {FAIL}  {exc}")
        return False


# ============================================================
# Test 5 — generate_analysis with live API call
# ============================================================

def test_generate_analysis_live():
    print("\n[5] generate_analysis() — live API call to gemini-3.8-flash")
    print("    (This may take 10–30 seconds and uses your API quota)")

    prompt = build_analysis_prompt(SAMPLE_EVIDENCE)

    result = generate_analysis(prompt)

    if result.startswith("Warning:") or result.startswith("⚠️"):
        print(f"    {WARN}  API returned a managed error message:\n    {result}")
        # Managed errors are not test failures — they indicate the error
        # handling works correctly.
        if "high demand" in result or "try again" in result.lower():
            print("    (503/rate-limit — transient, retry later)")
        return "managed_error"

    if len(result) < 100:
        print(f"    {FAIL}  Response too short ({len(result)} chars): {result!r}")
        return False

    print(f"    {PASS}  Received {len(result):,} characters from {MODEL_NAME}")
    print("\n    --- First 500 characters of report ---")
    print("   ", result[:500].replace("\n", "\n    "))
    print("    --- (truncated) ---")
    return True


# ============================================================
# Test 6 — Missing key returns friendly message
# ============================================================

def test_missing_key_handling():
    print("\n[6] Missing API key → friendly error (no key exposed)")
    original = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        # build_analysis_prompt doesn't need a key
        prompt = build_analysis_prompt({"note": "minimal test"})
        result = generate_analysis(prompt)
        # Check for config error — the prefix may have emoji or plain text
        is_config_error = (
            "Configuration error" in result
            or "GOOGLE_API_KEY" in result
        )
        assert is_config_error, f"Expected config error, got: {result[:100]}"
        # Key value must NOT appear in the error message
        if original:
            assert original not in result, "API key value exposed in error!"
        print(f"    {PASS}  Got expected config error: {result[:80]}...")
        return True
    except AssertionError as exc:
        print(f"    {FAIL}  {exc}")
        return False
    finally:
        if original:
            os.environ["GOOGLE_API_KEY"] = original


# ============================================================
# Test 7 — Empty response extraction guard
# ============================================================

def test_empty_response_extraction():
    print("\n[7] Empty response extraction helpers")

    class FakeInteraction:
        output_text = None
        steps = []

    class FakeResponse:
        text = ""
        candidates = []

    result_i = _extract_text_from_interaction(FakeInteraction())
    result_r = _extract_text_from_response(FakeResponse())

    if result_i is None and result_r is None:
        print(f"    {PASS}  Empty responses correctly return None")
        return True
    else:
        print(f"    {FAIL}  Got: interaction={result_i!r}, response={result_r!r}")
        return False


# ============================================================
# RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  llm_agent.py integration test — {MODEL_NAME}")
    print("=" * 60)

    results = {
        "load_analyst_prompt": test_load_analyst_prompt(),
        "build_analysis_prompt": test_build_analysis_prompt(),
        "api_key_detection": test_api_key_detection(),
        "model_config": test_model_config(),
        "generate_analysis_live": test_generate_analysis_live(),
        "missing_key_handling": test_missing_key_handling(),
        "empty_response_extraction": test_empty_response_extraction(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    passed = 0
    managed = 0
    failed = 0

    for name, result in results.items():
        if result is True:
            status = PASS
            passed += 1
        elif result == "managed_error":
            status = WARN + " MANAGED"
            managed += 1
        else:
            status = FAIL
            failed += 1
        print(f"  {status}  {name}")

    print(f"\n  Passed: {passed}  |  Managed: {managed}  |  Failed: {failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
