"""
tests/test_retry.py
===================
Unit tests for _call_with_retry in src/llm_agent.py.

All network calls are mocked — no real API access required.

Covers:
  - 503 followed by success → succeeds on retry
  - Repeated 503s → raises RuntimeError with friendly message after MAX_RETRIES
  - 429 retried; success on second attempt
  - 500 INTERNAL retried
  - 504 DEADLINE_EXCEEDED retried
  - ClientError (auth / 401 / 403 / 400) → NOT retried, re-raised immediately
  - Non-retryable ServerError code (501) → NOT retried, re-raised
  - Retry-After header is respected
  - Jitter stays within 25 % bounds
  - Non-API exceptions propagate immediately
"""

import sys
import time
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, "d:/Projects/Automated_eda_agent")

from src.llm_agent import _call_with_retry, MAX_RETRIES, RETRY_WAIT_MIN, RETRY_WAIT_MAX, _RETRYABLE_CODES
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Helpers to build mock API error objects
# ---------------------------------------------------------------------------

def _make_server_error(http_code: int, message: str = "test error") -> genai_errors.ServerError:
    """Create a ServerError with a specific .code value."""
    err = genai_errors.ServerError.__new__(genai_errors.ServerError)
    err.code = http_code
    err.status = str(http_code)
    err.message = message
    err.details = {"error": {"message": message}}
    err.response = None
    Exception.__init__(err, f"{http_code} {message}")
    return err


def _make_client_error(http_code: int = 401, message: str = "auth error") -> genai_errors.ClientError:
    """Create a ClientError (permanent 4xx) with a specific .code value."""
    err = genai_errors.ClientError.__new__(genai_errors.ClientError)
    err.code = http_code
    err.status = str(http_code)
    err.message = message
    err.details = {"error": {"message": message}}
    err.response = None
    Exception.__init__(err, f"{http_code} {message}")
    return err


def _make_server_error_with_retry_after(http_code: int, retry_after: str) -> genai_errors.ServerError:
    """Create a ServerError whose .response.headers contain a Retry-After header."""
    err = _make_server_error(http_code)
    mock_response = MagicMock()
    mock_response.headers = {"Retry-After": retry_after}
    err.response = mock_response
    return err


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetryableCodes:
    """Verify the _RETRYABLE_CODES constant contains the right values."""

    def test_503_in_retryable(self):
        assert 503 in _RETRYABLE_CODES

    def test_429_in_retryable(self):
        assert 429 in _RETRYABLE_CODES

    def test_500_in_retryable(self):
        assert 500 in _RETRYABLE_CODES

    def test_504_in_retryable(self):
        assert 504 in _RETRYABLE_CODES

    def test_400_not_retryable(self):
        assert 400 not in _RETRYABLE_CODES

    def test_401_not_retryable(self):
        assert 401 not in _RETRYABLE_CODES

    def test_403_not_retryable(self):
        assert 403 not in _RETRYABLE_CODES


class TestSuccessOnFirstAttempt:
    """No errors — function returns immediately."""

    def test_returns_value(self):
        fn = MagicMock(return_value="hello")
        with patch("time.sleep"):
            result = _call_with_retry(fn, "arg1", key="val")
        fn.assert_called_once_with("arg1", key="val")
        assert result == "hello"

    def test_no_sleep_on_success(self):
        fn = MagicMock(return_value=42)
        with patch("time.sleep") as mock_sleep:
            _call_with_retry(fn)
        mock_sleep.assert_not_called()


class TestRetryOn503:
    """503 UNAVAILABLE: should retry and succeed if a later attempt works."""

    def test_503_then_success(self):
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, "success"])
        with patch("time.sleep"):
            result = _call_with_retry(fn)
        assert result == "success"
        assert fn.call_count == 2

    def test_503_twice_then_success(self):
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, err, "success"])
        with patch("time.sleep"):
            result = _call_with_retry(fn)
        assert result == "success"
        assert fn.call_count == 3

    def test_503_exhausted_raises_runtime_error(self):
        err = _make_server_error(503, "Service Unavailable")
        fn = MagicMock(side_effect=[err, err, err])
        with patch("time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                _call_with_retry(fn)
        assert fn.call_count == MAX_RETRIES
        msg = str(exc_info.value)
        assert "temporarily unavailable" in msg.lower()
        assert "try again" in msg.lower()

    def test_friendly_message_does_not_contain_api_key(self):
        """Ensure the RuntimeError message never leaks an API key."""
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, err, err])
        with patch("time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                _call_with_retry(fn)
        assert "GOOGLE_API_KEY" not in str(exc_info.value)
        assert "sk-" not in str(exc_info.value)


class TestRetryOn429:
    def test_429_then_success(self):
        err = _make_server_error(429, "RESOURCE_EXHAUSTED")
        fn = MagicMock(side_effect=[err, "ok"])
        with patch("time.sleep"):
            result = _call_with_retry(fn)
        assert result == "ok"

    def test_429_exhausted_raises_runtime_error(self):
        err = _make_server_error(429)
        fn = MagicMock(side_effect=[err] * MAX_RETRIES)
        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                _call_with_retry(fn)


class TestRetryOn500:
    def test_500_then_success(self):
        err = _make_server_error(500, "INTERNAL")
        fn = MagicMock(side_effect=[err, "result"])
        with patch("time.sleep"):
            result = _call_with_retry(fn)
        assert result == "result"


class TestRetryOn504:
    def test_504_then_success(self):
        err = _make_server_error(504, "DEADLINE_EXCEEDED")
        fn = MagicMock(side_effect=[err, "done"])
        with patch("time.sleep"):
            result = _call_with_retry(fn)
        assert result == "done"


class TestNonRetryableErrors:
    """ClientError and non-retryable ServerError codes must NOT be retried."""

    def test_client_error_401_not_retried(self):
        auth_err = _make_client_error(401, "UNAUTHENTICATED")
        fn = MagicMock(side_effect=auth_err)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(genai_errors.ClientError):
                _call_with_retry(fn)
        fn.assert_called_once()           # only 1 attempt
        mock_sleep.assert_not_called()    # no sleep

    def test_client_error_403_not_retried(self):
        auth_err = _make_client_error(403, "PERMISSION_DENIED")
        fn = MagicMock(side_effect=auth_err)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(genai_errors.ClientError):
                _call_with_retry(fn)
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    def test_client_error_400_not_retried(self):
        bad_req = _make_client_error(400, "INVALID_ARGUMENT")
        fn = MagicMock(side_effect=bad_req)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(genai_errors.ClientError):
                _call_with_retry(fn)
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    def test_server_error_501_not_retried(self):
        """501 NOT_IMPLEMENTED is a ServerError but not in _RETRYABLE_CODES."""
        err = _make_server_error(501, "NOT_IMPLEMENTED")
        fn = MagicMock(side_effect=err)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(genai_errors.ServerError):
                _call_with_retry(fn)
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    def test_non_api_exception_not_retried(self):
        """ValueError, TypeError, etc. must propagate immediately."""
        fn = MagicMock(side_effect=ValueError("bad input"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                _call_with_retry(fn)
        fn.assert_called_once()
        mock_sleep.assert_not_called()


class TestRetryAfterHeader:
    """When the server supplies a Retry-After header, it must be respected."""

    def test_retry_after_header_used(self):
        """Sleep duration should come from Retry-After (here 7 s), not back-off."""
        err = _make_server_error_with_retry_after(503, "7")
        fn = MagicMock(side_effect=[err, "ok"])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            result = _call_with_retry(fn)
        assert result == "ok"
        assert len(sleep_calls) == 1
        # server_wait overrides jitter — exact value should be 7.0
        assert sleep_calls[0] == pytest.approx(7.0, abs=0.01)

    def test_retry_after_capped_at_max(self):
        """A very large Retry-After value must be capped at RETRY_WAIT_MAX."""
        err = _make_server_error_with_retry_after(429, "9999")
        fn = MagicMock(side_effect=[err, "ok"])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            _call_with_retry(fn)
        assert sleep_calls[0] <= RETRY_WAIT_MAX


class TestBackoffBehavior:
    """Verify exponential back-off and jitter properties."""

    def test_sleep_called_between_retries(self):
        """sleep() must be called once per retry (not on the final failure)."""
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, err, err])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with pytest.raises(RuntimeError):
                _call_with_retry(fn)
        # 3 attempts → 2 sleeps (between attempt 1→2 and 2→3)
        assert len(sleep_calls) == MAX_RETRIES - 1

    def test_sleep_duration_within_bounds(self):
        """Each sleep should stay within [RETRY_WAIT_MIN * 0.75, RETRY_WAIT_MAX]."""
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, err, err])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with pytest.raises(RuntimeError):
                _call_with_retry(fn)
        for s in sleep_calls:
            assert RETRY_WAIT_MIN * 0.75 <= s <= RETRY_WAIT_MAX

    def test_jitter_at_midpoint(self):
        """With jitter fixed at 1.0, sleep should equal RETRY_WAIT_MIN."""
        err = _make_server_error(503)
        fn = MagicMock(side_effect=[err, "ok"])
        sleep_calls = []
        with patch("random.uniform", return_value=1.0):
            with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                _call_with_retry(fn)
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(RETRY_WAIT_MIN, abs=0.01)
