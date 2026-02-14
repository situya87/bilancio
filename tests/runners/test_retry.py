"""Tests for the retry_transient decorator."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bilancio.runners.retry import retry_transient


class TestRetryTransient:
    """Tests for retry_transient decorator."""

    def test_success_no_retry(self) -> None:
        """Function succeeds on first call — no retries."""
        call_count = 0

        @retry_transient(max_retries=3, base_delay=0.0)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retry_then_success(self) -> None:
        """Function fails twice then succeeds on third attempt."""
        call_count = 0

        @retry_transient(max_retries=3, base_delay=0.0)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_max_retries_exceeded(self) -> None:
        """Function fails on all attempts — raises the exception."""

        @retry_transient(max_retries=2, base_delay=0.0)
        def always_fail() -> str:
            raise TimeoutError("timed out")

        with pytest.raises(TimeoutError, match="timed out"):
            always_fail()

    def test_non_retryable_exception_propagates(self) -> None:
        """Non-retryable exceptions propagate immediately without retry."""
        call_count = 0

        @retry_transient(max_retries=3, base_delay=0.0)
        def bad_logic() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            bad_logic()
        assert call_count == 1

    def test_custom_retryable_exceptions(self) -> None:
        """Custom retryable exception tuple is respected."""
        call_count = 0

        @retry_transient(max_retries=2, base_delay=0.0, retryable=(KeyError,))
        def key_flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KeyError("missing")
            return "found"

        assert key_flaky() == "found"
        assert call_count == 2

    def test_exponential_backoff(self) -> None:
        """Verify delays double each retry."""
        call_count = 0
        delays: list[float] = []

        @retry_transient(max_retries=3, base_delay=1.0)
        def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise OSError("fail")

        with patch("bilancio.runners.retry.time.sleep", side_effect=lambda d: delays.append(d)):
            with pytest.raises(OSError):
                always_fail()

        assert delays == [1.0, 2.0, 4.0]
        assert call_count == 4  # 3 retries + 1 final attempt

    def test_logs_warnings_on_retry(self) -> None:
        """Verify WARNING log emitted for each retry."""
        call_count = 0

        @retry_transient(max_retries=2, base_delay=0.0)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("down")
            return "up"

        with patch("bilancio.runners.retry.logger") as mock_logger:
            result = flaky()

        assert result == "up"
        assert mock_logger.warning.call_count == 2
