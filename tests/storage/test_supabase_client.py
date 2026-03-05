"""Unit tests for supabase_client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bilancio.storage.supabase_client import (
    SupabaseConfigError,
    _create_proxy_aware_httpx_client,
    get_supabase_client,
    is_supabase_configured,
    reset_client,
)


class TestIsSupabaseConfigured:
    """Tests for is_supabase_configured()."""

    def test_both_vars_present(self, monkeypatch):
        """Returns True when both env vars are set."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "some-key")
        assert is_supabase_configured() is True

    def test_url_missing(self, monkeypatch):
        """Returns False when URL is missing."""
        monkeypatch.delenv("BILANCIO_SUPABASE_URL", raising=False)
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "some-key")
        assert is_supabase_configured() is False

    def test_key_missing(self, monkeypatch):
        """Returns False when key is missing."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)
        assert is_supabase_configured() is False

    def test_neither_present(self, monkeypatch):
        """Returns False when neither env var is set."""
        monkeypatch.delenv("BILANCIO_SUPABASE_URL", raising=False)
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)
        assert is_supabase_configured() is False

    def test_empty_string_url(self, monkeypatch):
        """Returns False when URL is empty string."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "some-key")
        assert is_supabase_configured() is False


class TestCreateProxyAwareHttpxClient:
    """Tests for _create_proxy_aware_httpx_client()."""

    def test_returns_none_when_no_cert(self, monkeypatch):
        """Returns None when PROXY_CA_CERT does not exist."""
        with patch("bilancio.storage.supabase_client.PROXY_CA_CERT") as mock_cert:
            mock_cert.exists.return_value = False
            result = _create_proxy_aware_httpx_client()
        assert result is None

    def test_returns_none_when_no_proxy_env(self, monkeypatch):
        """Returns None when cert exists but no proxy env var."""
        monkeypatch.delenv("https_proxy", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        with patch("bilancio.storage.supabase_client.PROXY_CA_CERT") as mock_cert:
            mock_cert.exists.return_value = True
            result = _create_proxy_aware_httpx_client()
        assert result is None

    def test_returns_httpx_client_with_proxy(self, monkeypatch):
        """Returns an httpx.Client when cert exists and proxy is configured."""
        monkeypatch.setenv("https_proxy", "http://proxy.example.com:8080")
        with (
            patch("bilancio.storage.supabase_client.PROXY_CA_CERT") as mock_cert,
            patch("bilancio.storage.supabase_client.ssl") as mock_ssl,
            patch("httpx.Client") as mock_httpx_client,
        ):
            mock_cert.exists.return_value = True
            mock_cert.__str__ = lambda self: "/fake/cert.crt"
            mock_ssl_ctx = MagicMock()
            mock_ssl.create_default_context.return_value = mock_ssl_ctx

            result = _create_proxy_aware_httpx_client()

        assert result is not None
        mock_ssl.create_default_context.assert_called_once()
        mock_httpx_client.assert_called_once()


class TestGetSupabaseClient:
    """Tests for get_supabase_client()."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_client()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_client()

    def test_raises_when_url_missing(self, monkeypatch):
        """Raises SupabaseConfigError when URL is not set."""
        monkeypatch.delenv("BILANCIO_SUPABASE_URL", raising=False)
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)
        with pytest.raises(SupabaseConfigError, match="BILANCIO_SUPABASE_URL"):
            get_supabase_client()

    def test_raises_when_key_missing(self, monkeypatch):
        """Raises SupabaseConfigError when key is not set."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.delenv("BILANCIO_SUPABASE_ANON_KEY", raising=False)
        with pytest.raises(SupabaseConfigError, match="BILANCIO_SUPABASE_ANON_KEY"):
            get_supabase_client()

    def test_creates_client_with_valid_creds(self, monkeypatch):
        """Creates client when both env vars are set."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "test-key-123")

        mock_client = MagicMock()
        # create_client is imported locally inside get_supabase_client,
        # so we must patch at the supabase package level.
        with (
            patch("supabase.create_client", return_value=mock_client) as mock_create,
            patch(
                "bilancio.storage.supabase_client._create_proxy_aware_httpx_client",
                return_value=None,
            ),
        ):
            client = get_supabase_client()

        assert client is mock_client
        mock_create.assert_called_once_with("https://example.supabase.co", "test-key-123")

    def test_singleton_returns_same_client(self, monkeypatch):
        """Second call returns the same client (singleton pattern)."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "test-key-123")

        mock_client = MagicMock()
        with (
            patch("supabase.create_client", return_value=mock_client) as mock_create,
            patch(
                "bilancio.storage.supabase_client._create_proxy_aware_httpx_client",
                return_value=None,
            ),
        ):
            client1 = get_supabase_client()
            client2 = get_supabase_client()

        assert client1 is client2
        # create_client should only be called once
        mock_create.assert_called_once()

    def test_creates_client_with_proxy_httpx(self, monkeypatch):
        """Creates client with proxy-aware httpx when proxy is configured."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "test-key-123")

        mock_httpx_client = MagicMock()
        mock_client = MagicMock()
        mock_options_cls = MagicMock()

        with (
            patch("supabase.create_client", return_value=mock_client) as mock_create,
            patch(
                "bilancio.storage.supabase_client._create_proxy_aware_httpx_client",
                return_value=mock_httpx_client,
            ),
            patch(
                "supabase.lib.client_options.SyncClientOptions",
                mock_options_cls,
            ),
        ):
            client = get_supabase_client()

        assert client is mock_client
        # Should have been called with options kwarg
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert "options" in call_kwargs[1] or len(call_kwargs[0]) == 3


class TestResetClient:
    """Tests for reset_client()."""

    def setup_method(self):
        reset_client()

    def teardown_method(self):
        reset_client()

    def test_reset_clears_singleton(self, monkeypatch):
        """After reset, next get_supabase_client() creates a new client."""
        monkeypatch.setenv("BILANCIO_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("BILANCIO_SUPABASE_ANON_KEY", "test-key-123")

        mock_client_1 = MagicMock(name="client1")
        mock_client_2 = MagicMock(name="client2")

        with (
            patch(
                "supabase.create_client",
                side_effect=[mock_client_1, mock_client_2],
            ),
            patch(
                "bilancio.storage.supabase_client._create_proxy_aware_httpx_client",
                return_value=None,
            ),
        ):
            client1 = get_supabase_client()
            reset_client()
            client2 = get_supabase_client()

        assert client1 is mock_client_1
        assert client2 is mock_client_2
        assert client1 is not client2
