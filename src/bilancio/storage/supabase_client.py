"""Supabase client wrapper for bilancio storage.

This module provides a configured Supabase client for persisting simulation
results and job metadata to the Supabase database.

Environment variables required:
    BILANCIO_SUPABASE_URL: The Supabase project URL
    BILANCIO_SUPABASE_ANON_KEY: The Supabase anonymous/public key

Note: In environments with TLS-inspecting proxies (like Claude Code web),
this module automatically configures the httpx client with the custom CA
certificate to allow connections through the proxy.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from supabase import Client

# Custom CA certificate path for TLS inspection proxy
PROXY_CA_CERT = Path("/usr/local/share/ca-certificates/swp-ca-production.crt")

# Module-level singleton for client reuse
_supabase_client: Client | None = None


class SupabaseConfigError(Exception):
    """Raised when Supabase credentials are missing or invalid."""

    pass


def is_supabase_configured() -> bool:
    """Check if Supabase environment variables are set.

    Returns:
        True if both BILANCIO_SUPABASE_URL and BILANCIO_SUPABASE_ANON_KEY
        are set in the environment, False otherwise.
    """
    url = os.environ.get("BILANCIO_SUPABASE_URL")
    key = os.environ.get("BILANCIO_SUPABASE_ANON_KEY")
    return bool(url and key)


def _create_proxy_aware_httpx_client() -> Any:
    """Create an httpx client configured for TLS-inspecting proxies.

    If a custom CA certificate exists (for environments like Claude Code web
    that use TLS inspection), creates an httpx client with the custom CA.
    Otherwise returns None to use the default client.

    Returns:
        httpx.Client configured with custom CA, or None for default behavior.
    """
    if not PROXY_CA_CERT.exists():
        return None

    # Check if we're going through a proxy
    proxy_url = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not proxy_url:
        return None

    import httpx

    # Create SSL context with the proxy's CA certificate
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(str(PROXY_CA_CERT))

    # Create httpx client with custom SSL and proxy support
    return httpx.Client(
        verify=ssl_context,
        proxy=proxy_url,
        timeout=httpx.Timeout(120.0),
    )


def get_supabase_client() -> Client:
    """Get a configured Supabase client instance.

    This function implements a singleton pattern - the client is created
    once and reused for subsequent calls. This is efficient for connection
    pooling and reduces overhead.

    In environments with TLS-inspecting proxies (like Claude Code web),
    automatically configures the httpx client with the custom CA certificate.

    Returns:
        A configured Supabase Client instance.

    Raises:
        SupabaseConfigError: If BILANCIO_SUPABASE_URL or
            BILANCIO_SUPABASE_ANON_KEY environment variables are not set.

    Example:
        >>> client = get_supabase_client()
        >>> response = client.table("jobs").select("*").execute()
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("BILANCIO_SUPABASE_URL")
    key = os.environ.get("BILANCIO_SUPABASE_ANON_KEY")

    if not url:
        raise SupabaseConfigError(
            "BILANCIO_SUPABASE_URL environment variable is not set. "
            "Please set it to your Supabase project URL "
            "(e.g., https://xxxxx.supabase.co)"
        )

    if not key:
        raise SupabaseConfigError(
            "BILANCIO_SUPABASE_ANON_KEY environment variable is not set. "
            "Please set it to your Supabase anonymous/public key."
        )

    from supabase import create_client
    from supabase.lib.client_options import SyncClientOptions

    # Create proxy-aware httpx client if in a TLS-inspecting environment
    httpx_client = _create_proxy_aware_httpx_client()

    if httpx_client:
        options = SyncClientOptions(httpx_client=httpx_client)
        _supabase_client = create_client(url, key, options=options)
    else:
        _supabase_client = create_client(url, key)

    return _supabase_client


def reset_client() -> None:
    """Reset the singleton client instance.

    This is primarily useful for testing or when credentials change.
    The next call to get_supabase_client() will create a new client.
    """
    global _supabase_client
    _supabase_client = None
