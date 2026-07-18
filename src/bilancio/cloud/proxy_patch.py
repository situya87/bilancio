"""Patch grpclib and Modal's HTTP client for TLS-inspecting proxies.

This module patches:
1. grpclib's connection handling to work through HTTP CONNECT proxy
2. grpclib's SSL context to trust the proxy/egress gateway CA
3. Modal's HTTP client to use custom CA for TLS inspection

Required in environments where outbound connections go through either:
- An explicit HTTP CONNECT proxy with TLS inspection (MITM)
- A transparent egress gateway that re-signs TLS certificates

Usage:
    import bilancio.cloud.proxy_patch  # Apply patch before importing modal
    import modal
"""

from __future__ import annotations

import asyncio
import base64
import os
import socket
import ssl
from typing import Any
from urllib.parse import urlparse

import grpclib.client

# Store original methods
_original_create_connection = grpclib.client.Channel._create_connection
_original_http_client_with_tls = None  # Set lazily when modal is imported

# CA certificate paths (in priority order)
EGRESS_GATEWAY_CA = "/usr/local/share/ca-certificates/egress-gateway-ca.crt"
PROXY_CA_CERT = "/usr/local/share/ca-certificates/swp-ca-production.crt"


def _get_proxy_ca_path() -> str | None:
    """Find the appropriate proxy/egress CA certificate."""
    if os.path.exists(EGRESS_GATEWAY_CA):
        return EGRESS_GATEWAY_CA
    if os.path.exists(PROXY_CA_CERT):
        return PROXY_CA_CERT
    return None


def _should_use_proxy() -> bool:
    """Check if explicit HTTP CONNECT proxy should be used."""
    proxy_url = os.environ.get("https_proxy", "") or os.environ.get("HTTPS_PROXY", "")
    return bool(proxy_url) and _get_proxy_ca_path() is not None


def _should_patch_ssl() -> bool:
    """Check if SSL patching is needed (transparent proxy or explicit proxy)."""
    return _get_proxy_ca_path() is not None


def _create_proxy_ssl_context() -> ssl.SSLContext:
    """Create SSL context that trusts the proxy's CA certificate."""
    ctx = ssl.create_default_context()
    ca_path = _get_proxy_ca_path()
    if ca_path:
        ctx.load_verify_locations(ca_path)
    return ctx


# =============================================================================
# grpclib patch for Modal API (gRPC)
# =============================================================================


async def _proxied_create_connection(self: Any) -> Any:
    """Create connection through HTTP CONNECT proxy or with custom CA."""
    if not _should_use_proxy():
        # No explicit proxy - but may still need custom CA for transparent proxy
        ca_path = _get_proxy_ca_path()
        if ca_path and _should_patch_ssl():
            if self._ssl is True or self._ssl is None:
                self._ssl = _create_proxy_ssl_context()
            elif isinstance(self._ssl, ssl.SSLContext):
                self._ssl.load_verify_locations(ca_path)
        return await _original_create_connection(self)

    proxy_url = os.environ.get("https_proxy", "") or os.environ.get("HTTPS_PROXY", "")
    parsed = urlparse(proxy_url)

    # Create raw socket to proxy
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)

    loop = asyncio.get_event_loop()
    await loop.sock_connect(sock, (parsed.hostname, parsed.port))

    # Send HTTP CONNECT request
    proxy_auth = f"{parsed.username}:{parsed.password}" if parsed.username else None
    connect_req = f"CONNECT {self._host}:{self._port} HTTP/1.1\r\n"
    connect_req += f"Host: {self._host}:{self._port}\r\n"
    if proxy_auth:
        auth_b64 = base64.b64encode(proxy_auth.encode()).decode()
        connect_req += f"Proxy-Authorization: Basic {auth_b64}\r\n"
    connect_req += "\r\n"

    await loop.sock_sendall(sock, connect_req.encode())

    # Read proxy response
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = await loop.sock_recv(sock, 1024)
        if not chunk:
            break
        response += chunk

    if b"200" not in response.split(b"\r\n")[0]:
        sock.close()
        raise ConnectionError(f"Proxy CONNECT failed: {response.decode()}")

    # Pass raw socket to create_connection, let asyncio handle SSL
    ssl_ctx = _create_proxy_ssl_context() if self._ssl else None

    _, protocol = await loop.create_connection(
        self._protocol_factory,
        sock=sock,
        ssl=ssl_ctx,
        server_hostname=self._host if ssl_ctx else None,
    )

    return protocol


# =============================================================================
# Modal HTTP client patch for file downloads (aiohttp)
# =============================================================================


def _patched_http_client_with_tls(timeout: Any) -> Any:
    """Create HTTP client with custom CA for TLS inspection proxy."""
    from aiohttp import ClientSession, ClientTimeout, TCPConnector

    # Create SSL context with custom CA
    ssl_context = _create_proxy_ssl_context()

    connector = TCPConnector(ssl=ssl_context)

    # Enable trust_env to use HTTPS_PROXY environment variable
    return ClientSession(
        connector=connector,
        timeout=ClientTimeout(total=timeout),
        trust_env=True,  # Use HTTPS_PROXY from environment
    )


def _patch_modal_http_client() -> None:
    """Patch Modal's HTTP client to use custom CA and proxy."""
    global _original_http_client_with_tls

    try:
        import modal._utils.http_utils as http_utils

        if _original_http_client_with_tls is None:
            _original_http_client_with_tls = http_utils._http_client_with_tls

        http_utils._http_client_with_tls = _patched_http_client_with_tls

        # Also reset the client session registry to use new settings
        http_utils.ClientSessionRegistry._client_session_active = False

    except ImportError:
        pass  # Modal not installed or different version


# =============================================================================
# Patch application
# =============================================================================


def apply_proxy_patch() -> None:
    """Apply all proxy patches."""
    grpclib.client.Channel._create_connection = _proxied_create_connection  # type: ignore[method-assign]
    _patch_modal_http_client()


def is_patched() -> bool:
    """Check if the grpclib patch has been applied."""
    return grpclib.client.Channel._create_connection is _proxied_create_connection


# Auto-apply patch on import if proxy or transparent egress gateway is detected
if _should_use_proxy() or _should_patch_ssl():
    apply_proxy_patch()
    _patched = True
else:
    _patched = False
