"""Unit tests for Modal proxy patch behavior."""

from __future__ import annotations

import asyncio
import socket
import ssl
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bilancio.cloud import proxy_patch


def test_should_use_proxy_requires_proxy_url_and_ca(monkeypatch) -> None:
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setattr(proxy_patch.os.path, "exists", lambda _path: True)
    assert proxy_patch._should_use_proxy() is False

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setattr(proxy_patch.os.path, "exists", lambda path: path == proxy_patch.PROXY_CA_CERT)
    assert proxy_patch._should_use_proxy() is True

    monkeypatch.setattr(proxy_patch.os.path, "exists", lambda _path: False)
    assert proxy_patch._should_use_proxy() is False


def test_create_proxy_ssl_context_loads_custom_ca(monkeypatch) -> None:
    context = MagicMock(spec=ssl.SSLContext)
    monkeypatch.setattr(proxy_patch.ssl, "create_default_context", MagicMock(return_value=context))

    assert proxy_patch._create_proxy_ssl_context() is context
    context.load_verify_locations.assert_called_once_with(proxy_patch.PROXY_CA_CERT)


def test_apply_proxy_patch_patches_channel_and_modal_http_client(monkeypatch) -> None:
    original = proxy_patch.grpclib.client.Channel._create_connection
    try:
        monkeypatch.setattr(
            proxy_patch.grpclib.client.Channel,
            "_create_connection",
            proxy_patch._original_create_connection,
        )
        http_patch = MagicMock()
        monkeypatch.setattr(proxy_patch, "_patch_modal_http_client", http_patch)

        proxy_patch.apply_proxy_patch()

        assert proxy_patch.is_patched() is True
        http_patch.assert_called_once_with()
    finally:
        proxy_patch.grpclib.client.Channel._create_connection = original


def test_proxied_create_connection_delegates_without_proxy(monkeypatch) -> None:
    async def fake_original(self):
        return {"delegated": self}

    target = object()
    monkeypatch.setattr(proxy_patch, "_should_use_proxy", lambda: False)
    monkeypatch.setattr(proxy_patch, "_original_create_connection", fake_original)

    assert asyncio.run(proxy_patch._proxied_create_connection(target)) == {"delegated": target}


def test_proxied_create_connection_closes_socket_on_connect_failure(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def setblocking(self, _flag):
            return None

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()

    class FakeLoop:
        async def sock_connect(self, sock, address):
            assert sock is fake_socket
            assert address == ("proxy.example", 8080)

        async def sock_sendall(self, sock, payload):
            assert sock is fake_socket
            assert b"CONNECT modal.example:443 HTTP/1.1" in payload

        async def sock_recv(self, sock, _size):
            assert sock is fake_socket
            return b"HTTP/1.1 403 Forbidden\r\n\r\n"

    monkeypatch.setattr(proxy_patch, "_should_use_proxy", lambda: True)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy.example:8080")
    monkeypatch.setattr(asyncio, "get_event_loop", MagicMock(return_value=FakeLoop()))

    channel = SimpleNamespace(_host="modal.example", _port=443, _ssl=True)

    async def exercise_proxy_failure() -> None:
        monkeypatch.setattr(socket, "socket", MagicMock(return_value=fake_socket))
        with pytest.raises(ConnectionError, match="Proxy CONNECT failed"):
            await proxy_patch._proxied_create_connection(channel)

    asyncio.run(exercise_proxy_failure())

    assert fake_socket.closed is True
