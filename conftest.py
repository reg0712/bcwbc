"""Shared test setup. Tests never use the network: only loopback and Unix sockets work,
so no test can reach a real LLM or external API."""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models load from the local cache only


@pytest.fixture
def embedding_model() -> str:
    """The embedding model's name; skips the test when the model is not downloaded."""
    from huggingface_hub import try_to_load_from_cache

    from synapse.config import load_config

    name = load_config("models.yaml")["embedding_model"]
    if not isinstance(try_to_load_from_cache(name, "config.json"), str):
        pytest.skip(f"{name} is not downloaded; run make setup")
    return name


def _is_loopback(host: object) -> bool:
    """True when host names or addresses this machine."""
    if isinstance(host, bytes):
        host = host.decode()
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_unless_loopback(host: object) -> None:
    """Raise if host is anything other than this machine."""
    if not _is_loopback(host):
        raise RuntimeError(f"tests must not use the network (tried to reach {host!r})")


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block DNS lookups and connections to anything but loopback for every test."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _refuse_unless_loopback(address[0])
        return real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _refuse_unless_loopback(address[0])
        return real_connect_ex(sock, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None:
            _refuse_unless_loopback(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
