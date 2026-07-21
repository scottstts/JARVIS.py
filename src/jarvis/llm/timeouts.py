"""Shared transport-timeout primitives for provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import requests


@dataclass(slots=True, frozen=True)
class ProviderTransportTimeouts:
    """Connection and raw response-read limits owned by provider transports."""

    connect_seconds: float
    read_seconds: float

    def __post_init__(self) -> None:
        if self.connect_seconds <= 0:
            raise ValueError("connect_seconds must be > 0.")
        if self.read_seconds <= 0:
            raise ValueError("read_seconds must be > 0.")

    def as_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_seconds,
            read=self.read_seconds,
            write=self.connect_seconds,
            pool=self.connect_seconds,
        )

    def as_requests(self) -> tuple[float, float]:
        return (self.connect_seconds, self.read_seconds)


def transport_timeout_metadata(
    error: BaseException,
    *,
    timeouts: ProviderTransportTimeouts,
) -> dict[str, Any]:
    """Classify a transport timeout without relying on provider SDK messages."""

    timeout_kind = "transport"
    timeout_limit_seconds = timeouts.read_seconds
    for current in _exception_chain(error):
        if isinstance(current, (httpx.ConnectTimeout, requests.ConnectTimeout)):
            timeout_kind = "connect"
            timeout_limit_seconds = timeouts.connect_seconds
            break
        if isinstance(current, (httpx.ReadTimeout, requests.ReadTimeout)):
            timeout_kind = "read_idle"
            timeout_limit_seconds = timeouts.read_seconds
            break
        if isinstance(current, httpx.WriteTimeout):
            timeout_kind = "write"
            timeout_limit_seconds = timeouts.connect_seconds
            break
        if isinstance(current, httpx.PoolTimeout):
            timeout_kind = "pool"
            timeout_limit_seconds = timeouts.connect_seconds
            break

    return {
        "timeout_kind": timeout_kind,
        "timeout_limit_seconds": timeout_limit_seconds,
        "connect_timeout_seconds": timeouts.connect_seconds,
        "read_timeout_seconds": timeouts.read_seconds,
    }


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)
