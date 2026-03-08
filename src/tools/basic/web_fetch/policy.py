"""Policy checks for the web_fetch tool."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from ...types import ToolExecutionContext, ToolPolicyDecision

_MAX_URL_CHARS = 2_000
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}


class WebFetchPolicy:
    """Restricts web_fetch to explicit public HTTP(S) URLs."""

    def authorize(
        self,
        *,
        url: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        raw_url = url.strip()
        if not raw_url:
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch requires a non-empty 'url'.",
            )
        if len(raw_url) > _MAX_URL_CHARS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"web_fetch URL length must be <= {_MAX_URL_CHARS} characters.",
            )

        parsed = urlparse(raw_url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch only allows http:// or https:// URLs.",
            )
        if not parsed.netloc or parsed.hostname is None:
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch requires an absolute URL with a hostname.",
            )
        if parsed.username or parsed.password:
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch does not allow URLs with embedded credentials.",
            )

        hostname = parsed.hostname.strip().lower()
        if hostname in _LOCAL_HOSTS or hostname.endswith(".localhost"):
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch does not allow localhost targets.",
            )

        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None

        if address is not None and _is_non_public_address(address):
            return ToolPolicyDecision(
                allowed=False,
                reason="web_fetch does not allow private, loopback, or reserved IP targets.",
            )

        return ToolPolicyDecision(allowed=True)


def _is_non_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_reserved,
            address.is_multicast,
            address.is_unspecified,
        )
    )
