"""Web-fetch tool definition and execution runtime."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from llm import ToolDefinition

from ..config import ToolSettings
from ..types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_CLOUDFLARE_TOMARKDOWN_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/tomarkdown"
)
_DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)
_MAX_REDIRECTS = 5
_SUPPORTED_HTML_MIME_TYPES = {
    "application/xhtml+xml",
    "text/html",
}
_SUPPORTED_DIRECT_TEXT_MIME_TYPES = {
    "text/markdown",
    "text/plain",
}
_UNSUPPORTED_BINARY_PREFIXES = (
    "audio/",
    "font/",
    "image/",
    "video/",
)
_UNSUPPORTED_BINARY_MIME_TYPES = {
    "application/gzip",
    "application/epub+zip",
    "application/msword",
    "application/octet-stream",
    "application/pdf",
    "application/rtf",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
}
_CONTENT_SIGNAL_HEADER = "content-signal"
_MARKDOWN_TOKENS_HEADER = "x-markdown-tokens"
_APP_SHELL_PATTERNS = (
    "id=\"__next\"",
    "id=\"__nuxt\"",
    "id=\"app\"",
    "id=\"root\"",
    "data-reactroot",
    "ng-version",
)
_LOW_SIGNAL_MARKDOWN_PATTERNS = (
    "enable javascript",
    "javascript required",
    "loading...",
    "please wait",
)
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}


class WebFetchConfigurationError(RuntimeError):
    """Raised when web_fetch is missing required local configuration."""


class WebFetchRequestError(RuntimeError):
    """Raised when an upstream fetch fails or returns an invalid response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class WebFetchUnsupportedContentError(RuntimeError):
    """Raised when the URL resolves to unsupported non-text content."""

    def __init__(self, message: str, *, content_type: str | None = None) -> None:
        super().__init__(message)
        self.content_type = content_type


class WebFetchBodyTooLargeError(RuntimeError):
    """Raised when a fetched page exceeds the configured byte budget."""


@dataclass(slots=True, frozen=True)
class HTTPFetchResult:
    requested_url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    content_type: str | None
    body_text: str
    redirect_chain: tuple[str, ...]

    @property
    def body_chars(self) -> int:
        return len(self.body_text)


@dataclass(slots=True, frozen=True)
class MarkdownConversionResult:
    markdown: str
    markdown_tokens: int | None


@dataclass(slots=True, frozen=True)
class BrowserRenderResult:
    requested_url: str
    final_url: str
    html: str


class WebFetchToolExecutor:
    """Fetches a URL through markdown-first HTTP and Playwright fallback tiers."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        _ = context
        requested_url = str(arguments["url"]).strip()
        attempts: list[dict[str, Any]] = []

        try:
            tier1_result = await asyncio.to_thread(
                _fetch_http_text,
                url=requested_url,
                accept_markdown=True,
                settings=self._settings,
            )
            attempts.append(_summarize_http_attempt("tier1_request", tier1_result))
        except WebFetchUnsupportedContentError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
                content_type=exc.content_type,
            )
        except WebFetchBodyTooLargeError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
            )
        except WebFetchRequestError as exc:
            if not _should_continue_after_tier1_request_error(exc):
                return _web_fetch_error(
                    call_id=call_id,
                    requested_url=requested_url,
                    reason=str(exc),
                    attempts=attempts,
                    status_code=exc.status_code,
                )
            attempts.append(
                {
                    "stage": "tier1_request",
                    "status": "non_markdown_response",
                    "status_code": exc.status_code,
                    "reason": str(exc),
                }
            )

        if "tier1_result" in locals() and _tier1_response_is_usable_markdown(tier1_result):
            return _web_fetch_success(
                call_id=call_id,
                requested_url=requested_url,
                final_url=tier1_result.final_url,
                strategy="tier1_markdown_accept",
                status_code=tier1_result.status_code,
                content_type=tier1_result.content_type,
                redirect_chain=tier1_result.redirect_chain,
                markdown=tier1_result.body_text,
                markdown_tokens=_extract_markdown_tokens(tier1_result.headers),
                content_signal=_extract_header_value(
                    tier1_result.headers,
                    _CONTENT_SIGNAL_HEADER,
                ),
                markdown_source="native_markdown",
                browser_rendered=False,
                attempts=attempts,
                settings=self._settings,
            )

        try:
            tier2_result = await asyncio.to_thread(
                _fetch_http_text,
                url=requested_url,
                accept_markdown=False,
                settings=self._settings,
            )
            attempts.append(_summarize_http_attempt("tier2_request", tier2_result))
        except WebFetchUnsupportedContentError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
                content_type=exc.content_type,
            )
        except WebFetchBodyTooLargeError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
            )
        except WebFetchRequestError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
                status_code=exc.status_code,
            )

        if _is_direct_text_response(tier2_result) and _markdown_is_usable(tier2_result.body_text):
            return _web_fetch_success(
                call_id=call_id,
                requested_url=requested_url,
                final_url=tier2_result.final_url,
                strategy="tier2_direct_text",
                status_code=tier2_result.status_code,
                content_type=tier2_result.content_type,
                redirect_chain=tier2_result.redirect_chain,
                markdown=tier2_result.body_text,
                markdown_tokens=_extract_markdown_tokens(tier2_result.headers),
                content_signal=_extract_header_value(
                    tier2_result.headers,
                    _CONTENT_SIGNAL_HEADER,
                ),
                markdown_source="direct_text",
                browser_rendered=False,
                attempts=attempts,
                settings=self._settings,
            )

        if not _is_html_response(tier2_result):
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=(
                    "web_fetch only supports markdown, plain-text, or HTML pages in v1. "
                    "Binary and document conversion paths are intentionally excluded."
                ),
                attempts=attempts,
                status_code=tier2_result.status_code,
                content_type=tier2_result.content_type,
            )

        try:
            tier2_conversion = await asyncio.to_thread(
                _convert_html_to_markdown,
                html=tier2_result.body_text,
                source_url=tier2_result.final_url,
                settings=self._settings,
            )
            attempts.append(
                _summarize_conversion_attempt(
                    "tier2_conversion",
                    tier2_conversion,
                )
            )
        except (WebFetchConfigurationError, WebFetchRequestError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
                status_code=getattr(exc, "status_code", None),
                content_type=tier2_result.content_type,
            )

        if _markdown_is_usable(tier2_conversion.markdown):
            return _web_fetch_success(
                call_id=call_id,
                requested_url=requested_url,
                final_url=tier2_result.final_url,
                strategy="tier2_html_to_markdown",
                status_code=tier2_result.status_code,
                content_type=tier2_result.content_type,
                redirect_chain=tier2_result.redirect_chain,
                markdown=tier2_conversion.markdown,
                markdown_tokens=tier2_conversion.markdown_tokens,
                content_signal=_extract_header_value(
                    tier2_result.headers,
                    _CONTENT_SIGNAL_HEADER,
                ),
                markdown_source="cloudflare_tomarkdown",
                browser_rendered=False,
                attempts=attempts,
                settings=self._settings,
            )

        if not _should_try_browser_fallback(
            html=tier2_result.body_text,
            markdown=tier2_conversion.markdown,
        ):
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=(
                    "web_fetch could not extract acceptable markdown from the fetched HTML."
                ),
                attempts=attempts,
                status_code=tier2_result.status_code,
                content_type=tier2_result.content_type,
            )

        try:
            rendered_result = await _render_page_html(
                url=tier2_result.final_url,
                settings=self._settings,
            )
            attempts.append(_summarize_render_attempt(rendered_result))
            rendered_conversion = await asyncio.to_thread(
                _convert_html_to_markdown,
                html=rendered_result.html,
                source_url=rendered_result.final_url,
                settings=self._settings,
            )
            attempts.append(
                _summarize_conversion_attempt(
                    "tier3_conversion",
                    rendered_conversion,
                )
            )
        except (WebFetchConfigurationError, WebFetchRequestError, RuntimeError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                attempts=attempts,
                content_type=tier2_result.content_type,
            )

        if not _markdown_is_usable(rendered_conversion.markdown):
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=(
                    "web_fetch browser fallback completed, but the rendered page still did "
                    "not produce acceptable markdown."
                ),
                attempts=attempts,
                content_type=tier2_result.content_type,
            )

        return _web_fetch_success(
            call_id=call_id,
            requested_url=requested_url,
            final_url=rendered_result.final_url,
            strategy="tier3_playwright_html_to_markdown",
            status_code=tier2_result.status_code,
            content_type=tier2_result.content_type,
            redirect_chain=tier2_result.redirect_chain,
            markdown=rendered_conversion.markdown,
            markdown_tokens=rendered_conversion.markdown_tokens,
            content_signal=_extract_header_value(
                tier2_result.headers,
                _CONTENT_SIGNAL_HEADER,
            ),
            markdown_source="playwright_plus_cloudflare_tomarkdown",
            browser_rendered=True,
            attempts=attempts,
            settings=self._settings,
        )


def build_web_fetch_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the web_fetch registry entry."""

    return RegisteredTool(
        name="web_fetch",
        exposure="basic",
        definition=ToolDefinition(
            name="web_fetch",
            description=_build_web_fetch_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute http:// or https:// URL to fetch and convert into clean "
                            "markdown. The tool first asks for markdown natively, then falls "
                            "back to HTML-to-markdown conversion, and only uses Playwright "
                            "for JavaScript-heavy pages."
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        executor=WebFetchToolExecutor(settings),
    )


def _build_web_fetch_tool_description(settings: ToolSettings) -> str:
    return (
        "Fetch a specific web page and return clean markdown for downstream reasoning. "
        "The tool uses a three-tier strategy: first request markdown directly with "
        "Accept: text/markdown, then fetch normal HTML and convert it with Cloudflare "
        "toMarkdown, and only then fall back to local Playwright rendering for "
        "JavaScript-heavy pages. "
        f"Each fetch uses a {settings.web_fetch_timeout_seconds:.0f}s HTTP timeout, a "
        f"{settings.web_fetch_playwright_timeout_seconds:.0f}s Playwright timeout, and "
        f"caps fetched response bodies at {settings.web_fetch_max_response_bytes} bytes."
    )


def _fetch_http_text(
    *,
    url: str,
    accept_markdown: bool,
    settings: ToolSettings,
) -> HTTPFetchResult:
    with requests.Session() as session:
        current_url = url
        redirect_chain: list[str] = []

        for _ in range(_MAX_REDIRECTS + 1):
            _validate_public_url(current_url)

            headers = {
                "Accept": "text/markdown"
                if accept_markdown
                else "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": _DEFAULT_HTTP_USER_AGENT,
            }

            try:
                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=settings.web_fetch_timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout as exc:
                raise WebFetchRequestError("request timed out.") from exc
            except requests.RequestException as exc:
                raise WebFetchRequestError(f"request failed: {exc}") from exc

            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise WebFetchRequestError(
                        "redirect response did not include a Location header.",
                        status_code=response.status_code,
                    )
                next_url = urljoin(current_url, location)
                redirect_chain.append(next_url)
                current_url = next_url
                continue

            if response.status_code < 200 or response.status_code >= 300:
                error_message = _extract_http_error_message(
                    response=response,
                    max_bytes=settings.web_fetch_max_response_bytes,
                )
                raise WebFetchRequestError(
                    error_message,
                    status_code=response.status_code,
                )

            raw_content_type = _extract_header_value(response.headers, "Content-Type")
            normalized_content_type = _normalize_content_type(raw_content_type)
            if _is_unsupported_binary_content_type(normalized_content_type):
                response.close()
                raise WebFetchUnsupportedContentError(
                    (
                        "web_fetch does not support binary or document content in v1 "
                        f"(content type: {normalized_content_type or 'unknown'})."
                    ),
                    content_type=normalized_content_type,
                )

            try:
                body_bytes = _read_response_bytes(
                    response=response,
                    max_bytes=settings.web_fetch_max_response_bytes,
                )
            finally:
                response.close()

            if _looks_binary_bytes(body_bytes):
                raise WebFetchUnsupportedContentError(
                    "web_fetch received non-text content and will not attempt image or binary conversion.",
                    content_type=normalized_content_type,
                )

            body_text = _decode_body_text(
                body_bytes=body_bytes,
                response=response,
            )

            if normalized_content_type is None:
                normalized_content_type = (
                    "text/html" if _looks_like_html(body_text) else "text/plain"
                )

            return HTTPFetchResult(
                requested_url=url,
                final_url=current_url,
                status_code=response.status_code,
                headers=dict(response.headers),
                content_type=normalized_content_type,
                body_text=body_text,
                redirect_chain=tuple(redirect_chain),
            )

    raise WebFetchRequestError("request exceeded redirect limit.")


def _convert_html_to_markdown(
    *,
    html: str,
    source_url: str,
    settings: ToolSettings,
) -> MarkdownConversionResult:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_AI_WORKERS_REST_API_KEY")
    if not account_id:
        raise WebFetchConfigurationError(
            "CLOUDFLARE_ACCOUNT_ID is not configured."
        )
    if not api_key:
        raise WebFetchConfigurationError(
            "CLOUDFLARE_AI_WORKERS_REST_API_KEY is not configured."
        )

    source_host = _hostname_for_conversion(source_url)
    data: dict[str, str] = {}
    if source_host is not None:
        data["conversionOptions"] = json.dumps(
            {
                "html": {
                    "hostname": source_host,
                }
            }
        )

    try:
        response = requests.post(
            _CLOUDFLARE_TOMARKDOWN_URL_TEMPLATE.format(account_id=account_id),
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            files={
                "files": (
                    "page.html",
                    html.encode("utf-8"),
                    "text/html",
                ),
            },
            data=data,
            timeout=settings.web_fetch_timeout_seconds,
        )
    except requests.Timeout as exc:
        raise WebFetchRequestError("Cloudflare toMarkdown request timed out.") from exc
    except requests.RequestException as exc:
        raise WebFetchRequestError(
            f"Cloudflare toMarkdown request failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise WebFetchRequestError(
            _extract_cloudflare_error_message(response),
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise WebFetchRequestError(
            "Cloudflare toMarkdown returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise WebFetchRequestError(
            "Cloudflare toMarkdown returned an unexpected response shape."
        )

    result_entries = payload.get("result")
    if not isinstance(result_entries, list) or not result_entries:
        raise WebFetchRequestError(
            "Cloudflare toMarkdown returned no conversion results."
        )

    first_result = result_entries[0]
    if not isinstance(first_result, dict):
        raise WebFetchRequestError(
            "Cloudflare toMarkdown returned an invalid conversion result."
        )

    markdown = _normalize_markdown(first_result.get("data"))
    if markdown is None:
        raise WebFetchRequestError(
            "Cloudflare toMarkdown returned an empty markdown payload."
        )

    markdown_tokens = first_result.get("tokens")
    if not isinstance(markdown_tokens, int):
        markdown_tokens = None

    return MarkdownConversionResult(
        markdown=markdown,
        markdown_tokens=markdown_tokens,
    )


async def _render_page_html(
    *,
    url: str,
    settings: ToolSettings,
) -> BrowserRenderResult:
    await asyncio.to_thread(_validate_public_url, url)

    timeout_ms = int(settings.web_fetch_playwright_timeout_seconds * 1000)
    network_idle_timeout_ms = min(timeout_ms, 5_000)

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"],
            )
            context = await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=network_idle_timeout_ms,
                    )
                except PlaywrightTimeoutError:
                    pass
                html = await page.content()
                final_url = page.url
            finally:
                await context.close()
                await browser.close()
    except PlaywrightTimeoutError as exc:
        raise RuntimeError("Playwright page render timed out.") from exc
    except Exception as exc:
        raise RuntimeError(f"Playwright page render failed: {exc}") from exc

    return BrowserRenderResult(
        requested_url=url,
        final_url=final_url,
        html=_normalize_text(html),
    )


def _should_continue_after_tier1_request_error(exc: WebFetchRequestError) -> bool:
    return exc.status_code in {406, 415}


def _tier1_response_is_usable_markdown(result: HTTPFetchResult) -> bool:
    if result.content_type == "text/markdown":
        return _markdown_is_usable(result.body_text)
    if _extract_markdown_tokens(result.headers) is not None:
        return _markdown_is_usable(result.body_text)
    return (
        not _is_html_response(result)
        and _markdown_is_usable(result.body_text)
    )


def _is_direct_text_response(result: HTTPFetchResult) -> bool:
    if result.content_type in _SUPPORTED_DIRECT_TEXT_MIME_TYPES:
        return True
    return (
        result.content_type == "text/plain"
        or (result.content_type is None and not _looks_like_html(result.body_text))
    )


def _is_html_response(result: HTTPFetchResult) -> bool:
    if result.content_type in _SUPPORTED_HTML_MIME_TYPES:
        return True
    return _looks_like_html(result.body_text)


def _markdown_is_usable(markdown: str) -> bool:
    normalized = _normalize_markdown(markdown)
    if normalized is None:
        return False
    if _looks_like_html(normalized):
        return False

    visible = _visible_text(normalized)
    if len(visible) < 20:
        return False

    lowered = visible.lower()
    if any(pattern in lowered for pattern in _LOW_SIGNAL_MARKDOWN_PATTERNS):
        return len(visible) >= 120
    return True


def _should_try_browser_fallback(*, html: str, markdown: str) -> bool:
    if _markdown_is_usable(markdown):
        return False
    if _looks_like_app_shell(html):
        return True
    visible_html = _visible_text(_strip_scripts_and_styles(html))
    return len(visible_html) < 200


def _web_fetch_success(
    *,
    call_id: str,
    requested_url: str,
    final_url: str,
    strategy: str,
    status_code: int,
    content_type: str | None,
    redirect_chain: tuple[str, ...],
    markdown: str,
    markdown_tokens: int | None,
    content_signal: str | None,
    markdown_source: str,
    browser_rendered: bool,
    attempts: list[dict[str, Any]],
    settings: ToolSettings,
) -> ToolExecutionResult:
    truncated_markdown, markdown_truncated = _truncate_markdown(
        markdown=markdown,
        limit=settings.web_fetch_max_markdown_chars,
    )
    lines = [
        "Web fetch result",
        f"requested_url: {requested_url}",
        f"final_url: {final_url}",
        f"strategy: {strategy}",
        f"status_code: {status_code}",
        f"content_type: {content_type or 'unknown'}",
        f"markdown_source: {markdown_source}",
        f"browser_rendered: {'yes' if browser_rendered else 'no'}",
    ]
    if redirect_chain:
        lines.append(f"redirects: {len(redirect_chain)}")
    if content_signal is not None:
        lines.append(f"content_signal: {content_signal}")
    if markdown_tokens is not None:
        lines.append(f"markdown_tokens: {markdown_tokens}")
    if markdown_truncated:
        lines.append("markdown_truncated: true")
    lines.extend(
        [
            "markdown:",
            truncated_markdown,
        ]
    )

    metadata = {
        "requested_url": requested_url,
        "final_url": final_url,
        "strategy": strategy,
        "status_code": status_code,
        "content_type": content_type,
        "redirect_chain": list(redirect_chain),
        "markdown_source": markdown_source,
        "browser_rendered": browser_rendered,
        "markdown_chars": len(markdown),
        "markdown_truncated": markdown_truncated,
        "markdown_tokens": markdown_tokens,
        "content_signal": content_signal,
        "attempts": attempts,
    }
    return ToolExecutionResult(
        call_id=call_id,
        name="web_fetch",
        ok=True,
        content="\n".join(lines),
        metadata=metadata,
    )


def _web_fetch_error(
    *,
    call_id: str,
    requested_url: str,
    reason: str,
    attempts: list[dict[str, Any]],
    status_code: int | None = None,
    content_type: str | None = None,
) -> ToolExecutionResult:
    lines = [
        "Web fetch failed",
        f"url: {requested_url}",
        f"reason: {reason}",
    ]
    if status_code is not None:
        lines.append(f"status_code: {status_code}")
    if content_type is not None:
        lines.append(f"content_type: {content_type}")

    metadata: dict[str, Any] = {
        "requested_url": requested_url,
        "error": reason,
        "attempts": attempts,
    }
    if status_code is not None:
        metadata["status_code"] = status_code
    if content_type is not None:
        metadata["content_type"] = content_type

    return ToolExecutionResult(
        call_id=call_id,
        name="web_fetch",
        ok=False,
        content="\n".join(lines),
        metadata=metadata,
    )


def _summarize_http_attempt(stage: str, result: HTTPFetchResult) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "ok",
        "final_url": result.final_url,
        "status_code": result.status_code,
        "content_type": result.content_type,
        "body_chars": result.body_chars,
        "redirects": len(result.redirect_chain),
        "markdown_tokens": _extract_markdown_tokens(result.headers),
        "content_signal": _extract_header_value(result.headers, _CONTENT_SIGNAL_HEADER),
    }


def _summarize_conversion_attempt(
    stage: str,
    result: MarkdownConversionResult,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "ok",
        "markdown_chars": len(result.markdown),
        "markdown_tokens": result.markdown_tokens,
    }


def _summarize_render_attempt(result: BrowserRenderResult) -> dict[str, Any]:
    return {
        "stage": "tier3_render",
        "status": "ok",
        "requested_url": result.requested_url,
        "final_url": result.final_url,
        "html_chars": len(result.html),
    }


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise WebFetchRequestError("URL is missing a hostname.")
    lowered_hostname = hostname.strip().lower()
    if lowered_hostname in _LOCAL_HOSTS or lowered_hostname.endswith(".localhost"):
        raise WebFetchRequestError(
            "web_fetch does not allow localhost targets."
        )
    if parsed.username or parsed.password:
        raise WebFetchRequestError(
            "web_fetch does not allow URLs with embedded credentials."
        )

    try:
        address = ipaddress.ip_address(lowered_hostname)
    except ValueError:
        address = None

    if address is not None:
        if _is_non_public_address(address):
            raise WebFetchRequestError(
                "web_fetch does not allow private, loopback, or reserved IP targets."
            )
        return

    try:
        port = parsed.port
    except ValueError as exc:
        raise WebFetchRequestError(f"invalid URL port: {exc}") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80

    try:
        addresses = socket.getaddrinfo(
            lowered_hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise WebFetchRequestError(f"DNS lookup failed: {exc}") from exc

    if not addresses:
        raise WebFetchRequestError("DNS lookup returned no addresses.")

    for _, _, _, _, sockaddr in addresses:
        candidate = sockaddr[0]
        resolved = ipaddress.ip_address(candidate)
        if _is_non_public_address(resolved):
            raise WebFetchRequestError(
                "web_fetch resolved to a private, loopback, or reserved IP target."
            )


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


def _extract_http_error_message(
    *,
    response: requests.Response,
    max_bytes: int,
) -> str:
    status_code = response.status_code
    snippet = ""
    try:
        error_bytes = _read_response_bytes(response=response, max_bytes=max_bytes)
        if not _looks_binary_bytes(error_bytes):
            snippet = _visible_text(
                _decode_body_text(body_bytes=error_bytes, response=response)
            )[:300]
    except Exception:
        snippet = ""

    if snippet:
        return f"request returned HTTP {status_code}: {snippet}"
    return f"request returned HTTP {status_code}."


def _extract_cloudflare_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        errors = payload.get("errors")
        if isinstance(errors, list):
            details: list[str] = []
            for error in errors:
                if not isinstance(error, dict):
                    continue
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    details.append(message.strip())
            if details:
                return (
                    f"Cloudflare toMarkdown returned HTTP {response.status_code}: "
                    + "; ".join(details)
                )

    text = response.text.strip()
    if text:
        return f"Cloudflare toMarkdown returned HTTP {response.status_code}: {text[:300]}"
    return f"Cloudflare toMarkdown returned HTTP {response.status_code}."


def _read_response_bytes(*, response: requests.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total_bytes = 0
    for chunk in response.iter_content(chunk_size=16_384):
        if not chunk:
            continue
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise WebFetchBodyTooLargeError(
                f"response exceeded the {max_bytes}-byte limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_body_text(*, body_bytes: bytes, response: requests.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        text = body_bytes.decode(encoding, errors="replace")
    except LookupError:
        text = body_bytes.decode("utf-8", errors="replace")
    return _normalize_text(text)


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def _normalize_markdown(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalize_text(value)
    return normalized or None


def _normalize_content_type(raw_content_type: str | None) -> str | None:
    if raw_content_type is None:
        return None
    normalized = raw_content_type.split(";", 1)[0].strip().lower()
    return normalized or None


def _extract_header_value(headers: Any, name: str) -> str | None:
    if not hasattr(headers, "items"):
        return None
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() != target:
            continue
        normalized = str(value).strip()
        return normalized or None
    return None


def _extract_markdown_tokens(headers: Any) -> int | None:
    raw_value = _extract_header_value(headers, _MARKDOWN_TOKENS_HEADER)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _truncate_markdown(*, markdown: str, limit: int) -> tuple[str, bool]:
    if len(markdown) <= limit:
        return markdown, False
    head = markdown[:limit].rstrip()
    return f"{head}\n\n[markdown truncated]", True


def _looks_like_html(text: str) -> bool:
    sample = text.lstrip()[:500].lower()
    if sample.startswith("<!doctype html"):
        return True
    return bool(
        re.search(
            r"<(?:html|head|body|main|article|section|div|p|script|style|meta|title)\b",
            sample,
        )
    )


def _looks_like_app_shell(html: str) -> bool:
    lowered = html.lower()
    if any(pattern in lowered for pattern in _APP_SHELL_PATTERNS):
        visible = _visible_text(_strip_scripts_and_styles(html))
        return len(visible) < 250
    return False


def _strip_scripts_and_styles(html: str) -> str:
    without_scripts = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_styles = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        without_scripts,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return without_styles


def _visible_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    without_entities = without_tags.replace("&nbsp;", " ")
    normalized = re.sub(r"\s+", " ", without_entities).strip()
    return normalized


def _looks_binary_bytes(body_bytes: bytes) -> bool:
    if not body_bytes:
        return False
    if b"\x00" in body_bytes:
        return True
    sample = body_bytes[:512]
    control_bytes = sum(
        byte < 9 or (13 < byte < 32) for byte in sample
    )
    return control_bytes > max(8, len(sample) // 10)


def _is_unsupported_binary_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    if content_type in _UNSUPPORTED_BINARY_MIME_TYPES:
        return True
    return any(
        content_type.startswith(prefix) for prefix in _UNSUPPORTED_BINARY_PREFIXES
    )


def _hostname_for_conversion(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if parsed.hostname is None:
        return None
    return parsed.netloc or parsed.hostname
