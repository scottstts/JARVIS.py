"""Web-fetch tool definition and execution helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...remote_runtime_client import RemoteToolRuntimeClient, RemoteToolRuntimeError
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_CLOUDFLARE_TOMARKDOWN_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/tomarkdown"
)
_DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)
_DEFUDDLE_NODE_CA_CERTS_PATH = "/etc/ssl/certs/ca-certificates.crt"
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
_LOW_SIGNAL_MARKDOWN_PATTERNS = (
    "enable javascript",
    "javascript required",
    "loading...",
    "please wait",
)
_ACCESS_CHALLENGE_MARKDOWN_PATTERNS = (
    "performing security verification",
    "verify you are not a bot",
    "checking your browser before accessing",
    "please enable javascript and cookies to continue",
)
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}
_PLAYWRIGHT_NON_NETWORK_SCHEMES = {"data"}
_REDDIT_HOSTS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "np.reddit.com",
    "redd.it",
    "www.redd.it",
}
_TWITTER_HOSTS = {
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
    "x.com",
    "www.x.com",
}
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


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
    """Raised when a fetched page exceeds the configured size budget."""


@dataclass(slots=True, frozen=True)
class HTTPFetchResult:
    requested_url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    content_type: str | None
    body_text: str
    redirect_chain: tuple[str, ...]


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
    """Fetches webpages locally and uses tool_runtime only for Defuddle."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings
        self._remote_client = RemoteToolRuntimeClient(settings)

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        requested_url = str(arguments.get("url", "")).strip()
        if not requested_url:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason="web_fetch requires a non-empty 'url'.",
            )

        if _must_use_defuddle(requested_url):
            result, reason = await self._run_remote_defuddle_fetch(
                call_id=call_id,
                requested_url=requested_url,
                context=context,
            )
            if result is not None:
                return result
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=reason or "Defuddle fetch failed.",
            )

        try:
            tier1_result = await asyncio.to_thread(
                _fetch_http_text,
                url=requested_url,
                accept_markdown=True,
                settings=self._settings,
            )
        except WebFetchUnsupportedContentError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                content_type=exc.content_type,
            )
        except WebFetchBodyTooLargeError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
            )
        except WebFetchRequestError:
            tier1_result = None

        if tier1_result is not None and _tier1_response_is_usable_markdown(tier1_result):
            return _web_fetch_success(
                call_id=call_id,
                requested_url=requested_url,
                final_url=tier1_result.final_url,
                status_code=tier1_result.status_code,
                content_type=tier1_result.content_type,
                redirect_chain=tier1_result.redirect_chain,
                markdown=tier1_result.body_text,
                markdown_tokens=_extract_markdown_tokens(tier1_result.headers),
                content_signal=_extract_header_value(
                    tier1_result.headers,
                    _CONTENT_SIGNAL_HEADER,
                ),
                settings=self._settings,
            )

        defuddle_result, _defuddle_reason = await self._run_remote_defuddle_fetch(
            call_id=call_id,
            requested_url=requested_url,
            context=context,
        )
        if defuddle_result is not None:
            return defuddle_result

        fallback_result: HTTPFetchResult | None = None
        fallback_request_error: WebFetchRequestError | None = None
        try:
            fallback_result = await asyncio.to_thread(
                _fetch_http_text,
                url=requested_url,
                accept_markdown=False,
                settings=self._settings,
            )
        except WebFetchUnsupportedContentError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                content_type=exc.content_type,
            )
        except WebFetchBodyTooLargeError as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
            )
        except WebFetchRequestError as exc:
            fallback_request_error = exc

        if fallback_result is not None:
            if _is_direct_text_response(fallback_result) and _markdown_is_usable(
                fallback_result.body_text
            ):
                return _web_fetch_success(
                    call_id=call_id,
                    requested_url=requested_url,
                    final_url=fallback_result.final_url,
                    status_code=fallback_result.status_code,
                    content_type=fallback_result.content_type,
                    redirect_chain=fallback_result.redirect_chain,
                    markdown=fallback_result.body_text,
                    markdown_tokens=_extract_markdown_tokens(fallback_result.headers),
                    content_signal=_extract_header_value(
                        fallback_result.headers,
                        _CONTENT_SIGNAL_HEADER,
                    ),
                    settings=self._settings,
                )

            if not _is_html_response(fallback_result):
                return _web_fetch_error(
                    call_id=call_id,
                    requested_url=requested_url,
                    reason=(
                        "web_fetch only supports markdown, plain-text, or HTML pages."
                    ),
                    status_code=fallback_result.status_code,
                    content_type=fallback_result.content_type,
                )

            try:
                converted_result = await asyncio.to_thread(
                    _convert_html_to_markdown,
                    html=fallback_result.body_text,
                    source_url=fallback_result.final_url,
                    settings=self._settings,
                )
            except (WebFetchConfigurationError, WebFetchRequestError):
                converted_result = None
            else:
                if _markdown_is_usable(converted_result.markdown):
                    return _web_fetch_success(
                        call_id=call_id,
                        requested_url=requested_url,
                        final_url=fallback_result.final_url,
                        status_code=fallback_result.status_code,
                        content_type=fallback_result.content_type,
                        redirect_chain=fallback_result.redirect_chain,
                        markdown=converted_result.markdown,
                        markdown_tokens=converted_result.markdown_tokens,
                        content_signal=_extract_header_value(
                            fallback_result.headers,
                            _CONTENT_SIGNAL_HEADER,
                        ),
                        settings=self._settings,
                    )

        browser_target_url = _resolve_browser_fallback_target(
            requested_url=requested_url,
            fallback_result=fallback_result,
        )
        try:
            rendered_result = await _render_page_html(
                url=browser_target_url,
                settings=self._settings,
            )
        except (WebFetchConfigurationError, RuntimeError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                status_code=(
                    fallback_request_error.status_code
                    if fallback_request_error is not None
                    else None
                ),
                content_type=(
                    fallback_result.content_type
                    if fallback_result is not None
                    else None
                ),
            )

        rendered_html_path = _write_temporary_html_input(
            workspace_dir=context.workspace_dir,
            html=rendered_result.html,
        )
        try:
            rendered_defuddle_result, _rendered_defuddle_reason = (
                await self._run_remote_defuddle_fetch(
                    call_id=call_id,
                    requested_url=requested_url,
                    context=context,
                    input_path=rendered_html_path,
                )
            )
        finally:
            rendered_html_path.unlink(missing_ok=True)

        if rendered_defuddle_result is not None:
            rendered_markdown = _extract_markdown_from_tool_result(rendered_defuddle_result)
            if rendered_markdown is None or _markdown_is_usable(rendered_markdown):
                return rendered_defuddle_result

        try:
            rendered_conversion = await asyncio.to_thread(
                _convert_html_to_markdown,
                html=rendered_result.html,
                source_url=rendered_result.final_url,
                settings=self._settings,
            )
        except (WebFetchConfigurationError, WebFetchRequestError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
                status_code=getattr(exc, "status_code", None),
                content_type="text/html",
            )

        if not _markdown_is_usable(rendered_conversion.markdown):
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason="web_fetch could not extract acceptable markdown from the rendered page.",
                content_type="text/html",
            )

        return _web_fetch_success(
            call_id=call_id,
            requested_url=requested_url,
            final_url=rendered_result.final_url,
            status_code=200,
            content_type="text/html",
            redirect_chain=(),
            markdown=rendered_conversion.markdown,
            markdown_tokens=rendered_conversion.markdown_tokens,
            content_signal=None,
            settings=self._settings,
        )

    async def _run_remote_defuddle_fetch(
        self,
        *,
        call_id: str,
        requested_url: str,
        context: ToolExecutionContext,
        input_path: Path | None = None,
    ) -> tuple[ToolExecutionResult | None, str | None]:
        arguments: dict[str, Any] = {"url": requested_url}
        if input_path is not None:
            arguments["input_path"] = str(input_path.relative_to(context.workspace_dir))
        try:
            result = await self._remote_client.execute(
                tool_name="web_fetch",
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        except RemoteToolRuntimeError as exc:
            return None, str(exc)

        if result.ok:
            return result, None
        return None, _extract_tool_error_reason(result)


class DirectWebFetchToolExecutor:
    """Runs the Defuddle-only web_fetch path inside tool_runtime."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        requested_url = str(arguments.get("url", "")).strip()
        input_path = str(arguments.get("input_path", "")).strip() or None
        try:
            return await asyncio.to_thread(
                _run_defuddle_web_fetch,
                call_id=call_id,
                requested_url=requested_url,
                input_path=input_path,
                workspace_dir=context.workspace_dir,
                settings=self._settings,
            )
        except (WebFetchConfigurationError, WebFetchRequestError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
            )


def build_service_web_fetch_executor(settings: ToolSettings) -> DirectWebFetchToolExecutor:
    """Build the web_fetch executor used by the tool_runtime service."""

    return DirectWebFetchToolExecutor(settings)


def build_web_fetch_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the web_fetch registry entry."""

    return RegisteredTool(
        name="web_fetch",
        exposure="basic",
        definition=ToolDefinition(
            name="web_fetch",
            description=_build_web_fetch_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute public http:// or https:// URL to fetch and return as "
                            "clean markdown."
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        executor=WebFetchToolExecutor(settings),
    )


def _build_web_fetch_tool_description() -> str:
    return (
        "Fetch one specific public URL and return clean markdown. "
        "Use it when you already have the exact URL, including normal webpages, "
        "YouTube videos, X posts, and Reddit pages."
    )


def _run_defuddle_web_fetch(
    *,
    call_id: str,
    requested_url: str,
    input_path: str | None,
    workspace_dir: Path,
    settings: ToolSettings,
) -> ToolExecutionResult:
    if not requested_url and input_path is None:
        return _web_fetch_error(
            call_id=call_id,
            requested_url=requested_url,
            reason="web_fetch requires a non-empty 'url'.",
        )

    parse_target = requested_url
    if input_path is not None:
        parse_target = str(
            _resolve_workspace_input_path(
                requested_path=input_path,
                workspace_dir=workspace_dir,
            )
        )

    try:
        result = subprocess.run(
            [
                "npx",
                "defuddle",
                "parse",
                parse_target,
                "--markdown",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=str(workspace_dir),
            timeout=settings.web_fetch_timeout_seconds,
            env=_build_defuddle_environment(),
        )
    except FileNotFoundError as exc:
        raise WebFetchConfigurationError(
            "npx is required inside tool_runtime for web_fetch."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise WebFetchRequestError(
            f"Defuddle request timed out after {settings.web_fetch_timeout_seconds} seconds."
        ) from exc
    except OSError as exc:
        raise WebFetchRequestError(f"Defuddle execution failed: {exc}") from exc

    if result.returncode != 0:
        error_details = (
            _normalize_optional_text(result.stderr)
            or _normalize_optional_text(result.stdout)
            or f"defuddle exited with status {result.returncode}."
        )
        raise WebFetchRequestError(error_details)

    markdown = _normalize_optional_text(result.stdout)
    if markdown is None:
        raise WebFetchRequestError("Defuddle returned empty markdown.")

    truncated_markdown, markdown_truncated = _truncate_markdown(
        markdown=markdown,
        limit=settings.web_fetch_max_markdown_chars,
    )
    return ToolExecutionResult(
        call_id=call_id,
        name="web_fetch",
        ok=True,
        content="\n".join(
            [
                "Web fetch result",
                f"url: {requested_url}",
                "markdown:",
                truncated_markdown,
            ]
        ),
        metadata={
            "requested_url": requested_url,
            "markdown_chars": len(markdown),
            "markdown_truncated": markdown_truncated,
        },
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
                    max_bytes=settings.web_fetch_max_markdown_chars,
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
                        "web_fetch does not support binary or document content "
                        f"(content type: {normalized_content_type or 'unknown'})."
                    ),
                    content_type=normalized_content_type,
                )

            try:
                body_bytes = _read_response_bytes(
                    response=response,
                    max_bytes=settings.web_fetch_max_markdown_chars,
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

    markdown = _normalize_optional_text(first_result.get("data"))
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


def _must_use_defuddle(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False

    if hostname in _REDDIT_HOSTS:
        return True
    if _is_youtube_video_url(parsed):
        return True
    return _is_x_article_or_post_url(parsed)


def _is_youtube_video_url(parsed_url: Any) -> bool:
    hostname = (parsed_url.hostname or "").strip().lower()
    if hostname not in _YOUTUBE_HOSTS:
        return False

    if hostname.endswith("youtu.be"):
        return parsed_url.path.strip("/") != ""

    path = parsed_url.path.rstrip("/")
    if path == "/watch":
        return bool(parse_qs(parsed_url.query).get("v"))
    return any(
        path.startswith(prefix)
        for prefix in ("/shorts/", "/live/", "/embed/")
    )


def _is_x_article_or_post_url(parsed_url: Any) -> bool:
    hostname = (parsed_url.hostname or "").strip().lower()
    if hostname not in _TWITTER_HOSTS:
        return False

    path_segments = [segment for segment in parsed_url.path.split("/") if segment]
    return "status" in path_segments or "article" in path_segments


def _tier1_response_is_usable_markdown(result: HTTPFetchResult) -> bool:
    if result.content_type == "text/markdown":
        return _markdown_is_usable(result.body_text)
    if _extract_markdown_tokens(result.headers) is not None:
        return _markdown_is_usable(result.body_text)
    return not _is_html_response(result) and _markdown_is_usable(result.body_text)


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
    normalized = _normalize_optional_text(markdown)
    if normalized is None:
        return False
    if _looks_like_html(normalized):
        return False

    visible = _visible_text(normalized)
    if len(visible) < 20:
        return False

    lowered = visible.lower()
    if _looks_like_access_challenge(lowered):
        return False
    if any(pattern in lowered for pattern in _LOW_SIGNAL_MARKDOWN_PATTERNS):
        return len(visible) >= 120
    return True


def _looks_like_access_challenge(lowered_visible_text: str) -> bool:
    if "just a moment" in lowered_visible_text and (
        "cloudflare" in lowered_visible_text or "security verification" in lowered_visible_text
    ):
        return True
    return any(
        pattern in lowered_visible_text
        for pattern in _ACCESS_CHALLENGE_MARKDOWN_PATTERNS
    )


def _web_fetch_success(
    *,
    call_id: str,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str | None,
    redirect_chain: tuple[str, ...],
    markdown: str,
    markdown_tokens: int | None,
    content_signal: str | None,
    settings: ToolSettings,
) -> ToolExecutionResult:
    truncated_markdown, markdown_truncated = _truncate_markdown(
        markdown=markdown,
        limit=settings.web_fetch_max_markdown_chars,
    )
    lines = [
        "Web fetch result",
        f"url: {requested_url}",
    ]
    if final_url != requested_url:
        lines.append(f"final_url: {final_url}")
    lines.extend(
        [
            "markdown:",
            truncated_markdown,
        ]
    )

    metadata: dict[str, Any] = {
        "requested_url": requested_url,
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "redirect_chain": list(redirect_chain),
        "markdown_chars": len(markdown),
        "markdown_truncated": markdown_truncated,
    }
    if markdown_tokens is not None:
        metadata["markdown_tokens"] = markdown_tokens
    if content_signal is not None:
        metadata["content_signal"] = content_signal

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


def _extract_tool_error_reason(result: ToolExecutionResult) -> str:
    reason = result.metadata.get("error")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    lines = result.content.splitlines()
    for line in lines:
        if line.startswith("reason: "):
            return line.removeprefix("reason: ").strip()
    return "Defuddle fetch failed."


def _extract_markdown_from_tool_result(result: ToolExecutionResult) -> str | None:
    marker = "markdown:\n"
    if marker not in result.content:
        return None
    markdown = result.content.split(marker, 1)[1]
    return _normalize_optional_text(markdown)


def _resolve_browser_fallback_target(
    *,
    requested_url: str,
    fallback_result: HTTPFetchResult | None,
) -> str:
    if fallback_result is None:
        return requested_url
    return fallback_result.final_url


def _write_temporary_html_input(*, workspace_dir: Path, html: str) -> Path:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".html",
        prefix="web-fetch-render-",
        dir=workspace_dir,
        delete=False,
    ) as handle:
        handle.write(html)
        return Path(handle.name)


def _resolve_workspace_input_path(
    *,
    requested_path: str,
    workspace_dir: Path,
) -> Path:
    root = workspace_dir.resolve(strict=False)
    candidate = (root / requested_path).resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise WebFetchRequestError("web_fetch input_path must stay inside the workspace.")
    if not candidate.exists():
        raise WebFetchRequestError("web_fetch input_path does not exist inside the workspace.")
    if not candidate.is_file():
        raise WebFetchRequestError("web_fetch input_path must point to a file.")
    return candidate


async def _render_page_html(
    *,
    url: str,
    settings: ToolSettings,
) -> BrowserRenderResult:
    await asyncio.to_thread(_validate_public_url, url)

    timeout_ms = int(settings.web_fetch_timeout_seconds * 1000)
    network_idle_timeout_ms = min(timeout_ms, 5_000)
    validated_request_urls = {url}
    blocked_request_url: str | None = None
    blocked_request_reason: str | None = None

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"],
            )
            context = await browser.new_context(service_workers="block")
            page = await context.new_page()

            async def route_request(route: Any) -> None:
                nonlocal blocked_request_reason, blocked_request_url
                request_url = route.request.url
                if request_url not in validated_request_urls:
                    try:
                        await asyncio.to_thread(_validate_browser_request_url, request_url)
                    except WebFetchRequestError as exc:
                        if blocked_request_url is None:
                            blocked_request_url = request_url
                            blocked_request_reason = str(exc)
                        await route.abort("blockedbyclient")
                        return
                    validated_request_urls.add(request_url)
                await route.continue_()

            await page.route("**/*", route_request)
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
                await asyncio.to_thread(_validate_public_url, final_url)
            finally:
                await context.close()
                await browser.close()
    except PlaywrightTimeoutError as exc:
        raise RuntimeError("Playwright page render timed out.") from exc
    except WebFetchRequestError as exc:
        raise RuntimeError(
            f"Playwright resolved to a blocked final URL: {exc}"
        ) from exc
    except Exception as exc:
        if blocked_request_url is not None and blocked_request_reason is not None:
            raise RuntimeError(
                "Playwright blocked a non-public browser request "
                f"({blocked_request_url}): {blocked_request_reason}"
            ) from exc
        raise RuntimeError(f"Playwright page render failed: {exc}") from exc

    return BrowserRenderResult(
        requested_url=url,
        final_url=final_url,
        html=_normalize_text(html),
    )


def _validate_browser_request_url(url: str) -> None:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        _validate_public_url(url)
        return
    if scheme in _PLAYWRIGHT_NON_NETWORK_SCHEMES:
        return
    raise WebFetchRequestError(
        f"web_fetch Playwright does not allow browser requests with scheme '{scheme or 'unknown'}'."
    )


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
        resolved = ipaddress.ip_address(sockaddr[0])
        if _is_non_public_address(resolved):
            raise WebFetchRequestError(
                "web_fetch resolved to a private, loopback, or reserved IP target."
            )


def _is_non_public_address(address: Any) -> bool:
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
                "response exceeded the configured web_fetch size limit."
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


def _build_defuddle_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["NODE_EXTRA_CA_CERTS"] = _DEFUDDLE_NODE_CA_CERTS_PATH
    return environment


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def _normalize_optional_text(value: Any) -> str | None:
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
    return markdown[:limit].rstrip(), True


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
