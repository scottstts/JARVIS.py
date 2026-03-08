"""Web-search tool definition and execution runtime."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

import requests

from llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchConfigurationError(RuntimeError):
    """Raised when web_search is missing required local configuration."""


class WebSearchRequestError(RuntimeError):
    """Raised when the Brave request fails or returns an invalid response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class WebSearchToolExecutor:
    """Runs a basic Brave web search and normalizes the web results."""

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
        query = str(arguments["query"]).strip()

        try:
            payload, response_headers = await asyncio.to_thread(self._perform_request, query)
        except WebSearchConfigurationError as exc:
            return _web_search_error(
                call_id=call_id,
                query=query,
                reason=str(exc),
            )
        except WebSearchRequestError as exc:
            return _web_search_error(
                call_id=call_id,
                query=query,
                reason=str(exc),
                status_code=exc.status_code,
            )

        raw_web = payload.get("web")
        query_metadata = _normalize_query_metadata(payload.get("query"))
        results = _normalize_web_results(raw_web)
        content = _format_web_search_result(
            query=query,
            results=results,
            more_results_available=query_metadata.get("more_results_available"),
        )
        metadata = {
            "query": query_metadata,
            "results": results,
            "result_count": len(results),
            "configured_result_count": self._settings.web_search_result_count,
            "status_code": 200,
            "response_type": payload.get("type"),
            "family_friendly": _extract_family_friendly(raw_web),
            "rate_limit": _extract_rate_limit_headers(response_headers),
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="web_search",
            ok=True,
            content=content,
            metadata=metadata,
        )

    def _perform_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise WebSearchConfigurationError(
                "BRAVE_SEARCH_API_KEY is not configured."
            )

        try:
            response = requests.get(
                _BRAVE_WEB_SEARCH_URL,
                params={
                    "q": query,
                    "count": str(self._settings.web_search_result_count),
                    "result_filter": "web",
                    "spellcheck": "false",
                },
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                timeout=self._settings.web_search_timeout_seconds,
            )
        except requests.Timeout as exc:
            raise WebSearchRequestError("request timed out.") from exc
        except requests.RequestException as exc:
            raise WebSearchRequestError(f"request failed: {exc}") from exc

        if response.status_code != 200:
            raise WebSearchRequestError(
                _extract_error_message(response),
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchRequestError("Brave returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise WebSearchRequestError("Brave returned an unexpected response shape.")

        return payload, dict(response.headers)


def build_web_search_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the web_search registry entry."""

    return RegisteredTool(
        name="web_search",
        exposure="basic",
        definition=ToolDefinition(
            name="web_search",
            description=_build_web_search_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Plain web search query. Returns only normalized web results, "
                            "not news, videos, or other Brave verticals."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        executor=WebSearchToolExecutor(settings),
    )


def _build_web_search_tool_description(settings: ToolSettings) -> str:
    return (
        "Run a basic web search through Brave Search and return normalized web results. "
        f"Each search returns up to {settings.web_search_result_count} web results. "
        "This tool intentionally keeps the Brave request minimal: web results only, no "
        "advanced filters, no custom re-ranking, and no summary generation."
    )


def _normalize_query_metadata(raw_query: Any) -> dict[str, Any]:
    if not isinstance(raw_query, dict):
        return {}

    metadata: dict[str, Any] = {}
    for source_key, target_key in (
        ("original", "original"),
        ("cleaned", "cleaned"),
        ("altered", "altered"),
    ):
        value = _normalize_optional_string(raw_query.get(source_key))
        if value is not None:
            metadata[target_key] = value

    more_results_available = raw_query.get("more_results_available")
    if isinstance(more_results_available, bool):
        metadata["more_results_available"] = more_results_available

    return metadata


def _normalize_web_results(raw_web: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_web, dict):
        return []

    raw_results = raw_web.get("results")
    if not isinstance(raw_results, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        title = _normalize_optional_string(raw_result.get("title"))
        url = _normalize_optional_string(raw_result.get("url"))
        if title is None or url is None:
            continue

        result = {
            "title": title,
            "url": url,
        }
        snippet = _normalize_optional_string(raw_result.get("description"))
        if snippet is not None:
            result["snippet"] = snippet

        source = _extract_source(raw_result, url)
        if source is not None:
            result["source"] = source

        language = _normalize_optional_string(raw_result.get("language"))
        if language is not None:
            result["language"] = language

        page_age = _normalize_optional_string(raw_result.get("page_age"))
        if page_age is not None:
            result["page_age"] = page_age

        page_fetched = _normalize_optional_string(raw_result.get("page_fetched"))
        if page_fetched is not None:
            result["page_fetched"] = page_fetched

        normalized.append(result)

    return normalized


def _extract_source(raw_result: dict[str, Any], url: str) -> str | None:
    meta_url = raw_result.get("meta_url")
    if isinstance(meta_url, dict):
        hostname = _normalize_optional_string(meta_url.get("hostname"))
        if hostname is not None:
            return hostname

    parsed = urlparse(url)
    hostname = parsed.netloc.strip()
    return hostname or None


def _extract_family_friendly(raw_web: Any) -> bool | None:
    if not isinstance(raw_web, dict):
        return None

    family_friendly = raw_web.get("family_friendly")
    if isinstance(family_friendly, bool):
        return family_friendly
    return None


def _format_web_search_result(
    *,
    query: str,
    results: list[dict[str, Any]],
    more_results_available: bool | None,
) -> str:
    lines = [
        "Web search results",
        f"query: {query}",
        f"returned_results: {len(results)}",
    ]
    if more_results_available is not None:
        lines.append(f"more_results_available: {more_results_available}")

    lines.append("results:")
    if not results:
        lines.append("(none)")
        return "\n".join(lines)

    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}")
        lines.append(f"   url: {result['url']}")
        source = result.get("source")
        if isinstance(source, str):
            lines.append(f"   source: {source}")
        snippet = result.get("snippet")
        if isinstance(snippet, str):
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)


def _extract_rate_limit_headers(headers: dict[str, str]) -> dict[str, str]:
    rate_limit_headers: dict[str, str] = {}
    for header_name in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        value = headers.get(header_name)
        if value:
            rate_limit_headers[header_name] = value
    return rate_limit_headers


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("detail", "message", "code"):
                value = _normalize_optional_string(error.get(key))
                if value is not None:
                    return f"Brave API returned HTTP {response.status_code}: {value}"
        if isinstance(error, str):
            normalized_error = error.strip()
            if normalized_error:
                return f"Brave API returned HTTP {response.status_code}: {normalized_error}"

    body = response.text.strip()
    if body:
        return f"Brave API returned HTTP {response.status_code}: {body}"
    return f"Brave API returned HTTP {response.status_code}."


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _web_search_error(
    *,
    call_id: str,
    query: str,
    reason: str,
    status_code: int | None = None,
) -> ToolExecutionResult:
    metadata: dict[str, Any] = {
        "query": query,
        "error": reason,
    }
    if status_code is not None:
        metadata["status_code"] = status_code

    lines = [
        "Web search failed",
        f"query: {query}",
        f"reason: {reason}",
    ]
    if status_code is not None:
        lines.insert(2, f"status_code: {status_code}")

    return ToolExecutionResult(
        call_id=call_id,
        name="web_search",
        ok=False,
        content="\n".join(lines),
        metadata=metadata,
    )
