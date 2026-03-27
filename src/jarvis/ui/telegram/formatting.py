"""Render model markdown into Telegram-compatible HTML."""

from __future__ import annotations

import html
from urllib.parse import urlparse


_INLINE_MARKERS: tuple[tuple[str, str], ...] = (
    ("**", "b"),
    ("~~", "s"),
    ("||", "tg-spoiler"),
    ("*", "i"),
    ("_", "i"),
)

_ALLOWED_LINK_SCHEMES = {"http", "https", "mailto", "tg"}


def render_markdown_to_telegram_html(text: str) -> str:
    """Converts a markdown-like reply into Telegram HTML.

    The renderer is intentionally tolerant: unmatched markdown markers are left
    as literal text so partially-streamed drafts remain valid.
    """

    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    rendered: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.startswith("```"):
            code_html, next_index = _render_fenced_code_block(lines, index)
            rendered.append(code_html)
            index = next_index
            continue

        if _is_quote_line(line):
            quote_html, next_index = _render_blockquote(lines, index)
            rendered.append(quote_html)
            index = next_index
            continue

        heading_html = _render_heading(line)
        if heading_html is not None:
            rendered.append(heading_html)
        else:
            rendered.append(_render_inline(line))
        index += 1

    return "\n".join(rendered)


def _render_fenced_code_block(lines: list[str], start_index: int) -> tuple[str, int]:
    opening = lines[start_index]
    language = opening[3:].strip().split(maxsplit=1)[0] if opening[3:].strip() else ""

    for end_index in range(start_index + 1, len(lines)):
        if lines[end_index].startswith("```"):
            content = "\n".join(lines[start_index + 1 : end_index])
            escaped = html.escape(content)
            if language:
                return (
                    '<pre><code class="language-'
                    f'{html.escape(language, quote=True)}">{escaped}</code></pre>',
                    end_index + 1,
                )
            return (f"<pre>{escaped}</pre>", end_index + 1)

    literal = "\n".join(lines[start_index:])
    return (html.escape(literal), len(lines))


def _is_quote_line(line: str) -> bool:
    return line.startswith(">") and (len(line) == 1 or line[1] in {" ", "\t"})


def _render_blockquote(lines: list[str], start_index: int) -> tuple[str, int]:
    quote_lines: list[str] = []
    index = start_index
    while index < len(lines) and _is_quote_line(lines[index]):
        quote_line = lines[index][1:]
        if quote_line.startswith(" "):
            quote_line = quote_line[1:]
        quote_lines.append(_render_inline(quote_line))
        index += 1
    joined_quote_lines = "\n".join(quote_lines)
    return (f"<blockquote>{joined_quote_lines}</blockquote>", index)


def _render_heading(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None

    marker_length = len(stripped) - len(stripped.lstrip("#"))
    if marker_length == 0 or marker_length > 6:
        return None
    if len(stripped) <= marker_length or stripped[marker_length] != " ":
        return None
    return f"<b>{_render_inline(stripped[marker_length + 1 :])}</b>"


def _render_inline(text: str) -> str:
    parts: list[str] = []
    index = 0

    while index < len(text):
        link = _try_render_link(text, index)
        if link is not None:
            rendered, index = link
            parts.append(rendered)
            continue

        if text[index] == "`":
            code_span = _try_render_code_span(text, index)
            if code_span is not None:
                rendered, index = code_span
                parts.append(rendered)
                continue

        marker_match = _try_render_marker(text, index)
        if marker_match is not None:
            rendered, index = marker_match
            parts.append(rendered)
            continue

        parts.append(html.escape(text[index]))
        index += 1

    return "".join(parts)


def _try_render_link(text: str, start_index: int) -> tuple[str, int] | None:
    if text[start_index] != "[":
        return None

    label_end = text.find("]", start_index + 1)
    if label_end == -1 or label_end + 1 >= len(text) or text[label_end + 1] != "(":
        return None

    url_end = _find_link_url_end(text, label_end + 2)
    if url_end == -1:
        return None

    label = text[start_index + 1 : label_end]
    url = text[label_end + 2 : url_end].strip()
    if not label or not _is_safe_url(url):
        return None

    return (
        f'<a href="{html.escape(url, quote=True)}">{_render_inline(label)}</a>',
        url_end + 1,
    )


def _find_link_url_end(text: str, start_index: int) -> int:
    depth = 1
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_LINK_SCHEMES:
        return False
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    if parsed.scheme == "mailto":
        return bool(parsed.path)
    return True


def _try_render_code_span(text: str, start_index: int) -> tuple[str, int] | None:
    end_index = text.find("`", start_index + 1)
    if end_index == -1:
        return None
    content = text[start_index + 1 : end_index]
    return (f"<code>{html.escape(content)}</code>", end_index + 1)


def _try_render_marker(text: str, start_index: int) -> tuple[str, int] | None:
    for marker, tag in _INLINE_MARKERS:
        if not text.startswith(marker, start_index):
            continue

        end_index = text.find(marker, start_index + len(marker))
        if end_index == -1:
            if len(marker) == 1:
                return None
            return (html.escape(marker), start_index + len(marker))

        inner = _render_inline(text[start_index + len(marker) : end_index])
        return (f"<{tag}>{inner}</{tag}>", end_index + len(marker))
    return None
