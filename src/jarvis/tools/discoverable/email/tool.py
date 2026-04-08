"""Email discoverable tool definition and execution runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import html
import os
import re
import smtplib
from typing import Any
from urllib.parse import urlparse

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...types import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionContext,
    ToolExecutionResult,
)
from .shared import EmailArgumentError, EmailRequest, build_email_request

_ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}
_INLINE_MARKERS: tuple[tuple[str, str], ...] = (
    ("**", "strong"),
    ("~~", "del"),
    ("*", "em"),
    ("_", "em"),
)
_ORDERED_LIST_PATTERN = re.compile(r"^\d+\.\s+")


class EmailConfigurationError(RuntimeError):
    """Raised when SMTP configuration is incomplete."""


class EmailSendError(RuntimeError):
    """Raised when the SMTP send operation fails."""


@dataclass(slots=True, frozen=True)
class SentEmailPayload:
    """Normalized result payload for one SMTP send."""

    message_id: str
    from_email: str
    to_email: str
    subject: str
    attachment_count: int


class EmailToolExecutor:
    """Sends one email through the configured SMTP account."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            request = build_email_request(
                arguments=arguments,
                context=context,
                settings=self._settings,
            )
        except EmailArgumentError as exc:
            return _email_error(
                call_id=call_id,
                to_email=str(arguments.get("to_email", "")).strip() or None,
                subject=str(arguments.get("subject", "")).strip() or None,
                reason=str(exc),
            )

        try:
            payload = await asyncio.to_thread(
                self._send_email,
                request,
            )
        except (EmailConfigurationError, EmailSendError) as exc:
            return _email_error(
                call_id=call_id,
                to_email=request.to_email,
                subject=request.subject,
                reason=str(exc),
            )

        content_lines = [
            "Email sent",
            "provider: smtp",
            f"from_email: {payload.from_email}",
            f"to_email: {payload.to_email}",
            f"subject: {payload.subject}",
            f"attachment_count: {payload.attachment_count}",
            f"message_id: {payload.message_id}",
        ]
        if request.attachment_specs:
            content_lines.append("attachment_paths:")
            content_lines.extend(
                f"- {spec.resolved_path}"
                for spec in request.attachment_specs
            )

        return ToolExecutionResult(
            call_id=call_id,
            name="email",
            ok=True,
            content="\n".join(content_lines),
            metadata={
                "provider": "smtp",
                "smtp_host": self._settings.email_smtp_host,
                "smtp_port": self._settings.email_smtp_port,
                "smtp_security": self._settings.email_smtp_security,
                "from_email": payload.from_email,
                "to_email": payload.to_email,
                "subject": payload.subject,
                "attachment_count": payload.attachment_count,
                "attachment_paths": [
                    str(spec.resolved_path) for spec in request.attachment_specs
                ],
                "message_id": payload.message_id,
            },
        )

    def _send_email(self, request: EmailRequest) -> SentEmailPayload:
        sender_email = self._settings.email_sender_address
        if sender_email is None:
            raise EmailConfigurationError(
                "SENDER_EMAIL_ADDRESS is required for email."
            )

        smtp_password = os.getenv("SMTP_PASSWORD")
        if not smtp_password:
            raise EmailConfigurationError("SMTP_PASSWORD is required for email.")

        message = EmailMessage()
        message_id = make_msgid()
        message["Message-ID"] = message_id
        message["Date"] = formatdate(localtime=True)
        message["From"] = sender_email
        message["To"] = request.to_email
        message["Subject"] = request.subject
        plain_text_body = _render_markdown_to_plain_text(request.body_markdown)
        html_body = _render_markdown_to_email_html(request.body_markdown)
        message.set_content(plain_text_body)
        message.add_alternative(html_body, subtype="html")

        for spec in request.attachment_specs:
            maintype, subtype = spec.media_type.split("/", maxsplit=1)
            try:
                attachment_bytes = spec.resolved_path.read_bytes()
            except OSError as exc:
                raise EmailSendError(
                    f"failed to read attachment '{spec.raw_path}': {exc}"
                ) from exc
            message.add_attachment(
                attachment_bytes,
                maintype=maintype,
                subtype=subtype,
                filename=spec.filename,
            )

        try:
            security = self._settings.email_smtp_security
            if security == "ssl":
                with smtplib.SMTP_SSL(
                    self._settings.email_smtp_host,
                    self._settings.email_smtp_port,
                    timeout=self._settings.email_timeout_seconds,
                ) as client:
                    client.login(sender_email, smtp_password)
                    client.send_message(message)
            else:
                with smtplib.SMTP(
                    self._settings.email_smtp_host,
                    self._settings.email_smtp_port,
                    timeout=self._settings.email_timeout_seconds,
                ) as client:
                    client.ehlo()
                    if security == "starttls":
                        client.starttls()
                        client.ehlo()
                    client.login(sender_email, smtp_password)
                    client.send_message(message)
        except (smtplib.SMTPException, OSError) as exc:
            raise EmailSendError(f"SMTP send failed: {exc}") from exc

        return SentEmailPayload(
            message_id=message_id,
            from_email=sender_email,
            to_email=request.to_email,
            subject=request.subject,
            attachment_count=len(request.attachment_specs),
        )


def build_email_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the email registry entry."""

    return RegisteredTool(
        name="email",
        exposure="discoverable",
        definition=ToolDefinition(
            name="email",
            description=_build_email_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "to_email": {
                        "type": "string",
                        "description": "Required recipient address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Required subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Required markdown email body.",
                    },
                    "attachment_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional workspace attachments.",
                    },
                },
                "required": ["to_email", "subject", "body"],
                "additionalProperties": False,
            },
        ),
        executor=EmailToolExecutor(settings),
        allowed_agent_kinds=("main",),
    )


def build_email_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for email."""

    return DiscoverableTool(
        name="email",
        aliases=(
            "send email",
            "smtp email",
            "mail",
        ),
        purpose="Send an email to a recipient address through the configured SMTP account.",
        detailed_description=_build_email_tool_description(),
        backing_tool_name="email",
        allowed_agent_kinds=("main",),
    )


def _build_email_tool_description() -> str:
    return (
        "Send one email through the configured SMTP account. Provide `to_email`, "
        "`subject`, markdown `body`, and optional workspace attachments. Every send "
        "requires user approval."
    )


def _render_markdown_to_plain_text(text: str) -> str:
    body = text.strip()
    return f"{body}\n\n---\nSent by Jarvis"


def _render_markdown_to_email_html(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            rendered, index = _render_fenced_code_block(lines, index)
            blocks.append(rendered)
            continue
        if _is_unordered_list_item(line):
            rendered, index = _render_list(lines, index, ordered=False)
            blocks.append(rendered)
            continue
        if _is_ordered_list_item(line):
            rendered, index = _render_list(lines, index, ordered=True)
            blocks.append(rendered)
            continue
        if _is_quote_line(line):
            rendered, index = _render_blockquote(lines, index)
            blocks.append(rendered)
            continue

        heading = _render_heading(line)
        if heading is not None:
            blocks.append(heading)
            index += 1
            continue

        rendered, index = _render_paragraph(lines, index)
        blocks.append(rendered)

    footer = (
        '<hr style="border:none;border-top:1px solid #d0d7de;margin:24px 0 16px;">'
        '<p style="color:#57606a;font-size:12px;margin:0;">Sent by Jarvis</p>'
    )
    body_html = "\n".join(blocks)
    return (
        "<!DOCTYPE html>"
        '<html><body style="margin:0;padding:24px;font-family:Arial,sans-serif;'
        'line-height:1.6;color:#24292f;background:#ffffff;">'
        f'<div style="max-width:720px;margin:0 auto;">{body_html}{footer}</div>'
        "</body></html>"
    )


def _render_fenced_code_block(lines: list[str], start_index: int) -> tuple[str, int]:
    opening = lines[start_index]
    language = opening[3:].strip().split(maxsplit=1)[0] if opening[3:].strip() else ""
    for end_index in range(start_index + 1, len(lines)):
        if lines[end_index].startswith("```"):
            content = "\n".join(lines[start_index + 1 : end_index])
            escaped = html.escape(content)
            language_attr = (
                f' data-language="{html.escape(language, quote=True)}"'
                if language
                else ""
            )
            rendered = (
                '<pre style="background:#f6f8fa;border-radius:6px;padding:12px;overflow:auto;">'
                f"<code{language_attr}>{escaped}</code></pre>"
            )
            return rendered, end_index + 1

    literal = html.escape("\n".join(lines[start_index:]))
    return f"<p>{literal}</p>", len(lines)


def _render_list(
    lines: list[str],
    start_index: int,
    *,
    ordered: bool,
) -> tuple[str, int]:
    tag = "ol" if ordered else "ul"
    items: list[str] = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        if ordered:
            if not _is_ordered_list_item(line):
                break
            marker_end = line.find(" ")
        else:
            if not _is_unordered_list_item(line):
                break
            marker_end = 1
        item_text = line[marker_end + 1 :].strip()
        items.append(f"<li>{_render_inline(item_text)}</li>")
        index += 1
    rendered = (
        f'<{tag} style="padding-left:24px;margin:0 0 16px 0;">'
        f'{"".join(items)}'
        f"</{tag}>"
    )
    return rendered, index


def _render_blockquote(lines: list[str], start_index: int) -> tuple[str, int]:
    quote_lines: list[str] = []
    index = start_index
    while index < len(lines) and _is_quote_line(lines[index]):
        quote_line = lines[index][1:]
        if quote_line.startswith(" "):
            quote_line = quote_line[1:]
        quote_lines.append(quote_line)
        index += 1
    rendered = "<br>".join(_render_inline(line) for line in quote_lines)
    return (
        '<blockquote style="margin:0 0 16px 0;padding-left:16px;border-left:4px solid #d0d7de;'
        f'color:#57606a;">{rendered}</blockquote>',
        index,
    )


def _render_heading(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    marker_length = len(stripped) - len(stripped.lstrip("#"))
    if marker_length == 0 or marker_length > 6:
        return None
    if len(stripped) <= marker_length or stripped[marker_length] != " ":
        return None
    level = marker_length
    content = _render_inline(stripped[marker_length + 1 :])
    return (
        f'<h{level} style="margin:0 0 16px 0;line-height:1.25;">'
        f"{content}</h{level}>"
    )


def _render_paragraph(lines: list[str], start_index: int) -> tuple[str, int]:
    paragraph_lines: list[str] = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            break
        if line.startswith("```"):
            break
        if _is_unordered_list_item(line) or _is_ordered_list_item(line):
            break
        if _is_quote_line(line):
            break
        if _render_heading(line) is not None:
            break
        paragraph_lines.append(line.strip())
        index += 1
    joined = " ".join(paragraph_lines)
    return (
        f'<p style="margin:0 0 16px 0;">{_render_inline(joined)}</p>',
        index,
    )


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
        marker = _try_render_marker(text, index)
        if marker is not None:
            rendered, index = marker
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
        f'<a href="{html.escape(url, quote=True)}" '
        'style="color:#0969da;text-decoration:underline;">'
        f"{_render_inline(label)}</a>",
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
    return False


def _try_render_code_span(text: str, start_index: int) -> tuple[str, int] | None:
    end_index = text.find("`", start_index + 1)
    if end_index == -1:
        return None
    content = text[start_index + 1 : end_index]
    return (
        '<code style="background:#f6f8fa;border-radius:4px;padding:0.15em 0.35em;">'
        f"{html.escape(content)}</code>",
        end_index + 1,
    )


def _try_render_marker(text: str, start_index: int) -> tuple[str, int] | None:
    for marker, tag in _INLINE_MARKERS:
        if not text.startswith(marker, start_index):
            continue
        end_index = text.find(marker, start_index + len(marker))
        if end_index == -1:
            if len(marker) == 1:
                return None
            return html.escape(marker), start_index + len(marker)
        inner = _render_inline(text[start_index + len(marker) : end_index])
        return f"<{tag}>{inner}</{tag}>", end_index + len(marker)
    return None


def _is_unordered_list_item(line: str) -> bool:
    stripped = line.lstrip()
    return len(stripped) >= 2 and stripped[0] in {"-", "*", "+"} and stripped[1] == " "


def _is_ordered_list_item(line: str) -> bool:
    return bool(_ORDERED_LIST_PATTERN.match(line.lstrip()))


def _is_quote_line(line: str) -> bool:
    return line.startswith(">") and (len(line) == 1 or line[1] in {" ", "\t"})


def _email_error(
    *,
    call_id: str,
    reason: str,
    to_email: str | None = None,
    subject: str | None = None,
) -> ToolExecutionResult:
    content_lines = [
        "Email send failed",
        "provider: smtp",
        f"reason: {reason}",
    ]
    metadata: dict[str, Any] = {
        "provider": "smtp",
        "error": reason,
    }
    if to_email is not None:
        content_lines.append(f"to_email: {to_email}")
        metadata["to_email"] = to_email
    if subject is not None:
        content_lines.append(f"subject: {subject}")
        metadata["subject"] = subject
    return ToolExecutionResult(
        call_id=call_id,
        name="email",
        ok=False,
        content="\n".join(content_lines),
        metadata=metadata,
    )
