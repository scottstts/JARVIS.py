"""Policy checks for the email tool."""

from __future__ import annotations

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolPolicyDecision
from .shared import EmailArgumentError, build_email_request

_APPROVAL_BODY_PREVIEW_CHARS = 1_500


class EmailPolicy:
    """Requires exact approval before sending an email."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def authorize(
        self,
        *,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        try:
            request = build_email_request(
                arguments=arguments,
                context=context,
                settings=self._settings,
            )
        except EmailArgumentError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        approved_action = context.approved_action or {}
        if (
            approved_action.get("kind") == "send_email"
            and approved_action.get("request_hash") == request.request_hash
        ):
            return ToolPolicyDecision(allowed=True)

        return ToolPolicyDecision(
            allowed=False,
            reason="email requires explicit approval.",
            approval_request={
                "kind": "send_email",
                "summary": _build_approval_summary(request.to_email, request.subject),
                "details": _build_approval_details(request),
                "request_hash": request.request_hash,
                "to_email": request.to_email,
            },
        )


def _build_approval_summary(to_email: str, subject: str) -> str:
    return f"Send email to {to_email} with subject '{subject}'."


def _build_approval_details(request) -> str:
    preview = request.body_markdown
    if len(preview) > _APPROVAL_BODY_PREVIEW_CHARS:
        preview = (
            f"{preview[:_APPROVAL_BODY_PREVIEW_CHARS].rstrip()}\n\n"
            "[body preview truncated]"
        )

    lines = [
        f"to_email: {request.to_email}",
        f"subject: {request.subject}",
        f"attachment_count: {len(request.attachment_specs)}",
    ]
    if request.attachment_specs:
        lines.append("attachments:")
        lines.extend(
            f"- {spec.raw_path} ({spec.size_bytes} bytes)"
            for spec in request.attachment_specs
        )
    lines.extend(
        [
            "body_markdown:",
            preview,
        ]
    )
    return "\n".join(lines)
