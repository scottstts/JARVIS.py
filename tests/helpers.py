"""Shared testing helpers."""

from __future__ import annotations

import os
import json
import re
import tempfile
from pathlib import Path

from jarvis.core.compaction import CompactionCallTrace, CompactionOutcome
from jarvis.core.compaction_contract import (
    CompactionSourceEvent,
    apply_compaction_draft,
    build_source_manifest,
    compile_compaction_replay,
)
from jarvis.core.config import CompactionSettings, ContextPolicySettings, CoreSettings
from jarvis.llm import LLMRequest, LLMResponse, LLMUsage, TextPart, ToolCall


def build_core_settings(
    *,
    root_dir: Path,
    context_window_tokens: int = 400_000,
    compaction_provider: str = "openai",
) -> CoreSettings:
    if os.getenv("JARVIS_TOOL_RUNTIME_BASE_URL"):
        shared_root_dir = Path(tempfile.mkdtemp(prefix="jarvis-test-", dir="/workspace"))
        root_dir = shared_root_dir

    identities_dir = root_dir / "identities"
    workspace_dir = root_dir / "workspace"
    identities_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (identities_dir / "PROGRAM.md").write_text("PROGRAM PROMPT", encoding="utf-8")
    (identities_dir / "REACTOR.md").write_text("REACTOR PROMPT", encoding="utf-8")
    (identities_dir / "USER.md").write_text("USER PROMPT", encoding="utf-8")
    (identities_dir / "ARMOR.md").write_text("ARMOR PROMPT", encoding="utf-8")

    return CoreSettings(
        context_policy=ContextPolicySettings(context_window_tokens=context_window_tokens),
        compaction=CompactionSettings(provider=compaction_provider),
        workspace_dir=workspace_dir,
        transcript_archive_dir=root_dir / "archive" / "transcripts",
        identities_dir=identities_dir,
    )


def is_compaction_generate_request(request: LLMRequest) -> bool:
    return _request_system_text(request).startswith(
        "You compact a Jarvis session into a complete semantic continuation record."
    )


def is_compaction_verify_request(request: LLMRequest) -> bool:
    _ = request
    return False


def build_compaction_test_response(
    request: LLMRequest,
    *,
    marker: str = "Compacted summary",
) -> LLMResponse:
    if not is_compaction_generate_request(request):
        raise AssertionError("Request is not a compaction request.")
    user_text = _request_user_text(request)
    event_refs = re.findall(r"\[EVENT (E\d+) \|", user_text)
    payload: dict[str, object] = {
        "objective": marker,
        "background": [],
        "episodes": (
            [
                {
                    "summary": marker,
                    "outcomes": ["Continue from the compacted state."],
                }
            ]
            if event_refs
            else []
        ),
        "constraints": [],
        "decisions": [],
        "artifacts": [],
        "open_loops": [],
        "uncertainties": [],
        "handover": {
            "current_focus": "Continue from the latest task state.",
            "next_actions": ["Resume the task."],
            "do_not_repeat": [],
            "verification_needed": [],
        },
    }
    raw_payload = json.dumps(payload, ensure_ascii=False)
    return LLMResponse(
        provider=request.provider or "openai",
        model="fake-compactor",
        text="",
        tool_calls=[
            ToolCall(
                call_id="call_compact",
                name="submit_compaction",
                arguments=payload,
                raw_arguments=raw_payload,
            )
        ],
        finish_reason="tool_calls",
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_compact",
    )


def build_compaction_test_outcome() -> CompactionOutcome:
    source_event = CompactionSourceEvent(
        event_id="source_1",
        record_id="source_1",
        session_id="session_1",
        created_at="2026-08-21T00:00:00+00:00",
        sequence=1,
        generation=1,
        event_type="system_event",
        role="system",
        content="Compact this context.",
    )
    manifest = build_source_manifest(
        generation=1,
        previous_bundle=None,
        source_events=(source_event,),
    )
    draft = {
        "objective": "Preserve the current task context.",
        "background": [],
        "episodes": [
            {
                "summary": "The prior task context was compacted.",
                "outcomes": [],
            }
        ],
        "constraints": [],
        "decisions": [],
        "artifacts": [],
        "open_loops": [],
        "uncertainties": [],
        "handover": {
            "current_focus": "Continue the current task.",
            "next_actions": ["Resume from the compacted context."],
            "do_not_repeat": [],
            "verification_needed": [],
        },
    }
    bundle = apply_compaction_draft(
        draft,
        bundle_id="bundle_test",
        created_at="2026-08-21T00:00:00+00:00",
        source_manifest=manifest,
        recent_records=(),
    )
    return CompactionOutcome(
        bundle=bundle,
        items=compile_compaction_replay(bundle),
        draft_payload=draft,
        verification_payload={
            "valid": True,
            "method": "jarvis_deterministic_bundle",
            "schema_version": 4,
        },
        call_traces=(
            CompactionCallTrace(
                phase="generate",
                provider="fake-provider",
                model="fake-model",
                response_id="resp_fake",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        ),
        semantic_status="accepted",
        semantic_source="model",
    )


def _request_system_text(request: LLMRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages[:1]
        for part in message.parts
        if isinstance(part, TextPart)
    )


def _request_user_text(request: LLMRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.parts
        if isinstance(part, TextPart)
    )
