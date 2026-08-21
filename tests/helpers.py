"""Shared testing helpers."""

from __future__ import annotations

import os
import json
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
from jarvis.llm import LLMRequest, LLMResponse, LLMUsage, TextPart


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
        "You produce a complete canonical compaction draft for Jarvis."
    )


def is_compaction_verify_request(request: LLMRequest) -> bool:
    return _request_system_text(request).startswith(
        "You verify a candidate Jarvis compaction bundle"
    )


def build_compaction_test_response(
    request: LLMRequest,
    *,
    marker: str = "Compacted summary",
) -> LLMResponse:
    if is_compaction_verify_request(request):
        payload: dict[str, object] = {"valid": True, "issues": []}
    elif is_compaction_generate_request(request):
        request_payload = _compaction_request_payload(request)
        delta_events = request_payload.get("delta_events", [])
        if not isinstance(delta_events, list):
            raise AssertionError("Compaction request delta_events must be a list.")
        event_ids = [
            str(event["event_id"])
            for event in delta_events
            if isinstance(event, dict) and event.get("event_id")
        ]
        previous_bundle = request_payload.get("previous_bundle")
        if event_ids:
            objective = {
                "summary": marker,
                "evidence_event_ids": [event_ids[0]],
            }
            episode_actions = [
                {
                    "action": "add",
                    "episode_id": f"episode_{event_ids[0]}",
                    "summary": marker,
                    "source_ids": event_ids,
                    "outcomes": ["Continue from the compacted state."],
                }
            ]
            handover = {
                "current_focus": "Continue from the latest task state.",
                "next_actions": ["Resume the task."],
                "do_not_repeat": [],
                "verification_needed": [],
                "evidence_event_ids": [event_ids[-1]],
            }
            coverage = [
                {
                    "source_event_ids": event_ids,
                    "disposition": "episode",
                    "target_ids": [f"episode_{event_ids[0]}"],
                    "reason": "Represented by the test episode.",
                }
            ]
        elif isinstance(previous_bundle, dict):
            objective = previous_bundle["objective"]
            episode_actions = []
            handover = previous_bundle["handover"]
            coverage = []
        else:
            raise AssertionError("Compaction test request has no evidence.")
        payload = {
            "objective": objective,
            "preserved_actions": [],
            "episode_actions": episode_actions,
            "state_operations": [],
            "handover": handover,
            "coverage": coverage,
        }
    else:
        raise AssertionError("Request is not a compaction generator or verifier request.")
    return LLMResponse(
        provider=request.provider or "openai",
        model="fake-compactor",
        text=json.dumps(payload, ensure_ascii=False),
        tool_calls=[],
        finish_reason="stop",
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
        "objective": {
            "summary": "Preserve the current task context.",
            "evidence_event_ids": ["source_1"],
        },
        "preserved_actions": [],
        "episode_actions": [
            {
                "action": "add",
                "episode_id": "episode_source_1",
                "summary": "The prior task context was compacted.",
                "source_ids": ["source_1"],
                "outcomes": [],
            }
        ],
        "state_operations": [],
        "handover": {
            "current_focus": "Continue the current task.",
            "next_actions": ["Resume from the compacted context."],
            "do_not_repeat": [],
            "verification_needed": [],
            "evidence_event_ids": ["source_1"],
        },
        "coverage": [
            {
                "source_event_ids": ["source_1"],
                "disposition": "episode",
                "target_ids": ["episode_source_1"],
                "reason": "Represented in the compacted episode.",
            }
        ],
    }
    bundle = apply_compaction_draft(
        draft,
        bundle_id="bundle_test",
        created_at="2026-08-21T00:00:00+00:00",
        source_manifest=manifest,
        source_events=(source_event,),
        previous_bundle=None,
    )
    return CompactionOutcome(
        bundle=bundle,
        items=compile_compaction_replay(bundle),
        draft_payload=draft,
        verification_payload={"valid": True, "issues": []},
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
        repair_count=0,
    )


def _request_system_text(request: LLMRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages[:1]
        for part in message.parts
        if isinstance(part, TextPart)
    )


def _compaction_request_payload(request: LLMRequest) -> dict[str, object]:
    user_text = "\n".join(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.parts
        if isinstance(part, TextPart)
    )
    json_start = user_text.find("{")
    if json_start < 0:
        raise AssertionError("Compaction request is missing its JSON input.")
    payload = json.loads(user_text[json_start:])
    if not isinstance(payload, dict):
        raise AssertionError("Compaction request input must be an object.")
    return payload
