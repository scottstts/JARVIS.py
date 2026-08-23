"""High-fidelity test-only command-line runner for real Jarvis gateway sessions."""

from __future__ import annotations

import argparse
import atexit
import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import sys
from types import FrameType
from typing import Any
from uuid import uuid4

import uvicorn

from jarvis.core import CoreSettings
from jarvis.gateway import GatewaySettings, create_app
from jarvis.logging_setup import configure_application_logging
from jarvis.main import _wait_for_gateway_start
from jarvis.runtime_env import load_docker_secrets_if_present
from jarvis.runtime_provider_configuration import (
    RuntimeProviderConfiguration,
    load_runtime_provider_configuration,
)
from jarvis.settings import SETTINGS_SOURCE_PATH
from jarvis.ui.telegram.gateway_client import (
    GatewayApprovalRequestEvent,
    GatewayErrorEvent,
    GatewayRouteEvent,
    GatewayRouteSession,
    GatewayTaskStatusEvent,
    GatewayTurnDoneEvent,
    GatewayTurnStartedEvent,
)


_DEFAULT_FAILURE_PATTERNS = (
    "task_tool_round_budget_exhausted",
    "tool_progress_budget_exhausted",
    "unsupported mode 'service'",
    "terminal response without visible text or usable tool calls",
    "ProviderEmptyResponseError",
)
_TEST_WORKSPACE_ROOT = Path("/workspace")
_RUNS_DIR = _TEST_WORKSPACE_ROOT / ".jarvis_internal" / "dev_headless_runs"


class HeadlessRunFailure(RuntimeError):
    """Raised when the real headless run surfaces a visible system failure."""


@dataclass(slots=True, frozen=True)
class HeadlessRunConfig:
    prompts: tuple[str, ...]
    route_id: str
    max_runtime_seconds: float
    soak_seconds: float
    auto_approve: bool
    expected_main_provider: str
    expected_subagent_provider: str
    expected_model: str
    expected_settings_file: Path
    event_log: Path | None
    keep_workspace: bool


class EventAuditor:
    """Detect visible failure signatures and unchanged-notice retry storms."""

    def __init__(self, failure_patterns: Sequence[str] = _DEFAULT_FAILURE_PATTERNS) -> None:
        self._failure_patterns = tuple(pattern.casefold() for pattern in failure_patterns)
        self._notice_counts: dict[tuple[str, str, str], int] = {}

    def observe(self, event: GatewayRouteEvent) -> None:
        payload = asdict(event)
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        lowered = rendered.casefold()
        matched = [pattern for pattern in self._failure_patterns if pattern in lowered]
        if matched:
            raise HeadlessRunFailure(
                "Visible failure pattern detected: " + ", ".join(sorted(matched))
            )
        if event.type not in {"system_notice", "local_notice"}:
            return
        notice_kind = str(payload.get("notice_kind", ""))
        text = str(payload.get("text", ""))
        signature = (event.type, notice_kind, text)
        count = self._notice_counts.get(signature, 0) + 1
        self._notice_counts[signature] = count
        if count >= 3:
            raise HeadlessRunFailure(
                "Unchanged orchestrator notice was emitted three times: "
                f"{notice_kind or event.type}"
            )


class JsonlEventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, event: GatewayRouteEvent | dict[str, Any]) -> None:
        payload = asdict(event) if not isinstance(event, dict) else event
        self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def validate_provider_configuration(
    configuration: RuntimeProviderConfiguration,
    *,
    expected_main_provider: str,
    expected_subagent_provider: str,
    expected_model: str,
) -> None:
    targets = {target.role: target for target in configuration}
    expected = {
        "Main Agent": expected_main_provider,
        "Subagent": expected_subagent_provider,
    }
    mismatches: list[str] = []
    for role, provider in expected.items():
        target = targets.get(role)
        if target is None:
            mismatches.append(f"{role}: missing")
            continue
        if target.provider != provider or target.model != expected_model:
            mismatches.append(
                f"{role}: expected {provider}/{expected_model}, "
                f"got {target.provider}/{target.model}"
            )
    if mismatches:
        raise HeadlessRunFailure("Provider configuration mismatch: " + "; ".join(mismatches))


def build_sandboxed_prompt(prompt: str, *, workspace_dir: Path) -> str:
    return (
        "Dev-shim safety boundary (mandatory): perform all file creation, edits, builds, "
        f"services, and test artifacts only inside {workspace_dir}. Do not modify files "
        "elsewhere in /workspace. Do not send messages, email, publish, deploy, purchase, or "
        "change any external service. You may use real Jarvis tools and subagents extensively "
        "inside that test directory.\n\n"
        + prompt.strip()
    )


async def run_headless(config: HeadlessRunConfig) -> Path:
    load_docker_secrets_if_present()
    resolved_settings_path = SETTINGS_SOURCE_PATH.resolve(strict=False)
    expected_settings_path = config.expected_settings_file.resolve(strict=False)
    if resolved_settings_path != expected_settings_path:
        raise HeadlessRunFailure(
            f"Expected settings file {expected_settings_path}, got {resolved_settings_path}."
        )

    core_settings = CoreSettings.from_env()
    provider_configuration = load_runtime_provider_configuration(core_settings=core_settings)
    validate_provider_configuration(
        provider_configuration,
        expected_main_provider=config.expected_main_provider,
        expected_subagent_provider=config.expected_subagent_provider,
        expected_model=config.expected_model,
    )
    _print_provider_configuration(provider_configuration, settings_path=resolved_settings_path)

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    test_workspace = _TEST_WORKSPACE_ROOT / f"jarvis-test-headless-{run_stamp}-{uuid4().hex[:8]}"
    test_workspace.mkdir(parents=True, exist_ok=False)
    cleanup_test_workspace = partial(shutil.rmtree, test_workspace, ignore_errors=True)
    if not config.keep_workspace:
        atexit.register(cleanup_test_workspace)
    event_log_path = config.event_log or (
        _RUNS_DIR / f"{run_stamp}-{config.route_id}-{uuid4().hex[:8]}.jsonl"
    )
    event_log = JsonlEventLog(event_log_path)
    event_log.write(
        {
            "type": "headless_run_started",
            "created_at": datetime.now(UTC).isoformat(),
            "route_id": config.route_id,
            "settings_file": str(resolved_settings_path),
            "workspace_dir": str(test_workspace),
            "providers": [asdict(target) for target in provider_configuration],
        }
    )

    gateway_settings = GatewaySettings(
        host="127.0.0.1",
        port=_reserve_loopback_port(),
        websocket_path="/ws",
    )
    app = create_app(gateway_settings=gateway_settings, core_settings=core_settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host=gateway_settings.host,
            port=gateway_settings.port,
            lifespan="on",
            access_log=False,
            log_level="warning",
        )
    )
    server_task = asyncio.create_task(server.serve(), name="jarvis-dev-headless-gateway")
    route_session: GatewayRouteSession | None = None
    auditor = EventAuditor()
    try:
        await _wait_for_gateway_start(server, server_task, startup_timeout_seconds=30.0)
        route_session = GatewayRouteSession(
            route_id=config.route_id,
            websocket_base_url=f"ws://127.0.0.1:{gateway_settings.port}/ws",
            connect_timeout_seconds=30.0,
        )
        await route_session.connect()
        events = route_session.events().__aiter__()
        for prompt_index, prompt in enumerate(config.prompts, start=1):
            print(f"[headless] prompt {prompt_index}/{len(config.prompts)}", flush=True)
            await _run_prompt(
                route_session,
                events=events,
                prompt=build_sandboxed_prompt(prompt, workspace_dir=test_workspace),
                max_runtime_seconds=config.max_runtime_seconds,
                auto_approve=config.auto_approve,
                auditor=auditor,
                event_log=event_log,
            )
        if config.soak_seconds > 0:
            print(f"[headless] soaking for {config.soak_seconds:.0f}s", flush=True)
            await _soak(
                route_session,
                events=events,
                duration_seconds=config.soak_seconds,
                auto_approve=config.auto_approve,
                auditor=auditor,
                event_log=event_log,
            )
        event_log.write(
            {
                "type": "headless_run_completed",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        return event_log_path
    finally:
        if route_session is not None:
            with suppress(Exception):
                await _reset_route(route_session, event_log=event_log)
            with suppress(Exception):
                await route_session.request_stop()
            with suppress(Exception):
                await route_session.aclose()
        if not server_task.done():
            server.should_exit = True
        with suppress(asyncio.CancelledError, Exception):
            await server_task
        with suppress(Exception):
            event_log.close()
        if not config.keep_workspace:
            cleanup_test_workspace()
            atexit.unregister(cleanup_test_workspace)


async def _run_prompt(
    route_session: GatewayRouteSession,
    *,
    events: AsyncIterator[GatewayRouteEvent],
    prompt: str,
    max_runtime_seconds: float,
    auto_approve: bool,
    auditor: EventAuditor,
    event_log: JsonlEventLog,
) -> None:
    client_message_id = uuid4().hex
    await route_session.send_user_message(text=prompt, client_message_id=client_message_id)
    matched_turn_id: str | None = None
    turn_done = False
    route_active: bool | None = None
    deadline = asyncio.get_running_loop().time() + max_runtime_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise HeadlessRunFailure(
                f"Prompt did not become idle within {max_runtime_seconds:.0f} seconds."
            )
        event = await asyncio.wait_for(anext(events), timeout=remaining)
        event_log.write(event)
        auditor.observe(event)
        _print_event(event)
        if isinstance(event, GatewayErrorEvent):
            raise HeadlessRunFailure(f"Gateway error {event.code}: {event.message}")
        if isinstance(event, GatewayApprovalRequestEvent):
            if not auto_approve:
                raise HeadlessRunFailure(
                    f"Approval required ({event.kind}): rerun with --auto-approve if intended."
                )
            resolved = await route_session.submit_approval(
                approval_id=event.approval_id,
                approved=True,
            )
            if not resolved:
                raise HeadlessRunFailure(f"Approval {event.approval_id} was not resolved.")
        if (
            isinstance(event, GatewayTurnStartedEvent)
            and event.client_message_id == client_message_id
        ):
            matched_turn_id = event.turn_id
        if (
            isinstance(event, GatewayTurnDoneEvent)
            and matched_turn_id is not None
            and event.turn_id == matched_turn_id
        ):
            turn_done = True
            if event.completion_blocked:
                raise HeadlessRunFailure("The main turn ended with completion_blocked=true.")
        if isinstance(event, GatewayTaskStatusEvent):
            route_active = event.active
        if turn_done and route_active is False:
            return


async def _soak(
    route_session: GatewayRouteSession,
    *,
    events: AsyncIterator[GatewayRouteEvent],
    duration_seconds: float,
    auto_approve: bool,
    auditor: EventAuditor,
    event_log: JsonlEventLog,
) -> None:
    deadline = asyncio.get_running_loop().time() + duration_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            event = await asyncio.wait_for(anext(events), timeout=remaining)
        except TimeoutError:
            return
        event_log.write(event)
        auditor.observe(event)
        _print_event(event)
        if isinstance(event, GatewayErrorEvent):
            raise HeadlessRunFailure(f"Gateway error {event.code}: {event.message}")
        if isinstance(event, GatewayApprovalRequestEvent):
            if not auto_approve:
                raise HeadlessRunFailure("Approval was requested during soak.")
            await route_session.submit_approval(approval_id=event.approval_id, approved=True)


async def _reset_route(
    route_session: GatewayRouteSession,
    *,
    event_log: JsonlEventLog,
) -> None:
    """Use Jarvis's hard session boundary to cancel owned jobs and subagents."""

    client_message_id = uuid4().hex
    await route_session.send_user_message(text="/new", client_message_id=client_message_id)
    events = route_session.events().__aiter__()
    matched_turn_id: str | None = None
    turn_done = False
    inactive = False
    deadline = asyncio.get_running_loop().time() + 30.0
    while not (turn_done and inactive):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        event = await asyncio.wait_for(anext(events), timeout=remaining)
        event_log.write(event)
        if (
            isinstance(event, GatewayTurnStartedEvent)
            and event.client_message_id == client_message_id
        ):
            matched_turn_id = event.turn_id
        elif (
            isinstance(event, GatewayTurnDoneEvent)
            and (
                (
                    matched_turn_id is not None
                    and event.turn_id == matched_turn_id
                )
                or (
                    matched_turn_id is None
                    and event.client_message_id == client_message_id
                )
            )
        ):
            turn_done = True
        elif isinstance(event, GatewayTaskStatusEvent):
            inactive = not event.active


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _print_provider_configuration(
    configuration: RuntimeProviderConfiguration,
    *,
    settings_path: Path,
) -> None:
    print(f"[headless] settings={settings_path}")
    for target in configuration:
        print(f"[headless] {target.role}: {target.provider}/{target.model}")


def _print_event(event: GatewayRouteEvent) -> None:
    if event.type in {"assistant_delta"}:
        return
    payload = asdict(event)
    summary = (
        payload.get("text")
        or payload.get("response_text")
        or payload.get("message")
        or payload.get("notice_kind")
        or payload.get("reason")
        or ""
    )
    normalized = re.sub(r"\s+", " ", str(summary)).strip()
    if len(normalized) > 300:
        normalized = normalized[:297] + "..."
    print(f"[headless] {event.type}: {normalized}", flush=True)


def _parse_args(argv: Sequence[str] | None = None) -> HeadlessRunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path, action="append", default=[])
    parser.add_argument("--route-id", default=f"dev_headless_{uuid4().hex[:12]}")
    parser.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    parser.add_argument("--soak-seconds", type=float, default=0.0)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--expected-main-provider", default="openrouter")
    parser.add_argument("--expected-subagent-provider", default="openrouter")
    parser.add_argument("--expected-model", default="stealth/ox-alpha")
    parser.add_argument(
        "--expected-settings-file",
        type=Path,
        default=Path("/workspace/settings/settings.yml"),
    )
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args(argv)
    prompts = [str(prompt) for prompt in args.prompt]
    prompts.extend(path.read_text(encoding="utf-8") for path in args.prompt_file)
    if not prompts:
        parser.error("at least one --prompt or --prompt-file is required")
    if args.max_runtime_seconds <= 0 or args.soak_seconds < 0:
        parser.error("runtime values must be positive (soak may be zero)")
    return HeadlessRunConfig(
        prompts=tuple(prompts),
        route_id=str(args.route_id),
        max_runtime_seconds=float(args.max_runtime_seconds),
        soak_seconds=float(args.soak_seconds),
        auto_approve=bool(args.auto_approve),
        expected_main_provider=str(args.expected_main_provider),
        expected_subagent_provider=str(args.expected_subagent_provider),
        expected_model=str(args.expected_model),
        expected_settings_file=args.expected_settings_file,
        event_log=args.event_log,
        keep_workspace=bool(args.keep_workspace),
    )


def main() -> None:
    configure_application_logging()
    previous_signal_handlers = _install_shutdown_signal_handlers()
    try:
        try:
            path = asyncio.run(run_headless(_parse_args()))
        except (HeadlessRunFailure, TimeoutError) as exc:
            print(f"[headless] FAILED: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        except KeyboardInterrupt:
            print("[headless] interrupted", file=sys.stderr)
            raise SystemExit(130) from None
    finally:
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
    print(f"[headless] PASS event_log={path}")


def _install_shutdown_signal_handlers() -> dict[int, Any]:
    previous_handlers: dict[int, Any] = {}
    for name in ("SIGHUP", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        previous_handlers[signum] = signal.signal(signum, _raise_shutdown_interrupt)
    return previous_handlers


def _raise_shutdown_interrupt(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
