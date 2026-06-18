from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("jarvis")
    group.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run tests that make real API calls to external AI providers.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-live-api"):
        return

    skip_live_api = pytest.mark.skip(
        reason="requires --run-live-api to make real AI provider API calls"
    )
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_live_api)
