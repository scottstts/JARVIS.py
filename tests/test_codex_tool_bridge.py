"""Unit tests for Codex dynamic tool bridge behavior."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from jarvis.codex_backend.tool_bridge import CodexToolBridge
from jarvis.llm import ToolDefinition
from jarvis.tools import ToolExecutionResult


class CodexToolBridgeTests(unittest.TestCase):
    def test_build_dynamic_tools_uses_existing_tool_definitions(self) -> None:
        bridge = CodexToolBridge(
            tool_definitions_provider=lambda _activated: (
                ToolDefinition(
                    name="bash",
                    description="Run shell commands.",
                    input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
                ),
            )
        )

        specs = bridge.build_dynamic_tools(activated_discoverable_tool_names=())

        self.assertEqual(
            specs,
            [
                {
                    "name": "bash",
                    "description": "Run shell commands.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                    },
                }
            ],
        )

    def test_build_tool_response_includes_text_and_image_data_url(self) -> None:
        bridge = CodexToolBridge(tool_definitions_provider=lambda _activated: ())

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "chart.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
            response = bridge.build_tool_response(
                ToolExecutionResult(
                    call_id="call_1",
                    name="view_image",
                    ok=True,
                    content="Image ready.",
                    metadata={
                        "image_attachment": {
                            "path": str(image_path),
                            "media_type": "image/png",
                        }
                    },
                )
            )

        self.assertTrue(response["success"])
        self.assertEqual(response["contentItems"][0], {"type": "inputText", "text": "Image ready."})
        image_url = response["contentItems"][1]["imageUrl"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        encoded = image_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), b"\x89PNG\r\n\x1a\nfake-png-bytes")

