"""Unit tests for user command parsing."""

from __future__ import annotations

import unittest

from jarvis.core.commands import parse_user_command


class ParseUserCommandTests(unittest.TestCase):
    def test_plain_message_is_not_treated_as_command(self) -> None:
        parsed = parse_user_command("hello there")
        self.assertEqual(parsed.kind, "message")
        self.assertEqual(parsed.body, "hello there")

    def test_new_command_parses_optional_body(self) -> None:
        parsed = parse_user_command("/new continue this task")
        self.assertEqual(parsed.kind, "new")
        self.assertEqual(parsed.body, "continue this task")

    def test_compact_command_parses_empty_and_non_empty_body(self) -> None:
        empty = parse_user_command("/compact")
        with_text = parse_user_command("/compact keep deployment details")

        self.assertEqual(empty.kind, "compact")
        self.assertEqual(empty.body, "")
        self.assertEqual(with_text.kind, "compact")
        self.assertEqual(with_text.body, "keep deployment details")

    def test_unknown_slash_command_falls_back_to_message(self) -> None:
        parsed = parse_user_command("/unknown do not special-case this")
        self.assertEqual(parsed.kind, "message")
        self.assertEqual(parsed.body, "/unknown do not special-case this")
