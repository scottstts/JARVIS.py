"""Tests for passive subagent Acceptance Notes."""

from __future__ import annotations

import unittest

from jarvis.subagent.acceptance_notes import AcceptanceNotes


class AcceptanceNotesTests(unittest.TestCase):
    def test_notes_preserve_assignment_constraints_and_deliverable_without_gating(self) -> None:
        text = AcceptanceNotes(
            instructions="Implement the parser and keep the existing API.",
            user_constraints="Do not change generated files.",
            deliverable="Return the implementation plus verification limitations.",
        ).render()

        self.assertIn("Implement the parser", text)
        self.assertIn("Do not change generated files", text)
        self.assertIn("Return the implementation", text)
        self.assertIn("informational only", text)
        self.assertIn("never block handoff", text)
        self.assertIn("partial, or blocked", text)


if __name__ == "__main__":
    unittest.main()
