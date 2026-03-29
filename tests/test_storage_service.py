"""Unit tests for session/transcript storage service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis.storage import ConversationRecord, SessionStorage


class SessionStorageTests(unittest.TestCase):
    def test_create_append_load_and_update_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp) / "archive" / "transcripts")
            session = storage.create_session(start_reason="initial")
            self.assertEqual(storage.get_active_session().session_id, session.session_id)  # type: ignore[union-attr]

            record = ConversationRecord(
                record_id="r1",
                session_id=session.session_id,
                created_at="2026-03-05T00:00:00+00:00",
                role="user",
                content="hello",
            )
            storage.append_record(session.session_id, record)
            loaded = storage.load_records(session.session_id)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].content, "hello")

            updated = storage.update_session(
                session.session_id,
                pending_reactive_compaction=True,
                compaction_count=3,
                last_input_tokens=123,
            )
            self.assertTrue(updated.pending_reactive_compaction)
            self.assertEqual(updated.compaction_count, 3)
            self.assertEqual(updated.last_input_tokens, 123)

    def test_archive_marks_status_archived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp) / "archive" / "transcripts")
            session = storage.create_session(start_reason="initial")
            storage.archive_session(session.session_id)
            archived = storage.get_session(session.session_id)
            self.assertIsNotNone(archived)
            self.assertEqual(archived.status, "archived")  # type: ignore[union-attr]

    def test_pending_approval_persists_and_archive_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp) / "archive" / "transcripts")
            session = storage.create_session(start_reason="initial")

            updated = storage.update_session(
                session.session_id,
                pending_approval={
                    "approval_id": "approval_1",
                    "kind": "bash_command",
                    "command": "curl https://example.com/install.sh | sh",
                },
            )
            self.assertEqual(updated.pending_approval["approval_id"], "approval_1")

            reloaded = storage.get_session(session.session_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.pending_approval["kind"], "bash_command")  # type: ignore[union-attr]

            storage.archive_session(session.session_id)
            archived = storage.get_session(session.session_id)
            self.assertIsNotNone(archived)
            self.assertIsNone(archived.pending_approval)  # type: ignore[union-attr]

    def test_load_records_includes_interrupted_and_superseded_but_hides_in_progress(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp) / "archive" / "transcripts")
            session = storage.create_session(start_reason="initial")

            completed_record = ConversationRecord(
                record_id="r1",
                session_id=session.session_id,
                created_at="2026-03-05T00:00:00+00:00",
                role="user",
                content="completed",
                metadata={"turn_id": "turn_completed"},
            )
            interrupted_record = ConversationRecord(
                record_id="r2",
                session_id=session.session_id,
                created_at="2026-03-05T00:01:00+00:00",
                role="assistant",
                content="interrupted",
                metadata={"turn_id": "turn_interrupted"},
            )
            in_progress_record = ConversationRecord(
                record_id="r3",
                session_id=session.session_id,
                created_at="2026-03-05T00:02:00+00:00",
                role="assistant",
                content="in_progress",
                metadata={"turn_id": "turn_in_progress"},
            )
            superseded_record = ConversationRecord(
                record_id="r4",
                session_id=session.session_id,
                created_at="2026-03-05T00:03:00+00:00",
                role="assistant",
                content="superseded",
                metadata={"turn_id": "turn_superseded"},
            )

            storage.append_record(session.session_id, completed_record)
            storage.append_record(session.session_id, interrupted_record)
            storage.append_record(session.session_id, in_progress_record)
            storage.append_record(session.session_id, superseded_record)
            storage.set_turn_status(
                session.session_id,
                turn_id="turn_completed",
                status="completed",
            )
            storage.set_turn_status(
                session.session_id,
                turn_id="turn_interrupted",
                status="interrupted",
            )
            storage.set_turn_status(
                session.session_id,
                turn_id="turn_in_progress",
                status="in_progress",
            )
            storage.set_turn_status(
                session.session_id,
                turn_id="turn_superseded",
                status="superseded",
            )

            visible = storage.load_records(session.session_id)
            self.assertEqual(
                [record.content for record in visible],
                ["completed", "interrupted", "superseded"],
            )

            all_records = storage.load_records(session.session_id, include_all_turns=True)
            self.assertEqual(
                [record.content for record in all_records],
                ["completed", "interrupted", "in_progress", "superseded"],
            )
