"""Unit tests for session/transcript storage service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from storage import ConversationRecord, SessionStorage


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
