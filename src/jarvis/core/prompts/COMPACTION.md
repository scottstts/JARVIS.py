You are generating REPLACEMENT HISTORY for an AI agent session compaction.

Goal:
- maximize seamless handover into a fresh session
- preserve critical intent, corrections, decisions, flow, direction, and end-state
- compress the prior session into a shorter replacement history that still reads like condensed conversation history plus handover

You will receive transcript items as JSONL.
Each line is one whole transcript item.
Treat the item list as ordered chronology.

Hard rules:
- Output valid JSON only. No markdown fences. No prose before or after the JSON.
- Output one top-level object with exactly one key: `items`.
- `items` must be an ordered array of replacement history items.
- Every replacement history item must include:
  - `type` and it must equal `"compaction"`
  - `role` and it must be one of `"system"`, `"user"`, `"assistant"`
  - `kind` and it must be one of `"session_frame"`, `"preserved_message"`, `"condensed_span"`, `"handover_state"`
  - `content` as a non-empty string
- The first item must be:
  - `type="compaction"`
  - `role="system"`
  - `kind="session_frame"`
- The last item must be:
  - `type="compaction"`
  - `role="system"`
  - `kind="handover_state"`
- `preserved_message` items must preserve exact wording from the source and must include `verbatim: true`.
- Do not emit `tool` role items.
- Do not restate bootstrap identity, tool definitions, or harness policy.
- Do not invent facts.

What the replacement history should contain:
- one compact `session_frame` item that quickly frames the mission, durable constraints, corrections already in force, and the main thread of work
- exact critical preserved user or assistant messages when the exact wording matters for continuation
- one or more `condensed_span` assistant items that compress the general flow of the session:
  - what happened
  - what changed
  - what failed
  - what succeeded
  - the direction, pace, and nuance of the work
- one final `handover_state` item that states the true end-state, open threads, and the best next move

Writing rules:
- Preserve exact identifiers, file paths, commands, IDs, URLs, error strings, and other small literals when they matter.
- Preserve exact user corrections, redirects, acceptance criteria, and constraints when they matter.
- Preserve exact assistant commitments or explicit results when they matter.
- Use `condensed_span` items to collapse long execution-heavy stretches into compact assistant narration.
- The output should feel like compacted prior conversation history, not like a taxonomy-heavy checkpoint report.
- Keep it compact. Collapse repetitive execution chatter aggressively, but preserve the actual shape of the session.
- Source provenance is optional. If useful, include `source_record_ids` and/or `source_range`.

Preferred shape:
1. `session_frame`
2. critical preserved user or assistant messages when needed
3. condensed flow spans
4. more preserved messages only when they materially affect continuation
5. final `handover_state`

Return JSON only.
