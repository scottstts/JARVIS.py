# Compaction Design

## Status

Implemented on April 11, 2026.

This document describes the current Jarvis session compaction design.
Historical references to the old `summary_seed` path are included only to explain the redesign.

## Purpose

Jarvis previously compacted by:

- serializing prior transcript records into one prompt blob
- asking the compaction model for one monolithic summary string
- starting a fresh session with that summary injected as one `summary_seed` system message

That is too lossy and too flat.

The current design uses a structured transcript replacement flow:

- light item-level pruning before compaction
- compaction model produces a compact replacement history, not one summary blob
- fresh session is rebuilt from that replacement history
- light item-level pruning and validation after compaction

The goal is seamless handover into a fresh session while keeping the replacement history:

- compact
- replayable
- easy for the next agent to continue from
- closer to a condensed conversation history than a prose summary

## Goals

- Separate compaction provider choice from the main-agent and subagent chat provider choice.
- Use the selected compaction provider's existing model setting from `src/jarvis/settings.yml`.
- Keep pre-compaction pruning intentionally light and item-level only.
- Keep post-compaction pruning intentionally light and item-level only.
- Replace `summary_seed` with structured compacted transcript items.
- Preserve critical original user and assistant messages exactly when they matter.
- Preserve the general flow, direction, pace, corrections, and end-state of the prior session.
- Keep fresh bootstrap context authoritative rather than inheriting stale instruction layers.
- Use the same compaction design for normal LLM actors and Codex-backed actors.

## Non-Goals

- Do not add semantic pruning heuristics that inspect and rewrite content inside an item.
- Do not truncate, summarize, or rewrite the contents of kept transcript items during pre-compaction pruning.
- Do not treat replacement history as a rigid JSON checkpoint taxonomy with separate mission/state/todo fields only.
- Do not let compaction rewrite bootstrap identity, tool bootstrap, sandbox policy, or other harness-managed context.
- Do not introduce provider-specific compaction behavior into the core loop beyond provider selection and normal provider dispatch.

## Approved Decisions

### 1. Separate compaction provider

Add one explicit setting:

- `core.compaction.provider`

This setting chooses the provider used only for compaction.

The compaction flow then resolves the model from that provider's existing settings:

- `openai` -> `providers.openai.chat_model`
- `anthropic` -> `providers.anthropic.chat_model`
- `gemini` -> `providers.gemini.chat_model`
- `grok` -> `providers.grok.chat_model`
- `openrouter` -> `providers.openrouter.chat_model`
- `lmstudio` -> provider-selected runtime model as today

Do not add a separate compaction-model setting.

`codex` should not be a valid `core.compaction.provider` value because the compaction request itself runs through the normal `LLMService` path.
Codex-backed actors still use the compaction provider selected here, then rebuild onto a fresh Codex thread after compaction completes.

### 2. Light item-level pruning only

Pre-compaction and post-compaction pruning are both intentionally narrow.

Their contract is:

- decide whether to keep or drop a whole item
- optionally attach metadata or drop reasons outside the model-visible content path
- never rewrite an item's `content`
- never summarize within an item
- never trim head/tail slices out of a kept item

If the pruned compaction source is still too large for the chosen compaction provider, compaction fails closed.
Jarvis should not respond by introducing hidden within-item slicing heuristics.

### 3. Replacement history, not summary seed

The compaction model should return a structured replacement history.

That history is a mix of:

- one compact summary frame
- exact critical preserved user or assistant messages
- condensed assistant-written spans that explain the flow of the prior session
- one compact final handover state

It should read like a condensed conversation history plus handover, not like a prose status report and not like a single summary string in JSON clothing.

### 4. Fresh bootstrap remains authoritative

Compaction output is allowed to replace prior conversational state only.

It does not replace:

- identity bootstrap
- tool bootstrap
- memory bootstrap
- runtime-managed system context
- sandbox or approval policy
- Codex developer instructions or dynamic tools

Fresh session bootstrap is always rebuilt from current runtime state.

## Replacement Item Design

### Top-level response shape

The compaction model must return structured JSON with this top-level shape:

```json
{
  "items": [
    {
      "type": "compaction",
      "role": "system",
      "kind": "session_frame",
      "content": "..."
    }
  ]
}
```

The model-visible design target is an ordered list of compacted transcript items.

### Replacement item schema

Each item has this base shape:

```json
{
  "type": "compaction",
  "role": "system | user | assistant",
  "kind": "session_frame | preserved_message | condensed_span | handover_state",
  "content": "..."
}
```

Optional fields:

```json
{
  "verbatim": true,
  "source_record_ids": ["record_a", "record_b"],
  "source_range": {
    "start": 12,
    "end": 28
  }
}
```

### Meaning of each kind

- `session_frame`
  - role must be `system`
  - should be the first item
  - gives the compact frame for the next session: current mission, durable constraints, important corrections already in force, and the main thread of work

- `preserved_message`
  - role must match the original message role
  - should preserve the exact message text when it materially affects continuation
  - use it for important user instructions, corrections, constraints, redirects, acceptance criteria, or assistant commitments worth keeping verbatim
  - `verbatim` should be `true`

- `condensed_span`
  - role should normally be `assistant`
  - condenses a contiguous span of prior transcript into one compact conversation beat
  - this is where the general flow lives: what happened, what changed, what failed, what succeeded, and how the session moved
  - it should preserve the direction, flow, pace, and nuance of the session without replaying every message

- `handover_state`
  - role must be `system`
  - should be the last item
  - gives the exact end-state needed for continuation: what is true now, what remains open, and what the next agent should do first

### Required ordering invariants

Compaction output must obey these invariants:

- first item is `type=compaction`, `role=system`, `kind=session_frame`
- last item is `type=compaction`, `role=system`, `kind=handover_state`
- every item has non-empty `content`
- every item has `type=compaction`
- only `system`, `user`, and `assistant` roles are allowed
- `tool` role items are not allowed in replacement history
- `preserved_message` items must have `verbatim=true`
- order must follow the original session chronology

### Content-writing rules

The compaction prompt should instruct the model to follow these writing rules:

- Preserve exact user wording when that wording changes what Jarvis must do.
- Preserve exact assistant wording when it is a meaningful commitment, decision, or explicit result that the next session should inherit.
- Use `condensed_span` items to compress long execution-heavy spans into compact assistant narration.
- Preserve failures when they change future work.
- Preserve exact identifiers, commands, file paths, error strings, IDs, URLs, and other small literals that matter.
- Keep the whole replacement history compact enough to fit comfortably in a fresh session bootstrap.
- Do not restate bootstrap identity or tool descriptions.
- Do not emit raw tool logs or raw fetched page bodies unless the exact body text is itself the important thing.

### Design intent

Replacement history is intentionally a hybrid.

It should feel like:

- a compacted transcript
- a handover report
- a structured session recap

all at once.

It should not feel like:

- one prose summary split into labeled JSON fields
- a raw transcript replay
- a taxonomy-heavy checkpoint document

### Example

```json
{
  "items": [
    {
      "type": "compaction",
      "role": "system",
      "kind": "session_frame",
      "content": "The session focused on building Scott's personal React site. Durable constraints: keep the work bold and design-forward, preserve exact user corrections, and verify with lint and build before claiming completion."
    },
    {
      "type": "compaction",
      "role": "user",
      "kind": "preserved_message",
      "content": "Hi Jarvis. I want you to create a new react project and create a beautiful and apt personal website for me, I want to be completely dazzled by your sophisticated design and impeccable execution.",
      "verbatim": true
    },
    {
      "type": "compaction",
      "role": "assistant",
      "kind": "condensed_span",
      "content": "I inspected the existing site and memory context, scaffolded a Vite project, then moved through install, styling, component creation, and repeated repair loops. Several short planning utterances were emitted between tool actions, but the important flow was scaffold -> install -> build -> fix errors -> rerun verification."
    },
    {
      "type": "compaction",
      "role": "assistant",
      "kind": "preserved_message",
      "content": "Build is clean - zero warnings, zero errors. Now let me run the linter.",
      "verbatim": true
    },
    {
      "type": "compaction",
      "role": "system",
      "kind": "handover_state",
      "content": "End state: the site was largely built and verification was active. Resume by reopening the generated project, checking the latest lint and build state, and continuing from the last verified point without losing the bold design intent."
    }
  ]
}
```

## Pre-Compaction Pruning

### Contract

Pre-compaction pruning runs before the compaction request is built.

Its job is only to remove obviously unnecessary source items from the transcript.

It does not:

- summarize
- truncate
- rewrite
- inspect sub-item structure for content surgery

### Drop rules

Drop a source record entirely if any of the following are true:

- `record.kind == "compaction"`
- `record.metadata["bootstrap_identity"] == true`
- `record.metadata["transcript_only"] == true`
- record is a memory bootstrap record
- `record.metadata["summary_seed"] == true` from legacy sessions
- record is the auto-appended current-time turn-context record
- record is the subagent status snapshot system record
- `record.metadata["tool_call_validation_failed"] == true`
- record is an assistant message with empty `content`

Replayable replacement-history message records are not dropped here.
Only the non-replayable `kind="compaction"` audit record is pruneable by default.
Later compactions still need the prior compacted history available as real session context.

### Keep rules

Keep source records by default, especially:

- all user messages
- assistant messages with meaningful text
- successful tool results
- meaningful tool failures
- approval request and approval result records
- interruption and supersede records
- system notes that materially affect continuation and are not in the drop list

### No within-item pruning

If a kept tool record contains a large bash log, long fetched markdown, or a heredoc command, the whole item stays intact for compaction-source purposes.
The compaction model is responsible for collapsing that span into a replacement-history item later.

## Post-Compaction Pruning And Validation

### Contract

Post-compaction pruning runs only on the model's returned replacement items.

It is also intentionally light and item-level only.

It may:

- validate structure
- keep or drop whole returned items

It must not:

- rewrite `content`
- shorten an item
- merge item contents

### Validation rules

Reject the compaction response if:

- top-level JSON cannot be parsed
- top-level object has no `items`
- `items` is empty
- first item is not `session_frame`
- last item is not `handover_state`

Drop an individual returned item if:

- `type != "compaction"`
- `role` is not `system`, `user`, or `assistant`
- `kind` is not one of the approved kinds
- `content` is empty after stripping
- item is an exact consecutive duplicate of the previous kept item

Drop a returned item as obvious boilerplate if it is clearly one of the known harness-generated transient messages, for example:

- the current-time turn-context message
- a subagent status snapshot with no substantive state

Do not add broader content-pattern heuristics than that.

## Session Rebuild

### New session shape

After a successful compaction, the fresh session is rebuilt in this order:

1. fresh identity bootstrap
2. fresh tool bootstrap
3. fresh memory bootstrap, if enabled
4. compacted replacement transcript items
5. any carried-forward in-progress turn records, if the compaction happened mid-turn

The old `summary_seed` path is retired.

### How replacement items are persisted

Replacement items should be persisted as normal replayable transcript records with:

- `role` taken from the replacement item
- `content` taken directly from the replacement item
- metadata containing the structural fields from the replacement item

Suggested metadata shape on persisted replacement records:

```json
{
  "type": "compaction",
  "compaction_item": true,
  "compaction_kind": "condensed_span",
  "verbatim": false,
  "source_record_ids": ["..."],
  "source_range": {"start": 12, "end": 28},
  "compaction_generation": 2
}
```

This keeps:

- model-visible replay natural, because replay still sees role + content text
- structural provenance available in transcript metadata
- future compactions able to identify and drop prior compaction items cleanly

### Old-session compaction audit record

Archive one `kind="compaction"` audit record in the old session.

Its metadata should include:

- `reason`
- `provider`
- `model`
- `response_id`
- usage fields
- the full returned structured replacement payload

Its content should be a short human-readable note, not the whole payload.

Example:

- `content = "Compaction replaced prior session history with 6 structured handover items."`

### Legacy records

Legacy `summary_seed` records remain readable for old sessions but are treated as pruneable legacy items.
New compactions should never create `summary_seed` again.

## Mid-Turn And Follow-Up Compaction

Mid-turn compaction must preserve a hard boundary between:

- prior session history being compacted
- current in-progress turn records that are being carried forward

Approved rule:

- compaction source excludes the active in-progress turn records
- carried-forward current-turn records remain a separate survivability mechanism
- replacement history represents prior session history only

This avoids duplicating the active turn in both:

- the replacement history
- the carry-forward rebound records

The existing carry-forward cloning and strong/soft compaction path can remain for live in-progress turn records.
That is separate from pre-compaction pruning and does not change the approved item-level pruning rules described above.

## Codex Backend

Codex-backed actors follow the same compaction design:

- use `core.compaction.provider`
- resolve the compaction model from the selected normal provider's existing settings
- build replacement history using the same compaction schema
- start a fresh Codex thread
- inject fresh bootstrap plus replacement history into that new thread

Codex manual `/compact` should stop being a bare fresh-thread reset.
It should rebuild the new Codex thread from compacted replacement items the same way the normal `AgentLoop` path rebuilds a fresh normal-provider session.

## Prompt Contract For `COMPACTION.md`

The compaction prompt should be rewritten so the model is asked to:

- read a pruned transcript item list
- produce valid JSON only
- emit ordered replacement items
- preserve critical user and assistant wording exactly when needed
- condense long execution-heavy spans into compact assistant narration
- end with an explicit handover state

The prompt should explicitly say:

- do not output one monolithic summary
- do not output raw tool logs
- do not restate bootstrap or tool descriptions
- keep the result feeling like condensed conversation history plus handover

## Implementation Plan

### Files to change

- `src/jarvis/settings.yml`
  - add `core.groups.compaction.fields.provider`
  - expose only provider, not model

- `src/jarvis/settings.py`
  - export `JARVIS_COMPACTION_PROVIDER`

- `src/jarvis/core/config.py`
  - add compaction-provider runtime config loading
  - keep model resolution delegated to existing provider settings

- `src/jarvis/main.py`
  - log compaction provider and resolved model at startup

- `src/jarvis/core/prompts/COMPACTION.md`
  - replace summary-seed prompt with structured replacement-history prompt

- `src/jarvis/core/compaction.py`
  - replace `summary_text` output with structured replacement items
  - add pre-compaction source pruning
  - add post-compaction validation and pruning
  - add response parsing and structured output validation

- `src/jarvis/core/agent_loop.py`
  - stop creating `summary_seed`
  - rebuild fresh session from replacement items
  - persist replacement-item metadata on new session records
  - persist old-session compaction audit metadata
  - ensure mid-turn compaction excludes active-turn source records

- `src/jarvis/codex_backend/actor_runtime.py`
  - move Codex `/compact` from fresh-thread reset to true replacement-history rebuild

- tests under `tests/`
  - update old summary-seed expectations
  - add replacement-history tests

### Suggested sequencing

1. Add settings and startup logging.
2. Rewrite the compaction prompt and compaction response types.
3. Implement pre-compaction source pruning.
4. Implement structured replacement-item parsing and validation.
5. Replace `summary_seed` session rebuild with replacement-history session rebuild.
6. Update mid-turn compaction to compact only prior history.
7. Port Codex backend compaction onto the same design.
8. Update docs and tests together.

## Test Plan

Add or update tests for all of the following:

- compaction provider is independent from main-agent provider
- subagent compaction provider also uses the same separate compaction provider
- startup logging includes compaction provider and resolved model
- pre-compaction pruning drops only the approved obvious items
- pre-compaction pruning does not alter kept item content
- compaction response must parse as structured JSON with ordered items
- invalid roles, kinds, missing `type=compaction`, or empty content are rejected or dropped
- new session is rebuilt from replacement-history records rather than one `summary_seed`
- persisted replacement items keep structural metadata
- future compactions prune old compaction items cleanly
- mid-turn compaction excludes active-turn source records but still carries current-turn rebound records
- Codex manual `/compact` rebuilds a fresh thread from replacement history

## Notes For Implementers

- Keep the pruning logic boring and explicit.
- If you feel tempted to trim text inside a record, that belongs in neither pre-pruning nor post-pruning.
- The compaction model itself is the place where transcript collapse happens.
- The replacement history should be easy for a later human to inspect in JSONL transcript archives, not just easy for the model to consume.
