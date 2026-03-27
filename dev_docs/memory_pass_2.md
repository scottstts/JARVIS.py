# Jarvis Memory Improvement Pass 2

Last updated: 2026-03-13

## Purpose

This document is the implementation brief for the next pass on the Jarvis runtime memory system.

The goal is not to redesign memory from scratch. The current architecture is good and already works.

This pass is about improving:

- how accurately memory is retrieved
- how well the agent can make use of retrieved memory
- how well memory stays clean and useful over time
- how closely the live behavior matches the intended design in `dev_docs/memory_design_doc.md`

## Inputs Used For This Plan

This plan is based on:

- `dev_docs/memory_design_doc.md`
- `.codex/AGENTS.md`
- the current implementation under `src/jarvis/memory/`, `src/jarvis/tools/basic/memory_*`, `src/jarvis/tools/discoverable/memory_admin/`, `src/jarvis/core/agent_loop.py`, and `src/jarvis/identities/PROGRAM.md`
- current project notes in `notes/notes.md`

Important project rule:

- If this document and the implementation disagree, treat the code as the current truth and use this document as the intended next-step plan.

## Current State Summary

What is already strong:

- Canonical Markdown memory under `/workspace/memory`
- SQLite sidecar for search, graph lookup, dirty detection, and maintenance
- Semantic retrieval is operational through the vendored `sqlite-vec` path in current container builds
- Dedicated memory tools instead of generic file editing
- Opportunistic dirty-file reconciliation
- Runtime bootstrap injection for `core` and `ongoing`
- Post-turn reflection plumbing
- Clean degraded-mode behavior when semantic retrieval is unavailable

What is currently limiting effectiveness:

- lexical retrieval is too dependent on raw FTS5 query behavior
- hybrid scoring is still much simpler than the design intends
- graph/entity matching is too naive for natural language queries
- tool outputs are not optimized for model consumption
- maintenance and reflection are only partially implemented
- some fields that should improve retrieval quality exist in schema but are not yet used operationally

Operational note to keep in mind:

- Semantic retrieval is now operational in the current `jarvis_runtime` container/runtime path, so tuning should assume true hybrid retrieval is available.
- Degraded lexical-plus-graph fallback still matters because the design intentionally supports semantic-disabled runtime conditions.

## Scope And Constraints

Keep these constraints:

- Do not replace the Markdown-canonical model.
- Do not introduce a separate graph database.
- Do not replace the dedicated memory tools with generic file access.
- Do not perform a large architecture rewrite just to tune ranking or agent behavior.
- Keep degraded-mode behavior intact when semantic retrieval is unavailable.

Non-goals for this pass:

- rewriting the overall memory architecture
- rewriting transcript storage
- replacing SQLite
- introducing a large multi-stage retrieval service

## Priority Order

The implementation order should be:

1. Retrieval quality foundation
2. Agent memory consumption improvements
3. Maintenance and reflection hygiene
4. Advanced retrieval work and optional follow-ups

If time is tight, finish Phase 1 and the Phase 2 items that affect day-to-day agent use before doing anything else.

## Required Correctness Fixes Found During This Sweep

These are not the main purpose of the pass, but they should be fixed while touching this area because they affect memory quality directly.

1. `demote` currently duplicates instead of migrating.
   - Current behavior: `MemoryService._migrate_document()` only archives the source document when `operation == "promote"`.
   - Intended fix: `demote` must also archive or remove the source active document so the memory does not exist in two active places.
   - File: `src/jarvis/memory/service.py`

2. Search can return synthetic section paths that `memory_get` cannot open.
   - Current behavior: semantic and graph retrieval can return `section_path="facts"` or `section_path="relations"`, but `memory_get` only understands body sections.
   - Intended fix: `memory_get` must support synthetic sections for `facts` and `relations`, or search must stop returning section paths it cannot resolve. The preferred fix is to support synthetic sections.
   - Files: `src/jarvis/memory/service.py`, `src/jarvis/tools/basic/memory_get/tool.py`

3. `access_log.chunk_id` is currently populated with `section_path`, not the actual chunk id.
   - This will corrupt any later usage-based ranking or analytics.
   - Intended fix: store the real chunk id when available. If the result is a synthetic fact or relation hit, store a structured synthetic id instead of a section label.
   - Files: `src/jarvis/memory/service.py`, `src/jarvis/memory/types.py`, `src/jarvis/memory/index_db.py`

4. Recency scoring uses filesystem `mtime` instead of document `updated_at`.
   - The design says recency should come from memory metadata, not host file timestamps.
   - Intended fix: use indexed `updated_at` from the candidate/document row during fusion.
   - Files: `src/jarvis/memory/retrieval.py`, `src/jarvis/memory/index_db.py`, possibly `src/jarvis/memory/types.py`

5. `memory_get` accesses are not logged.
   - If later ranking uses access history, opening a memory document must count.
   - Intended fix: add access logging for `memory_get` and use correct identifiers.
   - Files: `src/jarvis/memory/service.py`, `src/jarvis/memory/index_db.py`

## Phase 1: Retrieval Quality Foundation

This phase is the highest-value work.

### 1. Lexical Query Planning

Problem:

- `document_chunks_fts MATCH ?` currently receives the raw query string.
- In practice this makes natural-language questions brittle because FTS5 treats them as implicit AND matches.
- This remains high-value even with semantic retrieval live because lexical recall is still part of hybrid retrieval and remains the fallback path when semantic retrieval is unavailable.

Implementation intent:

- Keep FTS5 as the lexical engine.
- Do not pass raw user text straight into `MATCH`.
- Add a small lexical query-planning helper and generate multiple lexical variants from the input query.

Recommended behavior:

1. Normalize the query.
   - lowercase
   - trim whitespace
   - strip punctuation that is only syntactic noise
   - remove obvious question framing and stop words
   - do not blindly drop short tokens if they may be meaningful, for example `go`, `c`, `c++`, `r`, version-like tokens, or initials

2. Produce at least these lexical variants:
   - a phrase-oriented variant for short meaningful phrases
   - a recall-oriented keyword variant using OR-joined tokens
   - a prefix variant for tokens where prefix matching helps recall

3. Merge lexical candidates across variants before fusion.
   - preserve a reason label for how the candidate matched, for example `lexical_phrase`, `lexical_keywords`, `lexical_prefix`
   - deduplicate by `(document_id, section_path)`

4. Escape or sanitize any FTS5 special syntax so normal user questions do not become malformed `MATCH` queries.

Important constraint:

- Do not replace lexical retrieval with one broad OR-only query. That will improve recall but can destroy precision. The right move is a small query planner with a few variants, not a single permissive query.

Files to change:

- `src/jarvis/memory/retrieval.py`
- `src/jarvis/memory/index_db.py`

Acceptance criteria:

- A natural-language query like `What does Scott prefer for backend work?` can retrieve the relevant memory even when the exact stop words are absent from the chunk text.
- Lexical retrieval no longer fails or degrades badly on punctuation-heavy or question-shaped queries.
- Search results include lexical match reasons that reflect which lexical plan matched.

### 2. Use Real Lexical And Semantic Scores

Problem:

- Lexical and semantic channels currently discard the actual FTS/vector score signal and use rank-position normalization only.
- Semantic candidates currently hardcode `distance = 0.0`.

Implementation intent:

- Carry the real score signal through the pipeline.
- Keep a rank-based fallback only when the raw score distribution is unusable.

Required changes:

1. Lexical channel:
   - keep the FTS5 `bm25()` value from SQL
   - normalize it in a stable way inside the returned candidate set
   - preserve the existing rank fallback if all raw scores collapse into the same value

2. Semantic channel:
   - capture the real vector distance from `sqlite-vec`
   - convert distance into a similarity-like normalized score for fusion
   - stop treating semantic order as the only signal

3. Candidate model:
   - extend the candidate data so fusion has access to raw and normalized lexical/semantic values plus `updated_at`

Files to change:

- `src/jarvis/memory/index_db.py`
- `src/jarvis/memory/retrieval.py`
- `src/jarvis/memory/types.py`

Acceptance criteria:

- Semantic candidates no longer return a fake `distance` value.
- Fusion can distinguish `clearly best hit` from `barely first hit`.
- Existing degraded-mode behavior still works when semantic search is unavailable.

### 3. Bring Hybrid Fusion Closer To The Design

Problem:

- The design expects more than `semantic + lexical + graph + recency`.
- The current code only applies kind multipliers and rough recency.

Implementation intent:

- Implement the cheap, high-confidence modifiers first.
- Do not pretend support/contradiction-aware ranking is working until those counters are populated meaningfully.

Phase 1 modifiers to implement now:

- kind multiplier
- pinned boost
- expired penalty
- archive penalty
- stale ongoing penalty

Recommended stale ongoing definition for this pass:

- ongoing document with `review_after` in the past or very old `updated_at`
- do not penalize pinned ongoing docs as aggressively

Do not implement yet unless the needed state is actually populated:

- support-count boosts
- contradiction-count penalties
- confirmation freshness boosts

Those belong in Phase 4 unless the next agent also implements population of those signals.

Files to change:

- `src/jarvis/memory/retrieval.py`
- `src/jarvis/memory/index_db.py`

Acceptance criteria:

- Two otherwise similar hits rank with pinned/current/active memory above stale or expired memory.
- Archived results are down-ranked compared with active results when archive scope is mixed in.

### 4. Use Existing Candidate Count Settings And Enforce Graph Caps

Problem:

- `MemorySettings` defines lexical/semantic/graph candidate counts, but retrieval still hardcodes `30` and does not enforce the graph result cap consistently.

Implementation intent:

- Use the settings that already exist.
- Avoid hardcoded candidate counts in the retrieval pipeline.

Files to change:

- `src/jarvis/memory/retrieval.py`
- `src/jarvis/memory/config.py`
- `src/jarvis/memory/index_db.py`

Acceptance criteria:

- lexical candidate count, semantic candidate count, graph candidate count, and final result count come from settings
- graph results are capped consistently

### 5. Improve Graph Entity Resolution For Natural Language

Problem:

- Graph resolution currently compares the whole query blob against entity names and aliases.
- Full-sentence queries are therefore weaker than they should be.

Implementation intent:

- Reuse the normalized query tokens from lexical query planning.
- Match entities against candidate tokens and short n-grams, not only the full normalized query string.

Recommended behavior:

- extract meaningful tokens and 1-2 token n-grams from the query
- resolve entities against each candidate token/ngram
- preserve exact canonical and exact alias matches as strongest
- keep fuzzy matching conservative
- default graph scoring should prefer current relations and penalize `past`, `uncertain`, and `superseded`

Files to change:

- `src/jarvis/memory/graph.py`
- `src/jarvis/memory/index_db.py`
- possibly `src/jarvis/memory/retrieval.py`

Acceptance criteria:

- Queries like `What does Scott think about TypeScript?` can match graph-backed memory through `Scott` and `TypeScript` even though the full sentence is not an entity name.
- Current relations rank above superseded relations by default.

### 6. Expose Alias And Entity Inputs Through `memory_write`

Problem:

- The schema and graph index support aliases and entity refs.
- The agent cannot actually write them through `memory_write`.

Implementation intent:

- Extend the tool schema and service normalization to support:
  - `aliases`
  - `tags`
  - `entity_refs`
  - `completion_criteria`

- Do not add unnecessary new fields beyond the existing design.

Files to change:

- `src/jarvis/tools/basic/memory_write/tool.py`
- `src/jarvis/memory/service.py`
- `src/jarvis/memory/types.py`
- tests

Acceptance criteria:

- A memory document created through `memory_write` can carry aliases and entity refs.
- Graph/entity search can benefit from these fields without manual file editing.

## Phase 2: Improve How The Agent Uses Memory

This phase is about the model-facing surface, not the index internals.

### 7. Make `memory_search` Output More Useful To The Model

Problem:

- The tool metadata is richer than what the model sees.
- Tool-result messages only carry `content`.
- Search output currently omits the document title and does not strongly guide the model toward `memory_get`.

Implementation intent:

- Make the text content of `memory_search` self-sufficient for the model.

Changes to make:

- include document title in each result
- keep path, kind, section, score, reasons, and snippet
- keep warnings explicit
- when `section_path` is synthetic, label it clearly
- keep formatting compact enough for follow-up tool rounds

Do not rely on tool metadata being visible to the model.

Files to change:

- `src/jarvis/tools/basic/memory_search/tool.py`
- `src/jarvis/memory/index_db.py`
- `src/jarvis/memory/types.py`

Acceptance criteria:

- A search result contains enough information for the model to decide whether to call `memory_get` without guessing from metadata it cannot see.

### 8. Make `memory_get` Better For Agent Consumption

Problem:

- `memory_get` defaults to `include_frontmatter=true`.
- For ordinary model use, YAML frontmatter is noisy compared with the document body.
- Synthetic `facts` and `relations` section paths are currently unsupported.

Implementation intent:

- Optimize the default for agent use.

Required changes:

1. Change the default to `include_frontmatter=false` for agent-facing tool use.
2. Support synthetic sections:
   - `facts`
   - `relations`
3. When a specific section is requested, return:
   - `# Title`
   - the requested section content
   - optional sources only when requested

Files to change:

- `src/jarvis/tools/basic/memory_get/tool.py`
- `src/jarvis/memory/service.py`

Acceptance criteria:

- After `memory_search`, the model can open either a body section or a synthetic facts/relations section cleanly.
- The default `memory_get` output is concise enough for normal follow-up reasoning.

### 9. Improve Bootstrap Density

Problem:

- Ongoing bootstrap currently includes open loops but not `Current State`, even though the design intends both.
- Bootstrap does not include relation-level information.
- The agent has little freshness context at session start.

Implementation intent:

- Keep bootstrap deterministic and cheap.
- Increase usefulness, not verbosity.

Required changes:

1. Ongoing bootstrap:
   - include summary
   - include compact `Current State`
   - include compact `Open Loops`

2. Core bootstrap:
   - keep summary first
   - include selected current facts
   - include selected current relations when they carry information the summary does not already express

3. Freshness hint:
   - add a compact freshness marker based on `updated_at`, for example `updated 3d ago`
   - keep it deterministic and token-cheap

4. Keep the token budget hard cap behavior.

Files to change:

- `src/jarvis/memory/bootstrap.py`
- `src/jarvis/memory/service.py`

Acceptance criteria:

- Bootstrap preview contains ongoing `Current State`
- Important relation-style memory can be visible at startup without forcing a memory search
- Token budgets are still respected

### 10. Tighten `PROGRAM.md` Memory Guidance

Problem:

- The prompt gives good high-level guidance, but it does not give enough tactical instruction on when and how to search memory.

Implementation intent:

- Improve the agent's default memory habits without making the prompt noisy.

Add guidance for:

- using short keyword-style `memory_search` queries rather than full natural-language questions
- searching memory before claiming not to know something that could plausibly be remembered
- using `memory_get` after a promising snippet instead of over-trusting snippets
- updating existing memory rather than creating duplicates
- checking memory when the user resumes an active project or asks about past preferences, commitments, or remembered facts

Files to change:

- `src/jarvis/identities/PROGRAM.md`

Acceptance criteria:

- The identity prompt more clearly pushes the agent toward concise search queries, follow-up reads, and update-over-duplicate behavior.

## Phase 3: Maintenance And Reflection Hygiene

This phase is about keeping stored memory useful over time.

### 11. Improve Reflection Prompt Quality

Problem:

- Reflection has the right shape but not enough tactical guidance.
- It only sees active titles, not rich enough context to prefer updates over creates reliably.

Implementation intent:

- Keep the single-call reflection approach.
- Improve the prompt and the input context before introducing more machinery.

Required changes:

1. Give the planner compact active-memory context, not just title lists.
   - include `document_id`
   - include title
   - include a short summary
   - include kind
   - keep this compact

2. Add explicit prompt guidance:
   - prefer updating an existing document over creating a duplicate
   - only emit actions for persistence-worthy content
   - routine turns should return `ignore`
   - list the valid daily sections explicitly
   - include a compact example of a good fact and a good relation triple

3. Remove standalone unsupported reflection actions from the prompt and parser for this pass.
   - remove `add_relation`
   - remove `supersede_relation`

Exact intention:

- Relation updates should be expressed through normal `create_*` or `update_*` payloads that include `relations`.
- Do not leave the planner emitting actions that the service silently ignores.

Files to change:

- `src/jarvis/memory/reflection.py`
- `src/jarvis/memory/service.py`
- tests

Acceptance criteria:

- Reflection prefers `update_ongoing` or `update_core` when an existing document already covers the topic.
- Routine turns are more likely to emit `ignore`.
- No reflection action types are emitted and then dropped on the floor.

### 12. Enforce `locked=true`

Problem:

- The design says locked documents must not be auto-mutated.
- Current auto-apply and maintenance paths do not enforce that rule strictly.

Implementation intent:

- Enforce lock protection in automatic paths.
- Manual writes through explicit user intent may still target locked docs if the product decision allows it, but automatic reflection/maintenance must not.

Required changes:

- reflection auto-apply must skip locked targets
- maintenance jobs must skip locked docs
- direct service mutation helpers should protect against accidental auto-mutation of locked docs

Files to change:

- `src/jarvis/memory/service.py`
- `src/jarvis/memory/maintenance.py`

Acceptance criteria:

- Locked documents are not changed by reflection or maintenance.

### 13. Fix Daily Rollover Behavior

Problem:

- The design says daily rollover closes the previous local-day file and ensures today's file exists.
- Current implementation only ensures today's file exists.

Implementation intent:

- Keep daily logs as a staging layer instead of a forever-open bucket.

Required changes:

- `daily_rollover` should close the previously active daily document when the date changes
- it should then ensure today's file exists
- keep the local timezone behavior

Files to change:

- `src/jarvis/memory/service.py`
- `src/jarvis/memory/maintenance.py`

Acceptance criteria:

- After rollover, the previous day's active daily file is no longer left open indefinitely.
- Archive sweep can eventually operate on closed daily docs as intended.

### 14. Stop Harmful Automatic Priority Drift

Problem:

- `recompute_priority_from_usage()` currently bumps priorities upward on every maintenance run.
- This flattens the ranking signal over time and does not actually use access data.

Implementation intent for this pass:

- Do not ship another unstable automatic priority formula.
- The immediate fix is to stop harmful behavior.

Exact instruction:

- Change `recompute_priority_from_usage` to a skipped or no-op maintenance job for now.
- Do not implement a new automatic formula in this pass unless the next agent also introduces a clean manual-vs-derived priority model. That is out of scope for this pass.

Files to change:

- `src/jarvis/memory/service.py`
- `src/jarvis/memory/maintenance.py`

Acceptance criteria:

- Maintenance no longer monotonically drives memory priorities upward.

## Phase 4: Advanced Retrieval Work And Optional Follow-Ups

These items are valuable, but they should come after Phases 1-3 unless the next agent can finish them cleanly.

### 15. Benchmark And Tune Semantic Retrieval End-To-End

Problem:

- Semantic retrieval is now expected to be live in the current container/runtime path.
- The next pass should treat semantic search as an active retrieval channel to measure and tune, not as missing infrastructure.
- Reminder from end-to-end transcript review: hybrid search can still surface semantic-only junk hits for weak or no-match queries. Concrete examples seen in the transcript: a morning-run query returned unrelated hike/Jarvis-improvement memories, and a Three.js project query returned daily-log and run-routine memories. Phase 4 should treat this as an explicit tuning target rather than assuming current semantic recall quality is acceptable.

Implementation intent:

- First verify live behavior in the current branch/container image rather than assuming semantic retrieval quality from code inspection alone.
- Confirm:
  - embeddings are actually being written
  - `semantic_disabled=false` in normal hybrid searches once embeddings exist
  - semantic candidates carry real distance/similarity values
  - semantic search materially improves recall when lexical overlap is weak
- Only if that verification fails should the next agent investigate the `sqlite-vec` override build path or embedding initialization path.
- This is also the place to add bounded weak-result suppression, such as minimum semantic-score thresholds or down-ranking semantic-only hits when lexical and graph support are both absent.

Files likely involved:

- `src/jarvis/memory/index_db.py`
- `src/jarvis/memory/retrieval.py`
- tests that exercise semantic search and hybrid ranking
- Docker/build files only if live verification shows semantic retrieval is broken again

Acceptance criteria:

- In the `jarvis_runtime` container, a live hybrid memory search reports `semantic_disabled=false` once embeddings exist.
- A weak-lexical but strong-semantic query can retrieve the expected memory.
- Semantic scores are meaningful enough to influence final ranking.

### 16. Populate Truth-Aware Signals Before Using Them For Ranking

Problem:

- The schema has fields like `support_count`, `contradiction_count`, `last_confirmed_at`, and `last_contradicted_at`.
- They are not yet populated in a way that ranking can trust.

Implementation intent:

- Do not fake these signals.
- If the next agent chooses to use them in ranking, first implement how they are computed and updated.

This is a separate step from the cheap modifiers in Phase 1.

Files likely involved:

- `src/jarvis/memory/index_db.py`
- `src/jarvis/memory/service.py`
- possibly maintenance/reconciliation paths

### 17. Add Retrieval Fallback Planning Only After Baseline Retrieval Is Strong

Problem:

- The design mentions LLM-assisted fallback retrieval planning for weak queries.

Implementation intent:

- Do not add this yet unless Phase 1 is already done.
- First fix lexical planning, scoring, graph matching, and semantic distance handling.

If implemented later:

- detect weak retrieval locally
- only then invoke a second-pass query-planning step
- keep it optional and bounded

Files likely involved:

- `src/jarvis/memory/retrieval.py`
- possibly a small helper in `src/jarvis/memory/`

### 18. Only Revisit Chunk Weighting If Search Noise Persists

Problem:

- Chunk text currently prepends title and heading, which helps grounding but may add FTS noise.

Implementation intent:

- Do not change chunk structure in this pass unless search noise remains a real problem after query planning and score fusion are improved.

If revisited later:

- consider separate FTS columns or weighted title/body handling instead of dropping title context entirely

File:

- `src/jarvis/memory/chunker.py`
- `src/jarvis/memory/index_db.py`

## Test Plan

The next agent should extend tests as part of the implementation. Do not rely only on manual checks.

### Retrieval Tests

Add or extend tests to cover:

- natural-language question query retrieves the correct document lexically
- stop-word-heavy query still retrieves the target memory
- query with punctuation or apostrophes does not break FTS matching
- pinned active memory outranks equivalent unpinned memory
- expired memory is penalized or excluded correctly
- stale ongoing memory is penalized correctly
- graph resolution works from full-sentence queries
- semantic candidates carry real distance/similarity values
- candidate counts obey settings rather than hardcoded constants

### Tool And Agent-Use Tests

Add or extend tests to cover:

- `memory_search` output includes title and remains compact
- `memory_get` defaults to body-oriented output
- `memory_get(section_path="facts")` works
- `memory_get(section_path="relations")` works
- access logging for `memory_get` occurs
- `access_log.chunk_id` stores a real id or a well-defined synthetic id

### Bootstrap Tests

Add or extend tests to cover:

- ongoing bootstrap includes `Current State`
- core bootstrap can include selected relations
- freshness hints are deterministic
- bootstrap remains inside token budget

### Reflection And Maintenance Tests

Add or extend tests to cover:

- reflection prefers update over duplicate create when an active memory already exists
- reflection emits `ignore` for routine turns
- locked docs are skipped by reflection
- locked docs are skipped by maintenance
- daily rollover closes the previous day
- `demote` does not leave the source active document in place
- priority recompute no longer drifts upward each run

## Manual Evaluation Checklist

After implementation, manually check these behaviors in a real runtime session:

1. Ask memory-style questions in natural language, not only keyword style.
2. Resume a known project and see whether the agent searches and opens the right memory.
3. Inspect `memory_admin render_bootstrap_preview` and confirm the bootstrap is denser but still compact.
4. Confirm a hybrid `memory_search` can retrieve a memory even when lexical overlap is weak.
5. Confirm `memory_search` still behaves well in semantic-disabled fallback conditions.

## File-Level Touch Points

The next agent will likely need to touch:

- `src/jarvis/memory/retrieval.py`
- `src/jarvis/memory/index_db.py`
- `src/jarvis/memory/graph.py`
- `src/jarvis/memory/service.py`
- `src/jarvis/memory/bootstrap.py`
- `src/jarvis/memory/reflection.py`
- `src/jarvis/memory/config.py`
- `src/jarvis/tools/basic/memory_search/tool.py`
- `src/jarvis/tools/basic/memory_get/tool.py`
- `src/jarvis/tools/basic/memory_write/tool.py`
- `src/jarvis/identities/PROGRAM.md`
- `tests/test_memory_service.py`
- `tests/test_tools.py`

If the existing tests become too crowded, it is reasonable to add a focused retrieval test module rather than keep everything in `tests/test_memory_service.py`.

## Final Implementation Guidance

Do this pass in small, verifiable slices.

Recommended order inside the work:

1. Fix the correctness issues that block reliable measurement.
2. Improve lexical query planning and score handling.
3. Improve graph/entity handling and write-surface support for aliases/entity refs.
4. Improve model-facing tool output and bootstrap density.
5. Improve reflection prompting and maintenance hygiene.
6. Only then decide whether to continue into deeper semantic tuning and advanced truth-aware ranking.

Do not let the scope expand into a rewrite. The right outcome is a memory system that behaves noticeably better with the same overall architecture.
