# Jarvis Memory System Design

## Status

This document is the draft source of truth for the Jarvis memory system design.

It is intentionally explicit. If a detail is written here, implementation should follow it unless we later revise this document.

This document is about the runtime memory system.

- Runtime memory files live under `/workspace/memory/`
- Memory implementation code lives under `src/jarvis/memory/`
- Transcript storage remains separate under `/workspace/archive/transcripts/`
- Identity bootstrap files remain separate under `/workspace/identities/`

## Purpose

The Jarvis memory system is the long-term memory layer for the agent.

It is not a transcript store.
It is not the same thing as the identity system.
It is not a single monolithic subsystem.

It is a distributed concern with one coordinated design:

- `src/jarvis/identities/` provides pinned procedural/bootstrap instruction memory
- `src/jarvis/storage/` provides raw episodic evidence in archived transcripts
- `src/jarvis/memory/` provides long-term semantic, ongoing, and graph-oriented memory orchestration
- `src/jarvis/tools/` provides the agent-facing memory control surface
- `src/jarvis/core/` decides when memory is injected and when maintenance/reflection runs

## Core Principles

1. Canonical memory state is Markdown under `/workspace/memory/`
2. SQLite is a derived operational index, not the source of truth
3. The agent and the human both inspect canonical memory through Markdown files
4. Jarvis must use dedicated memory tools for memory search, inspection, and mutation rather than generic file editing
5. Out-of-band edits by a human or external system may still happen, and the memory system must detect, reconcile, and reindex those edits without treating generic tools as part of Jarvis's normal memory workflow
6. The memory system includes graph memory from day one, but graph memory is implemented as a lightweight relations layer derived from canonical Markdown plus indexed in SQLite
7. Identity files are not ordinary memory files and are not auto-mutated by the memory system
8. Raw transcript archives remain immutable evidence and are never rewritten by memory maintenance
9. Memory search must support lexical search, graph expansion, semantic retrieval when physically available, and hybrid ranking that degrades cleanly when semantic retrieval is not ready
10. Memory maintenance must be mostly automatic, but manual inspection and intervention must stay simple

## What Counts As Memory

Jarvis runtime memory is split into three canonical categories:

1. `core`
2. `ongoing`
3. `daily`

### Core Memory

Core memory is the smallest set of runtime-learned facts that should be injected at session start without requiring a search.

Core memory is:

- durable
- broadly reusable
- behavior-shaping
- costly to forget before retrieval

Core memory is not:

- raw history
- recent chatter
- verbose evidence
- temporary plans
- one-off tasks

### Ongoing Memory

Ongoing memory is medium-horizon state.

This includes:

- active projects
- current life context
- temporary routines
- active commitments
- unresolved open loops
- current operational state that matters across turns

Ongoing memory is not automatically permanent.
Every ongoing memory document must have review and expiry semantics.

### Daily Memory

Daily memory is distilled episodic memory.

It is not a transcript mirror.
It is a running daily log of notable events, decisions, commitments, artifacts, and candidate promotions extracted from actual usage.

Daily memory exists so that:

- recent events are easy to inspect in a human-friendly form
- maintenance has a stable staging area
- not every useful event must immediately become core or ongoing memory

## Memory Representation Model

Jarvis memory is explicitly modeled as:

1. resources
2. facts and relations
3. summaries

### Resources

Resources are the evidence layer.

They are not a new canonical memory directory.
They are the source materials from which memory is derived.

Primary resource sources:

- transcript archives under `/workspace/archive/transcripts/`
- imported files or artifacts explicitly referenced by memory entries
- manual memory edits made by the user
- tool outputs explicitly promoted into memory provenance

Resources are:

- timestamped
- provenance-addressable
- mostly immutable from the memory system's perspective

Resources are not directly injected into bootstrap context.

### Facts And Relations

Facts and relations are the structured truth layer derived from resources.

Facts capture atomic descriptive statements.
Relations capture entity-linked structured statements.

They exist in two places:

- canonically in Markdown frontmatter for active `core/` and `ongoing/` documents
- operationally in SQLite for search, conflict handling, ranking, and maintenance

### Summaries

Summaries are the context layer.

They are the compressed, human-readable representations used for:

- bootstrap injection
- fast inspection
- memory browsing
- long-horizon coherence

In Jarvis, summaries live primarily in:

- `core/` documents
- `ongoing/` documents
- `daily/` logs as staging summaries before promotion or archival

### Representation Rule

The memory system should move upward through the stack:

- resources are the evidence
- facts and relations are the normalized learned content
- summaries are the context-efficient representation

The system must not treat retrieved summaries as stronger evidence than underlying facts and provenance.

## Runtime Filesystem Layout

All runtime memory files live under `/workspace/memory/`.

The full layout is:

```text
/workspace/memory/
├── core/
│   ├── <slug>.md
│   └── ...
├── ongoing/
│   ├── <slug>.md
│   └── ...
├── daily/
│   ├── YYYY-MM-DD.md
│   └── ...
├── archive/
│   ├── core/
│   ├── ongoing/
│   └── daily/
└── .index/
    ├── index.sqlite
    ├── index.sqlite-shm
    ├── index.sqlite-wal
    ├── embeddings.sqlite
    ├── extension/
    └── state.json
```

### Directory Semantics

- `core/` contains active canonical core memory documents
- `ongoing/` contains active canonical ongoing memory documents
- `daily/` contains current and recent distilled daily logs
- `archive/` contains retired or superseded Markdown memory files
- `.index/` contains derived, rebuildable operational artifacts

### Naming Rules

- All canonical memory files use kebab-case names
- File names are descriptive, human-readable, and stable where possible
- File names do not include timestamps except in `daily/` and `archive/`
- Canonical identity files are not stored under `/workspace/memory/`

Examples:

- `/workspace/memory/core/user-communication-style.md`
- `/workspace/memory/core/user-development-preferences.md`
- `/workspace/memory/ongoing/jarvis-memory-system.md`
- `/workspace/memory/ongoing/current-morning-routine.md`
- `/workspace/memory/daily/2026-03-12.md`

## Canonical Markdown Schemas

Canonical memory is Markdown with YAML frontmatter.

The frontmatter is machine-validated.
The body is human-readable narrative summary plus optional details.
Search indexing is derived from the structured document, not only raw body markdown.

### Common Rules

- UTF-8 text
- LF line endings
- one document per file
- no Markdown tables
- frontmatter is required for `core/` and `ongoing/`
- frontmatter is required for `daily/`
- all timestamps use ISO 8601 UTC
- all IDs are stable once created
- for `core/` and `ongoing/`, important narrative text should normally live in body sections, especially `## Summary`
- if a `core/` or `ongoing/` document carries meaningful `summary` text in frontmatter but the `## Summary` body section is blank, that summary text must still remain searchable

### Core Document Schema

Path:

- `/workspace/memory/core/<slug>.md`

Required frontmatter fields:

- `memory_id`: stable string ID
- `kind`: must be `core`
- `title`: human-readable title
- `status`: `active` or `archived`
- `created_at`
- `updated_at`
- `priority`: integer `0..100`
- `pinned`: boolean
- `locked`: boolean
- `confidence`: `low|medium|high`
- `review_after`: ISO timestamp or `null`
- `expires_at`: ISO timestamp or `null`
- `tags`: list of strings
- `aliases`: list of strings
- `facts`: list of fact objects
- `relations`: list of relation objects
- `source_refs`: list of source reference objects

Optional frontmatter fields:

- `summary`: short single-paragraph summary
- `entity_refs`: list of entity references

Core document body sections:

- `# <Title>`
- `## Summary`
- `## Details`
- `## Notes`

### Ongoing Document Schema

Path:

- `/workspace/memory/ongoing/<slug>.md`

Required frontmatter fields:

- `memory_id`: stable string ID
- `kind`: must be `ongoing`
- `title`
- `status`: `active|closed|archived`
- `created_at`
- `updated_at`
- `priority`: integer `0..100`
- `pinned`: boolean
- `locked`: boolean
- `confidence`: `low|medium|high`
- `review_after`: ISO timestamp
- `expires_at`: ISO timestamp or `null`
- `tags`: list of strings
- `aliases`: list of strings
- `facts`: list of fact objects
- `relations`: list of relation objects
- `source_refs`: list of source reference objects

Optional frontmatter fields:

- `summary`
- `entity_refs`
- `completion_criteria`: list of strings
- `close_reason`: string or `null`

Ongoing document body sections:

- `# <Title>`
- `## Summary`
- `## Current State`
- `## Open Loops`
- `## Artifacts`
- `## Notes`

### Daily Document Schema

Path:

- `/workspace/memory/daily/YYYY-MM-DD.md`

Required frontmatter fields:

- `memory_id`
- `kind`: must be `daily`
- `date`: local calendar date string
- `timezone`
- `status`: `active|closed|archived`
- `created_at`
- `updated_at`
- `route_ids`: list of route IDs seen that day
- `session_ids`: list of session IDs seen that day

Daily document body sections:

- `# Daily Log: YYYY-MM-DD`
- `## Notable Events`
- `## Decisions`
- `## Active Commitments`
- `## Open Loops`
- `## Artifacts`
- `## Candidate Promotions`

Daily files are semi-structured.
They are parseable by section heading and bullet grouping.

## Fact, Relation, Entity, And Source Shapes

### Fact Object

Each fact object in frontmatter has:

- `fact_id`
- `text`
- `status`: `current|past|uncertain|superseded`
- `confidence`: `low|medium|high`
- `first_seen_at`
- `last_seen_at`
- `valid_from`
- `valid_to`
- `source_ref_ids`: list of source reference IDs

### Relation Object

Each relation object in frontmatter has:

- `relation_id`
- `subject`
- `predicate`
- `object`
- `status`: `current|past|uncertain|superseded`
- `confidence`: `low|medium|high`
- `cardinality`: `single|multi`
- `first_seen_at`
- `last_seen_at`
- `valid_from`
- `valid_to`
- `source_ref_ids`: list of source reference IDs

### Entity Reference Object

Each entity reference object has:

- `entity_id`
- `name`
- `entity_type`
- `aliases`

### Source Reference Object

Each source reference object has:

- `source_ref_id`
- `source_type`: `transcript|manual|tool|import|maintenance`
- `route_id`: string or `null`
- `session_id`: string or `null`
- `record_id`: string or `null`
- `tool_name`: string or `null`
- `note`: string or `null`
- `captured_at`

## Canonical Meaning Of Markdown Files

Markdown is authoritative for:

- current memory content
- memory categorization into `core`, `ongoing`, or `daily`
- human inspection and manual edits
- pinned and locked state
- explicit facts and relations that should persist

Markdown is not authoritative for:

- search indexes
- embeddings
- access statistics
- ranking caches
- maintenance queues
- dirty-state tracking

Those live in SQLite and are derived.

## SQLite Sidecar Design

The SQLite sidecar exists for operational power, not as a second source of truth.

It is used for:

- FTS5 lexical search
- vector search through `sqlite-vec`
- graph traversal and relation lookup
- maintenance bookkeeping
- dirty-file detection
- ranking
- source/provenance joins
- access logging

If the sidecar is deleted, the memory system must be able to rebuild it from canonical Markdown plus available transcript archives.

### Sidecar Files

- `/workspace/memory/.index/index.sqlite`
- `/workspace/memory/.index/embeddings.sqlite`
- `/workspace/memory/.index/state.json`

`embeddings.sqlite` is separate so vector-heavy writes do not bloat the main operational database.

### SQLite Extensions

- FTS5 is required
- `sqlite-vec` is the intended vector index extension

If `sqlite-vec` is unavailable:

- the system enters degraded mode
- lexical and graph search still work
- semantic search is disabled
- the runtime logs a clear warning
- `memory_search` result metadata must indicate `semantic_disabled=true`

Semantic availability is determined at runtime, not just by package installation.
The system must treat semantic retrieval as unavailable if the extension cannot be loaded, if the vector table is missing, or if embeddings have not yet been initialized.

When semantic retrieval is unavailable:

- `hybrid`/`auto` search must fall back to the non-semantic path instead of failing the tool call
- the search response must include a clear warning explaining that semantic retrieval was skipped

### Main SQLite Tables

The main database must contain at least these tables:

- `resources`
- `documents`
- `document_chunks`
- `document_chunks_fts`
- `facts`
- `relations`
- `entities`
- `source_refs`
- `document_source_refs`
- `fact_source_refs`
- `relation_source_refs`
- `access_log`
- `maintenance_runs`
- `dirty_documents`
- `bootstrap_cache`
- `schema_info`

### `resources`

Columns:

- `resource_id`
- `source_type`
- `route_id`
- `session_id`
- `record_id`
- `tool_name`
- `artifact_path`
- `note`
- `captured_at`
- `checksum`
- `resource_status`

### `documents`

Columns:

- `document_id`
- `path`
- `kind`
- `title`
- `status`
- `priority`
- `pinned`
- `locked`
- `confidence`
- `created_at`
- `updated_at`
- `review_after`
- `expires_at`
- `checksum`
- `summary`
- `body_markdown`
- `archived_at`
- `support_count`
- `contradiction_count`
- `last_confirmed_at`
- `last_contradicted_at`
- `last_accessed_at`
- `access_count_30d`

### `document_chunks`

Columns:

- `chunk_id`
- `document_id`
- `ordinal`
- `section_path`
- `text`
- `token_estimate`
- `created_at`
- `updated_at`

### `document_chunks_fts`

FTS5 virtual table over:

- `chunk_id`
- `document_id`
- `section_path`
- `text`

### `facts`

Columns:

- `fact_id`
- `document_id`
- `text`
- `status`
- `confidence`
- `first_seen_at`
- `last_seen_at`
- `valid_from`
- `valid_to`
- `support_count`
- `contradiction_count`
- `last_confirmed_at`
- `last_contradicted_at`
- `last_accessed_at`
- `access_count_30d`

### `relations`

Columns:

- `relation_id`
- `document_id`
- `subject`
- `predicate`
- `object`
- `status`
- `confidence`
- `cardinality`
- `first_seen_at`
- `last_seen_at`
- `valid_from`
- `valid_to`
- `support_count`
- `contradiction_count`
- `last_confirmed_at`
- `last_contradicted_at`
- `last_accessed_at`
- `access_count_30d`

### `entities`

Columns:

- `entity_id`
- `canonical_name`
- `entity_type`
- `aliases_json`
- `first_seen_at`
- `last_seen_at`
- `last_accessed_at`
- `access_count_30d`

### `source_refs`

Columns:

- `source_ref_id`
- `source_type`
- `route_id`
- `session_id`
- `record_id`
- `tool_name`
- `note`
- `captured_at`

### `access_log`

Columns:

- `access_id`
- `occurred_at`
- `route_id`
- `session_id`
- `tool_name`
- `query`
- `mode`
- `document_id`
- `chunk_id`
- `result_rank`
- `result_score`

### `maintenance_runs`

Columns:

- `run_id`
- `job_name`
- `started_at`
- `finished_at`
- `status`
- `summary_json`

### `dirty_documents`

Columns:

- `path`
- `detected_at`
- `reason`

### `bootstrap_cache`

Columns:

- `cache_key`
- `generated_at`
- `content`
- `token_estimate`
- `checksum_bundle`

## Chunking And Indexing Rules

Chunking is deterministic.
The same Markdown file must produce the same chunk boundaries unless the file content changes.

### Core And Ongoing Chunking

- chunk first by top-level section
- if a section exceeds `1200` characters, split by paragraph
- if a paragraph exceeds `1200` characters, split by sentence groups
- target chunk size is `600..1200` characters
- hard max per chunk is `1600` characters
- chunk text is derived from canonical structured content, including title, section name, and section text
- if `## Summary` is blank but frontmatter `summary` is present, that summary text is used as searchable/indexable summary content
- if a structured document would otherwise yield no non-empty chunks, a minimal title-derived fallback chunk may be indexed so the document does not become invisible

### Daily Chunking

- chunk by daily section
- group adjacent bullets together
- target chunk size is `800..1500` characters
- hard max per chunk is `2000` characters

### What Gets Embedded

Embeddings are created for:

- chunk text
- fact text
- relation textualizations in the form `subject predicate object`

### What Does Not Get Embedded

- raw frontmatter YAML as a blob
- source reference notes alone
- empty or whitespace-only sections

## Search Modes

The memory system supports four search modes:

1. `lexical`
2. `semantic`
3. `graph`
4. `hybrid`

### Default Search Mode

The default mode for `memory_search` is `hybrid`.

Hybrid search performs:

- FTS lexical candidate retrieval
- vector candidate retrieval when semantic retrieval is ready
- graph candidate retrieval
- final score fusion

`hybrid` capability is dynamically determined at query time.
If semantic retrieval is not physically available or not yet initialized, hybrid search falls back to lexical plus graph retrieval and returns an explicit warning instead of failing.

Hybrid retrieval treats retrieved candidates as prospects, not truth.
Truthfulness is improved during score fusion by consulting status, contradiction, support, and recency signals.

### Default Search Scope

The default scope is:

- `core`
- `ongoing`
- `daily`

Archived memory is excluded by default.

### Default Daily Lookback

Default daily lookback is `30` local calendar days.

### Hybrid Candidate Counts

By default:

- lexical retrieves top `30`
- semantic retrieves top `30`
- graph retrieves top `20`
- final merged results return top `8`

### Hybrid Score Formula

All component scores are normalized to `0..1`.

Default fused score:

`0.40 * semantic + 0.35 * lexical + 0.15 * graph + 0.10 * recency`

Then apply modifiers:

- `core` document multiplier: `1.20`
- `ongoing` document multiplier: `1.10`
- `daily` document multiplier: `1.00`
- `archive` document multiplier: `0.85`
- `pinned=true` multiplier: `1.15`
- `status=expired` multiplier: `0.50`
- stale ongoing multiplier: `0.80`

Truth-aware adjustments:

- `status=current` boost when the query implies present state
- `status=past` penalty when the query implies present state
- `status=superseded` strong penalty
- contradiction penalty proportional to `contradiction_count`
- support boost proportional to `support_count`
- confirmation freshness boost based on `last_confirmed_at`
- unresolved-conflict penalty when both current and contradictory candidates coexist

### Recency

Recency boost uses document `updated_at`.

Default recency half-life:

- `core`: no recency bonus beyond freshness tie-break
- `ongoing`: `14` days
- `daily`: `7` days

### Retrieval Fallback Planning

If first-pass hybrid retrieval is weak, the system may run a fallback retrieval-planning step.

Weak retrieval conditions include:

- no results above minimum score
- high contradiction density among top results
- query appears underspecified or entity-ambiguous
- reminder from transcript review: even when the top hit is correct, weak hybrid retrieval can still leak semantic-only junk into the tail, so future fallback/tuning work should treat low-support tail suppression as an explicit requirement rather than assuming semantic ranking is acceptable as-is

Fallback planning behavior:

- synthesize a tighter retrieval query
- optionally infer likely present-state versus historical intent
- optionally infer likely target entities

Method details:

- first-pass retrieval and weakness checks: local computation only
- fallback query and entity planning: separate LLM call only when first-pass retrieval is weak
- second-pass retrieval after fallback: local computation plus semantic search as configured

## Graph Memory Design

Graph memory is part of the memory system from day one.

Graph memory is implemented as:

- canonical relations embedded in Markdown frontmatter
- indexed relations in SQLite
- entity resolution over aliases and canonical names
- graph-aware retrieval layered into `memory_search`

Graph memory is not implemented as a separate graph database.

### Graph Semantics

- entities are resolved from `entity_refs`, relation subjects and objects, and alias lists
- relations have explicit status and time bounds
- graph search expands from matched entities
- graph search returns relation-backed snippets and linked documents

### Graph Expansion Rules

Default graph expansion depth:

- `1` hop in `auto` and `hybrid`
- `2` hops only when the tool explicitly requests `expand=2`

Default graph result cap:

- `20`

### Relation Conflict Rules

If a new `current` relation is written and:

- it has `cardinality=single`
- it shares the same `subject` and `predicate`
- it has a different `object`

Then all prior active `current` relations with the same `subject` and `predicate` are marked `superseded` and their `valid_to` is set.

If `cardinality=multi`, coexistence is allowed.

### Entity Resolution Rules

Resolution order:

1. exact canonical name match
2. exact alias match
3. normalized token match
4. fuzzy alias match above `0.92`
5. optional LLM entity extraction fallback if no deterministic match exists

## Bootstrap Injection Design

At session start, Jarvis already loads:

- `PROGRAM.md`
- `REACTOR.md`
- `USER.md`
- `ARMOR.md`

The memory system adds runtime memory bootstrap after those files and before any user turn content.

Prompt policy note:

- `PROGRAM.md` explicitly instructs Jarvis to use dedicated memory tools for memory operations
- Jarvis should not treat generic-tool memory access as a normal fallback path
- this design still requires reconciliation of out-of-band edits made by a human or external system

### Bootstrap Order

1. identity system messages
2. runtime core memory bootstrap
3. runtime ongoing memory bootstrap
4. session compaction seed, if present

### Bootstrap Message Role

- core memory bootstrap is a `system` message
- ongoing memory bootstrap is a `system` message

### Bootstrap Token Budgets

- core bootstrap hard cap: `2500` tokens
- ongoing bootstrap hard cap: `2500` tokens
- combined runtime memory bootstrap hard cap: `5000` tokens

### Bootstrap Selection Rules

For `core`:

- include active core docs only
- sort by `pinned desc`, `priority desc`, `updated_at desc`
- render `summary` first
- include fact bullets only if budget remains

For `ongoing`:

- include active ongoing docs only
- exclude expired docs unless `pinned=true`
- sort by `pinned desc`, `priority desc`, `updated_at desc`
- render `summary`, `current state`, and `open loops`

### Bootstrap Rendering

Bootstrap rendering is deterministic and does not call the LLM.

It renders:

- document title
- short summary
- selected current facts
- selected open loops for ongoing memory

## Memory Tools

Dedicated memory tools are required.

Generic file tools are not enough for the standard path because they do not provide:

- index-aware search
- graph-aware retrieval
- structured validation
- atomic write plus reindex
- conflict handling
- provenance handling
- bootstrap-aware mutation semantics

Jarvis should use the memory tools as the full operating surface for memory work.
If a human or external system changes canonical memory files out of band, the memory system must detect drift and reconcile those edits back into the indexed state.

### Tool Exposure

Basic tools:

- `memory_search`
- `memory_get`
- `memory_write`

Discoverable/admin tool:

- `memory_admin`

Tool policy note:

- `memory_search`, `memory_get`, and `memory_write` are always basic tools
- `memory_admin` is discoverable only
- `memory_admin` description should explicitly state that it must not be used unless the user explicitly requests memory administration

### `memory_search`

Purpose:

- search canonical memory using lexical, semantic, graph, or hybrid search

Arguments:

- `query`
- `mode`: `auto|lexical|semantic|graph|hybrid`
- `scopes`: list of `core|ongoing|daily|archive`
- `top_k`
- `daily_lookback_days`
- `expand`: `0|1|2`
- `include_expired`: boolean

Returns:

- ranked result list with
- `document_id`
- `path`
- `kind`
- `section_path`
- `score`
- `snippet`
- `match_reasons`
- `source_ref_ids`
- top-level `semantic_disabled` status when semantic retrieval is unavailable for the query
- warning text when hybrid search had to skip semantic retrieval and fall back

### `memory_get`

Purpose:

- open a full memory document or a specific section after discovery

Arguments:

- `document_id` or `path`
- `section_path` optional
- `include_frontmatter` boolean
- `include_sources` boolean

Returns:

- full Markdown text or requested section

### `memory_write`

Purpose:

- create or update canonical memory documents through validated structured operations

Arguments:

- `operation`: `create|upsert|append_daily|close|archive|promote|demote`
- `target_kind`: `core|ongoing|daily`
- `document_id` optional
- `title`
- `summary`
- `priority`
- `pinned`
- `locked`
- `review_after`
- `expires_at`
- `facts`
- `relations`
- `body_sections`
- `source_refs`

Behavior:

- validates payload
- reads canonical Markdown
- applies deterministic mutation
- `upsert` revises an existing canonical document and is the normal correction path when active memory is wrong
- `append_daily` appends a new daily entry instead of revising prior daily content
- daily corrections are explicit section rewrites: fetch the current daily doc with `memory_get`, then send replacement `body_sections`; summary-only daily `upsert` is rejected
- provided `body_sections` overwrite matching canonical sections; omitted sections remain unchanged
- provided `facts` and `relations` replace that document's structured truth sets; omitted ones remain unchanged
- writes Markdown
- updates or invalidates sidecar rows
- returns changed path and summary of mutations

### `memory_admin`

Purpose:

- manual maintenance and inspection

Actions:

- `reindex_all`
- `reindex_dirty`
- `rebuild_embeddings`
- `repair_canonical_drift`
- `run_due_maintenance`
- `integrity_check`
- `render_bootstrap_preview`

`memory_admin` is discoverable and not part of the normal agent turn surface.

## Generic File Tool Policy For Memory

`bash` and `file_patch` are allowed to read `/workspace/memory/`.

Generic tools may also write to `/workspace/memory/`, but that is outside Jarvis's intended memory workflow.

Agent-side write policy:

- Jarvis should use `memory_search`, `memory_get`, and `memory_write` for memory operations
- Jarvis should not use generic tools as a memory fallback path
- when memory files are changed by a human or external system through generic tools, those edits are treated as out-of-band memory mutations that must be detected and reconciled

Prompt policy:

- `PROGRAM.md` makes the memory-tool-only operating model explicit
- the prompt should not advertise generic-tool memory writes as a fallback path for Jarvis
- reconciliation of out-of-band edits remains required even though Jarvis is not instructed to use that path

If a canonical memory file is changed outside the dedicated memory tools by a human, by generic tools, or by out-of-band system action:

- the next memory service access computes checksum drift
- the file is added to `dirty_documents`
- the sidecar reparses and reindexes it before serving memory search results
- relation conflicts and schema validity are re-evaluated during reconciliation

No file watcher dependency is required.
Dirty detection is checksum-based and opportunistic.

## Memory Maintenance

Memory maintenance is a first-class part of the design.

There are three maintenance lanes:

1. immediate sync maintenance
2. post-turn reflection maintenance
3. due-time background maintenance

These lanes are the primary architecture.
Jarvis does not depend on a rigid daily/weekly/monthly cron pyramid.

However, Jarvis does use soft backstop sweeps.
These sweeps are opportunistic and threshold-based rather than hard calendar requirements.

Soft sweep examples:

- run a consolidation sweep if one has not run in roughly 24 hours or enough new daily material has accumulated
- run a broader review sweep if one has not run in roughly 7 days or enough stale active documents have accumulated
- run a cold archive sweep if one has not run in roughly 30 days or enough archive candidates have accumulated

These sweeps are safety nets, not the primary maintenance model.

All maintenance operations that require an LLM call must use a separate maintenance-model configuration from `src/jarvis/settings.py`.
They must not be implicitly tied to the main chat model used to power the live agent turn.

This includes:

- post-turn reflection planning
- pre-compaction reflection planning
- consolidation
- review
- summary refresh

The purpose of this split is:

- to keep maintenance cost controllable
- to allow a cheaper or more specialized model for background memory work
- to avoid coupling memory quality and maintenance behavior to the user-facing agent model choice

### Immediate Sync Maintenance

Runs after:

- `memory_write`
- `memory_admin reindex_*`
- dirty-file detection on a memory read/search call

Responsibilities:

- validate canonical Markdown
- update sidecar rows
- refresh chunks
- refresh fact and relation indexes
- refresh embeddings for changed items

Method details:

- canonical Markdown validation: local computation only
- frontmatter and section parsing: local computation only
- chunk refresh from canonical sections plus searchable summary-bearing structured fields: local computation only
- SQLite row updates: local computation and indexing only
- FTS refresh: local computation and indexing only
- graph relation refresh: local computation and indexing only
- embeddings for changed chunks, facts, and relations: separate embedding calls through `src/jarvis/llm/`
- no LLM generation calls occur in immediate sync maintenance
- this same sync path is used to reconcile out-of-band edits made by a human or external system

### Post-Turn Reflection Maintenance

Runs after each completed user turn.

It does not rewrite transcripts.
It looks at the completed turn evidence and decides whether there are memory actions worth applying.

Reflection output is a structured action set:

- `append_daily`
- `create_ongoing`
- `update_ongoing`
- `close_ongoing`
- `create_core`
- `update_core`
- `add_relation`
- `supersede_relation`
- `ignore`

Method details:

- transcript evidence loading: local computation only
- candidate extraction and memory action planning: separate LLM call through the maintenance LLM settings in `src/jarvis/settings.py`
- validation of the proposed action set: local computation only
- writes for approved actions: local computation through the dedicated memory write path
- sidecar refresh for changed documents: local computation plus embedding calls for changed indexed content
- reflection does not rewrite transcripts

### Post-Turn Auto-Apply Policy

`append_daily`:

- always auto-apply if reflection found a notable item

`ongoing` create or update:

- auto-apply when confidence is `medium` or `high`

`core` create or update:

- immediate core promotion is allowed only when the user explicitly asked Jarvis to remember it
- otherwise, proposed core promotions should wait for due review or consolidation
- non-explicit core promotion requires:
- repeated support across turns or sources
- no unresolved recent contradiction
- broad future usefulness
- a representation compact enough to stay bootstrap-worthy
- a manual memory edit may refresh an existing core document without re-earning promotion from scratch

`locked=true` documents:

- never auto-mutated

Method details:

- confidence thresholding: local computation only
- repeated-fact support checks across turns: local computation over transcript and provenance references
- final application of approved actions: local computation plus any required embedding refresh
- no additional LLM call is made here beyond the maintenance-LLM reflection planning call unless reflection output must be retried after validation failure

### Pre-Compaction Flush

Before session compaction runs:

- the memory system performs a final reflection pass over the soon-to-be-archived session
- applies any due daily or ongoing updates
- attempts core promotions that satisfy the auto-apply policy

This prevents compaction from being the only summarization layer.

Method details:

- session evidence loading: local computation only
- final reflection plan: separate LLM call through the maintenance LLM settings in `src/jarvis/settings.py`
- validation and policy gating: local computation only
- memory writes: local computation through the dedicated memory write path
- sidecar refresh: local computation plus embedding calls for changed indexed content
- this reflection step is separate from the existing context compaction LLM call

### Due-Time Background Maintenance

No separate cron service is required.

Due-time maintenance is opportunistic:

- it runs at startup
- it runs before serving a new user turn if due timestamps have passed
- it can be triggered manually through `memory_admin`

Jobs:

- `daily_rollover`
- `consolidate_recent_daily`
- `refresh_ongoing_summaries`
- `refresh_core_summaries`
- `recompute_priority_from_usage`
- `review_due_ongoing`
- `review_due_core`
- `expire_due_ongoing`
- `archive_closed_ongoing`
- `cold_archive_sweep`
- `integrity_check`
- `embedding_model_drift_check`
- `repair_missing_embeddings`

Per-job method details:

`daily_rollover`

- closes the current local-calendar daily file and ensures the new day file exists
- method: local computation and indexing only
- embedding calls: no
- LLM calls: no

`consolidate_recent_daily`

- reviews recent daily logs as a staging layer and merges redundant or closely related daily material
- may promote stable daily material into `ongoing/` or queue it for later core review
- method: local computation plus separate LLM consolidation call through the maintenance LLM settings in `src/jarvis/settings.py`
- embedding calls: only if promoted or rewritten indexed content changes
- LLM calls: yes

`refresh_ongoing_summaries`

- rewrites or tightens ongoing summaries from their current facts, relations, and recent daily support
- method: local computation plus separate LLM summarization call through the maintenance LLM settings in `src/jarvis/settings.py`
- embedding calls: only if summary-bearing documents change
- LLM calls: yes

`refresh_core_summaries`

- rewrites or tightens core summaries to keep them compact, stable, and bootstrap-efficient
- method: local computation plus separate LLM summarization call through the maintenance LLM settings in `src/jarvis/settings.py`
- embedding calls: only if summary-bearing documents change
- LLM calls: yes

`recompute_priority_from_usage`

- recomputes document priority hints from access patterns, support signals, contradiction signals, and recency
- method: local computation only
- embedding calls: no
- LLM calls: no

`review_due_ongoing`

- reviews active ongoing documents whose `review_after` is due
- decides whether to refresh, split, merge, close, or leave unchanged
- method: local computation plus separate LLM review call through the maintenance LLM settings in `src/jarvis/settings.py`
- embedding calls: only if a document changes
- LLM calls: yes

`review_due_core`

- reviews active core documents whose `review_after` is due
- decides whether they remain core-worthy, need refresh, should demote, or should expire
- method: local computation plus separate LLM review call through the maintenance LLM settings in `src/jarvis/settings.py`
- embedding calls: only if a document changes
- LLM calls: yes

`expire_due_ongoing`

- marks due ongoing documents expired or closed when `expires_at` has passed
- method: local computation only
- embedding calls: only if metadata refresh requires sidecar update
- LLM calls: no

`archive_closed_ongoing`

- moves closed ongoing documents into `archive/ongoing/` and updates index state
- method: local computation and indexing only
- embedding calls: no new embedding call required unless archive embeddings are rebuilt later
- LLM calls: no

`cold_archive_sweep`

- archives stale daily material after consolidation and moves inactive, already-rolled-up daily context out of the active search set
- may also archive demoted or expired low-value memory that no longer belongs in active search scope
- method: local computation only
- embedding calls: no unless archive indexes are rebuilt
- LLM calls: no

`integrity_check`

- validates Markdown schemas, cross-reference consistency, checksum state, and SQLite integrity
- method: local computation only
- embedding calls: no
- LLM calls: no

`embedding_model_drift_check`

- detects whether the configured embedding provider/model differs from the one recorded in sidecar metadata
- method: local computation only
- embedding calls: no for the check itself
- LLM calls: no
- if drift is detected, affected indexed content is marked dirty for later re-embedding or an admin-triggered rebuild

`repair_missing_embeddings`

- detects indexed memory documents whose expected embedding items are missing from the vector sidecar
- attempts to backfill those embeddings automatically when semantic indexing is physically available
- method: local computation plus embedding calls only for the missing items
- embedding calls: yes, but only for documents missing expected semantic rows
- LLM calls: no

### Heavy Reindex And Rebuild Operations

These are not scheduled by fixed calendar tier.
They are operator-triggered or condition-triggered.

`reindex_all`

- reparses all canonical memory Markdown and rebuilds all non-embedding indexes
- method: local computation and indexing only
- embedding calls: no unless the caller separately requests embedding rebuild
- LLM calls: no

`rebuild_embeddings`

- re-embeds all indexed chunks, facts, and relations
- method: embedding calls plus local indexing only
- embedding calls: yes
- LLM calls: no

`render_bootstrap_preview`

- renders the exact runtime memory bootstrap Jarvis would inject
- method: local computation only
- embedding calls: no
- LLM calls: no

## Promotion And Demotion Policy

### Promotion To Core

A memory item is eligible for `core` only if it is:

- durable
- broadly reusable
- behavior-shaping or identity-shaping
- costly to forget before retrieval

And one of:

- explicitly requested by the user
- strongly repeated across turns
- manually curated by a human

Promotion timing rule:

- explicit user-requested memory may promote immediately
- non-explicit promotion should normally occur during due review or consolidation, not routine turn-end reflection

Promotion safety rule:

- unresolved recent contradiction blocks promotion
- low-support, high-noise daily material should not promote directly to core
- anything too verbose to remain bootstrap-efficient should stay in `ongoing/` or `daily/`

### Demotion From Core

A core document is demoted when it is:

- stale
- superseded
- no longer broadly useful
- clearly temporary in hindsight

Demoted core documents move to:

- `ongoing/` if still active but no longer core-worthy
- `archive/core/` if no longer active

### Closing Ongoing Documents

An ongoing document is closed when:

- completion criteria are met
- it expires without renewal
- it is superseded by a replacement document

Closed ongoing documents move to:

- `archive/ongoing/`

## Daily Roll-Up And Retention Policy

`daily/` is a staging layer, not a forever-growing memory bucket.

Rules:

- daily documents are not long-term bootstrap memory
- only recent daily content inside the configured lookback window participates in normal active search by default
- older daily material should be consolidated into `ongoing/`, `core/`, or `archive/`
- daily items that remain unresolved but still active should be rolled into `ongoing/`
- stable, broadly reusable, contradiction-free material may be queued for later core review
- once daily material has been sufficiently rolled up, it may be archived out of the active set

Bootstrap rule:

- daily documents are not injected directly into session bootstrap
- only the information that has been rolled into `core/` or `ongoing/` may become bootstrap content

Retention rule:

- unrolled recent daily logs remain searchable within the active daily lookback window
- rolled-up or stale daily logs may be moved to archive and down-ranked from normal search

## Identity System Boundary

The memory system does not auto-edit:

- `/workspace/identities/PROGRAM.md`
- `/workspace/identities/REACTOR.md`
- `/workspace/identities/ARMOR.md`

`USER.md` remains part of the identity bootstrap system and is still treated as pinned bootstrap context, not as a normal runtime memory file.

Runtime-learned user facts should live in `/workspace/memory/core/` or `/workspace/memory/ongoing/`, not by rewriting identity files.

This preserves the distinction:

- identities are normative instructions
- runtime memory is descriptive state

## Transcript Boundary

Transcript archives remain under `/workspace/archive/transcripts/`.

The memory system may read transcript archives as evidence.
It must never modify them.

When memory stores provenance from transcripts, it stores references only:

- `route_id`
- `session_id`
- `record_id`

It does not copy whole transcript files into memory.

## Embeddings

Embeddings are produced through the existing embedding path in `src/jarvis/llm/`.

Memory embedding settings should be configurable separately from chat settings only if we later need that.
For now, the memory subsystem uses the global embedding provider and model already configured for Jarvis.

Embedding targets:

- changed document chunks
- changed facts
- changed relations

Re-embedding triggers:

- document content changed
- embedding model changed
- manual rebuild via `memory_admin`

## Config Surface

The memory system should define these runtime settings:

- `JARVIS_MEMORY_DIR`
- `JARVIS_MEMORY_INDEX_DIR`
- `JARVIS_MEMORY_MAINTENANCE_LLM_PROVIDER`
- `JARVIS_MEMORY_MAINTENANCE_LLM_MODEL`
- `JARVIS_MEMORY_MAINTENANCE_LLM_MAX_OUTPUT_TOKENS`
- `JARVIS_MEMORY_BOOTSTRAP_MAX_TOKENS`
- `JARVIS_MEMORY_CORE_BOOTSTRAP_MAX_TOKENS`
- `JARVIS_MEMORY_ONGOING_BOOTSTRAP_MAX_TOKENS`
- `JARVIS_MEMORY_SEARCH_DEFAULT_TOP_K`
- `JARVIS_MEMORY_DAILY_LOOKBACK_DAYS`
- `JARVIS_MEMORY_ENABLE_REFLECTION`
- `JARVIS_MEMORY_ENABLE_AUTO_APPLY_CORE`
- `JARVIS_MEMORY_ENABLE_AUTO_APPLY_ONGOING`
- `JARVIS_MEMORY_GRAPH_DEFAULT_EXPAND`

Default values:

- `JARVIS_MEMORY_DIR=/workspace/memory`
- `JARVIS_MEMORY_INDEX_DIR=/workspace/memory/.index`
- `JARVIS_MEMORY_MAINTENANCE_LLM_PROVIDER=openai`
- `JARVIS_MEMORY_MAINTENANCE_LLM_MODEL=<separate maintenance model>`
- `JARVIS_MEMORY_MAINTENANCE_LLM_MAX_OUTPUT_TOKENS=4000`
- `JARVIS_MEMORY_BOOTSTRAP_MAX_TOKENS=5000`
- `JARVIS_MEMORY_CORE_BOOTSTRAP_MAX_TOKENS=2500`
- `JARVIS_MEMORY_ONGOING_BOOTSTRAP_MAX_TOKENS=2500`
- `JARVIS_MEMORY_SEARCH_DEFAULT_TOP_K=8`
- `JARVIS_MEMORY_DAILY_LOOKBACK_DAYS=30`
- `JARVIS_MEMORY_ENABLE_REFLECTION=true`
- `JARVIS_MEMORY_ENABLE_AUTO_APPLY_CORE=true`
- `JARVIS_MEMORY_ENABLE_AUTO_APPLY_ONGOING=true`
- `JARVIS_MEMORY_GRAPH_DEFAULT_EXPAND=1`

Maintenance-model rule:

- any maintenance operation that requires an LLM call must resolve that call from the dedicated maintenance settings above
- it must not implicitly inherit the main agent chat provider or chat model

## `src/jarvis/memory/` Module Layout

The implementation code under `src/jarvis/memory/` should be split like this:

- `config.py`
- `types.py`
- `service.py`
- `markdown_store.py`
- `parser.py`
- `validator.py`
- `chunker.py`
- `index_db.py`
- `retrieval.py`
- `graph.py`
- `bootstrap.py`
- `reflection.py`
- `maintenance.py`
- `dirty_scan.py`

### Module Responsibilities

`config.py`

- environment parsing
- path resolution
- constants

`types.py`

- dataclasses and validation enums

`service.py`

- orchestration entry point

`markdown_store.py`

- canonical Markdown file reads and writes

`parser.py`

- YAML frontmatter and section parsing

`validator.py`

- schema validation

`chunker.py`

- deterministic chunking

`index_db.py`

- SQLite reads and writes

`retrieval.py`

- search pipeline and score fusion

`graph.py`

- entity resolution and graph expansion

`bootstrap.py`

- runtime memory bootstrap rendering

`reflection.py`

- post-turn reflection planning

`maintenance.py`

- due-time maintenance jobs

`dirty_scan.py`

- checksum drift detection

## Failure Modes

### Sidecar Corruption

If SQLite integrity fails:

- log the error
- move corrupt DB aside with a timestamp suffix
- rebuild from Markdown

### Invalid Markdown Schema

If a canonical memory file is invalid:

- do not index it
- record the validation error
- surface it through `memory_admin integrity_check`
- do not silently rewrite it

### Reflection Failure

If post-turn reflection fails:

- do not fail the user turn
- log the failure
- continue with existing memory state

### Embedding Failure

If embedding creation fails for a changed item:

- keep lexical and graph indexing
- mark semantic index incomplete or dirty
- do not fail hybrid search; hybrid must continue with non-semantic retrieval and surface a clear warning
- retry on the next maintenance pass, including the missing-embedding repair sweep

## Non-Goals

The memory system does not aim to:

- replace transcript archives
- mutate identities as ordinary memory
- become a separate graph database service
- expose raw SQL concepts to the agent
- keep hidden canonical state outside Markdown

## Open Design Decisions Explicitly Settled By This Doc

1. Canonical runtime memory lives under `/workspace/memory/`
2. Memory code lives under `src/jarvis/memory/`
3. Dedicated memory tools are required even though Markdown is canonical
4. Jarvis uses dedicated memory tools rather than generic file-tool fallback, while out-of-band edits must still be detected and reconciled
5. SQLite is a derived index and maintenance substrate, not a second source of truth
6. Semantic search uses SQLite plus vector support through `sqlite-vec` only when the runtime can actually load and use that vector path
7. Graph memory exists from day one as a relations layer, not a separate graph service
8. `ongoing/` is the medium-horizon memory abstraction
9. The memory model is explicitly `resources -> facts/relations -> summaries`
10. Maintenance is primarily event-driven and due-driven, with soft opportunistic sweeps rather than a rigid daily/weekly/monthly cron pyramid
11. Identity files remain outside ordinary memory mutation
12. The memory system is intentionally distributed across `core`, `tools`, `storage`, `identities`, and `memory`
