# Agent Memory

## Purpose

Jarvis memory is the long-term runtime memory layer for the agent.

It is not a transcript store, not the identity system, and not a generic file-editing convention. It is a coordinated subsystem made of:

- `src/jarvis/identities/`: pinned procedural/bootstrap instruction memory
- `src/jarvis/storage/`: raw episodic evidence in archived transcripts
- `src/jarvis/memory/`: long-term semantic, ongoing, daily, and graph-oriented memory
- `src/jarvis/tools/basic/memory_*`: the agent-facing memory tools
- `src/jarvis/core/`: memory bootstrap, reflection, and pre-compaction flush integration

Canonical runtime memory files live under `/workspace/memory/`. The SQLite index is derived operational state and can be rebuilt from canonical Markdown.

## Core Principles

1. Canonical memory state is Markdown under `/workspace/memory/`.
2. SQLite is a derived operational index, not the source of truth.
3. The agent and human inspect canonical memory through Markdown.
4. Jarvis uses dedicated memory tools for search, inspection, and mutation.
5. Out-of-band edits are detected by checksum drift and reconciled opportunistically.
6. Graph memory is a lightweight relations layer in Markdown frontmatter and SQLite.
7. Identity files are not ordinary memory files and are not auto-mutated.
8. Transcript archives remain immutable evidence.
9. Memory search supports lexical, graph, semantic, and hybrid retrieval with degraded fallback.
10. Maintenance is mostly automatic, while manual admin stays available through `memory_admin`.

## Memory Categories

### Core

Core memory is the smallest runtime-learned set that should be injected at session start without a search.

Core memory is durable, broadly reusable, behavior-shaping, and costly to forget before retrieval. It is not raw history, verbose evidence, temporary planning, or one-off task state.

### Ongoing

Ongoing memory is medium-horizon state:

- active projects
- current life context
- temporary routines
- active commitments
- unresolved open loops
- current operational state that matters across turns

Every ongoing document has review and expiry semantics.

### Daily

Daily memory is distilled episodic memory. It is a daily log of notable events, decisions, commitments, artifacts, and candidate promotions extracted from usage. It is a staging layer, not a transcript mirror and not a forever-growing bootstrap source.

## Runtime Layout

```text
/workspace/memory/
├── core/
│   └── <slug>.md
├── ongoing/
│   └── <slug>.md
├── daily/
│   └── YYYY-MM-DD.md
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

Canonical memory filenames use stable kebab-case names. Daily files use local calendar dates.

## Markdown Schemas

Canonical memory is Markdown with YAML frontmatter. UTF-8, LF line endings, and one document per file are expected.

Core and ongoing documents include frontmatter fields such as:

- `memory_id`
- `kind`
- `title`
- `status`
- `created_at`
- `updated_at`
- `priority`
- `pinned`
- `locked`
- `confidence`
- `review_after`
- `expires_at`
- `tags`
- `aliases`
- `facts`
- `relations`
- `source_refs`
- optional `summary`
- optional `entity_refs`

Ongoing documents may also include `completion_criteria` and `close_reason`.

Daily documents include:

- `memory_id`
- `kind=daily`
- `date`
- `timezone`
- `status`
- `created_at`
- `updated_at`
- `route_ids`
- `session_ids`

Daily body sections:

- `## Notable Events`
- `## Decisions`
- `## Active Commitments`
- `## Open Loops`
- `## Artifacts`
- `## Candidate Promotions`

## Facts, Relations, Entities, And Sources

Facts are atomic descriptive statements. Relations are structured entity-linked statements.

Fact fields include:

- `fact_id`
- `text`
- `status`: `current`, `past`, `uncertain`, or `superseded`
- `confidence`
- first/last seen timestamps
- valid-from/valid-to timestamps
- `source_ref_ids`

Relation fields include:

- `relation_id`
- `subject`
- `predicate`
- `object`
- `status`
- `confidence`
- `cardinality`: `single` or `multi`
- first/last seen timestamps
- valid-from/valid-to timestamps
- `source_ref_ids`

Entity refs carry `entity_id`, `name`, `entity_type`, and `aliases`.

Source refs connect memory back to transcript, manual, tool, import, or maintenance evidence.

## SQLite Sidecar

The sidecar supports:

- FTS5 lexical search
- `sqlite-vec` semantic search
- graph traversal and relation lookup
- dirty-file detection
- ranking
- provenance joins
- access logging
- maintenance bookkeeping
- bootstrap caching

Main tables include:

- `resources`
- `documents`
- `document_chunks`
- `document_chunks_fts`
- `facts`
- `relations`
- `entities`
- `source_refs`
- source-ref join tables
- `access_log`
- `maintenance_runs`
- `dirty_documents`
- `bootstrap_cache`
- `schema_info`

If `sqlite-vec` is unavailable, semantic retrieval is disabled and hybrid search falls back to lexical plus graph. Tool metadata and result text report `semantic_disabled=true` when this happens.

## Indexing

Chunking is deterministic.

Core and ongoing documents are chunked by section, then paragraph, then sentence groups when needed. Daily logs are chunked by daily section and adjacent bullet groups.

Embeddings are created for:

- chunk text
- fact text
- relation textualizations in the form `subject predicate object`

Raw YAML blobs, source-reference notes alone, and empty sections are not embedded.

## Retrieval

Search modes:

- `lexical`
- `semantic`
- `graph`
- `hybrid`
- `auto`

Default search is hybrid over active `core`, `ongoing`, and recent `daily` memory. Archive is excluded unless requested.

Hybrid retrieval combines:

- FTS lexical candidates
- semantic candidates when embeddings are ready
- graph candidates
- score fusion with kind, pinned, expiry, archive, stale ongoing, support, contradiction, and recency signals

Lexical retrieval uses query planning instead of passing raw user text directly to FTS5. It produces phrase, keyword, and prefix variants, escapes FTS syntax, and preserves match reasons.

Semantic retrieval carries real vector distance/similarity through fusion.

Graph retrieval resolves entities through canonical names, aliases, tokens, and short n-grams from the query. Current relations rank above past, uncertain, and superseded relations.

Weak semantic-only tails are suppressed so unrelated semantic matches do not pollute otherwise useful hybrid results.

## Memory Tools

### `memory_search`

Searches canonical memory through lexical, semantic, graph, or hybrid retrieval.

Arguments include:

- `query`
- `mode`
- `scopes`
- `top_k`
- `daily_lookback_days`
- `expand`
- `include_expired`

Result text includes enough model-visible information to decide whether to call `memory_get`: title, path, kind, section, score, reasons, warnings, and snippet.

### `memory_get`

Opens a full memory document or one section by `document_id` or path.

The default output is body-oriented and omits frontmatter unless `include_frontmatter=true`. Synthetic sections `facts` and `relations` are supported.

### `memory_write`

Creates or updates canonical memory through structured operations:

- `create`
- `upsert`
- `append_daily`
- `close`
- `archive`
- `promote`
- `demote`

The tool supports structured fields including aliases, tags, entity refs, completion criteria, facts, relations, body sections, and source refs.

`upsert` revises existing canonical documents. `append_daily` appends a new daily entry. Daily corrections use explicit section rewrites through `body_sections`.

Relation-conflict reconciliation supersedes older `single` cardinality current relations when a newer conflicting current relation is written.

### `memory_admin`

`memory_admin` is discoverable only. It is intended for explicit user-requested administration:

- `reindex_all`
- `reindex_dirty`
- `rebuild_embeddings`
- `repair_canonical_drift`
- `run_due_maintenance`
- `integrity_check`
- `render_bootstrap_preview`

## Generic File Tools And Memory

Generic tools may physically read or write `/workspace/memory`, but Jarvis's normal memory workflow uses only memory tools.

If a human or external system changes a canonical memory file out of band:

- the next memory service access detects checksum drift
- the file is marked dirty
- the sidecar reparses and reindexes it before serving results
- relation conflicts and schema validity are re-evaluated

No file watcher is required.

## Bootstrap

Runtime memory bootstrap is injected after identity bootstrap and before user turn content.

Order:

1. identity system messages
2. core memory bootstrap
3. ongoing memory bootstrap
4. compacted replacement-history records, when present

Core and ongoing bootstrap are system messages.

Token caps:

- core: `2500`
- ongoing: `2500`
- combined: `5000`

Core bootstrap includes active core docs sorted by pinned, priority, and updated time. It renders summary first, then selected current facts and relations when useful.

Ongoing bootstrap includes summary, current state, open loops, and freshness hints for active ongoing docs. Expired ongoing docs are excluded unless pinned.

Bootstrap rendering is deterministic and does not call an LLM.

## Reflection And Maintenance

Memory maintenance has three lanes:

- immediate sync maintenance
- post-turn reflection maintenance
- due-time background maintenance

Immediate sync runs after memory writes, admin reindex actions, and dirty-file detection. It validates Markdown, updates SQLite, refreshes chunks, facts, relations, and embeddings. It does not call a generation model.

Post-turn reflection runs after completed user turns. It uses the maintenance model, not the main chat model, to decide whether to apply memory actions. Routine turns can be ignored. The planner receives compact active-memory context and prefers updating existing documents over creating duplicates.

Pre-compaction flush runs a final reflection pass over the soon-to-be-archived session before compaction.

Due-time maintenance runs opportunistically at startup, before new user turns when due, and through `memory_admin`.

Jobs include:

- `daily_rollover`
- `consolidate_recent_daily`
- `refresh_ongoing_summaries`
- `refresh_core_summaries`
- `review_due_ongoing`
- `review_due_core`
- `expire_due_ongoing`
- `archive_closed_ongoing`
- `cold_archive_sweep`
- `integrity_check`
- `embedding_model_drift_check`
- `repair_missing_embeddings`

`locked=true` documents are not auto-mutated by reflection or maintenance.

## Promotion And Retention

Core promotion is reserved for durable, broadly reusable, behavior-shaping memory that is costly to forget.

Immediate core promotion is allowed when the user explicitly asks Jarvis to remember something. Non-explicit promotion normally happens during review or consolidation and requires repeated support, no unresolved contradiction, broad future usefulness, and a compact representation.

Ongoing documents close when completion criteria are met, they expire without renewal, or they are superseded. Closed ongoing documents move to `archive/ongoing/`.

Daily memory is periodically rolled up into ongoing/core candidates or archived after consolidation.

## Maintenance Rules

When changing memory behavior:

- keep Markdown canonical
- keep SQLite rebuildable
- do not use generic file edits as the standard memory workflow
- preserve degraded behavior when semantic search is unavailable
- expose model-useful context in tool result content, not only metadata
- keep bootstrap deterministic and token-capped
- do not auto-mutate locked documents
- update tests for retrieval, tool output, bootstrap, reflection, and maintenance changes
