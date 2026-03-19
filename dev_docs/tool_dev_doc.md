# Tool Development Doc

## Purpose

This is the running source of truth for the `src/tools/` suite.

Update this document whenever:

- a tool is added
- a tool changes exposure class
- a tool schema changes
- a tool executor changes behavior
- a tool policy changes

Keep this doc structured and status-oriented. It is meant to be updated throughout the full tool-development phase, not written like a one-off article.

## Protocol And Structure

### Tool Packages

- Basic tools live in `src/tools/basic/<tool_name>/`
- Discoverable executable tools live in `src/tools/discoverable/<tool_name>/`
- Shared cross-tool interfaces stay at the top level of `src/tools/`
- Synthetic core runtime primitives such as subagent control live outside `src/tools/`; they are defined under `src/subagent/` and injected by the route runtime rather than registered as normal tools
- Current shared top-level modules:
  - `registry.py`
  - `runtime.py`
  - `policy.py`
  - `config.py`
  - `types.py`

### Registry

Responsibilities:

- register executable tools
- define exposure class for executable tools
- expose executable tool definitions to the LLM
- register discoverable catalog entries for `tool_search`
- resolve discoverable executable tools when `tool_search` activates them for the current turn

Exposure classes:

- `basic`: auto-exposed to the agent at the start of every session
- `discoverable`: not auto-exposed; the agent must find them through `tool_search`

Current rule:

- only `basic` tools are injected into the initial `LLMRequest`
- the session transcript also logs the raw basic `ToolDefinition` payloads available at session start for auditability
- that transcript-only tool-definition record is not replayed back into later `LLMRequest.messages`
- discoverable entries live in a separate catalog inside `ToolRegistry`
- discoverable entries may be docs-only or may link to a backing executable tool
- low-verbosity `tool_search` stays informational only
- high-verbosity `tool_search` may transiently surface matched backed discoverable tools for the rest of the current turn
- `ToolRegistry.filtered_view(...)` provides agent-scoped visibility so different actor types can share the same registry with different allowed tools
- the current subagent filtered view hides the built-in memory tools by settings-backed blocklist while still allowing runtime manifest discoverables to remain visible through `tool_search`
- every executable tool must have an explicit developer-decided subagent allow status from day 1
- in code, do not rely on default agent visibility for new tools; set `allowed_agent_kinds` explicitly on every `RegisteredTool` and every `DiscoverableTool` entry that can activate a backing tool
- treat subagent allow status as required tool-spec input during implementation, not as a later cleanup once subagent compatibility questions arise

### Discoverable Catalog

Each discoverable catalog entry is separate from `ToolDefinition`.

Purpose:

- let `tool_search` return richer discovery docs than normal tool schemas can express
- avoid forcing basic tools to carry search-only metadata they do not use
- support future discoverables that are not just normal JSON-schema LLM tools

Current discoverable entry shape:

- `name`
- `purpose`
- `aliases` optional
- `detailed_description` optional
- `usage` flexible
- `metadata` optional
- `backing_tool_name` optional

Current catalog rules:

- discoverable entries are what `tool_search` searches and formats
- discoverable entries are not automatically callable just because they exist in the catalog
- if `backing_tool_name` is set, it must point at a separately registered executable tool
- `usage` is intentionally flexible and does not need to mirror `ToolDefinition.input_schema`
- `usage` and `metadata` should be omitted by default when they do not add unique model-useful signal
- `usage` and `metadata` are returned by `tool_search`, but current search indexing only uses `name`, `aliases`, `purpose`, and `detailed_description`

### Discoverable Surface Simplification

Current built-in discoverables were deliberately simplified to reduce prompt noise.

Current baseline:

- discoverable surfaces are activation hints for the model, not human-facing code docs
- low verbosity should usually expose only name plus one-line `purpose`
- high verbosity should stay compact enough that surfacing several discoverables in one turn does not materially bloat context
- most backed discoverables now expose only `purpose`, a short alias set, one concise `detailed_description`, a minimal `usage.arguments` list, and `backing_tool_name`
- `examples`, long `notes`, large `metadata` blobs, and repeated default or limit explanations were removed from built-in discoverables unless they carry unique operational value
- docs-only discoverables may still put their primary operator instruction in `usage`, but that instruction should also stay compact

### Discoverable Description Surface Rules

Ethos for future discoverable-tool text:

- optimize for model runtime, not human documentation
- keep `purpose` to one search-friendly sentence
- keep `detailed_description` to the minimum needed for activation choice or mode selection
- for backed discoverables, `detailed_description` should normally reuse the executable `ToolDefinition.description`
- keep `usage` minimal; it should add only what is not already obvious from the name, purpose, long description, or executable schema
- do not include examples of obvious literals such as sample YouTube URLs, ordinary workspace paths, or trivial commands
- do not repeat the same default, limit, or path rule more than twice across description, argument descriptions, notes, and metadata combined; once is preferred
- if a detail matters for search ranking, put it in `purpose`, `aliases`, or `detailed_description`; `usage` and `metadata` are not indexed
- treat `metadata` as exceptional, not standard; omit it unless it adds unique operational signal the model would otherwise miss
- when deciding between adding another sentence and deleting text, bias toward deleting text

### How To Add A Discoverable Tool

There are two valid patterns. Pick the smallest one that matches the capability.

#### Pattern A: Backed Executable Discoverable Tool

Use this when the discoverable capability should eventually become an actual callable tool after the agent finds it through `tool_search`.

Required wiring:

1. Create the discoverable executable tool package under `src/tools/discoverable/<tool_name>/`.
2. Implement the executor and `build_<tool_name>_tool(...)` exactly like any other tool.
3. Set `exposure="discoverable"` on the returned `RegisteredTool`.
4. Explicitly set the tool's subagent allow status through `allowed_agent_kinds`; never leave this implicit.
5. Register that executable tool in `ToolRegistry.default(...)` with `registry.register(...)`.
6. Register a separate discoverable catalog entry with `registry.register_discoverable(DiscoverableTool(...))`.
7. Explicitly set the discoverable entry's `allowed_agent_kinds` to match the intended subagent allow status when it maps to a backing tool.
8. Set `backing_tool_name` on the discoverable entry to the executable tool name.
9. Add the policy branch in `src/tools/policy.py`.

Important:

- the executable tool registration and the discoverable catalog registration are separate steps by design
- if you forget step 9, the tool may appear in the follow-up request after `tool_search`, but runtime execution will still be denied by policy
- the recommended convention is for discoverable `name` and executable tool name to match unless there is a strong reason not to
- for backed discoverables, keep one long-form description source of truth by reusing the executable `ToolDefinition.description` text for the discoverable entry's `detailed_description` instead of maintaining two separate long descriptions
- do not mirror the same defaults, limits, or path rules across `purpose`, `detailed_description`, `usage`, and `metadata`; keep each field doing distinct work

Minimal mental model:

- `RegisteredTool` makes a tool executable
- `DiscoverableTool` makes a capability searchable through `tool_search`
- `backing_tool_name` is the bridge between those two worlds

#### Pattern B: Docs-Only / Skill-Backed Discoverable Entry

Use this when the capability should be discoverable but should not be surfaced as a callable tool.

Typical examples:

- a workflow that should be carried out through an agent skill
- a capability that is not implemented yet, but should still be discoverable as a roadmap hint
- a non-tool integration where `usage` should explain what the agent should do next
- any tool that does not use the backend execution style (i.e. fixed and pre-defined schema tool call)

Required wiring:

1. Register only a `DiscoverableTool` entry.
2. Leave `backing_tool_name=None`.
3. Put the real operator instructions in `usage`.

Behavior:

- `tool_search` can find and explain it
- high verbosity will return richer docs
- it will never be activated as a callable tool because there is no backing executable tool

### Discoverable Search Behavior

Current search intentionally stays simple and deterministic.

Behavior:

- empty query returns all discoverable entries sorted by name
- non-empty query is normalized to lowercase with collapsed whitespace
- search ranking favors exact `name` and `aliases` matches first
- substring hits in `name` / `aliases` rank above `purpose`
- `purpose` ranks above `detailed_description`
- token overlap gives additional score so multi-word queries can still match even when the words are separated

Non-goals for the current implementation:

- no embeddings
- no fuzzy package dependency
- no indexing of arbitrary nested `usage` / `metadata` payloads

### Discoverable Activation Flow

This is the current end-to-end flow for backed discoverable tools:

1. The initial turn request contains only `basic` tools.
2. The model calls `tool_search`.
3. If `tool_search` is used with `verbosity="low"`, the result is informational only.
4. If `tool_search` is used with `verbosity="high"`, matched discoverable entries with a `backing_tool_name` are written into tool-result metadata.
5. The agent loop scans current-turn pending tool results for that activation metadata.
6. The agent loop merges the resolved discoverable executable `ToolDefinition`s into the next follow-up `LLMRequest`.
7. Those discoverable tools are available for the rest of the current turn only.

Important:

- activation is transient by design and does not carry into future user turns
- this keeps initial tool lists small and avoids turning every discoverable capability into permanent context noise
- low verbosity is intentionally non-activating so the agent can scout first and expand later

### Runtime Tools

Runtime tools are discoverable manifest entries loaded dynamically from `/workspace/runtime_tools/*.json`.

Current runtime-tool rules:

- runtime tools are not repo-defined executable tools
- runtime tools are merged into `tool_search` results at execution time, not startup
- runtime tools are usually used through existing operators such as `bash`
- runtime tool registration happens through the basic `tool_register` tool
- runtime tool manifests are validated by `src/tools/runtime_tool_manifest.py`
- runtime tool loading is handled by `src/tools/runtime_tools.py`

Current runtime manifest shape:

- `name`
- `purpose`
- `aliases` optional
- `detailed_description` optional
- `usage` optional
- `notes` optional
- `operator`
- `invocation` optional
- `provisioning` optional
- `artifacts` optional
- `rebuild` optional
- `safety` optional

Important:

- runtime tool manifests are data only and do not create new repo executors
- `tool_search` should clearly label runtime entries with `source: runtime_tools`
- the manifest should contain enough information for another agent to rebuild the capability

### Recommended Test Coverage For New Discoverable Tools

When you add the first real discoverable executable tool, add tests for all of the following:

- the tool is registered with `exposure="discoverable"`
- the discoverable catalog entry is registered and searchable
- low-verbosity `tool_search` finds it without activating it
- high-verbosity `tool_search` returns richer docs and activation metadata
- the agent loop follow-up request includes the discoverable tool after the high-verbosity search
- the tool executes successfully once surfaced
- policy denies malformed inputs and allows valid ones

### Executor Runtime

Responsibilities:

- resolve a tool from the registry
- run policy before execution
- execute the tool
- return a normalized result object

Normalized tool result contract:

- `call_id`
- `name`
- `ok`
- `content`
- `metadata`

Important runtime behavior:

- executor failures are converted into structured tool-error results
- tool failures should feed back into the agent loop instead of crashing the turn

### Policy

Structure:

- `src/tools/policy.py` is the universal policy interface/router
- each basic tool owns its own policy logic under `src/tools/basic/<tool_name>/policy.py`
- each discoverable executable tool owns its own policy logic under `src/tools/discoverable/<tool_name>/policy.py`

Current active policy:

- `bash` is policy-checked in `dev` and then executed in the isolated `tool_runtime` container over internal HTTP; it scrubs the environment, shares only `/workspace`, enforces approval for install/build/system-mutation commands, and hard-denies pointless or harmful container-admin commands even inside `tool_runtime` unless `BASH_DANGEROUSLY_SKIP_PERMISSION=True`
- `python_interpreter` is policy-checked in `dev` and then executed in the isolated `tool_runtime` container over internal HTTP; it keeps the workspace-only script-path model, subprocess blocking, and no-network sandbox behavior
- `view_image` may only read explicit image files inside `/workspace`
- `tool_search` allows an optional short query and `low` / `high` verbosity only
- `tool_register` always requires exact-action approval and binds approval to the manifest payload hash

### Transcript And Follow-Up Tool Rounds

Current standard:

- assistant tool calls are stored as structured records plus metadata
- tool results are stored as structured records plus metadata
- approval requests and approval decisions are stored as structured records plus metadata
- internal tool-call payloads are not sent to user-facing Telegram output
- follow-up tool rounds are rebuilt into provider-native request shapes inside `src/llm/`
- `tool_search` may attach discoverable-activation metadata to its tool result
- only current-turn pending tool results can activate discoverable tools; activations do not persist across turns

Current provider-native follow-up handling:

- OpenAI: `function_call` + `function_call_output`
- Anthropic: `tool_use` + `tool_result`
- Gemini: `function_call` + `function_response` with preserved `thought_signature`
- OpenRouter: assistant `tool_calls` + `tool` role messages

### Standard Per-Tool Entry Format

When adding a new tool below, keep the tool entry in this shape:

- `Status`
- `Exposure`
- `Package`
- `Purpose`
- `Input Schema`
- `Executor Behavior`
- `Policy`
- `Current Limitations`

### Standard Discoverable Entry Format

When documenting a discoverable entry or a discoverable-capable tool below, keep the discoverable shape in mind:

- `Name`
- `Aliases`
- `Purpose`
- `Detailed Description`
- `Usage`
- `Metadata`
- `Backing Tool`

For backed discoverables, `Detailed Description` should normally mirror the executable tool's `ToolDefinition.description`; docs-only discoverables still define their own discoverable-only long description.
`Usage` is compact operator guidance, not a miniature tutorial. `Metadata` is exceptional and should usually be omitted.

## Tools Implemented

### `bash`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/bash/`
- Purpose: run bash commands inside the isolated `tool_runtime` container with `/workspace` as the shared handoff boundary

#### Input Schema

- `command: string` required
- `timeout_seconds: number` optional

#### Executor Behavior

- policy is evaluated in `dev`, but execution happens in the sibling `tool_runtime` container over internal HTTP
- runs directly in the `tool_runtime` container rather than inside a second inner sandbox layer
- mounts the shared workspace at `/workspace`
- the project repo is not mounted there, so `/repo` is absent by construction
- no app secrets are mounted in `tool_runtime`
- runs bash with `--noprofile --norc`
- clears the environment and sets only a minimal runtime env (`PATH`, `HOME`, `PWD`, `TMPDIR`, `LANG`, `LC_ALL`)
- keeps the `tool_runtime` container network available so tools like `curl` can still work
- `set -o pipefail` is enabled
- default timeout is `10s`
- max timeout is `30s`
- captures both `stdout` and `stderr`
- truncates large output to the configured cap
- returns a normalized tool result even when the command produces no `stdout`

#### Policy

- rejects empty commands
- rejects commands containing null bytes
- requires approval for install/build/system-mutation commands targeting the isolated `tool_runtime` container
- hard-denies upgrade, service/init-control, mount/kernel-admin, and container-runtime-recursion commands even inside `tool_runtime`

#### Current Limitations

- only `/workspace` is shared back to the app; if the command writes elsewhere, those changes stay local to the long-lived `tool_runtime` container
- command availability depends on what is installed in the runtime image

### `python_interpreter`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/python_interpreter/`
- Purpose: run constrained Python code or stored workspace scripts for parsing, tabular work, PDF/text extraction, image processing, and structured transformations that are awkward in shell

#### Input Schema

- `code: string | null` optional; exactly one of `code` or `script_path` is required
- `script_path: string | null` optional; exactly one of `code` or `script_path` is required
- `args: string[]` optional
- `read_paths: string[]` optional
- `write_paths: string[]` optional
- `timeout_seconds: number` optional

#### Executor Behavior

- policy is evaluated in `dev`, but execution happens in the sibling `tool_runtime` container over internal HTTP
- still runs inside `bubblewrap` in `tool_runtime` so network stays disabled and writes stay constrained to `/workspace`
- mounts the real workspace directly at `/workspace`
- only `/workspace` is writable; writes outside `/workspace` are denied by the sandbox
- disables network with `--unshare-net`
- only mounts the minimal Python runtime roots needed for the interpreter plus the tool-runtime Python environment
- executes through an internal runner that applies resource limits, blocks process-spawn APIs, and enforces a low-interference import trust model
- supports inline code and stored scripts under `/workspace`
- captures both `stdout` and `stderr`
- truncates large output to the configured cap

#### Security / Capability Ethos

- The python tool is capability-constrained, not package-by-package micromanaged.
- Hard security lives at the sandbox boundary first:
  - no network namespace
  - writable filesystem limited to `/workspace`
  - scrubbed environment
  - process spawning blocked
  - resource and output limits enforced
- Python-level guarding should be minimal and low-friction because invasive import hooks can break legitimate native and SWIG-backed libraries.
- Imports are therefore judged mainly by trust root, not by an ever-growing library exception list:
  - stdlib modules are allowed
  - packages installed in the dedicated curated venv are allowed
  - workspace-local helper modules are allowed so stored scripts can be composed normally
- The small hard import denylist is reserved for true escape hatches such as direct native FFI (`ctypes`, `_ctypes`, `cffi`, `_cffi_backend`).
- This means adding ordinary curated packages should usually require no policy changes; policy should only move when a new capability class is intentionally granted.

#### Policy

- exactly one of `code` or `script_path` must be provided
- `script_path` must stay inside `/workspace`
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- intentionally one-shot and stateless; no persistent kernel/session memory across tool calls
- `read_paths` and `write_paths` are deprecated compatibility fields and no longer control filesystem access
- third-party availability is defined by the curated package setting and the dependency closure of those installed packages in the tool-runtime Python environment
- workspace imports are intended for normal pure-Python helper modules; importing arbitrary native extensions from `/workspace` is intentionally disallowed

### `file_patch`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/file_patch/`
- Purpose: perform structured one-file text edits through explicit patch operations instead of shell editing

#### Input Schema

- `path: string` required
- `operations: array` required
- supported operation objects:
  - `{"type":"write","content":string}`
  - `{"type":"replace","old":string,"new":string}`
  - `{"type":"insert_before","anchor":string,"text":string}`
  - `{"type":"insert_after","anchor":string,"text":string}`
  - `{"type":"delete","text":string}`

#### Executor Behavior

- resolves exactly one workspace file path per call
- allows creating a new file or fully overwriting an existing file through a single `write` operation
- applies non-`write` operations sequentially in memory, using exact literal text matching only
- for broad holistic rewrites, the agent should usually prefer one `write` operation over many granular patch operations
- for small-to-medium targeted edits, the agent should usually prefer one `file_patch` call with a modest set of operations
- multiple `file_patch` calls in the same turn should be a fallback only when one patch payload would otherwise become too large or unreliable
- requires edit targets to match exactly once; missing or ambiguous matches fail the full call
- writes the final content atomically to avoid partial-file corruption on later-operation failure
- reads and writes UTF-8 text only

#### Policy

- `path` must be non-empty
- only explicit file paths inside `/workspace` are allowed
- `.env` files and paths inside `.env` directories are denied
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- v1 is text-only and UTF-8-only; it does not support binary patching
- `write` must be the only operation in the call
- parent directories must already exist; the tool does not create directories
- no regex, fuzzy matching, unified diff parsing, or multi-file patch sets

### `view_image`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/view_image/`
- Purpose: attach a local workspace image to the next model turn so multimodal providers can inspect it through a single tool path

#### Input Schema

- `path: string` required
- `detail: string` optional enum `auto | low | high | original`

#### Executor Behavior

- resolves relative paths from `/workspace`
- validates that the target exists and is a file
- inspects file bytes and only accepts image types shared across all current provider adapters
- returns a normalized tool result with `image_attachment` metadata so the agent loop can inject a transient multimodal follow-up message
- the image attachment is only guaranteed for the immediate tool-follow-up request and is not persisted into stored transcript history

#### Policy

- only explicit paths are allowed
- path must stay inside `/workspace`
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- currently limited to the common provider-safe MIME set: `image/png`, `image/jpeg`, and `image/webp`
- the tool does not upload provider file handles; it re-reads the local file and injects inline image data for the next model call only

### `send_file`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/send_file/`
- Purpose: send a local workspace file to the user through the Telegram file-send interface

#### Input Schema

- `path: string` required
- `caption: string` optional
- `filename: string` optional

#### Executor Behavior

- resolves relative paths from `/workspace`
- validates that the target exists and is a file
- uses the Telegram file-send interface already implemented under `src/ui/telegram/`
- prefers the active Telegram route when available and falls back to the configured owner chat when route context is unavailable
- returns a normalized tool result with delivery metadata including the resolved Telegram chat id when available

#### Policy

- only explicit paths are allowed
- path must stay inside `/workspace`
- `.env` files and paths inside `.env` directories are denied
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- depends on the Telegram UI runtime being configured with a valid bot token
- if the current route is not a Telegram route, delivery falls back to `JARVIS_UI_TELEGRAM_ALLOWED_USER_ID` when set

### `web_search`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/web_search/`
- Purpose: run a basic Brave web search query and return normalized web results for current information gathering

#### Input Schema

- `query: string` required

#### Executor Behavior

- reads `BRAVE_SEARCH_API_KEY` from runtime environment
- calls Brave `GET /res/v1/web/search`
- always forces `result_filter=web`
- disables Brave spellcheck so the query stays deterministic
- returns up to `JARVIS_TOOL_WEB_SEARCH_RESULT_COUNT` results (default `10`)
- normalizes each result to a concise structure centered on `title`, `url`, `snippet`, and source hostname
- includes query metadata such as `original`, `cleaned`, and `more_results_available` when Brave returns it
- converts request failures, missing API key errors, timeouts, and non-200 Brave responses into normalized tool-error results

#### Policy

- `query` must be non-empty
- query length must stay within Brave-compatible limits (`<= 400` chars and `<= 50` words)

#### Current Limitations

- intentionally minimal; no news, videos, locations, goggles, summary generation, or other Brave advanced features are exposed
- count is controlled by app settings, not by tool arguments
- no response-body fetching; downstream page reading belongs in `web_fetch`

### `web_fetch`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/web_fetch/`
- Purpose: fetch a specific web page and return clean markdown through a markdown-first three-tier strategy

#### Input Schema

- `url: string` required

#### Executor Behavior

- reads `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AI_WORKERS_REST_API_KEY` from runtime environment for HTML-to-markdown conversion
- Tier 1 requests the target URL with `Accept: text/markdown` and returns the response directly when usable markdown is available
- Tier 2 re-fetches the URL as normal HTML/text content and converts fetched HTML through Cloudflare `toMarkdown`
- Tier 3 uses local Playwright only as a fallback for JavaScript-heavy or render-dependent pages, then sends the rendered HTML through the same Cloudflare `toMarkdown` path
- manually validates redirect targets and rejects private / localhost / reserved destinations before following them
- caps fetched response bodies at `JARVIS_TOOL_WEB_FETCH_MAX_RESPONSE_BYTES` and truncates oversized markdown output at `JARVIS_TOOL_WEB_FETCH_MAX_MARKDOWN_CHARS`
- stores strategy, redirect chain, content type, markdown token count, and attempt summaries in normalized tool-result metadata

#### Policy

- `url` must be non-empty
- only absolute `http://` and `https://` URLs are allowed
- embedded credentials are denied
- localhost and literal private / loopback / reserved IP targets are denied

#### Current Limitations

- HTML conversion and browser-render fallback both depend on Cloudflare `toMarkdown`; if `CLOUDFLARE_ACCOUNT_ID` is missing, only the Tier 1 markdown-native path can succeed
- v1 intentionally excludes image conversion and general binary/document conversion paths
- browser rendering is fallback-only and does not expose multi-step browser automation

### `tool_search`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/tool_search/`
- Purpose: search the discoverable catalog at low or high verbosity and optionally surface backed discoverable tools for the current turn

#### Input Schema

- `query: string` optional; omit or pass empty string to list all discoverable tools
- `verbosity: string` optional; accepts `low` or `high`, defaults to `low`

#### Executor Behavior

- low verbosity returns only discoverable tool name and one-line purpose
- high verbosity returns the discoverable fields that are actually present on the entry; current built-in discoverables keep that surface intentionally sparse
- search is simple deterministic text matching across discoverable `name`, `aliases`, `purpose`, and `detailed_description`
- high-verbosity results add activation metadata so matched backed discoverable tools can be included in follow-up requests for the rest of the current turn
- low-verbosity results do not activate discoverable tools, to avoid inflating the tool list too early

#### Policy

- `query` is optional
- query length must stay within the configured hard caps
- `verbosity` must be `low` or `high`

#### Current Limitations

- activation only applies to discoverable entries that link to a real executable tool
- search does not currently index arbitrary `usage` or `metadata` payload contents

### `memory_search`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/memory_search/`
- Purpose: search canonical runtime memory through lexical, semantic, graph, or hybrid retrieval before opening or mutating memory

#### Input Schema

- `query: string` required
- `mode: string` optional enum `auto | lexical | semantic | graph | hybrid`
- `scopes: array[string]` optional enum members `core | ongoing | daily | archive`
- `top_k: integer` optional
- `daily_lookback_days: integer` optional
- `expand: integer` optional enum `0 | 1 | 2`
- `include_expired: boolean` optional

#### Executor Behavior

- routes all retrieval through the canonical memory service rather than raw filesystem reads
- reconciles checksum drift before search so out-of-band edits are reindexed opportunistically
- returns normalized ranked matches with `document_id`, `path`, `kind`, `section_path`, `score`, `snippet`, `match_reasons`, and `source_ref_ids`
- propagates `semantic_disabled=true` in metadata when `sqlite-vec` is unavailable and the runtime falls back to lexical + graph only

#### Policy

- `query` must be non-empty
- all other validation is delegated to the memory service

#### Current Limitations

- fallback retrieval planning for weak hybrid results is not yet implemented
- semantic search depends on `sqlite-vec`; on current container builds it may legitimately run in degraded lexical/graph mode

### `memory_get`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/memory_get/`
- Purpose: open a full canonical memory document or one named section after discovery

#### Input Schema

- `document_id: string` optional
- `path: string` optional
- `section_path: string` optional
- `include_frontmatter: boolean` optional
- `include_sources: boolean` optional

#### Executor Behavior

- resolves the target by `document_id` or canonical path through the memory index
- can return the full Markdown document, just the body, or a single named section
- can append a compact source-ref section for provenance inspection

#### Policy

- requires either `document_id` or `path`
- direct `path` access is restricted to the workspace memory directory

#### Current Limitations

- section access is by exact top-level section name only
- source rendering is intentionally compact and does not yet expand transcript provenance inline

### `memory_write`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/memory_write/`
- Purpose: create or update canonical memory documents through validated structured operations instead of generic file edits

#### Input Schema

- `operation: string` required enum `create | upsert | append_daily | close | archive | promote | demote`
- `target_kind: string` required enum `core | ongoing | daily`
- `document_id: string` optional
- `title: string` optional
- `summary: string` optional
- `priority: integer` optional
- `pinned: boolean` optional
- `locked: boolean` optional
- `review_after: string` optional
- `expires_at: string` optional
- `facts: array[object]` optional
- `relations: array[object]` optional
- `body_sections: object` optional
- `source_refs: array[object]` optional
- `date: string` optional
- `timezone: string` optional
- `close_reason: string` optional

#### Executor Behavior

- routes all normal memory mutations through the memory service, not raw file patching
- writes canonical Markdown, reindexes affected documents, refreshes lexical/graph state, and updates embeddings when semantic indexing is available
- supports explicit daily appends, ongoing close/archive flows, and cross-kind promote/demote migrations
- applies relation-conflict reconciliation so older `single` cardinality current relations are superseded when a newer conflicting relation is written

#### Policy

- `operation` and `target_kind` must be valid enums
- deeper payload validation is delegated to the memory service and canonical Markdown validator

#### Current Limitations

- the tool schema is intentionally non-strict because nested structured payloads are too broad for provider-strict JSON schema without a large nullable expansion
- automatic LLM-backed maintenance promotions and reviews are still conservative compared with the full design ambitions in `dev_docs/memory_doc.md`

### `memory_admin`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/memory_admin/`
- Purpose: run manual reindex, integrity, maintenance, embedding-rebuild, and bootstrap-preview actions when the user explicitly asks for memory administration

#### Input Schema

- `action: string` required enum `reindex_all | reindex_dirty | rebuild_embeddings | run_due_maintenance | integrity_check | render_bootstrap_preview`

#### Executor Behavior

- stays hidden by default and only becomes callable after `tool_search` high-verbosity activation
- runs the requested operator action through the shared memory service instance
- returns structured metadata for maintenance runs, integrity issues, and rebuild summaries

#### Policy

- `action` must be one of the implemented admin operations

#### Current Limitations

- the current due-maintenance lane includes local jobs plus explicit placeholders for the LLM-backed review/consolidation jobs that are not yet richly implemented
- admin actions are operator-grade but still route through the same in-process memory service rather than a separate maintenance worker

### Discoverable Entry: `memory_admin`

- Name: `memory_admin`
- Aliases: `memory maintenance`, `memory reindex`, `memory integrity`
- Purpose: expose manual memory maintenance capabilities without permanently inflating the default tool list
- Detailed Description: run explicit memory maintenance or inspection actions; use only when the user asks for memory admin work
- Usage: choose one of `reindex_all`, `reindex_dirty`, `rebuild_embeddings`, `repair_canonical_drift`, `run_due_maintenance`, `integrity_check`, or `render_bootstrap_preview`
- Metadata: none beyond the discoverable purpose/usage payload
- Backing Tool: `memory_admin`

### `generate_edit_image`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/generate_edit_image/`
- Purpose: generate a new image from a prompt or edit an existing workspace image through Gemini or OpenAI after discovery through `tool_search`

#### Input Schema

- `prompt: string` required
- `image_path: string` optional; include it only for edit mode
- `output_path: string` required; output file path inside `/workspace`
- `provider: string` optional enum `gemini | openai`; defaults to `gemini`
- `quality: string` optional enum `low | medium | high`; OpenAI only, defaults to `medium`
- `resolution: string` optional enum `512 | 1K | 2K | 4K`; Gemini only, defaults to `1K`

#### Executor Behavior

- stays hidden by default and only becomes callable after `tool_search` high-verbosity activation
- treats omitted `image_path` as generation mode and provided `image_path` as edit mode
- requires an explicit `output_path` inside `/workspace`
- creates missing parent directories for `output_path` automatically before writing the image
- appends a provider-safe image extension automatically when `output_path` has no suffix
- resolves edit inputs from `/workspace` and only accepts provider-safe image formats shared by the current implementation (`image/png`, `image/jpeg`, `image/webp`)
- defaults to Gemini with model `gemini-3.1-flash-image-preview`
- uses OpenAI model `gpt-image-1.5` when `provider="openai"`
- applies OpenAI `quality` only for the OpenAI path, defaulting to `medium`
- applies Gemini `resolution` only for the Gemini path, defaulting to `1K`
- forces Gemini image-only responses with `response_modalities=['Image']` so the tool does not intermittently receive text-only payloads
- returns normalized tool-result metadata including operation, provider, model, output path, MIME type, and provider usage metadata when available

#### Policy

- `prompt` must be non-empty and stay within the configured hard caps
- `output_path` must be non-empty
- `provider` must be either `gemini` or `openai`
- `quality`, when present, must be one of `low`, `medium`, or `high`
- `resolution`, when present, must be one of `512`, `1K`, `2K`, or `4K`
- `image_path` is optional, but if present it must be an explicit path inside `/workspace`
- `output_path` must also stay inside `/workspace`
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected for both `image_path` and `output_path`

#### Current Limitations

- v1 supports a single input image only; no masks or multi-image compositing yet
- output settings are intentionally minimal; the tool does not yet expose aspect ratio or background controls
- the tool saves the image locally but does not automatically send it to Telegram; a later `send_file` call is still needed for delivery

### `transcribe`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/transcribe/`
- Purpose: transcribe spoken audio from one workspace media file to plain text through OpenAI after discovery through `tool_search`

#### Input Schema

- `audio_path: string` required; workspace path to one supported local media file

#### Executor Behavior

- stays hidden by default and only becomes callable after `tool_search` high-verbosity activation
- uses OpenAI model `gpt-4o-mini-transcribe`
- uploads exactly one local media file and returns the transcript text directly in the tool result content
- pre-validates the file extension before upload and accepts the currently wired endpoint-safe formats: `flac`, `m4a`, `mp3`, `mp4`, `mpeg`, `mpga`, `ogg`, `wav`, `webm`
- pre-validates the current OpenAI Audio API upload limit of `25 MB` before sending the request
- tool guidance tells the agent to split larger sources into smaller chunks before transcription
- uses `response_format="json"` so the runtime can reliably extract transcript text from the structured response
- stores normalized metadata including model, input format, file size, transcript character count, and optional language/duration fields when returned

#### Policy

- `audio_path` must be non-empty
- `audio_path` must stay inside `/workspace`
- `audio_path` must use one of the supported filename extensions
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- v1 exposes only a single `audio_path` argument; it does not expose optional language, prompt, timestamp, or diarization controls
- files larger than `25 MB` must be trimmed, split, or converted first, typically via `ffmpeg`
- the tool relies on `OPENAI_API_KEY` at runtime and does not currently support alternate transcription providers

### `youtube`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/youtube/`
- Purpose: understand one or more public YouTube videos by URL through Gemini after discovery through `tool_search`

#### Input Schema

- `video_urls: string[]` required; list of one or more valid YouTube video URLs
- `objectives: string` optional; replaces the default summary task while preserving the tool's shared system instruction when provided
- `transcript: boolean` optional; defaults to `false`; when `true`, ignores `objectives` and returns transcript text for each video via Defuddle instead of running Gemini

#### Executor Behavior

- stays hidden by default and only becomes callable after `tool_search` high-verbosity activation
- when `transcript=false`, uses Gemini model `gemini-3-flash-preview`
- when `transcript=false`, sends each public YouTube URL to Gemini as a video input part and appends a short text prompt after the video parts, following the current Gemini video-understanding guidance
- when `transcript=false`, always keeps a built-in shared system instruction with universal context and guidelines
- when `transcript=false`, uses a built-in summary-oriented task objective by default
- when `transcript=false`, replaces only that default task objective when `objectives` is provided
- when `transcript=true`, ignores `objectives` and internally runs `curl` against `https://defuddle.md/<encoded-youtube-url>` for each provided URL, returning the transcript text as the tool result
- use `transcript=true` when exact dialogue or narration wording matters
- use `transcript=false` when the agent wants overview, focused question answering, or analysis that depends on visual content or broader audio cues beyond the transcript
- validates all `video_urls` before execution and returns an indexed invalid-URL error instead of making the Gemini call when any URL is malformed
- returns normalized metadata including provider, mode, video count, and response size; Gemini mode additionally records model, objectives source, and usage metadata when available

#### Policy

- `video_urls` must be a non-empty list
- every entry in `video_urls` must be a valid YouTube video URL
- the current implementation allows at most `10` video URLs per call

#### Current Limitations

- v1 only does simple regex validation; it does not verify that a valid-looking YouTube URL is actually reachable, public, or still available
- the tool is text-output only; transcript mode returns raw Defuddle text and Gemini mode does not expose structured extraction or timestamp-specific controls beyond what the caller writes into `objectives`
- Gemini analysis mode relies on `GOOGLE_API_KEY`; transcript mode instead depends on local `curl` plus Defuddle service availability

### `email`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/email/`
- Purpose: send one email through the configured SMTP account after discovery through `tool_search`

#### Input Schema

- `to_email: string` required; one recipient email address
- `subject: string` required; one non-empty subject line
- `body: string` required; markdown email body
- `attachment_paths: string[]` optional; workspace file paths to attach

#### Executor Behavior

- stays hidden by default and only becomes callable after `tool_search` high-verbosity activation
- is main-agent-only; subagents cannot discover or execute it
- authenticates and sends through the configured SMTP account using `SENDER_EMAIL_ADDRESS` as both the login username and the `From` address
- renders the markdown `body` into HTML, also includes a plain-text alternative, and always appends a `Sent by Jarvis` footer to both variants
- attaches any provided local workspace files and guesses attachment MIME types from their filenames
- returns normalized metadata including SMTP host, port, security mode, sender, recipient, subject, attachment count, and message id

#### Policy

- every send requires explicit approval matched against an exact request hash
- `to_email` must be a valid single email address
- `subject` and `body` must be non-empty and stay within the configured hard caps
- `attachment_paths`, when provided, must be explicit file paths inside `/workspace`
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected for attachments
- `.env` files or paths inside `.env` directories are rejected as attachments
- total attachment size and attachment count must stay within the configured hard caps

#### Current Limitations

- supports one `to_email` per call only; there is no `cc` or `bcc` surface yet
- does not support inline images, remote URLs as attachments, or arbitrary custom SMTP identities per call
- the built-in defaults target Gmail over SMTP SSL unless overridden by environment configuration

### `ffmpeg`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/ffmpeg/`
- Purpose: help the agent discover that ffmpeg and ffprobe are already installed and should be used through the basic `bash` tool

#### Discoverable Entry

- Name: `ffmpeg`
- Aliases: `ffprobe`, `media convert`
- Purpose: use the installed ffmpeg or ffprobe CLI through bash for audio or video conversion, trimming, muxing, probing, and stream extraction
- Detailed Description: docs-only entry; run `ffmpeg` or `ffprobe` through `bash`
- Usage: use the basic `bash` tool to run `ffmpeg` or `ffprobe` on files in `/workspace`
- Metadata: none
- Backing Tool: none

#### Current Limitations

- not a callable tool; `tool_search` can explain it but cannot activate a separate runtime
- success depends on constructing a valid `bash` command and keeping file paths inside `/workspace`

## Tools To Be Implemented

### Basic Tools

- No additional basic tools currently planned beyond the memory tool surface added in this pass.

### Discoverable Tools

These should stay hidden by default and only be surfaced through `tool_search`.

#### `tabular_query`

- Purpose: query and transform CSV, JSON, and SQLite-style tabular data through a higher-level interface

#### `workspace_index`

- Purpose: build and query a structured index of workspace files so the agent can find relevant project artifacts faster than raw shell search alone

## Current Snapshot

- Implemented tools: `bash`, `file_patch`, `memory_search`, `memory_get`, `memory_write`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`, `tool_search`, `email`, `memory_admin`, `generate_edit_image`, `transcribe`, `youtube`
- Implemented basic tools: `bash`, `file_patch`, `memory_search`, `memory_get`, `memory_write`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`, `tool_search`
- Implemented discoverable tools: `email`, `ffmpeg` (docs-only), `memory_admin`, `generate_edit_image`, `transcribe`, `youtube`
