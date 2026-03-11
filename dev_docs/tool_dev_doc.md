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
- discoverable entries live in a separate catalog inside `ToolRegistry`
- discoverable entries may be docs-only or may link to a backing executable tool
- low-verbosity `tool_search` stays informational only
- high-verbosity `tool_search` may transiently surface matched backed discoverable tools for the rest of the current turn

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
- `usage` and `metadata` are returned by `tool_search`, but current search indexing only uses `name`, `aliases`, `purpose`, and `detailed_description`

### How To Add A Discoverable Tool

There are two valid patterns. Pick the smallest one that matches the capability.

#### Pattern A: Backed Executable Discoverable Tool

Use this when the discoverable capability should eventually become an actual callable tool after the agent finds it through `tool_search`.

Required wiring:

1. Create the discoverable executable tool package under `src/tools/discoverable/<tool_name>/`.
2. Implement the executor and `build_<tool_name>_tool(...)` exactly like any other tool.
3. Set `exposure="discoverable"` on the returned `RegisteredTool`.
4. Register that executable tool in `ToolRegistry.default(...)` with `registry.register(...)`.
5. Register a separate discoverable catalog entry with `registry.register_discoverable(DiscoverableTool(...))`.
6. Set `backing_tool_name` on the discoverable entry to the executable tool name.
7. Add the policy branch in `src/tools/policy.py`.

Important:

- the executable tool registration and the discoverable catalog registration are separate steps by design
- if you forget step 7, the tool may appear in the follow-up request after `tool_search`, but runtime execution will still be denied by policy
- the recommended convention is for discoverable `name` and executable tool name to match unless there is a strong reason not to

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

- `bash` only does thin validation up front; actual access control is enforced by a `bubblewrap` sandbox that mounts the workspace at `/workspace`, scrubs the environment, and omits `/repo` and `/run/secrets`
- `view_image` may only read explicit image files inside `/workspace`
- `tool_search` allows an optional short query and `low` / `high` verbosity only

### Transcript And Follow-Up Tool Rounds

Current standard:

- assistant tool calls are stored as structured records plus metadata
- tool results are stored as structured records plus metadata
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

## Tools Implemented

### `bash`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/basic/bash/`
- Purpose: run bash commands inside a real sandbox with workspace-only user data access

#### Input Schema

- `command: string` required
- `timeout_seconds: number` optional

#### Executor Behavior

- runs inside `bubblewrap`
- mounts the real workspace at `/workspace`
- binds `/tmp` to a workspace-owned internal temp directory
- mounts `/usr` and the full system `/etc` read-only so dynamically linked CLI tools can resolve loader config, alternatives, certificates, and other runtime metadata
- does not mount `/repo` or `/run/secrets`
- runs bash with `--noprofile --norc`
- clears the environment and sets only a minimal runtime env (`PATH`, `HOME`, `PWD`, `TMPDIR`, `LANG`, `LC_ALL`)
- keeps the container network available so tools like `curl` can still work
- `set -o pipefail` is enabled
- default timeout is `10s`
- max timeout is `30s`
- captures both `stdout` and `stderr`
- truncates large output to the configured cap
- returns a normalized tool result even when the command produces no `stdout`

#### Policy

- rejects empty commands
- rejects commands containing null bytes
- all filesystem boundary enforcement is delegated to the `bubblewrap` sandbox rather than command parsing

#### Current Limitations

- not a full general-purpose container shell: user-controlled data access is limited to `/workspace`
- system runtime paths such as `/usr`, `/etc`, and `/dev` are present, but `/etc` is mounted read-only and user-writable data is still limited to `/workspace`
- command availability depends on what is installed in the runtime image
- requires `bubblewrap` to be installed in the runtime environment

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

- runs inside `bubblewrap` with a dedicated interpreter venv created at container build time
- mounts the real workspace directly at `/workspace`
- only `/workspace` is writable; writes outside `/workspace` are denied by the sandbox
- disables network with `--unshare-net`
- only mounts the minimal Python runtime roots needed for the interpreter plus the dedicated venv
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
- third-party availability is defined by the curated package setting and the dependency closure of those installed packages in the dedicated venv
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
- high verbosity returns richer discovery docs including aliases, detailed description, flexible usage payload, metadata, and backing tool name when present
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
- files larger than `25 MB` must be trimmed, split, or converted first, typically via `ffmpeg_cli`
- the tool relies on `OPENAI_API_KEY` at runtime and does not currently support alternate transcription providers

### `ffmpeg_cli`

- Status: implemented
- Exposure: `discoverable`
- Package: `src/tools/discoverable/ffmpeg_cli/`
- Purpose: help the agent discover that ffmpeg and ffprobe are already installed and should be used through the basic `bash` tool

#### Discoverable Entry

- Name: `ffmpeg_cli`
- Aliases: `ffmpeg`, `ffprobe`, `media_convert`
- Purpose: use the installed ffmpeg or ffprobe CLI through bash for audio or video conversion, trimming, muxing, probing, and stream extraction
- Detailed Description: this is a docs-only discoverable entry with no separate runtime; after discovery, invoke `ffmpeg` or `ffprobe` through the basic `bash` tool
- Usage: use the basic `bash` tool to run `ffmpeg` or `ffprobe` directly inside the container, for example `ffmpeg -i /workspace/input.mp4 /workspace/output.mp3`, and keep all input and output paths inside `/workspace`
- Metadata: `operator=bash`, `commands=[ffmpeg, ffprobe]`, `runtime=docs_only_discoverable`
- Backing Tool: none

#### Current Limitations

- not a callable tool; `tool_search` can explain it but cannot activate a separate runtime
- success depends on constructing a valid `bash` command and keeping file paths inside `/workspace`

## Tools To Be Implemented

### Basic Tools

None currently planned.

### Discoverable Tools

These should stay hidden by default and only be surfaced through `tool_search`.

#### `view_youtube`

- Purpose: get a summary of a given youtube video.

#### `tabular_query`

- Purpose: query and transform CSV, JSON, and SQLite-style tabular data through a higher-level interface

#### `workspace_index`

- Purpose: build and query a structured index of workspace files so the agent can find relevant project artifacts faster than raw shell search alone

## Current Snapshot

- Implemented tools: `bash`, `file_patch`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`, `tool_search`, `generate_edit_image`, `transcribe`
- Implemented basic tools: `bash`, `file_patch`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`, `tool_search`
- Implemented discoverable tools: `ffmpeg_cli` (docs-only), `generate_edit_image`, `transcribe`
