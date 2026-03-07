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

- Each tool lives in its own package under `src/tools/<tool_name>/`
- Shared cross-tool interfaces stay at the top level of `src/tools/`
- Current shared top-level modules:
  - `registry.py`
  - `runtime.py`
  - `policy.py`
  - `config.py`
  - `types.py`

### Registry

Responsibilities:

- register tools
- define exposure class
- expose tool definitions to the LLM
- support future tool discovery

Exposure classes:

- `basic`: auto-exposed to the agent at the start of every session
- `discoverable`: not auto-exposed; the agent must find them through `tool_search`

Current rule:

- only `basic` tools are injected into every `LLMRequest`
- discoverable tools are hidden until explicitly surfaced later

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
- each tool owns its own policy logic under `src/tools/<tool_name>/policy.py`

Current active policy:

- `bash` may read across the container filesystem except `.env` paths
- `bash` may only write inside `/workspace`, and `.env` paths are denied even there
- `view_image` may only read explicit image files inside `/workspace`

### Transcript And Follow-Up Tool Rounds

Current standard:

- assistant tool calls are stored as structured records plus metadata
- tool results are stored as structured records plus metadata
- internal tool-call payloads are not sent to user-facing Telegram output
- follow-up tool rounds are rebuilt into provider-native request shapes inside `src/llm/`

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

## Tools Implemented

### `bash`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/bash/`
- Purpose: run validated shell commands inside the container for file inspection and workspace-limited file manipulation

#### Input Schema

- `command: string` required
- `timeout_seconds: number` optional

#### Executor Behavior

- runs with `/bin/bash -lc`
- default working directory is `/workspace`
- `set -o pipefail` is enabled
- default timeout is `10s`
- max timeout is `30s`
- captures both `stdout` and `stderr`
- truncates large output to the configured cap
- returns a normalized tool result even when the command produces no `stdout`

#### Policy

Filesystem rule:

- reads allowed anywhere in the container except `.env` paths
- writes allowed only inside `/workspace`, except `.env` paths are denied there too

Allowed read / inspect commands:

- `pwd`
- `ls`
- `find`
- `stat`
- `file`
- `du`
- `cat`
- `head`
- `tail`
- `grep`
- `rg`
- `wc`
- `cut`
- `sort`
- `uniq`
- `diff`
- `printf`
- `echo`

Allowed write / mutate commands:

- `mkdir`
- `touch`
- `cp`
- `mv`
- `rm`
- `truncate`
- `tee`
- `sed`

Write restrictions:

- `cp` may read from anywhere, but its destination must be inside `/workspace`
- `mv` may only move paths entirely inside `/workspace`
- `rm`, `mkdir`, `touch`, `truncate`, `tee`, and `sed -i` may only target paths inside `/workspace`
- any explicit `.env` path is denied for both reads and writes
- relative write paths resolve from `/workspace`
- write operands using `~`, `*`, `?`, or `[` are rejected

Allowed shell syntax:

- plain command invocation
- quoted arguments
- pipelines with `|`

Rejected shell syntax:

- `;`
- `&&`
- `||`
- `>`
- `>>`
- `<`
- heredocs
- command substitution
- subshells
- environment-variable expansion
- background execution
- multiline command strings

Command-specific restrictions:

- `find` rejects `-delete`, `-exec`, `-execdir`, `-ok`, `-okdir`, `-fprint`, `-fprint0`, `-fprintf`, `-fls`
- `find` may not explicitly target `.env` via path/name/regex predicates
- `grep` rejects recursive forms like `-r`, `-R`, and `--recursive`
- `rg` always runs with a config that excludes `.env`, rejects `--no-config`, and rejects glob filters that would re-include `.env`
- `sort` rejects `-o` and `--output`
- `sed` only allows read-only `-n` line printing and limited `sed -i` substitution
- `cp` / `mv` reject target-directory forms like `-t`, `--target-directory`, `-T`, `--no-target-directory`

#### Current Limitations

- intentionally conservative; many harmless shell patterns are blocked
- sequential tool execution only
- no parallel shell tool calls yet
- if more expressive editing is needed later, prefer a dedicated edit tool over relaxing shell policy too much

### `python_interpreter`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/python_interpreter/`
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
- uses a staged `/workspace` copy instead of mounting the real workspace directly
- mounts the staged workspace read-only, then re-binds only explicit `write_paths` as writable
- disables network with `--unshare-net`
- only mounts the minimal Python runtime roots needed for the interpreter plus the dedicated venv
- executes through an internal runner that applies resource limits and blocks process-spawn APIs/imports
- supports inline code and stored scripts under `/workspace`
- captures both `stdout` and `stderr`
- truncates large output to the configured cap
- syncs explicit write targets back into the real workspace after execution, even if the script exits with an error

#### Policy

- exactly one of `code` or `script_path` must be provided
- `script_path`, `read_paths`, and `write_paths` must stay inside `/workspace`
- scripts may not execute from protected workspace paths
- `read_paths` and `write_paths` must already exist and must be explicit files or directories
- `write_paths` may not target or contain protected workspace paths
- `.env` paths are denied
- shell-expanded forms like `~`, `*`, `?`, and `[` are rejected

#### Current Limitations

- intentionally one-shot and stateless; no persistent kernel/session memory across tool calls
- writable targets must already exist; to create new files, pass an existing writable directory
- protected workspace paths are excluded from staged reads, so broad reads like `/workspace` are partial by design
- third-party availability is defined by the curated package setting and the dependency closure of those installed packages in the dedicated venv

### `view_image`

- Status: implemented
- Exposure: `basic`
- Package: `src/tools/view_image/`
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
- Package: `src/tools/send_file/`
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
- Package: `src/tools/web_search/`
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
- Package: `src/tools/web_fetch/`
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

## Tools To Be Implemented

### Basic Tools

These should be auto-exposed at session start once implemented.

#### `tool_search`

- Purpose: search the registry for discoverable tools and return concise usage docs so the agent can opt into additional capabilities

#### `file_patch`

- Purpose: perform structured file edits with explicit patch operations instead of freeform shell editing

### Discoverable Tools

These should stay hidden by default and only be surfaced through `tool_search`.

#### `git_readonly`

- Purpose: inspect repository status, diff, and history through a safer read-only interface

#### `archive`

- Purpose: list, extract, and create common archive formats like `zip` and `tar` inside the workspace

#### `tabular_query`

- Purpose: query and transform CSV, JSON, and SQLite-style tabular data through a higher-level interface

#### `document_convert`

- Purpose: convert between common document formats or extract clean text from them for downstream reasoning

#### `browser_automation`

- Purpose: handle multi-step web tasks that simple search or fetch cannot cover, such as navigation, clicks, or form interaction

#### `http_api`

- Purpose: make structured API requests with controlled auth/header handling for tool-like external integrations

#### `workspace_index`

- Purpose: build and query a structured index of workspace files so the agent can find relevant project artifacts faster than raw shell search alone

## Current Snapshot

- Implemented tools: `bash`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`
- Implemented basic tools: `bash`, `python_interpreter`, `web_search`, `web_fetch`, `view_image`, `send_file`
- Implemented discoverable tools: none
