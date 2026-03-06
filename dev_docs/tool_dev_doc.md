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

- `bash` may read across the container filesystem
- `bash` may only write inside `/workspace`

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

- reads allowed anywhere in the container
- writes allowed only inside `/workspace`

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
- `sort` rejects `-o` and `--output`
- `sed` only allows read-only `-n` line printing and limited `sed -i` substitution
- `cp` / `mv` reject target-directory forms like `-t`, `--target-directory`, `-T`, `--no-target-directory`

#### Current Limitations

- intentionally conservative; many harmless shell patterns are blocked
- sequential tool execution only
- no parallel shell tool calls yet
- if more expressive editing is needed later, prefer a dedicated edit tool over relaxing shell policy too much

## Tools To Be Implemented

### Basic Tools

These should be auto-exposed at session start once implemented.

#### `tool_search`

- Purpose: search the registry for discoverable tools and return concise usage docs so the agent can opt into additional capabilities

#### `web_search`

- Purpose: run web search queries and return structured search results for current information gathering

#### `web_fetch`

- Purpose: fetch and normalize the contents of a specific URL or page for reading and extraction

#### `send_file`

- Purpose: send a local file from the agent workspace to the user through the Telegram file-send interface (such interface has been implemented in src/ui/telegram/, use it)

#### `python_interpreter`

- Purpose: run constrained Python code for data processing, parsing, small scripts, and structured transformations that are awkward in shell

#### `image_inspect`

- Purpose: inspect images for metadata, OCR, dimensions, and lightweight visual extraction tasks

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

- Implemented tools: `bash`
- Implemented basic tools: `bash`
- Implemented discoverable tools: none

