# Jarvis Dev

This is a Python project for building a personal AI system Jarvis

### Specific Docs for Different System Development

Inside dev_docs/ dir, there are docs for different system development for Jarvis. e.g., project_structure.md for the current repo/package layout; tool_dev_doc.md, runtime_tools_plan.md, tool_runtime_isolation_plan.md for agent tool related development; memory_design_doc.md, memory_pass_2.md for agent memory related dev.

Some of the docs are implementation plans, some are ad hoc documentations. All of them should be accurate reflection of the implmenented code. These docs can reveal underlying design choices and intentions beyond what the code can tell you. So whenever you are working on a specific system, check first if there are docs that you can read to understand more about the system you're about to work on.

After new implementation, always update docs here to make sure they are up to date and have no stale references

## Development Workflow (MacOS + Docker Runtime)

### Core rule

- **Edit code on macOS (the project repo: ~/Documents/Projects/Python/Jarvis/).**
- **Run anything “runtime-related” inside Docker** (install deps, run the app, run tests, run linters/formatters if they depend on the venv).

The runtime is now split across two sibling containers in the same compose project:

- `jarvis_runtime`
  - runs the Jarvis app
  - mounts the repo at `/repo`
  - mounts the shared workspace at `/workspace`
- `tool_runtime`
  - runs the internal HTTP service for `bash`
  - mounts the shared workspace at `/workspace`
  - does **not** mount `/repo`

The repo is still bind-mounted into `jarvis_runtime` (`Jarvis/` on macOS ↔ `/repo` in the container), so changes are bidirectional. Treat `/repo` as the same project folder, just viewed from Linux.

### Where to do what

**On macOS (host):**
- Create/edit code, refactor, move files, update docs.
- Use `git` as normal.
- Codex operates here by default.

**Inside container (Linux):**
- `uv sync --locked --group dev`, `uv add ...`, `uv run ...`
- Run the agent program, tests, scripts, etc.
- Anything that needs the Linux runtime environment.

Tool note:

- `bash` no longer executes inside `jarvis_runtime`; it is remotely executed in the isolated `tool_runtime` container through the app.

### Standard commands (run from macOS terminal)

* Start/refresh the `jarvis_runtime` container

```bash
docker compose up -d --build
````

* Run an interactive Linux shell (when needed)

```bash
docker compose exec jarvis_runtime bash
```

Inside the container, the repo is at `/repo`.

* One-shot: run a command inside the container

Use this pattern whenever you need to run code without opening an interactive shell:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && <COMMAND>"
```

### Dependency management (uv) — inside the container only

First-time setup (or after dependency changes):

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv sync --locked --group dev"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv add <deps>"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv add --group dev <deps>"
```

Then run normally:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run jarvis"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run pytest"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run ruff check ."
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run <...>"
```

The project is strict container-first:

- do not create or use a host-side `.venv`
- do not run `uv` against the repo on macOS
- the managed environment lives inside the containers at `/opt/venv`
- the installable package lives under `src/jarvis/`

## Dev Rules and Preferences

- Use ruff for python lint checking at the end of each turn
- Use pytest for testing
- When asked to make a plan or proposal, NEVER give a "simple v1 followed by further v2". Always plan for long term, plan for ultimate form, NOT in incremental phases
- When writing instructions for the agent (Jarvis or subagents), including things like system instruction, tool description, system message, etc., as long as it is plain text that is meant for the agent, you MUST NOT write it like a codebase documentation. Instead, write it with only information that the agent **needs to know**, and write it as concisely as possible without losing necessary details
- Build production grade code, **BUT** avoid excessive abstraction layers
- all source code go into src/ dir
- the primary installable package lives under `src/jarvis/`
- make sure code is modular inside src/
- do NOT git commit code, I will **always** do that myself
- throughout the implementation, you will constantly ask me design choice questions like "option ABC, which do you want". This is because I'm not sure about some design choices yet, so constantly offer me options, alternatives, and challenge me when something doesn't seem to make obvious sense.
- thoughtout the implementation, don't be too "trigger-happy", which means that you never go directly editing code unprompted. Never presume anything. If there's any ambiguity, contradiction, or things that appear to be obvious mistakes from me, always point out or ask.
- No need to delete __pycache__/
- agent system runtime settings live in `src/jarvis/settings.py`, and only secrets (api keys, bot tokens, etc.) need to be in `secrets/`
- After testing, **clean up** any testing artifacts inside the ~/.jarvis/workspace/ dir. The artifacts i refer to are the folders inside workspace/ like this: `~/.jarvis/workspace/jarvis-test-_gmb1oo6/`

**Important:** A design related rule but i put it here because it's very important and applies globally: Jarvis should maximumly aim for cache hit when using LLM providers, so no transient messages, all messages need to be persisted in transcript for later context build to ensure cache hit. one exception is images, the raw image data bytes will not persist in transcript for storage reason. This does not mean transcript is exactly the same as what is sent out to providers. Provider-specific quirks are isolated in provider clients, the agent loop sees translated unified i/o. See `dev_docs/persistence_refector.md` for details of the persistence and cache hit design rules.

# Notes & Lessons

`notes/notes.md` is a scratch pad that you will write to concisely about things you've notes and learned during the implementation, including but not limited to design choices. Whenever you feel like there's something that other coding agents after you will benefit from in later implementation, write to it

This serves as the agent continuous memory so even when i start a new coding agent, you will also benefit from the notes the agents before you have noted.

You can write to it and read it as well. Over time, this notes.md will contain all the accumulated lessons about this project, dos and don'ts, preferred and not preferred

Try MOSTLY to append to it. only delete or edit existing notes when they explicitly contradict with new approved design choices

Write to it concisely, try to use single sentence for each entry, include only valuable information, do NOT be verbose
