# Jarvis Dev

This is a Python project for building a personal AI system similar to OpenClaw. This project will be built from scratch and not related to OpenClaw. However, some design choices will be taking inspirations from OpenClaw

## Design

See dev_docs/design.md for design choices. Note that this document will be updated by me as we go because certain design choices might not be fully decided at a given point. However:

1. I may have code implemented different from what design.md says, that most likely means that i decided to deviate from design.md in actual implementation but didn't update it to reflect acutal implementation. In that case, treat code as the source of truth.
2. The design.md at any given point might not have complete design. That could either mean certain parts of the design is not in place yet or certain design details of a part is not in place yet.

So treat this document as a helper doc hinting you what has been built and what we're still building out, but not as a definitive documentation.

## Development Workflow (MacOS + Docker Runtime)

### Core rule

- **Edit code on macOS (the project repo: ~/Documents/Projects/Python/Jarvis/).**
- **Run anything “runtime-related” inside Docker** (install deps, run the app, run tests, run linters/formatters if they depend on the venv).

The runtime is now split across two sibling containers in the same compose project:

- `dev`
  - runs the Jarvis app
  - mounts the repo at `/repo`
  - mounts the shared workspace at `/workspace`
- `tool_runtime`
  - runs the internal HTTP service for `bash` and `python_interpreter`
  - mounts the shared workspace at `/workspace`
  - does **not** mount `/repo`

The repo is still bind-mounted into `dev` (`Jarvis/` on macOS ↔ `/repo` in the container), so changes are bidirectional. Treat `/repo` as the same project folder, just viewed from Linux.

### Where to do what

**On macOS (host):**
- Create/edit code, refactor, move files, update docs.
- Use `git` as normal.
- Codex operates here by default.

**Inside container (Linux):**
- `uv venv`, `uv pip install ...`, `uv run ...`
- Run the agent program, tests, scripts, etc.
- Anything that needs the Linux runtime environment.

Tool note:

- `bash` and `python_interpreter` no longer execute inside `dev`; they are remotely executed in the isolated `tool_runtime` container through the app.

### Standard commands (run from macOS terminal)

* Start/refresh the dev container

```bash
docker compose up -d --build
````

* Run an interactive Linux shell (when needed)

```bash
docker compose exec dev bash
```

Inside the container, the repo is at `/repo`.

* One-shot: run a command inside the container

Use this pattern whenever you need to run code without opening an interactive shell:

```bash
docker compose exec dev bash -lc "cd /repo && <COMMAND>"
```

### Dependency management (uv) — inside the container only

First-time setup (or after dependency changes):

```bash
docker compose exec dev bash -lc "cd /repo && uv venv"
docker compose exec dev bash -lc "cd /repo && uv add <deps>"
```

Then run normally:

```bash
docker compose exec dev bash -lc "cd /repo && uv run <...>"
```

### Specific Docs for Different System Development

Inside dev_docs/ dir, there are docs for different system development for Jarvis. e.g., tool_dev_doc.md, runtime_tools_plan.md, tool_runtime_isolation_plan.md for agent tool related development; memory_design_doc.md, memory_pass_2.md for agent memory related dev.

Some of the docs are implementation plans, some are ad hoc documentations. All of them should be accurate reflection of the implmenented code. These docs can reveal underlying design choices and intentions beyond what the code can tell you. So whenever you are working on a specific system, check first if there are docs that you can read to understand more about the system you're about to work on.

## Dev Rules and Preferences

- Use ruff for python lint checking at the end of each turn
- Use pytest for testing
- Build production grade code, **BUT** avoid excessive abstraction layers
- all source code go into src/ dir
- make sure code is modular inside src/
- do NOT git commit code, I will **always** do that myself
- throughout the implementation, you will constantly ask me design choice questions like "option ABC, which do you want". This is because I'm not sure about some design choices yet, so constantly offer me options, alternatives, and challenge me when something doesn't seem to make obvious sense.
- thoughtout the implementation, don't be too "trigger-happy", which means that you never go directly editing code unprompted. Never presume anything. If there's any ambiguity, contradiction, or things that appear to be obvious mistakes from me, always point out or ask.
- No need to delete __pycache__/
- agent system runtime settings live in src/settings.py, and only secrets (api keys, bot tokens, etc.) need to be in secrets/

# Notes & Lessons

`notes/notes.md` is a scratch pad that you will write to concisely about things you've notes and learned during the implementation, including but not limited to design choices. Whenever you feel like there's something that other coding agents after you will benefit from in later implementation, write to it

This serves as the agent continuous memory so even when i start a new coding agent, you will also benefit from the notes the agents before you have noted.

You can write to it and read it as well. Over time, this notes.md will contain all the accumulated lessons about this project, dos and don'ts, preferred and not preferred

Try MOSTLY to append to it. only delete or edit existing notes when they explicitly contradict with new approved design choices

Write to it concisely, try to use single sentence for each entry, include only valuable information, do NOT be verbose
