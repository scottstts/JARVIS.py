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
- **Run anything “runtime-related” inside the Docker container** (install deps, run the app, run tests, run linters/formatters if they depend on the venv).

The repo is **bind-mounted** into the container (`Javis/` on macOS ↔ `/repo` in the container), so changes are bidirectional. Treat `/repo` as the same project folder, just viewed from Linux.

### Where to do what

**On macOS (host):**
- Create/edit code, refactor, move files, update docs.
- Use `git` as normal.
- Codex operates here by default.

**Inside container (Linux):**
- `uv venv`, `uv pip install ...`, `uv run ...`
- Run the agent program, tests, scripts, etc.
- Anything that needs the Linux runtime environment.

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

## Dev Rules and Preferences

- Use ruff for python lint checking at the end of each turn
- Build production grade code, **BUT** avoid excessive abstraction layers
- all source code go into src/ dir
- make sure code is modular inside src/
- do NOT git commit code, I will **always** do that myself
- throughout the implementation, you will constantly ask me design choice questions like "option ABC, which do you want". This is because I'm not sure about some design choices yet, so constantly offer me options, alternatives, and challenge me when something doesn't seem to make obvious sense.
- thoughtout the implementation, don't be too "trigger-happy", which means that you never go directly editing code unprompted. Never presume anything. If there's any ambiguity, contradiction, or things that appear to be obvious mistakes from me, always point out or ask.