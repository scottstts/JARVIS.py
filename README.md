# JARVIS.py

![Jarvis and Tony Stark](assets/jarvis-stark.jpeg)

Jarvis is arguably my favorite AI assistant from any sci-fi stories. He's capable, low key, intelligent, and witty. He is my idea of how a personal AI assistant should be like. Watching Iron Man has been a great inspiration.

Now we finally have the technology to build this out, or ... close enough.

There are great personal AI assistants out there I have no doubt, but for me it is more fun to build out my own version of Jarvis, shaping it to be the way I want. That's what I am doing with JARVIS.py.

## Setup

### Spin Up Docker Containers

Run Jarvis inside Docker. Bring the containers up first:

```bash
docker compose up -d --build
```

Docker Compose should create `~/.jarvis/workspace/` on first run; if it does not, create it manually with

```bash
mkdir -p ~/.jarvis/workspace
```

`~/.jarvis/workspace/` will be the workspace dir for the agent.

### Deposit Secrets & Settings

Set up secrets by creating files under `secrets/`. The full expected list and doc URLs are in [`secrets/README.md`](secrets/README.md).

In telegram BotFather, add commands:

```plaintext
new - start a new session
stop - pause Jarvis
compact - compact current session
```

### BTWs

- At container startup time, `jarvis_runtime` reseeds workspace starter files from the repo and overwrites the previous copies: `workspace/settings/settings.yml`, `workspace/settings/settings_gui.html`, `workspace/identities/*`, and `workspace/migrate.sh`
- Jarvis reads runtime settings from `workspace/settings/settings.yml` when it exists, and falls back to the packaged template YAML only if that workspace file is absent.
- Settings GUI is also available in `workspace/settings/settings_gui.html`. Open or drag in `settings.yml` there, edit the settings in the GUI, and save edited settings.
- `workspace/migrate.sh` creates a zip archive of `archive/`, `memory/`, `runtime_tools/`, and `settings/` from the current directory. Pass `--all` to archive everything in the current directory instead.

## Run Jarvis

### Run Dev

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv sync --locked --group dev"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run jarvis"
```

For tests and linting, use the same container-managed environment:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run pytest"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run ruff check ."
```

### Build & Run App

To build and install the packaged artifact into `jarvis_runtime`, run this on host:

```bash
bash utils/install_build.sh
```

To force a fresh rebuild and reinstall:

```bash
bash utils/install_build.sh --reinstall
```

Run installed artifact with:

```bash
docker compose exec jarvis_runtime bash -lc "jarvis"
```
