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

Docker Compose should create `~/.jarvis/workspace/` on first run; if it does not, create it manually with `mkdir -p ~/.jarvis/workspace`.

`~/.jarvis/workspace/` will be the workspace dir for the agent. This dir on your host OS will be bind mounted in the containers as `workspace/` dir.

This repo is strict container-first for Python: do not create or use a host-side `.venv`; run `uv` against the project only inside the `jarvis_runtime` container.

### Deposit Secrets & Settings

Set up secrets by creating files under `secrets/`. The full expected list and doc URLs are in [`secrets/README.md`](secrets/README.md).

In telegram BotFather, add commands:

```plaintext
new - start a new session
stop - pause Jarvis
compact - compact current session
```

At container startup time, Jarvis settings YAML is copied to `workspace/settings/settings.yml` if the workspace copy does not already exist. Jarvis reads runtime settings from it. If that YAML file is missing from `workspace/settings/`, Jarvis falls back to the packaged template YAML.

Settings GUI is also available in `workspace/settings/settings_gui.html`. Open or drag in `settings.yml` there, edit the settings in the GUI, and save edited settings.

## Run Jarvis

### Run Dev

Run dev inside the `jarvis_runtime` container:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv sync --locked --group dev"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run jarvis"
```

After changing `~/.jarvis/workspace/settings/settings.yml`, restart the affected runtime processes so they reload the file. If you change tool-runtime settings, restart the containers with `docker compose restart`.

For tests and linting, use the same container-managed environment:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run pytest"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv run ruff check ."
```

### Build & Run App

To use the built package instead, build it from the `jarvis_runtime` container:

```bash
docker compose exec jarvis_runtime bash -lc "cd /repo && uv build"
docker compose exec jarvis_runtime bash -lc "cd /repo && uv tool install dist/jarvis-0.1.0-py3-none-any.whl"
docker compose exec jarvis_runtime bash -lc "jarvis"
```
