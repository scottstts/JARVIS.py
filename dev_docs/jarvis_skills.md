# Jarvis Skills

## Status

This document is the source of truth for Jarvis agent skill support.

It records the implemented behavior and the constraints future changes must preserve.

## Purpose

Jarvis should be able to:

- use installed agent skills from the workspace
- install skills for itself when the user asks
- create or update skills for itself when the user asks
- keep skill discovery compact enough for long-running sessions
- preserve transcript and context continuity according to [persistence_refector.md](/Users/scott/Documents/Projects/Python/Jarvis/dev_docs/persistence_refector.md)

## Resolved Decisions

1. The canonical Jarvis skill root is `/workspace/skills/`.
2. A skill is one directory under that root:
   `/workspace/skills/<skill_id>/SKILL.md`.
3. Skill directories may include optional `scripts/`, `references/`, and `assets/` directories.
4. Installed skills are discovered from the workspace at runtime. They are not repo assets.
5. `skills.bootstrap_headers` is a boolean user-facing setting in [src/jarvis/settings.yml](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/settings.yml).
6. When `skills.bootstrap_headers=true`, Jarvis injects concise skill headers at session start.
7. When `skills.bootstrap_headers=false`, Jarvis injects only compact search guidance, not skill headers; the agent must use `get_skills mode=search` before choosing a relevant skill.
8. `get_skills mode=get` is available in both settings modes.
9. `get_skills mode=search` is exposed and allowed only when `skills.bootstrap_headers=false`.
10. Skill installation commands should run with `/workspace` as both the current working area and home, usually:
    `cd /workspace && HOME=/workspace npx ...`
11. Jarvis should automatically normalize common installer output into `/workspace/skills/` instead of expecting the agent to move files manually.
12. If automatic import cannot safely determine the target skill, Jarvis reports the issue and the agent handles the unusual case explicitly.
13. Skill support must not add replay-time synthesis of prompt-visible context.

## Skill Format

Jarvis should support the common Agent Skills shape:

```text
/workspace/skills/<skill_id>/
  SKILL.md
  scripts/
  references/
  assets/
```

`SKILL.md` should use YAML frontmatter when possible:

```markdown
---
name: skill-name
description: Short activation description.
---

Skill instructions...
```

Required runtime metadata:

- `name`: use frontmatter `name` when present, otherwise the directory name
- `description`: required for discovery; invalid or missing descriptions should make the skill non-discoverable
- `skill_id`: canonical directory name under `/workspace/skills`
- `path`: canonical workspace path

Optional metadata:

- `compatibility`
- `metadata`
- any other frontmatter fields should be preserved in parsed metadata but not exposed unless useful

Invalid skills should not crash Jarvis startup. They should be skipped with an application log entry and surfaced as scanner warnings in `get_skills` metadata.

## Workspace Layout

Canonical paths:

- `/workspace/skills/`
- `/workspace/skills/<skill_id>/SKILL.md`

Installer staging paths that Jarvis should scan and normalize:

- `/workspace/.codex/skills/`
- `/workspace/.claude/skills/`
- `/workspace/.agents/skills/`
- `/workspace/.mdskills/skills/`

Jarvis should not recursively scan all of `/workspace` for skills. Keep import scanning bounded to known installer output roots and the canonical root.

Container startup should create `/workspace/skills` if missing, alongside the existing workspace setup for `identities` and `settings`.

## Settings

Add this top-level user-facing section:

```yaml
skills:
  title: "Skills"
  description: "Agent skill discovery and startup context behavior."
  fields:
    bootstrap_headers:
      label: "Bootstrap Skill Headers"
      description: "Show installed skill names and descriptions at session start."
      type: "boolean"
      value: true
```

Add `JARVIS_SKILLS_BOOTSTRAP_HEADERS` to [src/jarvis/settings.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/settings.py).

Keep internal limits out of `settings.yml`. Put scanner and import caps in `src/jarvis/skills/config.py`.

Recommended internal defaults:

- max `SKILL.md` read size
- max rendered `get` output chars
- max resource listing count
- max imported skill file count
- max imported single-file bytes
- ignored import directories such as `node_modules`, `.git`, `.venv`, and cache directories

## Skill Service

Add a small `src/jarvis/skills/` package.

Recommended files:

- `src/jarvis/skills/__init__.py`
- `src/jarvis/skills/config.py`
- `src/jarvis/skills/types.py`
- `src/jarvis/skills/catalog.py`
- `src/jarvis/skills/importer.py`
- `src/jarvis/skills/rendering.py`

Responsibilities:

- resolve `/workspace/skills`
- parse skill headers from `SKILL.md`
- search installed skills deterministically
- render compact bootstrap headers
- render compact search results
- render one full skill for `get_skills mode=get`
- list bundled resources without eagerly reading references/assets
- normalize installer output into `/workspace/skills`

Search should stay deterministic and dependency-light:

- exact `skill_id` and `name` matches first
- substring hits next
- token overlap against `name`, `description`, and `compatibility`
- stable ordering by score then `skill_id`
- no embeddings

## Automatic Skill Import

The importer exists so `HOME=/workspace npx ...` can work even when the installer writes to client-specific default homes instead of `/workspace/skills`.

### Install Command Convention

Agent-facing install guidance should be concise:

```text
Skills live in /workspace/skills. For npx skill installers, run from /workspace with HOME=/workspace. Jarvis imports common installer output into /workspace/skills automatically.
```

Expected command shape:

```bash
cd /workspace && HOME=/workspace npx ...
```

This makes home-relative installer defaults land under the shared workspace, for example:

- `/workspace/.codex/skills/...`
- `/workspace/.claude/skills/...`
- `/workspace/.agents/skills/...`

### Import Trigger Points

Run the importer opportunistically at these points:

1. before rendering session skill bootstrap headers
2. before every `get_skills` execution
3. after successful foreground `bash` results
4. after a background `bash` job reaches a terminal successful state

The importer should be quiet when no changes are found.

When a `bash` command looks like a skill install command, the agent-facing `bash` result is normalized after the command exits and after import cleanup runs. On success, hide installer sequencing details and return only the canonical result:

```text
Skill install result
status: success
skill: ai-image-prompts-skill
installed_at: /workspace/skills/ai-image-prompts-skill/SKILL.md
```

For multi-skill installs, use the same fields under a short `skills:` list.

If the shell command fails, return `status: failed` with `failed_stage: install` and include the original bash result. If the shell command succeeds but import, conflict detection, or cleanup fails, return `status: failed` with `failed_stage: normalization` and enough detail for the agent to finish or report the issue.

For ordinary non-install `bash` commands, import changes may still append a short import normalization notice to the normal bash output.

### Import Algorithm

For each known staging root:

1. Look for immediate child directories containing `SKILL.md`.
2. Parse the skill header.
3. Skip invalid skills and record a warning.
4. Compute `skill_id`:
   - prefer a valid source directory basename
   - otherwise slugify frontmatter `name`
5. Set target path to `/workspace/skills/<skill_id>/`.
6. If target does not exist, import the skill.
7. If target exists and content hash matches, treat as already imported.
8. If target exists and content differs, do not overwrite automatically; report a conflict.
9. After a successful import or already-present match, remove the staging source directory so the canonical tree is the durable installed location.

Import means copying a filtered skill payload into the canonical target, not blindly moving installer caches. Copy these by default:

- `SKILL.md`
- `scripts/`
- `references/`
- `assets/`
- small top-level support files that are not ignored

Ignore heavy or unsafe directories:

- `node_modules`
- `.git`
- `.venv`
- `__pycache__`
- cache directories

The importer should never follow symlinks outside the source skill directory or outside `/workspace`.

Conflicts are intentionally not auto-resolved. Updating an existing skill is a user-visible operation and should be handled explicitly by the agent with normal workspace tools after reporting the conflict.

## `get_skills` Tool

Add a basic tool package:

- `src/jarvis/tools/basic/get_skills/`

Files:

- `__init__.py`
- `tool.py`
- `policy.py`

Register it in [src/jarvis/tools/registry.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/tools/registry.py), and add policy routing in [src/jarvis/tools/policy.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/tools/policy.py).

### Modes

`search`:

- available only when `skills.bootstrap_headers=false`
- accepts optional `query`
- returns compact headers and metadata only
- never returns full `SKILL.md`

`get`:

- available in both settings modes
- accepts `skill_id`
- returns the full `SKILL.md` content for one skill
- includes a bounded resource listing
- does not read full referenced files automatically

### Schema Shape

When `skills.bootstrap_headers=true`, expose only `get`:

```json
{
  "mode": "get",
  "skill_id": "..."
}
```

When `skills.bootstrap_headers=false`, expose both:

```json
{
  "mode": "search",
  "query": "..."
}
```

```json
{
  "mode": "get",
  "skill_id": "..."
}
```

The builder should generate the enum dynamically from `SkillsSettings`.

### Agent-Facing Text

Keep the description short.

For bootstrap mode:

```text
Open an installed skill by id after matching a bootstrapped skill header.
```

For search mode:

```text
Search installed skills, then open one by id before using it.
```

Argument descriptions should not repeat obvious schema facts. Do not include examples of ordinary skill ids.

### Policy

Policy should enforce:

- `mode` must be valid for the current settings mode
- `search` query length and word count are bounded
- `get` requires a non-empty explicit `skill_id`
- `skill_id` must be a simple canonical directory name, not a path
- all file reads must stay inside `/workspace/skills/<skill_id>/`

The executor should still handle missing or invalid skills gracefully with a normal tool-error result.

## Bootstrap Behavior

### Normal Provider Loop

In [src/jarvis/core/agent_loop.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/core/agent_loop.py), session startup does this when `skills.bootstrap_headers=true`:

1. run skill import normalization
2. render installed skill headers
3. persist the rendered headers as a normal prompt-visible system record before the first provider call
4. mark the record with `skills_bootstrap: "headers"`

This record must not be `transcript_only`, because the model sees it and later replay depends on it.

If `skills.bootstrap_headers=false`, session startup persists a compact prompt-visible guidance record with `skills_bootstrap: "search_guidance"`. It must not include installed skill names or descriptions.

If no valid skills exist and headers are enabled, omit the headers record. The install convention may still be included in the base identity/program guidance if needed.

### Codex Backend

In [src/jarvis/codex_backend/actor_runtime.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/codex_backend/actor_runtime.py), `_build_developer_instructions()` should:

1. run skill import normalization when rendering headers
2. append the same concise skill header section when `skills.bootstrap_headers=true`
3. append the same compact search guidance when `skills.bootstrap_headers=false`
4. include the same skill install convention from the base identity guidance

The persisted Codex developer-instructions snapshot remains `transcript_only`, matching the existing Codex backend design, because the provider-visible developer instructions are managed by the Codex thread API rather than normal transcript replay.

### Header Format

Keep headers compact:

```text
Installed skills:
- skill_id: description

Use get_skills mode=get before applying a skill.
```

If compatibility metadata exists and is short, include it only when useful:

```text
- skill_id: description (compatibility: ...)
```

Do not include paths, resource listings, examples, or full instructions in bootstrap headers.

## Transcript And Context Continuity

Skill support must preserve the rules in [persistence_refector.md](/Users/scott/Documents/Projects/Python/Jarvis/dev_docs/persistence_refector.md).

Rules:

1. If the model sees skill headers in a normal provider request, persist those headers as a replayable transcript record before the request.
2. If the model sees no-header skill search guidance in a normal provider request, persist that guidance as a replayable transcript record before the request.
3. If a `bash` result tells the model that skills were imported or import conflicts occurred, persist that information as part of the normal `bash` tool result.
4. If `get_skills` returns search results or a full skill, persist the tool result like every other tool result.
5. Do not synthesize skill headers or no-header guidance during replay. They must already be in the transcript for that session.
6. Do not use `transcript_only` for prompt-visible normal-provider skill bootstrap.
7. Add skill bootstrap metadata to compaction pruning so old bootstrap records are not summarized into future sessions.
8. New sessions and compacted sessions should render fresh skill bootstrap records from the current workspace state and setting.
9. Skill installation or skill file changes are legitimate request changes and may affect provider cache reuse; that is not a transcript-fidelity bug.

Compaction update:

- [src/jarvis/core/compaction.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/core/compaction.py) should drop records with `metadata["skills_bootstrap"]`.
- The compacted replacement session will get fresh skill bootstrap from current workspace state during `_start_session`.

No replay-time behavior should scan skills and inject missing historical headers. That would break transcript continuity.

## Subagents

Subagents should be able to use skills unless explicitly blocked later.

Initial behavior:

- `get_skills` is allowed for both `main` and `subagent`
- subagents receive the same skill header bootstrap behavior as their configured settings mode
- subagents may read `/workspace/skills`
- subagents should not install, create, or update skills unless the assignment explicitly asks for it

Update [src/jarvis/subagent/prompts/OPERATING_RULES.md](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/subagent/prompts/OPERATING_RULES.md):

- `/workspace/skills` is managed skill storage
- reading skills through `get_skills` is normal
- writing skill files should happen only when the task explicitly requires installing, creating, or updating a skill

Codex-backed subagents should receive the same skill instructions through their Codex developer instructions path.

## Installation Workflows

### NPM/Npx Installer

When the user asks Jarvis to install skills with an npm-based installer, the agent should run:

```bash
cd /workspace && HOME=/workspace npx ...
```

Jarvis should import any resulting skill directories from known staging roots into `/workspace/skills`.

The agent should inspect the `bash` result:

- if imported skills are listed, use `get_skills mode=get` before using one
- if conflicts are listed, report or resolve explicitly
- if no skills were imported, inspect the installer output and handle the unusual case

### GitHub Or Direct Source

When no installer exists, the agent may use `bash` to fetch source files into workspace staging, then rely on the importer if the source has a recognizable skill directory.

If the source layout is unusual, the agent can use normal workspace tools to create:

```text
/workspace/skills/<skill_id>/SKILL.md
```

### Self-Created Skills

When the user asks Jarvis to create a skill for itself, the agent should write directly under:

```text
/workspace/skills/<skill_id>/SKILL.md
```

No special tool is needed. Existing `file_patch` or `bash` can create the files, subject to normal workspace policy.

## Tool Runtime Boundary

`bash` executes in the isolated `tool_runtime` container. The project repo is not mounted there. `/workspace` is the durable shared boundary.

Implications:

- skill installers must write under `/workspace`
- `HOME=/workspace` is required for deterministic home-relative installer output
- any files written outside `/workspace` are not part of the durable Jarvis skill system
- skill import normalization runs from Jarvis against the shared workspace, not from repo paths

## Security And Safety

Skill files are instructions, not trusted code.

Rules:

- `get_skills` reads only from `/workspace/skills`
- `get_skills` must not read `.env` files
- symlinks must not escape the skill directory or workspace
- resource listings should not include huge file contents
- scripts inside skills are not executed by `get_skills`
- if the agent later runs a skill script, that happens through existing tools and existing policy
- installer commands still go through `bash` approval policy when they match install/build/system-mutation patterns

Do not add a separate skill-install tool unless later design work shows a real need. The combination of `bash`, `HOME=/workspace`, automatic importer, and `get_skills` should be enough.

## Implemented Checklist

1. Add `skills.bootstrap_headers` to [src/jarvis/settings.yml](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/settings.yml).
2. Export `JARVIS_SKILLS_BOOTSTRAP_HEADERS` from [src/jarvis/settings.py](/Users/scott/Documents/Projects/Python/Jarvis/src/jarvis/settings.py).
3. Add `src/jarvis/skills/` with settings, types, catalog, importer, and rendering helpers.
4. Create `/workspace/skills` during container startup.
5. Add `src/jarvis/tools/basic/get_skills/`.
6. Register `get_skills` in `ToolRegistry.default(...)`.
7. Add `GetSkillsPolicy` routing in `src/jarvis/tools/policy.py`.
8. Add skill bootstrap rendering to normal `AgentLoop` session startup.
9. Add skill bootstrap rendering to Codex developer instructions.
10. Add skill import normalization before skill bootstrap and before `get_skills`.
11. Add skill import normalization after successful foreground bash results.
12. Add skill import normalization after successful terminal background bash jobs.
13. Add `skills_bootstrap` compaction pruning.
14. Update subagent operating rules for `/workspace/skills`.
15. Update [dev_docs/tool_dev_doc.md](/Users/scott/Documents/Projects/Python/Jarvis/dev_docs/tool_dev_doc.md) with the new `get_skills` tool.
16. Update [dev_docs/project_structure.md](/Users/scott/Documents/Projects/Python/Jarvis/dev_docs/project_structure.md) if it lists top-level source packages.

## Tests

Focused coverage should include:

- settings extraction for `skills.bootstrap_headers`
- valid skill parsing
- invalid skill handling without startup failure
- symlinked canonical skill directories skipped
- deterministic skill search ordering
- `get_skills mode=search` available when `bootstrap_headers=false`
- `get_skills mode=search` hidden and policy-denied when `bootstrap_headers=true`
- `get_skills mode=get` returns full `SKILL.md`
- `get_skills mode=get` rejects path-like skill ids
- resource listing stays bounded
- importer moves/copies staged skills from `/workspace/.codex/skills` into `/workspace/skills`
- importer handles already-imported matching skills
- importer reports conflicts without overwriting
- normal provider skill bootstrap persists as replayable system context
- no-header mode persists search guidance without skill headers
- skill bootstrap is pruned from compaction source
- compacted sessions receive fresh skill headers
- Codex developer instructions include skills when bootstrapping is enabled
- `bash` result metadata/content reports imported skills when the importer finds new skills
- subagent filtered registry includes `get_skills`

## External References

- Agent Skills specification: `https://agentskills.io/specification`
- Agent Skills client implementation guide: `https://agentskills.io/client-implementation/adding-skills-support`
- OpenAI Codex skills docs: `https://developers.openai.com/codex/skills`
- mdskills marketplace: `https://www.mdskills.ai/`
- Claude Code skills docs: `https://code.claude.com/docs/en/skills`
