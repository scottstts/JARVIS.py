# Agent Skills

## Purpose

Jarvis supports workspace-installed agent skills. Skills provide reusable instructions, scripts, references, and assets that the agent can discover and open before applying.

Canonical skills live under:

```text
/workspace/skills/<skill_id>/SKILL.md
```

Installed skills are runtime workspace assets, not repo assets.

## Skill Format

Supported layout:

```text
/workspace/skills/<skill_id>/
  SKILL.md
  scripts/
  references/
  assets/
```

`SKILL.md` should use YAML frontmatter:

```markdown
---
name: skill-name
description: Short activation description.
---

Skill instructions...
```

Runtime metadata:

- `name`: frontmatter `name` or directory name
- `description`: required for discovery
- `skill_id`: canonical directory name
- `path`: canonical workspace path

Optional metadata such as compatibility is preserved and exposed only when useful.

Invalid skills are skipped without crashing startup. Scanner warnings are surfaced through `get_skills` metadata.

## Settings

Skill bootstrap behavior is controlled by:

```yaml
skills:
  fields:
    bootstrap_headers:
      value: true
```

`JARVIS_SKILLS_BOOTSTRAP_HEADERS` is exported from `src/jarvis/settings.py`.

When `skills.bootstrap_headers=true`, session startup injects concise installed skill headers.

When `skills.bootstrap_headers=false`, session startup injects compact search guidance instead; the agent must use `get_skills mode=search` before choosing a skill.

`get_skills mode=get` is available in both modes.

## Skill Service

The `src/jarvis/skills/` package handles:

- resolving `/workspace/skills`
- parsing `SKILL.md` headers
- deterministic skill search
- compact bootstrap rendering
- compact search-result rendering
- full skill rendering for `get_skills mode=get`
- bounded bundled-resource listings
- staged installer output import

Search is deterministic and dependency-light:

- exact `skill_id` and `name` matches first
- substring hits next
- token overlap against `name`, `description`, and compatibility
- stable ordering by score then `skill_id`
- no embeddings

## Automatic Skill Import

Jarvis normalizes common installer output into `/workspace/skills`.

Known staging roots:

- `/workspace/.codex/skills/`
- `/workspace/.claude/skills/`
- `/workspace/.agents/skills/`
- `/workspace/.mdskills/skills/`

Import runs:

- before rendering session skill bootstrap headers
- before every `get_skills` execution
- after successful foreground `bash` results
- after terminal successful background `bash` jobs

Jarvis does not recursively scan all of `/workspace`.

The importer copies filtered skill payloads into the canonical directory:

- `SKILL.md`
- `scripts/`
- `references/`
- `assets/`
- small top-level support files

It ignores heavy or unsafe directories such as `node_modules`, `.git`, `.venv`, caches, and `__pycache__`. Symlinks must not escape the skill directory or workspace.

If a matching canonical skill already exists with identical content, the staged copy is treated as already imported. If the content differs, the importer reports a conflict and does not overwrite automatically.

## Install Workflow

Skill installers should run from `/workspace` with `HOME=/workspace`:

```bash
cd /workspace && HOME=/workspace npx ...
```

This makes home-relative installer output land in known staging roots.

When a bash command looks like a skill install command, Jarvis post-processes the successful output into a normalized result:

```text
Skill install result
status: success
skill: <skill_id>
installed_at: /workspace/skills/<skill_id>/SKILL.md
```

For shell command failure, the result reports `failed_stage: install`. For import, conflict, or cleanup failure, it reports `failed_stage: normalization`.

## `get_skills`

`get_skills` is a basic tool.

Modes:

- `search`: available only when `skills.bootstrap_headers=false`; returns compact headers and metadata
- `get`: available in both modes; returns the full `SKILL.md` for one skill plus bounded resource listing

Policy enforces:

- valid mode for the current settings mode
- bounded search query length and word count
- explicit canonical `skill_id` for `get`
- no path-like skill ids
- reads stay inside `/workspace/skills/<skill_id>/`

The tool imports staged skill output before execution.

## Bootstrap Behavior

For normal provider sessions, startup persists prompt-visible skill context before the provider call:

- `skills_bootstrap="headers"` when headers are enabled and valid skills exist
- `skills_bootstrap="search_guidance"` when headers are disabled

These records are replayable transcript records, not `transcript_only`, because the model sees them.

Header format is compact:

```text
Installed skills:
- skill_id: description

Use get_skills mode=get before applying a skill.
```

Paths, resource listings, examples, and full instructions are not included in bootstrap headers.

For Codex-backed actors, skill context is appended to Codex developer instructions. The persisted Codex developer-instructions snapshot remains `transcript_only` because Codex thread state owns provider-visible developer instructions.

## Transcript And Compaction

Skill support follows the persistence rules in `agent_design.md`.

Rules:

- prompt-visible skill headers or search guidance persist before normal provider calls
- `bash` import notices persist as part of normal bash tool results
- `get_skills` results persist like other tool results
- replay does not synthesize missing historical skill headers
- skill bootstrap records are pruned from compaction source
- new and compacted sessions render fresh skill context from current workspace state and settings
- skill changes are legitimate provider-cache boundaries

## Subagents

Subagents can use skills.

Current behavior:

- `get_skills` is allowed for main and subagent actors
- subagents receive the same skill bootstrap mode as configured
- subagents may read `/workspace/skills`
- subagents should not install, create, or update skills unless the assignment explicitly asks for it
- `subagent_invoke.skill_ids` lets Jarvis select up to four canonical installed skills; Jarvis validates those ids and embeds each selected `SKILL.md` into the child assignment bootstrap

Codex-backed subagents receive skill context through their Codex developer-instructions path.

## Security

Skill files are instructions, not trusted code.

Rules:

- `get_skills` reads only from `/workspace/skills`
- `.env` files are not read
- symlinks cannot escape the skill directory or workspace
- resource listings do not include large file contents
- scripts inside skills are not executed by `get_skills`
- if the agent runs a skill script, it goes through existing tools and policy
- installer commands still go through `bash` approval policy when they match install/build/system-mutation patterns

There is no separate skill-install tool. The supported workflow is `bash`, `HOME=/workspace`, automatic importer, and `get_skills`.

## Tests To Preserve

Coverage should include:

- skill setting extraction
- valid and invalid skill parsing
- symlink escape prevention
- deterministic search ordering
- search mode exposure based on `bootstrap_headers`
- full skill rendering with bounded resources
- staged importer success, already-imported handling, and conflicts
- replayable normal-provider skill bootstrap
- no-header search guidance
- compaction pruning of skill bootstrap
- fresh skill context in compacted sessions
- Codex developer instruction skill rendering
- bash import result normalization
- subagent `get_skills` visibility
