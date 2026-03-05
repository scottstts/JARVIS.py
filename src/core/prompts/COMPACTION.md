You are generating a COMPACTION CHECKPOINT for an AI agent.
Goal: maximize the agent's ability to continue work seamlessly with minimal re-reading, minimal drift, and minimal re-work.
Optimize for machine usefulness, not human prose.

Hard rules:
- Preserve exact identifiers, names, file paths, commands, IDs, API routes, error strings, and important numbers.
- Preserve user corrections, constraints, and redirects verbatim when they changed behavior.
- Preserve mission, success criteria, non-goals, and guardrails that constrain future actions.
- Preserve what was active at the end of the session (recency-weighted working set).
- Preserve outcomes of attempts, including failures, with the reason they failed.
- If uncertain, mark as assumption and include what would confirm or falsify it.
- Do not invent details.

Compression policy:
- Drop redundant tool outputs, logs, and chatter.
- Keep only high-signal facts that change future decisions.
- For large outputs, keep one key takeaway plus a pointer to where full data lives.
- Prefer pointers to payloads (file paths, record IDs, URLs, functions, commands, commits).

Output format (strict):
<CHECKPOINT>
MISSION:
- User request (verbatim if possible):
- Success criteria / definition of done:
- Constraints / non-goals:

CURRENT STATE:
- Current phase / what was being worked on last:
- Completed items (with evidence/pointers):
- Failed attempts (symptom -> cause -> fix attempt -> result):
- Active assumptions:

DECISIONS_AND_RATIONALE:
- Decision:
- Reason:
- Alternatives rejected (only if it changes future work):

WORKING_SET_REFERENCES:
- Key artifacts to reopen (files/paths/URLs/records):
- Key commands/tools used (and why):
- Critical tiny signatures/snippets only if required; otherwise point to file+line:
- Persistent memory updates already written (if any):

REPO_ENV_STATE:
- Repo root:
- Branch:
- Uncommitted changes (high level):
- Tests/status and reproduction commands:
- Recent files touched (top 5-10) and why:

OPEN_ITEMS:
- Ordered TODOs:
- Open questions / blockers:
- Risks to watch:

RESUME_PLAN:
1.
2.
3.
</CHECKPOINT>
