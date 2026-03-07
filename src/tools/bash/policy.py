"""Bash-specific policy implementation."""

from __future__ import annotations

from fnmatch import fnmatchcase
import re
import shlex
from pathlib import Path

from ..types import ToolExecutionContext, ToolPolicyDecision

_FORBIDDEN_SHELL_SNIPPETS = ("&&", "||", ";", ">", "<", "`", "$", "\n", "\r", "(", ")")
_PROTECTED_ENV_BASENAME = ".env"
_READ_ONLY_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "stat",
    "file",
    "du",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "wc",
    "cut",
    "sort",
    "uniq",
    "diff",
    "printf",
    "echo",
}
_WRITE_COMMANDS = {
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "truncate",
    "tee",
    "sed",
}
_DANGEROUS_FIND_TOKENS = {
    "-delete",
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-fls",
}
_WRITE_GLOB_PATTERN = re.compile(r"[*?\[]")
_SAFE_SED_PRINT_PATTERN = re.compile(r"^[0-9,$ ]+p$")
_PATH_GLOB_PATTERN = re.compile(r"[*?\[]")
_GREP_RECURSIVE_TOKENS = {"-r", "-R", "--recursive"}
_READ_OPTIONS_WITH_VALUES: dict[str, set[str]] = {
    "cut": {
        "-b",
        "-c",
        "-d",
        "-f",
        "--bytes",
        "--characters",
        "--delimiter",
        "--fields",
        "--output-delimiter",
    },
    "head": {"-c", "-n", "--bytes", "--lines"},
    "tail": {
        "-c",
        "-n",
        "-s",
        "--bytes",
        "--lines",
        "--max-unchanged-stats",
        "--pid",
        "--sleep-interval",
    },
}
_GREP_OPTIONS_WITH_VALUES = {
    "-A",
    "-B",
    "-C",
    "-D",
    "-d",
    "-e",
    "-f",
    "-m",
    "--after-context",
    "--before-context",
    "--binary-files",
    "--color",
    "--colour",
    "--context",
    "--devices",
    "--directories",
    "--exclude",
    "--exclude-dir",
    "--exclude-from",
    "--file",
    "--include",
    "--label",
    "--max-count",
    "--regexp",
}
_RG_OPTIONS_WITH_VALUES = {
    "-A",
    "-B",
    "-C",
    "-M",
    "-T",
    "-e",
    "-f",
    "-g",
    "-j",
    "-m",
    "-t",
    "--after-context",
    "--before-context",
    "--context",
    "--encoding",
    "--engine",
    "--file",
    "--glob",
    "--iglob",
    "--ignore-file",
    "--max-columns",
    "--max-count",
    "--max-depth",
    "--max-filesize",
    "--pre",
    "--pre-glob",
    "--regexp",
    "--replace",
    "--sort",
    "--sortr",
    "--type",
    "--type-add",
    "--type-not",
}
_GREP_PATTERN_OPTIONS = {"-e", "-f", "--file", "--regexp"}
_RG_PATTERN_OPTIONS = {"-e", "-f", "--file", "--regexp"}
_RG_GLOB_OPTIONS = {"-g", "--glob", "--iglob"}


class BashCommandPolicy:
    """Validates the controlled bash subset used by the v1 tool runtime."""

    def authorize(
        self,
        *,
        command: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        for snippet in _FORBIDDEN_SHELL_SNIPPETS:
            if snippet in command:
                return ToolPolicyDecision(
                    allowed=False,
                    reason=f"bash syntax '{snippet}' is not allowed in tool mode.",
                )

        try:
            segments = _parse_pipeline(command)
        except ValueError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        if not segments:
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")

        for tokens in segments:
            decision = self._authorize_segment(tokens=tokens, context=context)
            if not decision.allowed:
                return decision

        return ToolPolicyDecision(allowed=True)

    def _authorize_segment(
        self,
        *,
        tokens: list[str],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        command = tokens[0]
        args = tokens[1:]

        if command not in _READ_ONLY_COMMANDS and command not in _WRITE_COMMANDS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Command '{command}' is not allowed in bash tool v1.",
            )

        if command == "find":
            return _authorize_find(args)
        if command == "grep":
            return _authorize_grep(args)
        if command == "rg":
            return _authorize_rg(args)
        if command == "sort":
            decision = _reject_if_tokens_present(
                args,
                forbidden={"-o", "--output"},
                message="sort output-file flags are not allowed.",
            )
            if not decision.allowed:
                return decision
            return _authorize_read_operands(command=command, args=args)
        if command == "sed":
            return _authorize_sed(args, context)
        if command == "cp":
            return _authorize_copy_like(
                command="cp",
                args=args,
                context=context,
                require_sources_in_workspace=False,
            )
        if command == "mv":
            return _authorize_copy_like(
                command="mv",
                args=args,
                context=context,
                require_sources_in_workspace=True,
            )
        if command == "rm":
            return _authorize_workspace_operands(
                args=args,
                context=context,
                command_name="rm",
            )
        if command == "mkdir":
            return _authorize_workspace_operands(
                args=args,
                context=context,
                command_name="mkdir",
            )
        if command == "touch":
            return _authorize_workspace_operands(
                args=args,
                context=context,
                command_name="touch",
                options_with_values={"-d", "-t", "-r", "--date", "--reference"},
            )
        if command == "truncate":
            return _authorize_workspace_operands(
                args=args,
                context=context,
                command_name="truncate",
                options_with_values={"-s", "-r", "-o", "--size", "--reference"},
            )
        if command == "tee":
            return _authorize_tee(args=args, context=context)
        if command in _READ_ONLY_COMMANDS:
            return _authorize_read_operands(command=command, args=args)

        return ToolPolicyDecision(allowed=True)


def _parse_pipeline(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    lexer.commenters = ""

    segments: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token == "|":
            if not current:
                raise ValueError("bash pipeline cannot contain an empty segment.")
            segments.append(current)
            current = []
            continue
        current.append(token)

    if not current:
        raise ValueError("bash pipeline cannot end with '|'.")
    segments.append(current)
    return segments


def _authorize_find(args: list[str]) -> ToolPolicyDecision:
    for token in args:
        if token in _DANGEROUS_FIND_TOKENS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"find primary '{token}' is not allowed in bash tool v1.",
            )

    start_paths: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"!", "(", ")"} or token.startswith("-"):
            break
        start_paths.append(token)
        index += 1

    decision = _deny_protected_path_operands(start_paths, command_name="find")
    if not decision.allowed:
        return decision

    pattern_primaries = {"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename"}
    regex_primaries = {"-regex", "-iregex"}
    while index < len(args):
        token = args[index]
        if token in pattern_primaries and index + 1 < len(args):
            if _pattern_targets_protected_env(args[index + 1]):
                return ToolPolicyDecision(
                    allowed=False,
                    reason="find may not target '.env' paths in bash tool v1.",
                )
            index += 2
            continue
        if token in regex_primaries and index + 1 < len(args):
            if _regex_targets_protected_env(args[index + 1]):
                return ToolPolicyDecision(
                    allowed=False,
                    reason="find may not target '.env' paths in bash tool v1.",
                )
            index += 2
            continue
        index += 1
    return ToolPolicyDecision(allowed=True)


def _authorize_sed(
    args: list[str],
    context: ToolExecutionContext,
) -> ToolPolicyDecision:
    for token in args:
        if token in {"-e", "-f", "--expression", "--file"}:
            return ToolPolicyDecision(
                allowed=False,
                reason="sed -e/-f forms are not allowed in bash tool v1.",
            )
        if token.startswith("--expression=") or token.startswith("--file="):
            return ToolPolicyDecision(
                allowed=False,
                reason="sed --expression/--file forms are not allowed in bash tool v1.",
            )

    positionals = _extract_positionals(args)
    if len(positionals) < 2:
        return ToolPolicyDecision(
            allowed=False,
            reason="sed requires a script and at least one target file.",
        )

    script = positionals[0]
    file_operands = positionals[1:]
    decision = _deny_protected_path_operands(file_operands, command_name="sed")
    if not decision.allowed:
        return decision
    inplace = any(token == "-i" or token.startswith("--in-place") for token in args)

    if inplace:
        if not _is_safe_sed_substitution(script):
            return ToolPolicyDecision(
                allowed=False,
                reason="Only simple sed in-place substitution scripts are allowed.",
            )
        return _authorize_explicit_workspace_paths(
            file_operands,
            context=context,
            command_name="sed -i",
        )

    if not any(token in {"-n", "--quiet", "--silent"} for token in args):
        return ToolPolicyDecision(
            allowed=False,
            reason="Read-only sed requires -n, --quiet, or --silent in v1.",
        )
    if not _SAFE_SED_PRINT_PATTERN.fullmatch(script):
        return ToolPolicyDecision(
            allowed=False,
            reason="Read-only sed is limited to simple line-print scripts like '1,20p'.",
        )
    return ToolPolicyDecision(allowed=True)


def _authorize_copy_like(
    *,
    command: str,
    args: list[str],
    context: ToolExecutionContext,
    require_sources_in_workspace: bool,
) -> ToolPolicyDecision:
    forbidden_options = {"-t", "--target-directory", "-T", "--no-target-directory"}
    for token in args:
        if token in forbidden_options or any(token.startswith(f"{option}=") for option in forbidden_options):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"{command} target-directory flags are not allowed in bash tool v1.",
            )

    positionals = _extract_positionals(args)
    if len(positionals) < 2:
        return ToolPolicyDecision(
            allowed=False,
            reason=f"{command} requires at least one source and one destination.",
        )

    decision = _deny_protected_path_operands(positionals, command_name=command)
    if not decision.allowed:
        return decision

    sources = positionals[:-1]
    destination = positionals[-1]
    if require_sources_in_workspace:
        decision = _authorize_explicit_workspace_paths(
            sources,
            context=context,
            command_name=command,
        )
        if not decision.allowed:
            return decision

    return _authorize_explicit_workspace_paths(
        [destination],
        context=context,
        command_name=command,
    )


def _authorize_workspace_operands(
    *,
    args: list[str],
    context: ToolExecutionContext,
    command_name: str,
    options_with_values: set[str] | None = None,
) -> ToolPolicyDecision:
    positionals = _extract_positionals(args, options_with_values=options_with_values or set())
    if not positionals:
        return ToolPolicyDecision(
            allowed=False,
            reason=f"{command_name} requires at least one explicit path operand.",
        )
    return _authorize_explicit_workspace_paths(
        positionals,
        context=context,
        command_name=command_name,
    )


def _authorize_tee(
    *,
    args: list[str],
    context: ToolExecutionContext,
) -> ToolPolicyDecision:
    positionals = _extract_positionals(args)
    if not positionals:
        return ToolPolicyDecision(allowed=True)
    return _authorize_explicit_workspace_paths(
        positionals,
        context=context,
        command_name="tee",
    )


def _authorize_explicit_workspace_paths(
    operands: list[str],
    *,
    context: ToolExecutionContext,
    command_name: str,
) -> ToolPolicyDecision:
    decision = _deny_protected_path_operands(operands, command_name=command_name)
    if not decision.allowed:
        return decision

    for operand in operands:
        if operand == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason=f"{command_name} path operand '-' is not allowed in bash tool v1.",
            )
        if operand.startswith("~") or _WRITE_GLOB_PATTERN.search(operand):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"{command_name} does not allow shell-expanded path operand '{operand}'.",
            )

        resolved = _resolve_workspace_relative_path(operand, context)
        if not _is_within_workspace(resolved, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    f"{command_name} may only write inside {context.workspace_dir}; "
                    f"got '{operand}'."
                ),
            )
    return ToolPolicyDecision(allowed=True)


def _authorize_read_operands(
    *,
    command: str,
    args: list[str],
) -> ToolPolicyDecision:
    if command in {"echo", "printf", "pwd"}:
        return ToolPolicyDecision(allowed=True)

    positionals = _extract_positionals(
        args,
        options_with_values=_READ_OPTIONS_WITH_VALUES.get(command, set()),
    )
    return _deny_protected_path_operands(positionals, command_name=command)


def _authorize_grep(args: list[str]) -> ToolPolicyDecision:
    if _has_recursive_grep_flag(args):
        return ToolPolicyDecision(
            allowed=False,
            reason="Recursive grep is not allowed in bash tool v1.",
        )

    decision = _deny_protected_option_path_values(
        args,
        option_names={"-f", "--file", "--exclude-from"},
        command_name="grep",
    )
    if not decision.allowed:
        return decision

    path_operands = _extract_grep_like_path_operands(
        args,
        options_with_values=_GREP_OPTIONS_WITH_VALUES,
        pattern_options=_GREP_PATTERN_OPTIONS,
    )
    return _deny_protected_path_operands(path_operands, command_name="grep")


def _authorize_rg(args: list[str]) -> ToolPolicyDecision:
    decision = _reject_if_tokens_present(
        args,
        forbidden={"--no-config"},
        message="rg --no-config is not allowed in bash tool v1.",
    )
    if not decision.allowed:
        return decision

    decision = _deny_protected_option_path_values(
        args,
        option_names={"-f", "--file", "--ignore-file"},
        command_name="rg",
    )
    if not decision.allowed:
        return decision

    decision = _deny_protected_rg_globs(args)
    if not decision.allowed:
        return decision

    path_operands = _extract_grep_like_path_operands(
        args,
        options_with_values=_RG_OPTIONS_WITH_VALUES,
        pattern_options=_RG_PATTERN_OPTIONS,
    )
    return _deny_protected_path_operands(path_operands, command_name="rg")


def _has_recursive_grep_flag(args: list[str]) -> bool:
    for index, token in enumerate(args):
        if token in _GREP_RECURSIVE_TOKENS:
            return True
        if token == "-d" and index + 1 < len(args) and args[index + 1] == "recurse":
            return True
        if token.startswith("--directories=") and token.partition("=")[2] == "recurse":
            return True
        if token.startswith("-d") and len(token) > 2 and token[2:] == "recurse":
            return True
    return False


def _extract_grep_like_path_operands(
    args: list[str],
    *,
    options_with_values: set[str],
    pattern_options: set[str],
) -> list[str]:
    positionals = _extract_positionals(args, options_with_values=options_with_values)
    if _has_any_option(args, pattern_options):
        return positionals
    if not positionals:
        return []
    return positionals[1:]


def _has_any_option(args: list[str], option_names: set[str]) -> bool:
    for token in args:
        if token == "--":
            break
        if token in option_names:
            return True
        for option_name in option_names:
            if option_name.startswith("--") and token.startswith(f"{option_name}="):
                return True
            if option_name.startswith("-") and not option_name.startswith("--") and len(option_name) == 2:
                if token.startswith(option_name) and len(token) > len(option_name):
                    return True
    return False


def _deny_protected_option_path_values(
    args: list[str],
    *,
    option_names: set[str],
    command_name: str,
) -> ToolPolicyDecision:
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        if token.startswith("--"):
            option_name, has_inline_value, value = token.partition("=")
            if option_name in option_names:
                candidate = value if has_inline_value else (args[index + 1] if index + 1 < len(args) else "")
                if _path_token_targets_protected_env(candidate):
                    return ToolPolicyDecision(
                        allowed=False,
                        reason=(
                            f"{command_name} may not access '.env' paths in bash tool v1; "
                            f"got '{candidate}'."
                        ),
                    )
                index += 1 if has_inline_value else 2
                continue
        if token.startswith("-") and token != "-":
            option_name = token[:2]
            if option_name in option_names:
                candidate = token[2:] if len(token) > 2 else (args[index + 1] if index + 1 < len(args) else "")
                if _path_token_targets_protected_env(candidate):
                    return ToolPolicyDecision(
                        allowed=False,
                        reason=(
                            f"{command_name} may not access '.env' paths in bash tool v1; "
                            f"got '{candidate}'."
                        ),
                    )
                index += 1 if len(token) > 2 else 2
                continue
        index += 1
    return ToolPolicyDecision(allowed=True)


def _deny_protected_rg_globs(args: list[str]) -> ToolPolicyDecision:
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        if token.startswith("--"):
            option_name, has_inline_value, value = token.partition("=")
            if option_name in _RG_GLOB_OPTIONS:
                candidate = value if has_inline_value else (args[index + 1] if index + 1 < len(args) else "")
                if _rg_glob_targets_protected_env(candidate):
                    return ToolPolicyDecision(
                        allowed=False,
                        reason="rg may not include '.env' via glob filters in bash tool v1.",
                    )
                index += 1 if has_inline_value else 2
                continue
        if token.startswith("-") and token != "-":
            option_name = token[:2]
            if option_name in _RG_GLOB_OPTIONS:
                candidate = token[2:] if len(token) > 2 else (args[index + 1] if index + 1 < len(args) else "")
                if _rg_glob_targets_protected_env(candidate):
                    return ToolPolicyDecision(
                        allowed=False,
                        reason="rg may not include '.env' via glob filters in bash tool v1.",
                    )
                index += 1 if len(token) > 2 else 2
                continue
        index += 1
    return ToolPolicyDecision(allowed=True)


def _rg_glob_targets_protected_env(pattern: str) -> bool:
    if not pattern.startswith("!"):
        return _pattern_targets_protected_env(pattern)
    return False


def _deny_protected_path_operands(
    operands: list[str],
    *,
    command_name: str,
) -> ToolPolicyDecision:
    for operand in operands:
        if _path_token_targets_protected_env(operand):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    f"{command_name} may not access '.env' paths in bash tool v1; "
                    f"got '{operand}'."
                ),
            )
    return ToolPolicyDecision(allowed=True)


def _path_token_targets_protected_env(operand: str) -> bool:
    return _pattern_targets_protected_env(operand)


def _pattern_targets_protected_env(pattern: str) -> bool:
    last_segment = pattern.rstrip("/").split("/")[-1]
    if not last_segment:
        return False
    if not _PATH_GLOB_PATTERN.search(last_segment):
        return last_segment == _PROTECTED_ENV_BASENAME
    if not last_segment.startswith("."):
        return False
    return fnmatchcase(_PROTECTED_ENV_BASENAME, last_segment)


def _regex_targets_protected_env(pattern: str) -> bool:
    return _PROTECTED_ENV_BASENAME in pattern


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _is_within_workspace(path: Path, workspace_dir: Path) -> bool:
    workspace = workspace_dir.resolve(strict=False)
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False


def _extract_positionals(
    args: list[str],
    *,
    options_with_values: set[str] | None = None,
) -> list[str]:
    positionals: list[str] = []
    value_options = options_with_values or set()
    literal_mode = False
    index = 0

    while index < len(args):
        token = args[index]
        if literal_mode:
            positionals.append(token)
            index += 1
            continue

        if token == "--":
            literal_mode = True
            index += 1
            continue

        if token.startswith("--"):
            option_name, has_inline_value, _value = token.partition("=")
            if option_name in value_options and not has_inline_value:
                index += 2
                continue
            if option_name in value_options:
                index += 1
                continue
            if len(token) > 2:
                index += 1
                continue

        if token.startswith("-") and token != "-":
            short_name = token[:2]
            if short_name in value_options and len(token) == 2:
                index += 2
                continue
            if short_name in value_options and len(token) > 2:
                index += 1
                continue
            index += 1
            continue

        positionals.append(token)
        index += 1

    return positionals


def _reject_if_tokens_present(
    args: list[str],
    *,
    forbidden: set[str],
    message: str,
) -> ToolPolicyDecision:
    for token in args:
        if token in forbidden:
            return ToolPolicyDecision(allowed=False, reason=message)
        if any(token.startswith(f"{name}=") for name in forbidden):
            return ToolPolicyDecision(allowed=False, reason=message)
    return ToolPolicyDecision(allowed=True)


def _is_safe_sed_substitution(script: str) -> bool:
    if len(script) < 4 or not script.startswith("s"):
        return False

    delimiter = script[1]
    if delimiter.isalnum() or delimiter.isspace():
        return False

    separators: list[int] = []
    index = 2
    while index < len(script):
        char = script[index]
        if char == "\\":
            index += 2
            continue
        if char == delimiter:
            separators.append(index)
            if len(separators) == 2:
                break
        index += 1

    if len(separators) != 2:
        return False

    flags = script[separators[-1] + 1 :]
    return all(flag in "gIi0123456789" for flag in flags)
