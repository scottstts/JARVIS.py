"""Small shell lexer for policy decisions that must not execute the command."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ShellOperator:
    """One shell control operator outside quoted, commented, or data-only regions."""

    value: str
    offset: int


@dataclass(slots=True, frozen=True)
class _Heredoc:
    delimiter: str
    strip_tabs: bool


def shell_control_operators(command: str) -> tuple[ShellOperator, ...]:
    """Return control operators while excluding redirections and shell data."""

    masked = masked_shell_syntax(command)
    operators: list[ShellOperator] = []
    index = 0
    arithmetic_depth = 0
    while index < len(masked):
        if masked.startswith("$((", index) or masked.startswith("((", index):
            arithmetic_depth += 1
            index += 3 if masked.startswith("$((", index) else 2
            continue
        if arithmetic_depth:
            if masked.startswith("))", index):
                arithmetic_depth -= 1
                index += 2
            else:
                index += 1
            continue

        if masked[index] == "\n":
            operators.append(ShellOperator("newline", index))
            index += 1
            continue
        matched = next(
            (
                operator
                for operator in (";;&", "&&", "||", "|&", ";&")
                if masked.startswith(operator, index)
            ),
            None,
        )
        if matched is not None:
            operators.append(ShellOperator(matched, index))
            index += len(matched)
            continue

        redirect = next(
            (
                operator
                for operator in ("<<<", "<<-", "<<", ">>", "<>", "<&", ">&", ">|", "<", ">")
                if masked.startswith(operator, index)
            ),
            None,
        )
        if redirect is not None:
            index += len(redirect)
            continue

        char = masked[index]
        if char == "&":
            before = masked[index - 1] if index else ""
            after = masked[index + 1] if index + 1 < len(masked) else ""
            # Descriptor duplication (2>&1, <&3) and combined redirects (&>, &>>)
            # do not create an asynchronous command list.
            if before not in {">", "<"} and after != ">":
                operators.append(ShellOperator("&", index))
        elif char in {";", "|"}:
            operators.append(ShellOperator(char, index))
        index += 1
    return tuple(operators)


def background_operator(command: str) -> ShellOperator | None:
    """Return the first asynchronous-list operator in *command*, if present."""

    return next(
        (operator for operator in shell_control_operators(command) if operator.value == "&"),
        None,
    )


def masked_shell_syntax(command: str) -> str:
    """Mask quotes, comments, and heredoc bodies while preserving offsets/newlines."""

    output: list[str] = []
    pending_heredocs: list[_Heredoc] = []
    quote: str | None = None
    escaped = False
    lines = command.splitlines(keepends=True)
    if not lines and command:
        lines = [command]

    for line in lines:
        if pending_heredocs:
            candidate = line[:-1] if line.endswith("\n") else line
            heredoc = pending_heredocs[0]
            comparable = candidate.lstrip("\t") if heredoc.strip_tabs else candidate
            output.extend("\n" if char == "\n" else " " for char in line)
            if comparable == heredoc.delimiter:
                pending_heredocs.pop(0)
            continue

        visible: list[str] = []
        comment = False
        index = 0
        while index < len(line):
            char = line[index]
            if comment:
                visible.append("\n" if char == "\n" else " ")
                index += 1
                continue
            if escaped:
                visible.append("\n" if char == "\n" else " ")
                escaped = False
                index += 1
                continue
            if quote is not None:
                visible.append("\n" if char == "\n" else " ")
                if char == quote:
                    quote = None
                elif char == "\\" and quote == '"':
                    escaped = True
                index += 1
                continue
            if char == "\\":
                visible.append(" ")
                escaped = True
                index += 1
                continue
            if char == "#" and (
                index == 0 or line[index - 1].isspace() or line[index - 1] in ";|&()"
            ):
                visible.append(" ")
                comment = True
                index += 1
                continue
            if line.startswith("<<", index) and not line.startswith("<<<", index):
                heredoc, end = _parse_heredoc(line, index)
                if heredoc is not None:
                    pending_heredocs.append(heredoc)
                    visible.extend(line[index : index + 2])
                    visible.extend(" " for _ in line[index + 2 : end])
                    index = end
                    continue
            if char in {"'", '"'}:
                visible.append(" ")
                quote = char
                index += 1
                continue
            visible.append(char)
            index += 1
        output.extend(visible)
    return "".join(output)


def _parse_heredoc(line: str, start: int) -> tuple[_Heredoc | None, int]:
    index = start + 2
    strip_tabs = index < len(line) and line[index] == "-"
    if strip_tabs:
        index += 1
    while index < len(line) and line[index] in " \t":
        index += 1
    token_start = index
    delimiter: list[str] = []
    quote: str | None = None
    escaped = False
    while index < len(line):
        char = line[index]
        if escaped:
            delimiter.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote is not None:
            if char == quote:
                quote = None
            else:
                delimiter.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char.isspace() or char in ";|&()<>":
            break
        delimiter.append(char)
        index += 1
    if index == token_start or not delimiter or quote is not None:
        return None, start + 2
    return _Heredoc(delimiter="".join(delimiter), strip_tabs=strip_tabs), index
