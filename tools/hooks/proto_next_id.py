# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Checks, and can fix, the `// Next ID: N` comment and the `reserved N to max;` line every protobuf message carries.

Every `message` in a .proto file is preceded by a comment naming the next free field number, and opens by reserving
every number from it on:

    // One MPC solution.
    // Next ID: 15
    message MpcPolicy {
      option (nproto.generate_struct) = "ocs2::humanoid::msgs::MpcPolicy";

      reserved 15 to max;

      uint64 resets_served = 1;
      ...
      ViewerAnnotations annotations = 14;
    }

The comment lets whoever adds a field see which number to take without scanning the message, and never reuse the
number of a deleted field. The reservation makes protoc itself refuse a field that takes that number while the
reservation still claims it, so the two cannot be left behind: adding a field means raising both. The comment is the
line directly above `message Name {`, after the message's own documentation; the reservation is the first statement of
the message body after its `option` statements (the nproto option, tools/hooks/proto_file_layout.py).

The check fails when a message has no such comment, when its number is not greater than every other number the message
uses (its fields - in oneofs too, map fields and proto2 groups included - its other `reserved` numbers and ranges, and
its `extensions` ranges), or when the message does not open with `reserved <that number> to max;`. A number larger
than the largest one in use is fine - a field was deleted, and its number must not come back. Enums are not checked.

    python3 -m tools.hooks.proto_next_id <files>        # check
    python3 -m tools.hooks.proto_next_id --fix <files>  # add or raise the comments and reservations
    python3 -m tools.hooks.proto_next_id --git-staged   # check the staged .proto files

`python3 -m tools.hooks.lint_code` (make lint) and the pre-commit hook run it as the registry check proto-next-id.
"""

import argparse
from collections.abc import Iterable
import os
import re
import sys
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import lint_files

NAME = "proto-next-id"

PROTO_EXTENSION = ".proto"

# The comment, alone on the line directly above `message Name {`.
NEXT_ID_COMMENT = re.compile(r"^\s*//\s*Next ID:\s*(\S+)\s*$")
# Any line that looks like an attempt at the comment, to report a malformed one instead of a missing one.
NEXT_ID_ATTEMPT = re.compile(r"^\s*//\s*next\s*id\b", re.IGNORECASE)

# Field numbers above this are invalid in protobuf; `max` in a reserved or extensions range means it.
MAX_FIELD_NUMBER = 536870911

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Violation(NamedTuple):
    path: str
    line: int  # 1-based line of the `message` keyword
    message_name: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: message {self.message_name}: {self.text}"


class Token(NamedTuple):
    text: str
    line: int  # 1-based


class MessageInfo(NamedTuple):
    name: str
    line: int  # 1-based line of the `message` keyword
    # Every number the message uses (fields, reserved, extensions), except the opening `reserved N to max`.
    numbers: tuple[int, ...]
    largest_label: str  # what uses the largest number, for the error message
    brace_line: int  # 1-based line of the message's opening '{'
    # N of the opening `reserved N to max;` and the line of that statement; None when the message opens otherwise.
    top_reservation: int | None
    top_reservation_line: int | None
    # Line of the last `option` statement the body opens with (before anything else); None when it opens otherwise.
    leading_option_line: int | None


def tokenize(source: str) -> list[Token]:
    """Splits a .proto source into tokens, dropping comments and keeping string literals whole."""
    tokens: list[Token] = []
    line = 1
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += source.count("\n", i, end)
            i = end
        elif c in "\"'":
            j = i + 1
            while j < n and source[j] != c:
                j += 2 if source[j] == "\\" else 1
            j = min(j + 1, n)
            tokens.append(Token(source[i:j], line))
            line += source.count("\n", i, j)
            i = j
        elif c.isalnum() or c in "_.":
            j = i
            while j < n and (source[j].isalnum() or source[j] in "_."):
                j += 1
            tokens.append(Token(source[i:j], line))
            i = j
        else:
            tokens.append(Token(c, line))
            i += 1
    return tokens


def _parse_number(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


def _range_numbers(statement: list[Token]) -> list[int]:
    """The numbers of a `reserved` or `extensions` statement: `1, 3 to 5, 9 to max` -> [1, 5, 9]; names are ignored.

    Only the largest number of a range matters for the check; `to max` closes the message for good, so it counts as the
    start of the range (nothing can follow it anyway).

    Args:
        statement: The tokens of the statement after its keyword, up to the `;`.

    Returns:
        One number per single number or range, in the order of the statement.
    """
    numbers: list[int] = []
    for index, token in enumerate(statement):
        value = _parse_number(token.text)
        if value is None:
            continue
        following = statement[index + 1 : index + 3]
        if len(following) == 2 and following[0].text == "to":
            end = (
                MAX_FIELD_NUMBER
                if following[1].text == "max"
                else _parse_number(following[1].text)
            )
            if end is not None and end != MAX_FIELD_NUMBER:
                numbers.append(end)
                continue
        numbers.append(value)
    return numbers


def _is_aggregate_value(head: list[str]) -> bool:
    """Whether a `{` after the tokens `head` of a statement opens an aggregate value rather than a body.

    `option (nproto.retired_field) = { name: "x" ... };` and a field's `[(tuning) = { unit: "m" }]` keep their statement
    going after the `}`: the option is still a leading option, and the field keeps its number.

    Args:
      head: The texts of the statement's tokens before the `{`.

    Returns:
      True when the `{` follows an `=` or a `:` of the statement.
    """
    return bool(head) and head[-1] in ("=", ":")


def _field_number(statement: list[Token]) -> tuple[int, str] | None:
    """(number, field name) of a field statement `[label] type name = N [options]`, or None for anything else."""
    texts = [t.text for t in statement]
    if not texts or texts[0] in ("option", "reserved", "extensions", "extend"):
        return None
    if "=" not in texts:
        return None
    equals = texts.index("=")
    if equals == 0 or equals + 1 >= len(texts):
        return None
    number = _parse_number(texts[equals + 1])
    name = texts[equals - 1]
    if number is None or not _IDENTIFIER.fullmatch(name):
        return None
    return number, name


def parse_messages(source: str) -> list[MessageInfo]:
    """Every message of a .proto source, nested ones included, with the numbers it uses."""
    tokens = tokenize(source)
    messages: list[MessageInfo] = []

    def parse_block(start: int, kind: str, name: str, line: int) -> int:
        """Parses the body of a block opened at tokens[start - 1] == '{'; returns the index after its '}'."""
        numbers: list[int] = []
        labels: list[str] = []
        statement: list[Token] = []
        first_item_seen = False
        top_reservation: int | None = None
        top_reservation_line: int | None = None
        leading_option_line: int | None = None
        i = start
        while i < len(tokens):
            text = tokens[i].text
            if text == "}":
                break
            if text == "{":
                head = [t.text for t in statement]
                if _is_aggregate_value(head):
                    # `option (x) = { ... };` or a field's `[(x) = { ... }]`: a value inside the statement, which goes on.
                    i = skip_block(i + 1)
                    continue
                first_item_seen = True
                # A nested block: a message or enum declaration, a oneof (its fields belong to this message), a proto2
                # group (a field that opens a message of its own), or another block.
                if len(head) >= 2 and head[0] in (
                    "message",
                    "enum",
                    "service",
                    "extend",
                ):
                    i = parse_block(i + 1, head[0], head[1], statement[0].line)
                elif len(head) >= 2 and head[0] == "oneof":
                    i, oneof_numbers, oneof_labels = parse_oneof(i + 1)
                    numbers += oneof_numbers
                    labels += oneof_labels
                elif "group" in head:
                    field = _field_number(statement)
                    if field is not None:
                        numbers.append(field[0])
                        labels.append(f"group {field[1]} = {field[0]}")
                    group_name = (
                        head[head.index("group") + 1]
                        if head.index("group") + 1 < len(head)
                        else "group"
                    )
                    i = parse_block(i + 1, "message", group_name, statement[0].line)
                else:
                    i = skip_block(i + 1)
                statement = []
                continue
            if text == ";":
                if kind == "message":
                    texts = [t.text for t in statement]
                    reserved_to_max = (
                        len(texts) == 4
                        and texts[0] == "reserved"
                        and texts[2:] == ["to", "max"]
                        and _parse_number(texts[1]) is not None
                    )
                    if not first_item_seen and texts and texts[0] == "option":
                        # The body's leading options (the nproto option first) come before the reservation.
                        leading_option_line = tokens[i].line
                        statement = []
                        i += 1
                        continue
                    if not first_item_seen and reserved_to_max:
                        top_reservation = _parse_number(texts[1])
                        top_reservation_line = statement[0].line
                    else:
                        collect(statement, numbers, labels)
                    first_item_seen = True
                statement = []
                i += 1
                continue
            statement.append(tokens[i])
            i += 1
        if kind == "message":
            largest = max(numbers) if numbers else 0
            largest_label = labels[numbers.index(largest)] if numbers else ""
            brace_line = tokens[start - 1].line if start >= 1 else line
            messages.append(
                MessageInfo(
                    name,
                    line,
                    tuple(numbers),
                    largest_label,
                    brace_line,
                    top_reservation,
                    top_reservation_line,
                    leading_option_line,
                )
            )
        return i + 1

    def collect(statement: list[Token], numbers: list[int], labels: list[str]) -> None:
        if not statement:
            return
        head = statement[0].text
        if head in ("reserved", "extensions"):
            for value in _range_numbers(statement[1:]):
                numbers.append(value)
                labels.append(f"{head} {value}")
            return
        field = _field_number(statement)
        if field is not None:
            numbers.append(field[0])
            labels.append(f"field {field[1]} = {field[0]}")

    def parse_oneof(start: int) -> tuple[int, list[int], list[str]]:
        numbers: list[int] = []
        labels: list[str] = []
        statement: list[Token] = []
        i = start
        while i < len(tokens) and tokens[i].text != "}":
            if tokens[i].text == "{":
                i = skip_block(i + 1)
                if not _is_aggregate_value([t.text for t in statement]):
                    statement = []
                continue
            if tokens[i].text == ";":
                collect(statement, numbers, labels)
                statement = []
            else:
                statement.append(tokens[i])
            i += 1
        return i + 1, numbers, labels

    def skip_block(start: int) -> int:
        depth = 1
        i = start
        while i < len(tokens) and depth:
            if tokens[i].text == "{":
                depth += 1
            elif tokens[i].text == "}":
                depth -= 1
            i += 1
        return i

    parse_block(0, "file", "", 0)
    return sorted(messages, key=lambda m: m.line)


def _next_free(message: MessageInfo) -> int:
    return (max(message.numbers) if message.numbers else 0) + 1


def check_source(source: str, path: str = "<string>") -> list[Violation]:
    """Every message of `source` whose `// Next ID:` comment or opening `reserved N to max;` is missing or stale."""
    lines = source.splitlines()
    violations: list[Violation] = []
    for message in parse_messages(source):
        next_free = _next_free(message)
        above = lines[message.line - 2] if message.line >= 2 else ""
        match = NEXT_ID_COMMENT.match(above)
        if not match:
            if NEXT_ID_ATTEMPT.match(above):
                text = f"malformed comment '{above.strip()}'; write exactly '// Next ID: {next_free}'"
            else:
                text = f"no '// Next ID: N' comment on the line directly above it; write '// Next ID: {next_free}'"
            violations.append(Violation(path, message.line, message.name, text))
            continue
        declared = _parse_number(match.group(1))
        if declared is None or declared < 1:
            violations.append(
                Violation(
                    path,
                    message.line,
                    message.name,
                    f"'{above.strip()}' is not a positive number; write '// Next ID: {next_free}'",
                )
            )
        elif declared < next_free:
            violations.append(
                Violation(
                    path,
                    message.line,
                    message.name,
                    f"says 'Next ID: {declared}' but {message.largest_label} is already taken; "
                    f"the next free number is {next_free}",
                )
            )
        else:
            violations += _check_top_reservation(path, message, declared)
    return violations


def _check_top_reservation(
    path: str, message: MessageInfo, next_id: int
) -> list[Violation]:
    """The message must open with `reserved <next_id> to max;`."""
    if message.top_reservation is None:
        return [
            Violation(
                path,
                message.line,
                message.name,
                f"does not open with 'reserved {next_id} to max;' (the first statement of the message after its "
                "options reserves every number from its Next ID on)",
            )
        ]
    if message.top_reservation != next_id:
        return [
            Violation(
                path,
                message.top_reservation_line or message.line,
                message.name,
                f"opens with 'reserved {message.top_reservation} to max;' but its Next ID is {next_id}; write "
                f"'reserved {next_id} to max;'",
            )
        ]
    return []


def fix_source(source: str) -> str:
    """`source` with every comment and opening reservation added, or raised to the next free number.

    A comment that already names a number above every number in use is kept (a field was deleted), and the
    reservation follows it.

    Args:
        source: The text of a .proto file.

    Returns:
        The fixed text; `source` itself when it needs no fix.
    """
    lines = source.splitlines(keepends=True)
    # Bottom-up, so inserting a line does not move the messages still to be fixed; within a message the reservation
    # (below the `message` line) is fixed before the comment (above it).
    for message in reversed(parse_messages(source)):
        next_id = _next_free(message)
        message_line = lines[message.line - 1]
        indent = message_line[: len(message_line) - len(message_line.lstrip())]
        above_index = message.line - 2
        above = lines[above_index] if above_index >= 0 else ""
        match = NEXT_ID_COMMENT.match(above.rstrip("\n"))
        declared = _parse_number(match.group(1)) if match else None
        if declared is not None and declared >= next_id:
            next_id = declared

        reservation = f"{indent}  reserved {next_id} to max;\n"
        if message.top_reservation_line is not None:
            if message.top_reservation != next_id:
                lines[message.top_reservation_line - 1] = reservation
        elif message.leading_option_line is not None:
            option_index = message.leading_option_line - 1
            insert_at = option_index + 1
            if insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1
                lines[insert_at:insert_at] = [reservation, "\n"]
            else:
                lines[insert_at:insert_at] = ["\n", reservation, "\n"]
        else:
            brace_index = message.brace_line - 1
            brace_text = lines[brace_index]
            brace_column = brace_text.find("{")
            after_brace = brace_text[brace_column + 1 :]
            if after_brace.strip().startswith("}"):
                # `message Empty {}` on one line: open it up.
                lines[brace_index] = (
                    brace_text[: brace_column + 1]
                    + "\n"
                    + reservation
                    + indent
                    + after_brace.lstrip()
                )
            else:
                body_follows = after_brace.strip() != ""
                if body_follows:
                    lines[brace_index] = brace_text[: brace_column + 1] + "\n"
                    lines.insert(brace_index + 1, f"{indent}  {after_brace.lstrip()}")
                lines.insert(brace_index + 1, reservation + "\n")

        comment = f"{indent}// Next ID: {next_id}\n"
        if match:
            if declared != next_id:
                lines[above_index] = comment
        elif NEXT_ID_ATTEMPT.match(above):
            lines[above_index] = comment
        else:
            lines.insert(message.line - 1, comment)
    return "".join(lines)


def check_files(paths: Iterable[str], root: str) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


def fix_files(paths: Iterable[str]) -> list[str]:
    """Fixes the files in place; returns the ones that changed."""
    changed: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        fixed = fix_source(source)
        if fixed != source:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
            changed.append(path)
    return changed


def check_staged(repository: str) -> list[Violation]:
    """Checks the STAGED version of every .proto file staged for commit, as the pre-commit hook needs."""
    violations: list[Violation] = []
    for path in lint_files.staged_files(repository):
        if path.endswith(PROTO_EXTENSION):
            violations += check_source(lint_files.staged_source(repository, path), path)
    return violations


def _findings(source: str, path: str) -> list[check_types.Finding]:
    return [
        check_types.Finding(
            path, v.line, 1, NAME, f"message {v.message_name}: {v.text}"
        )
        for v in check_source(source, path)
    ]


def _fix(source: str, path: str) -> str:
    del path  # Unused.
    return fix_source(source)


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.PROTO}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        fix_source=_fix,
        description="every protobuf message names its next free field number (`// Next ID: N`) and opens with "
        "`reserved N to max;`.",
        hint="Add or raise them with: python3 -m tools.hooks.proto_next_id --fix <files>.",
    )
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help=".proto files")
    parser.add_argument(
        "--fix", action="store_true", help="add or raise the comments in place"
    )
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="check the staged version of every .proto file staged for commit (the pre-commit hook)",
    )
    args = parser.parse_args(argv)
    if args.git_staged == bool(args.paths):
        parser.error("give either .proto paths or --git-staged")
    if args.git_staged and args.fix:
        parser.error("--fix works on paths, not on the index")
    if args.fix:
        for path in fix_files(args.paths):
            print(f"fixed {path}")
        return 0
    if args.git_staged:
        violations = check_staged(os.getcwd())
    else:
        violations = check_files(args.paths, os.getcwd())
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
