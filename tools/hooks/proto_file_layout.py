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

"""Checks the layout of the .proto files: google3's "1-1-1" rule and the nproto struct option.

Every .proto file defines exactly ONE top-level message, enum or service, and the file is named after it in
snake_case: `message MpcPolicy` lives in mpc_policy.proto, `enum ControllerType` in controller_type.proto, `message
Vector3` in vector3.proto. Types that only make sense inside their parent (a message's own enum, such as
TargetContactPatch.Kind) stay nested in it. A file that only declares options (`extend google.protobuf.MessageOptions
{...}`, such as tools/nproto/options.proto) is the one exception to the naming rule.

One definition per file keeps every build dependency, every generated header and every import exactly as large as
what is used, and it makes the file of a type obvious from its name.

Every top-level message or enum also names the C++ type nproto generates for it, as the FIRST statement of its body
(tools/nproto/README.md), the file importing "nproto/options.proto":

    // Next ID: 4
    message Vector3 {
      option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

      reserved 4 to max;
      ...
    }

    enum ControllerType {
      option (nproto.generate_enum) = "ocs2::humanoid::msgs::ControllerType";
      ...
    }

The value is a fully qualified C++ name whose last component is the definition's name. Nested types are generated as
nested C++ types of their parent's struct and carry no option; neither does the file.

No message reserves a field NAME, in either spelling (`reserved "old_name";`, edition 2023's `reserved old_name;`):
protobuf's C++ text parser skips a reserved name without a word, so a configuration file that still carries a retired
key would silently stop meaning anything. List the name in `(nproto.retired_field)` instead, which turns it into an
error that says what replaced it (tools/nproto/README.md, "Retired fields"). Reserving field numbers is fine.

    python3 -m tools.hooks.proto_file_layout <files>       # check
    python3 -m tools.hooks.proto_file_layout --git-staged  # check the staged .proto files

`python3 -m tools.hooks.lint_code` (make lint) and the pre-commit hook run it as the registry check proto-file-layout.
"""

import argparse
from collections.abc import Iterable
import dataclasses
import os
import re
import sys
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import lint_files
from tools.hooks import proto_next_id

NAME = "proto-file-layout"

PROTO_EXTENSION = ".proto"
DEFINITION_KEYWORDS = ("message", "enum", "service", "extend")
NPROTO_OPTIONS_IMPORT = "nproto/options.proto"
NPROTO_STRUCT_OPTION = "nproto.generate_struct"  # in a message
NPROTO_ENUM_OPTION = "nproto.generate_enum"  # in an enum
NPROTO_OPTIONS = (NPROTO_STRUCT_OPTION, NPROTO_ENUM_OPTION)
_CPP_QUALIFIED_NAME = re.compile(
    r"(::)?[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Definition(NamedTuple):
    keyword: str  # message, enum, service or extend
    name: str
    line: int  # 1-based


class Violation(NamedTuple):
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.text}"


def snake_case(name: str) -> str:
    """`MpcPolicy` -> `mpc_policy`, `Vector3` -> `vector3`, `HTTPRequest` -> `http_request`."""
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words)
    return words.lower()


def top_level_definitions(source: str) -> list[Definition]:
    """The messages, enums, services and extend blocks declared at file scope (nested types are not listed)."""
    definitions: list[Definition] = []
    tokens = proto_next_id.tokenize(source)
    depth = 0
    for index, token in enumerate(tokens):
        if token.text == "{":
            depth += 1
        elif token.text == "}":
            depth = max(0, depth - 1)
        elif (
            depth == 0 and token.text in DEFINITION_KEYWORDS and index + 2 < len(tokens)
        ):
            name = tokens[index + 1].text
            # `message Name {` (`extend pkg.Type {`); anything else with these words at file scope is not a definition.
            pattern = (
                r"[A-Za-z_.][A-Za-z0-9_.]*"
                if token.text == "extend"
                else r"[A-Za-z_][A-Za-z0-9_]*"
            )
            if tokens[index + 2].text == "{" and re.fullmatch(pattern, name):
                definitions.append(Definition(token.text, name, token.line))
    return definitions


class NprotoOption(NamedTuple):
    name: str  # nproto.generate_struct or nproto.generate_enum
    value: str
    line: int
    # The chain of definitions the option is declared in, outermost first: () at file scope, (("message", "Foo"),) in
    # the body of a top-level message, and so on.
    scope: tuple[tuple[str, str], ...]
    first_in_body: bool  # it is the first statement of the body that declares it


@dataclasses.dataclass
class _Block:
    """An open `{` block of a .proto file, while nproto_options() walks it."""

    # The (keyword, name) of the definition the block belongs to; None for an option value or an rpc body.
    owner: tuple[str, str] | None
    # Whether a statement has been seen in the block yet.
    has_statement: bool = False


def nproto_options(source: str) -> list[NprotoOption]:
    """Every `option (nproto.generate_*) = "...";` in `source`, with where it is declared."""
    tokens = proto_next_id.tokenize(source)
    found: list[NprotoOption] = []
    stack: list[_Block] = []
    statement_start = True
    for index, token in enumerate(tokens):
        text = token.text
        if text == "{":
            previous = [t.text for t in tokens[max(0, index - 2) : index]]
            owner = None
            if len(previous) == 2 and previous[0] in DEFINITION_KEYWORDS + ("oneof",):
                owner = (previous[0], previous[1])
            stack.append(_Block(owner))
            statement_start = True
            continue
        if text == "}":
            if stack:
                stack.pop()
            if stack:
                stack[-1].has_statement = True
            statement_start = True
            continue
        if text == ";":
            if stack:
                stack[-1].has_statement = True
            statement_start = True
            continue
        if statement_start and text == "option" and index + 6 < len(tokens):
            window = [t.text for t in tokens[index + 1 : index + 7]]
            if (
                window[0] == "("
                and window[1] in NPROTO_OPTIONS
                and window[2:4] == [")", "="]
                and window[5] == ";"
            ):
                value = window[4]
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                scope = tuple(block.owner for block in stack if block.owner is not None)
                first = bool(stack) and not stack[-1].has_statement
                found.append(NprotoOption(window[1], value, token.line, scope, first))
        statement_start = False
    return found


def reserved_names(source: str) -> list[tuple[int, str]]:
    """(1-based line, name) of every field name a message's `reserved` statement lists, quoted or not.

    `reserved 4, 9 to 11, 20 to max;` reserves numbers only; `reserved "old";` (proto2, proto3) and `reserved old;`
    (edition 2023) reserve a name. Enums are not looked at: their reserved names are value names, not keys of a file.

    Args:
      source: A .proto file.

    Returns:
      The reserved names, in file order.
    """
    tokens = proto_next_id.tokenize(source)
    found: list[tuple[int, str]] = []
    owners: list[str | None] = (
        []
    )  # the keyword of each open block's definition (None: not a definition)
    statement_start = True
    index = 0
    while index < len(tokens):
        text = tokens[index].text
        if text == "{":
            previous = [t.text for t in tokens[max(0, index - 2) : index]]
            owners.append(
                previous[0]
                if len(previous) == 2 and previous[0] in DEFINITION_KEYWORDS
                else None
            )
            statement_start = True
        elif text == "}":
            if owners:
                owners.pop()
            statement_start = True
        elif text == ";":
            statement_start = True
        elif (
            statement_start
            and text == "reserved"
            and owners
            and owners[-1] == "message"
        ):
            index += 1
            while index < len(tokens) and tokens[index].text != ";":
                item = tokens[index].text
                if item[:1] in "\"'" or (
                    _IDENTIFIER.fullmatch(item) and item not in ("to", "max")
                ):
                    found.append((tokens[index].line, item.strip("\"'")))
                index += 1
            statement_start = True
        else:
            statement_start = False
        index += 1
    return found


def check_reserved_names(source: str, path: str) -> list[Violation]:
    """A violation for every field name a message reserves (reserved_names())."""
    return [
        Violation(
            path,
            line,
            f"reserves the field name '{name}': protobuf's C++ text parser silently skips a reserved name, so a file "
            "that still sets it would parse and mean nothing. Reserve the number only, and list the name in "
            '(nproto.retired_field) (tools/nproto/README.md, "Retired fields").',
        )
        for line, name in reserved_names(source)
    ]


def imports(source: str) -> list[str]:
    """The paths of the file's import statements."""
    tokens = proto_next_id.tokenize(source)
    paths: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token.text == "import":
            following = tokens[index + 1].text
            if following in ("public", "weak") and index + 2 < len(tokens):
                following = tokens[index + 2].text
            paths.append(following.strip("\"'"))
    return paths


def check_struct_option(
    source: str, path: str, definition: Definition
) -> list[Violation]:
    """The nproto option of a file whose one definition is `definition` (a message or an enum)."""
    expected = (
        NPROTO_STRUCT_OPTION if definition.keyword == "message" else NPROTO_ENUM_OPTION
    )
    example = f'option ({expected}) = "<namespace>::{definition.name}";'
    violations: list[Violation] = []
    own: NprotoOption | None = None
    for option in nproto_options(source):
        if not option.scope:
            violations.append(
                Violation(
                    path,
                    option.line,
                    f"({option.name}) is declared at file scope; move it into {definition.keyword} "
                    f"{definition.name}, as the first statement of its body: '{example}'",
                )
            )
        elif option.scope == ((definition.keyword, definition.name),):
            if own is not None:
                violations.append(
                    Violation(
                        path,
                        option.line,
                        f"a second nproto option in {definition.keyword} {definition.name}",
                    )
                )
            else:
                own = option
        else:
            nested = ".".join(name for _, name in option.scope)
            violations.append(
                Violation(
                    path,
                    option.line,
                    f"({option.name}) in nested type {nested}: nested types are generated as nested C++ types of "
                    f"their parent's struct and carry no option",
                )
            )
    if own is None:
        if not any(not o.scope for o in nproto_options(source)):
            violations.append(
                Violation(
                    path,
                    definition.line,
                    f"{definition.keyword} {definition.name} does not name the C++ type nproto generates for it: write "
                    f"'{example}' as the first statement of its body (with 'import \"{NPROTO_OPTIONS_IMPORT}\";')",
                )
            )
        return violations
    if own.name != expected:
        violations.append(
            Violation(
                path,
                own.line,
                f"a {definition.keyword} takes ({expected}), not ({own.name})",
            )
        )
    if not own.first_in_body:
        violations.append(
            Violation(
                path,
                own.line,
                f"({own.name}) must be the first statement of {definition.keyword} {definition.name}",
            )
        )
    if (
        not _CPP_QUALIFIED_NAME.fullmatch(own.value)
        or own.value.split("::")[-1] != definition.name
    ):
        violations.append(
            Violation(
                path,
                own.line,
                f"'{own.value}' is not a fully qualified C++ name ending in ::{definition.name} (e.g. "
                f'"ocs2::humanoid::msgs::{definition.name}")',
            )
        )
    if NPROTO_OPTIONS_IMPORT not in imports(source):
        violations.append(
            Violation(
                path,
                own.line,
                f"uses ({own.name}) without 'import \"{NPROTO_OPTIONS_IMPORT}\";'",
            )
        )
    return violations


def check_source(source: str, path: str) -> list[Violation]:
    """Violations of the one-definition-per-file rule in `source`, which is the file at `path`."""
    definitions = top_level_definitions(source)
    stem = os.path.basename(path)
    if stem.endswith(PROTO_EXTENSION):
        stem = stem[: -len(PROTO_EXTENSION)]
    if not definitions:
        return [
            Violation(
                path,
                1,
                "defines no message, enum or service; every .proto file defines exactly one",
            )
        ]
    if all(d.keyword == "extend" for d in definitions):
        # An options file (tools/nproto/options.proto): it declares extensions, not a type to name the file after.
        return []
    violations: list[Violation] = check_reserved_names(source, path)
    if len(definitions) > 1:
        names = ", ".join(f"{d.keyword} {d.name}" for d in definitions)
        for definition in definitions[1:]:
            violations.append(
                Violation(
                    path,
                    definition.line,
                    f"{definition.keyword} {definition.name} is a further top-level definition here ({names}); move "
                    f"it to its own file, {snake_case(definition.name)}{PROTO_EXTENSION}",
                )
            )
    first = definitions[0]
    if snake_case(first.name) != stem:
        violations.append(
            Violation(
                path,
                first.line,
                f"{first.keyword} {first.name} belongs in {snake_case(first.name)}{PROTO_EXTENSION}, not in "
                f"{stem}{PROTO_EXTENSION}",
            )
        )
    if first.keyword in ("message", "enum"):
        violations += check_struct_option(source, path, first)
    return violations


def check_files(paths: Iterable[str], root: str) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


def check_staged(repository: str) -> list[Violation]:
    """Checks the STAGED version of every .proto file staged for commit, as the pre-commit hook needs."""
    violations: list[Violation] = []
    for path in lint_files.staged_files(repository):
        if path.endswith(PROTO_EXTENSION):
            violations += check_source(lint_files.staged_source(repository, path), path)
    return violations


def _findings(source: str, path: str) -> list[check_types.Finding]:
    return [
        check_types.Finding(path, v.line, 1, NAME, v.text)
        for v in check_source(source, path)
    ]


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.PROTO}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        description="one top-level definition per .proto file, named after it, with its nproto option (google3's "
        "1-1-1 rule); no message reserves a field name.",
        hint="Give each .proto file one definition named after the file (message MpcPolicy -> mpc_policy.proto) whose "
        "body opens with 'option (nproto.generate_struct) = \"<namespace>::MpcPolicy\";'. Retire a field with "
        '(nproto.retired_field), not with `reserved "name"`.',
    )
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help=".proto files")
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="check the staged version of every .proto file staged for commit (the pre-commit hook)",
    )
    args = parser.parse_args(argv)
    if args.git_staged == bool(args.paths):
        parser.error("give either .proto paths or --git-staged")
    if args.git_staged:
        violations = check_staged(os.getcwd())
    else:
        violations = check_files(args.paths, os.getcwd())
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
