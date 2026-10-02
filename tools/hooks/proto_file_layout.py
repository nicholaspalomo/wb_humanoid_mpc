#!/usr/bin/env python3
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

    python3 -m tools.hooks.proto_file_layout <files>       # check
    python3 -m tools.hooks.proto_file_layout --git-staged  # check the staged .proto files

`python3 -m tools.hooks.lint_code` (make lint) and the pre-commit hook run it as the registry check proto-file-layout.
"""

import argparse
import os
import re
import sys
from typing import Iterable, List, NamedTuple, Optional, Tuple

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


def top_level_definitions(source: str) -> List[Definition]:
    """The messages, enums, services and extend blocks declared at file scope (nested types are not listed)."""
    definitions: List[Definition] = []
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
    scope: Tuple[Tuple[str, str], ...]
    first_in_body: bool  # it is the first statement of the body that declares it


def nproto_options(source: str) -> List[NprotoOption]:
    """Every `option (nproto.generate_*) = "...";` in `source`, with where it is declared."""
    tokens = proto_next_id.tokenize(source)
    found: List[NprotoOption] = []
    # One entry per open block: the definition it belongs to (None for an option value or rpc body) and whether a
    # statement has been seen in it yet.
    stack: List[List[object]] = []
    statement_start = True
    for index, token in enumerate(tokens):
        text = token.text
        if text == "{":
            previous = [t.text for t in tokens[max(0, index - 2) : index]]
            owner = (
                (previous[0], previous[1])
                if len(previous) == 2
                and previous[0] in DEFINITION_KEYWORDS + ("oneof",)
                else None
            )
            stack.append([owner, False])
            statement_start = True
            continue
        if text == "}":
            if stack:
                stack.pop()
            if stack:
                stack[-1][1] = True
            statement_start = True
            continue
        if text == ";":
            if stack:
                stack[-1][1] = True
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
                scope = tuple(entry[0] for entry in stack if entry[0] is not None)
                first = bool(stack) and not stack[-1][1]
                found.append(NprotoOption(window[1], value, token.line, scope, first))
        statement_start = False
    return found


def imports(source: str) -> List[str]:
    """The paths of the file's import statements."""
    tokens = proto_next_id.tokenize(source)
    paths: List[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token.text == "import":
            following = tokens[index + 1].text
            if following in ("public", "weak") and index + 2 < len(tokens):
                following = tokens[index + 2].text
            paths.append(following.strip("\"'"))
    return paths


def check_struct_option(
    source: str, path: str, definition: Definition
) -> List[Violation]:
    """The nproto option of a file whose one definition is `definition` (a message or an enum)."""
    expected = (
        NPROTO_STRUCT_OPTION if definition.keyword == "message" else NPROTO_ENUM_OPTION
    )
    example = f'option ({expected}) = "<namespace>::{definition.name}";'
    violations: List[Violation] = []
    own: Optional[NprotoOption] = None
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


def check_source(source: str, path: str) -> List[Violation]:
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
    violations: List[Violation] = []
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


def check_files(paths: Iterable[str], root: str) -> List[Violation]:
    violations: List[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


def check_staged(repository: str) -> List[Violation]:
    """Checks the STAGED version of every .proto file staged for commit, as the pre-commit hook needs."""
    violations: List[Violation] = []
    for path in lint_files.staged_files(repository):
        if path.endswith(PROTO_EXTENSION):
            violations += check_source(lint_files.staged_source(repository, path), path)
    return violations


def _findings(source: str, path: str) -> List[check_types.Finding]:
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
        "1-1-1 rule).",
        hint="Give each .proto file one definition named after the file (message MpcPolicy -> mpc_policy.proto) whose "
        "body opens with 'option (nproto.generate_struct) = \"<namespace>::MpcPolicy\";'.",
    )
]


def main(argv: Optional[List[str]] = None) -> int:
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
