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


"""Checks that every .textproto file names its schema in its leading comment block, and that the schema exists.

A configuration file is a textproto of one message (tools/nproto/README.md, "Textproto configuration files"), and it
says which in its first lines:

    # proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto
    # proto-message: humanoid_mpc_config.TaskFile

The check fails when the leading comment block (the comment lines before the first line that is neither blank nor a
comment) lacks either line, when the proto-file is not a file of the repository (its path from the repository root),
and when the message is not a top-level message of that file, written with the file's package. The parsers refuse a
file whose header names another message than the one they parse into (nproto/Textproto.h, nproto_textproto.py),
so the header is checked against the schema here and against the reader there.

    python3 -m tools.hooks.lint_code --only textproto-header

`python3 -m tools.hooks.lint_code` (make lint) and the pre-commit hook run it as the registry check textproto-header.
"""

import os
import re
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import lint_files
from tools.hooks import proto_file_layout

NAME = "textproto-header"

TEXTPROTO_EXTENSION = ".textproto"
# The header lines, as the parsers read them.
# LINT.IfChange(header_lines)
_PROTO_FILE = re.compile(r"#\s*proto-file:\s*(\S+)")
_PROTO_MESSAGE = re.compile(r"#\s*proto-message:\s*(\S+)")
# LINT.ThenChange(//tools/nproto/src/Textproto.cpp:header_lines, //tools/nproto/nproto_textproto.py:header_lines)
_PACKAGE = re.compile(r"^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;", re.MULTILINE)

# The repository root, against which the proto-file paths are resolved; the tests pass another one.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_EXAMPLE = (
    "'# proto-file: <the .proto file, from the repository root>' and '# proto-message: <package>.<Message>', "
    "e.g. '# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto' and "
    "'# proto-message: humanoid_mpc_config.TaskFile'"
)


class Violation(NamedTuple):
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.text}"


class Header(NamedTuple):
    """The header lines of a textproto: (1-based line, value) of each, or None when it is missing."""

    proto_file: tuple[int, str] | None
    proto_message: tuple[int, str] | None


def read_header(source: str) -> Header:
    """The `# proto-file:` and `# proto-message:` lines of the leading comment block of `source`."""
    proto_file: tuple[int, str] | None = None
    proto_message: tuple[int, str] | None = None
    for number, line in enumerate(source.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break  # The leading comment block has ended.
        file_match = _PROTO_FILE.match(stripped)
        message_match = _PROTO_MESSAGE.match(stripped)
        if file_match and proto_file is None:
            proto_file = (number, file_match.group(1))
        elif message_match and proto_message is None:
            proto_message = (number, message_match.group(1))
    return Header(proto_file, proto_message)


def schema_messages(proto_source: str) -> list[str]:
    """The full names (`package.Message`) of the top-level messages of a .proto file."""
    package = _PACKAGE.search(proto_source)
    prefix = package.group(1) + "." if package else ""
    return [
        prefix + definition.name
        for definition in proto_file_layout.top_level_definitions(proto_source)
        if definition.keyword == "message"
    ]


def check_header(source: str, path: str, root: str = ROOT) -> list[Violation]:
    """The problems of the header of the textproto `source`, the file at `path`; `root` resolves its proto-file."""
    if not path.endswith(TEXTPROTO_EXTENSION):
        return []
    header = read_header(source)
    if header.proto_file is None or header.proto_message is None:
        missing = [
            name
            for name, line in (
                ("# proto-file:", header.proto_file),
                ("# proto-message:", header.proto_message),
            )
            if line is None
        ]
        return [
            Violation(
                path,
                1,
                f"the leading comment block has no {' and no '.join(missing)} line; a configuration file names its "
                f"schema with {_EXAMPLE}",
            )
        ]
    file_line, proto_path = header.proto_file
    message_line, message_name = header.proto_message
    proto = os.path.join(root, proto_path)
    if not os.path.isfile(proto):
        return [
            Violation(
                path,
                file_line,
                f"'# proto-file: {proto_path}' names no file of the repository (the path is from its root)",
            )
        ]
    with open(proto, encoding="utf-8") as f:
        messages = schema_messages(f.read())
    if message_name not in messages:
        return [
            Violation(
                path,
                message_line,
                f"'# proto-message: {message_name}' is not a top-level message of {proto_path}, which defines "
                f"{', '.join(messages) or 'none'}",
            )
        ]
    return []


def _findings(source: str, path: str) -> list[check_types.Finding]:
    return [
        check_types.Finding(path, v.line, 1, NAME, v.text)
        for v in check_header(source, path)
    ]


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.TEXT}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        description="every .textproto names its schema (# proto-file: / # proto-message:) in its leading comment "
        "block, and the schema is a message of a .proto file of the repository.",
        hint="Start the file with '# proto-file: <path of the .proto from the repository root>' and "
        "'# proto-message: <package>.<Message>' (tools/nproto/README.md).",
    )
]
