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

"""TODO comments say who or what tracks them: `TODO: bug 123 - <text>` or `TODO(<owner or bug>): <text>`.

The Google C++ Style Guide (TODO comments) and the Google Python Style Guide (3.12) want a TODO to carry a bug, a link
or a person, so that whoever reads it can find out more. Both forms are accepted, in C++, Python, shell and Starlark
comments:

    // TODO: bug 123 - Remove this once the solver reports its own iterations.
    # TODO: https://github.com/org/repo/issues/42 - Drop the Python 3.11 workaround.
    // TODO(npalomo): Tune the swing height on hardware.

A reference is `bug <number>`, `#<number>`, `b/<number>` or a URL. cpplint's readability/todo and clang-tidy's
google-readability-todo stay off, because they accept only `TODO(user)`.
"""

import re

from tools.hooks import check_types
from tools.hooks import lint_files
from tools.hooks import nolint

NAME = "todo-format"

_TODO = re.compile(r"\bTODO\b")
_WITH_OWNER = re.compile(r"^TODO\([^)\s][^)]*\):\s*\S")
_WITH_REFERENCE = re.compile(r"^TODO:\s*(bug \d+|#\d+|b/\d+|https?://\S+)\s+-\s+\S")


def check_source(source: str, path: str = "<source>") -> list[check_types.Finding]:
    """The TODO comments of `source` in neither accepted form."""
    style = nolint.comment_style(path)
    if style == nolint.WHOLE_LINE:
        return []
    findings = []
    for number, text in enumerate(nolint.comment_lines(source, style), start=1):
        for match in _TODO.finditer(text):
            if match.start() > 0 and text[match.start() - 1] in "`'\"":
                continue  # quoted, so prose
            rest = text[match.start() :]
            if _WITH_OWNER.match(rest) or _WITH_REFERENCE.match(rest):
                continue
            findings.append(
                check_types.Finding(
                    path,
                    number,
                    match.start() + 1,
                    NAME,
                    "TODO without a reference: write `TODO: bug 123 - <text>` (or #123, b/123, a URL) or "
                    "`TODO(<owner>): <text>` (Google C++ style, TODO comments; Python style 3.12).",
                )
            )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset(
            {
                check_types.Language.CPP,
                check_types.Language.PYTHON,
                check_types.Language.SHELL,
                check_types.Language.STARLARK,
            }
        ),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_source,
        description="a TODO names a bug, a link or an owner (G: TODO comments; Python style 3.12).",
    )
]
