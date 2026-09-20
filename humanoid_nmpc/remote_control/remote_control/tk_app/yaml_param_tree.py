"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""Turns any YAML configuration into the tree of tunable parameters a GUI can be built from.

Nothing here knows the name of a single parameter. A file is read, every numeric leaf becomes a tunable, its label is
built from the key and the trailing comment the file already carries next to it, and its slider range is derived from
its own magnitude. A parameter added to a configuration therefore reaches the GUI with no code written anywhere, which
is the point: the names live in the YAML, not in Python.
"""

import os
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from remote_control.tk_app.yaml_editor_utils import load_yaml_safe

#: A key that is a matrix entry, e.g. "(2,2)" or "(11,0)". Matched by shape, never by the matrix's name.
MATRIX_KEY = re.compile(r'^"?\(\s*\d+\s*,\s*\d+\s*\)"?$')

#: `key: value  # comment` on one line, capturing the indent, the key and the comment. Quoted keys are kept as written
#: so that a path built from them matches the file, which is what the writer needs to find the line again.
_KEY_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<key>\"[^\"]+\"|'[^']+'|[^\s:#][^:#]*?)\s*:\s*(?P<rest>.*)$"
)


class Tunable(NamedTuple):
    """One numeric leaf of a configuration."""

    path: List[str]  #: the key path, as the file spells it, e.g. ["Q_com", '"(2,2)"']
    value: float
    label: (
        str  #: `key [trailing comment]` where the file has a comment, else the bare key
    )
    minimum: float
    maximum: float

    @property
    def dotted(self) -> str:
        return ".".join(self.path)


def slider_range(value: float) -> Tuple[float, float]:
    """The range a slider gets for a value, derived from the value alone.

    Four times the magnitude covers a weight someone wants to raise substantially while keeping the useful part of the
    travel usable, and the minimum of one keeps a slider around a value of zero from collapsing to a point. A negative
    value opens the range symmetrically, since a quantity that is negative at all is one that has a sign.
    """
    magnitude = max(abs(value) * 4.0, 1.0)
    return (-magnitude, magnitude) if value < 0.0 else (0.0, magnitude)


def trailing_comments(file_path: str) -> Dict[str, str]:
    """Every `key: value  # comment` of a file, as dotted path -> comment text.

    The file is read as text rather than through the parser because a YAML loader discards comments, and the comments
    are where this repository keeps what each number means: `"(2,2)": 15  # p_com_z (matches p_base_z)`.
    """
    comments: Dict[str, str] = {}
    if not file_path or not os.path.exists(file_path):
        return comments
    stack: List[Tuple[int, str]] = []  # (indent, key) of the blocks currently open
    with open(file_path, "r") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _KEY_LINE.match(line)
            if match is None:
                continue
            indent = len(match.group("indent"))
            key = match.group("key").strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = [entry[1] for entry in stack] + [key]
            rest = match.group("rest")
            hashPosition = rest.find("#")
            if hashPosition >= 0:
                comment = rest[hashPosition + 1 :].strip()
                if comment:
                    comments[".".join(path)] = comment
            stack.append((indent, key))
    return comments


def label_for(key: str, comment: Optional[str]) -> str:
    """The label a GUI shows for one parameter: `key [comment]`, or the bare key where the file has no comment.

    Both halves earn their place. The KEY is what the task file calls the parameter, so it is what an engineer greps
    for, what an error message names and what they have to type to change it somewhere other than the GUI; a label
    that showed only the comment made the slider unsearchable against the file it edits. The COMMENT is what the
    number means, which the key rarely says on its own - `"(2,2)"` is unreadable, `(2,2) [p_com_z - effective weight
    1275, matches p_base_z]` is not.

    The comment is taken verbatim, including any units or derivation the file records, because editing it here would
    put a second description of the parameter in Python and that is exactly what this module exists to avoid.
    """
    return "%s [%s]" % (key, comment) if comment else key


def tunables(
    file_path: str, root: Optional[dict] = None, prefix: Optional[List[str]] = None
) -> List[Tunable]:
    """Every numeric leaf of a configuration file, in file order, labelled from its trailing comment.

    Booleans are left out: a slider would write a float back into a bool key and break the next reload. Strings and
    lists are left out because a slider cannot express them.
    """
    data = load_yaml_safe(file_path) if root is None else root
    comments = trailing_comments(file_path)

    found: List[Tunable] = []

    def walk(node, path: List[str]):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [str(key)])
            return
        if isinstance(node, bool) or node is None or not isinstance(node, (int, float)):
            return
        dotted = ".".join(path)
        # The file may spell a matrix key quoted; the comment index is keyed as written, so try both.
        comment = comments.get(dotted) or comments.get(
            ".".join(path[:-1] + ['"%s"' % path[-1]])
        )
        minimum, maximum = slider_range(float(node))
        found.append(
            Tunable(
                path=list(path),
                value=float(node),
                label=label_for(path[-1], comment),
                minimum=minimum,
                maximum=maximum,
            )
        )

    walk(data, list(prefix) if prefix else [])
    return found


def group_by_block(found: List[Tunable]) -> Dict[str, List[Tunable]]:
    """The tunables grouped by their top-level block, in the order the file lists them.

    This is what the GUI turns into its categories: the configuration's own structure, rather than a curated list of
    group names that has to be maintained alongside it.
    """
    grouped: Dict[str, List[Tunable]] = {}
    for tunable in found:
        block = tunable.path[0] if len(tunable.path) > 1 else ""
        grouped.setdefault(block, []).append(tunable)
    return grouped
