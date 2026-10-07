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

"""Python docstrings and license headers as the Google Python Style Guide writes them (3.8).

- `py-license-docstring` (3.8.2): the license is a `#` comment block above the module docstring, never the docstring
  (a module docstring whose first line, below any `****` rule, starts with "Copyright").
- `py-license-header` (3.8.2; decision D5): every file starts with the BSD-3 license comment, "Copyright (c) <year>,
  <holder>. All rights reserved.", the three conditions and the disclaimer (the C++ files' header, as `#` comments;
  tools/hooks/license_header.py reads it).
- `py-docstring-summary` (3.8.1): the summary is one physical line of at most 140 columns, ending in `.`, `?` or `!`,
  followed by a blank line or the end of the docstring.
- `py-docstring-sections` (3.8.3): a function whose docstring goes beyond its summary documents its parameters under
  `Args:`, and what it returns under `Returns:` (a generator: `Yields:`), unless the summary already starts with
  "Return" or "Yield". One-line docstrings and functions shorter than DOCSTRING_MIN_LENGTH lines (pylint's
  docstring-min-length) are exempt.
- `py-property-docstring` (3.8.3): a property's docstring describes the attribute ("The robot's name."), not a getter
  ("Returns the robot's name.").
"""

import ast
import re
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import license_header
from tools.hooks import lint_files

LICENSE_DOCSTRING = "py-license-docstring"
LICENSE_HEADER = "py-license-header"
DOCSTRING_SUMMARY = "py-docstring-summary"
DOCSTRING_SECTIONS = "py-docstring-sections"
PROPERTY_DOCSTRING = "py-property-docstring"

PYTHON = frozenset({check_types.Language.PYTHON})
MAX_SUMMARY_COLUMNS = 140
DOCSTRING_MIN_LENGTH = 12
_ARGS = re.compile(r"^\s*(Args|Arguments):\s*$", re.MULTILINE)
_RETURNS = re.compile(r"^\s*(Returns|Return):", re.MULTILINE)
_YIELDS = re.compile(r"^\s*(Yields|Yield):", re.MULTILINE)


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


class _Docstring(NamedTuple):
    node: ast.Constant  # where it is
    text: str


def _docstring_node(node: ast.AST) -> _Docstring | None:
    """The docstring of a module, class or function, or None."""
    body = getattr(node, "body", None)
    if not body or not isinstance(body[0], ast.Expr):
        return None
    value = body[0].value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return _Docstring(value, value.value)
    return None


def _finding(path: str, node: ast.AST, name: str, message: str) -> check_types.Finding:
    return check_types.Finding(
        path,
        getattr(node, "lineno", 1),
        getattr(node, "col_offset", 0) + 1,
        name,
        message,
    )


def _starts_with_copyright(text: str) -> bool:
    """True when the first line of `text` that is not a `****` rule starts with "Copyright": a license, not prose."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped) - set("*#=-/"):
            return stripped.startswith("Copyright")
    return False


def check_license_docstring(source: str, path: str) -> list[check_types.Finding]:
    """A module docstring that holds the license."""
    tree = _parse(source)
    docstring = _docstring_node(tree) if tree is not None else None
    if docstring is None or not _starts_with_copyright(docstring.text):
        return []
    return [
        _finding(
            path,
            docstring.node,
            LICENSE_DOCSTRING,
            "the license is a `#` comment block above the module docstring, not the docstring (Python style 3.8.2).",
        )
    ]


def _leading_comment(source: str) -> tuple[list[str], int, int]:
    """The `#` lines that start a file (below a shebang and blank lines), the line of the first, and the line after."""
    lines: list[str] = []
    start = 1
    end = 1
    for number, line in enumerate(source.splitlines(), start=1):
        end = number
        stripped = line.strip()
        if not lines and (not stripped or (number == 1 and stripped.startswith("#!"))):
            continue
        if not stripped.startswith("#"):
            break
        if not lines:
            start = number
        lines.append(line)
    return lines, start, end


def check_license_header(source: str, path: str) -> list[check_types.Finding]:
    """A file whose first comment is not the complete BSD-3 license block (tools/hooks/license_header.py)."""
    if not source.strip():
        return []
    lines, start, end = _leading_comment(source)
    return [
        check_types.Finding(
            path,
            line,
            1,
            LICENSE_HEADER,
            f"{message} (Python style 3.8.2; decision D5).",
        )
        for line, message in license_header.finding_lines(
            license_header.comment_lines("\n".join(lines)), start, end
        )
    ]


def _documented(tree: ast.Module) -> list[ast.AST]:
    nodes: list[ast.AST] = [tree]
    nodes += [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return nodes


def check_docstring_summary(source: str, path: str) -> list[check_types.Finding]:
    """Docstring summaries that are not one line ending in punctuation, followed by a blank line."""
    tree = _parse(source)
    if tree is None:
        return []
    lines = source.splitlines()
    findings = []
    for node in _documented(tree):
        docstring = _docstring_node(node)
        if docstring is None:
            continue
        text = docstring.text
        doc_lines = text.split("\n")
        summary = doc_lines[0].strip()
        problem = None
        if not summary:
            problem = "the summary starts on the line of the opening quotes"
        elif summary[-1] not in ".?!":
            problem = "the summary ends in `.`, `?` or `!`"
        elif len(doc_lines) > 1 and doc_lines[1].strip():
            problem = "the summary is one line, followed by a blank line"
        else:
            physical = (
                lines[docstring.node.lineno - 1]
                if docstring.node.lineno - 1 < len(lines)
                else ""
            )
            if len(physical) > MAX_SUMMARY_COLUMNS:
                problem = f"the summary line fits in {MAX_SUMMARY_COLUMNS} columns"
        if problem:
            findings.append(
                _finding(
                    path,
                    docstring.node,
                    DOCSTRING_SUMMARY,
                    f"docstring summary: {problem} (Python style 3.8.1).",
                )
            )
    return findings


def _own_nodes(function: ast.AST) -> list[ast.AST]:
    """The nodes of `function`'s body, without those of the functions, lambdas and classes nested in it."""
    found = []
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = function.args
    names = [
        a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    ]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def _is_property(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id in ("property", "cached_property"))
        or (
            isinstance(d, ast.Attribute)
            and d.attr in ("property", "cached_property", "setter")
        )
        for d in function.decorator_list
    )


def check_docstring_sections(source: str, path: str) -> list[check_types.Finding]:
    """Multi-line function docstrings without their Args:, Returns: or Yields: sections."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) or _is_property(node):
            continue
        docstring = _docstring_node(node)
        if docstring is None:
            continue
        text = docstring.text
        if (
            not text.strip()
            or len([line for line in text.strip().split("\n") if line.strip()]) <= 1
        ):
            continue  # a one-line docstring
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        if length < DOCSTRING_MIN_LENGTH:
            continue
        summary = text.strip().split("\n")[0].strip()
        missing = []
        if _parameters(node) and not _ARGS.search(text):
            missing.append("Args:")
        own = _own_nodes(node)
        yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own)
        returns = any(isinstance(n, ast.Return) and n.value is not None for n in own)
        if yields and not _YIELDS.search(text) and not summary.startswith("Yield"):
            missing.append("Yields:")
        elif (
            returns
            and not yields
            and not _RETURNS.search(text)
            and not summary.startswith("Return")
        ):
            missing.append("Returns:")
        if missing:
            findings.append(
                _finding(
                    path,
                    node,
                    DOCSTRING_SECTIONS,
                    f"the docstring of `{node.name}` has no {' or '.join(missing)} section; each says something the "
                    "signature does not (Python style 3.8.3).",
                )
            )
    return findings


def check_property_docstring(source: str, path: str) -> list[check_types.Finding]:
    """Property docstrings written like a getter's."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) or not _is_property(node):
            continue
        docstring = _docstring_node(node)
        if docstring is not None and docstring.text.strip().startswith(
            ("Returns", "Return ")
        ):
            findings.append(
                _finding(
                    path,
                    docstring.node,
                    PROPERTY_DOCSTRING,
                    f'the property `{node.name}` describes the attribute ("The robot\'s name."), not a getter '
                    '("Returns ...") (Python style 3.8.3).',
                )
            )
    return findings


def _check(
    name: str, function: check_types.CheckFunction, description: str
) -> check_types.Check:
    return check_types.Check(
        name=name,
        languages=PYTHON,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=function,
        description=description,
    )


CHECKS = [
    _check(
        LICENSE_DOCSTRING,
        check_license_docstring,
        "the license is a comment, not the module docstring (3.8.2).",
    ),
    _check(
        LICENSE_HEADER,
        check_license_header,
        "every Python file starts with the BSD-3 license comment (3.8.2).",
    ),
    _check(
        DOCSTRING_SUMMARY,
        check_docstring_summary,
        "a docstring summary is one line ending in punctuation (3.8.1).",
    ),
    _check(
        DOCSTRING_SECTIONS,
        check_docstring_sections,
        "documented functions have Args:, Returns: / Yields: (3.8.3).",
    ),
    _check(
        PROPERTY_DOCSTRING,
        check_property_docstring,
        "property docstrings describe the attribute (3.8.3).",
    ),
]
