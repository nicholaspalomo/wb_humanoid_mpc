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

"""Python language rules of the Google Python Style Guide that pylint does not check (2.4-2.19, 3.2, 3.7, 3.17, 3.19).

- `py-staticmethod` (2.17): no `@staticmethod`; write a module-level function. An external API that forces one takes
  `# NOLINT(py-staticmethod): <the API>`.
- `py-complex-comprehension` (2.7): at most one `for` clause and one condition.
- `py-long-ternary` (2.11): each part of a conditional expression fits on one line.
- `py-long-lambda` (2.10): a lambda is one line of at most 80 characters.
- `py-length-test` (2.14): `if not seq:`, not `if len(seq) == 0:`.
- `py-assert` (2.4): no `assert` outside tests; raise an exception.
- `py-exception-name` (2.4): an exception class name ends in `Error`.
- `py-power-feature` (2.19): no `__del__`, metaclasses, `exec` or `eval` outside tests.
- `py-backslash` (3.2): no backslash line continuations; use parentheses.
- `py-type-comment` (3.19): no `# type:` comments other than `# type: ignore[code]`; annotate.
- `py-pragma-reason` (AGENTS.md): every `# pylint: disable=...` / `disable-next=...` and every `# type: ignore[code]`
  says why after it, `  # <reason>` (a standalone pylint pragma may carry its reason on the comment line above it
  instead), and `# pylint: skip-file` is not used.
- `py-shebang` (3.7): a shebang if and only if the file is executable, and then exactly `#!/usr/bin/env python3`.
- `py-main-guard` (3.17): the `__main__` guard calls `main()`, and nothing else runs at import time.
"""

import ast
import io
import os
import re
import tokenize

from tools.hooks import check_types
from tools.hooks import lint_files

STATICMETHOD = "py-staticmethod"
COMPLEX_COMPREHENSION = "py-complex-comprehension"
LONG_TERNARY = "py-long-ternary"
LONG_LAMBDA = "py-long-lambda"
LENGTH_TEST = "py-length-test"
ASSERT = "py-assert"
EXCEPTION_NAME = "py-exception-name"
POWER_FEATURE = "py-power-feature"
BACKSLASH = "py-backslash"
TYPE_COMMENT = "py-type-comment"
PRAGMA_REASON = "py-pragma-reason"
SHEBANG = "py-shebang"
MAIN_GUARD = "py-main-guard"

PYTHON = frozenset({check_types.Language.PYTHON})
SHEBANG_LINE = "#!/usr/bin/env python3"
MAX_LAMBDA_LENGTH = 80
# Calls a module may make at import time, outside its __main__ guard.
TOP_LEVEL_CALLS = frozenset(
    {
        "sys.path.insert",
        "sys.path.append",
        "os.environ.setdefault",
        "warnings.filterwarnings",
        "warnings.simplefilter",
        "matplotlib.use",
        "logging.basicConfig",
        "jax.config.update",
        "np.set_printoptions",
        "absl.flags.DEFINE_string",
    }
)

# The repository root, against which the executable bit of a file is read; the tests point it elsewhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _finding(path: str, node: ast.AST, name: str, message: str) -> check_types.Finding:
    return check_types.Finding(
        path,
        getattr(node, "lineno", 1),
        getattr(node, "col_offset", 0) + 1,
        name,
        message,
    )


def _name(node: ast.AST) -> str:
    """The dotted name of a Name / Attribute chain, or ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def check_staticmethod(source: str, path: str) -> list[check_types.Finding]:
    """`@staticmethod`."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if _name(decorator) == "staticmethod":
                findings.append(
                    _finding(
                        path,
                        decorator,
                        STATICMETHOD,
                        f"`@staticmethod` on `{node.name}`: write a module-level function (Python style 2.17).",
                    )
                )
    return findings


def check_complex_comprehension(source: str, path: str) -> list[check_types.Finding]:
    """Comprehensions with more than one `for` or more than one `if`."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            continue
        conditions = sum(len(g.ifs) for g in node.generators)
        if len(node.generators) > 1 or conditions > 1:
            findings.append(
                _finding(
                    path,
                    node,
                    COMPLEX_COMPREHENSION,
                    "a comprehension has one `for` clause and one condition at most: write a loop "
                    "(Python style 2.7).",
                )
            )
    return findings


def check_long_ternary(source: str, path: str) -> list[check_types.Finding]:
    """Conditional expressions with a part that spans lines."""
    tree = _parse(source)
    if tree is None:
        return []
    return [
        _finding(
            path,
            node,
            LONG_TERNARY,
            "each part of a conditional expression fits on one line: write an `if` statement (Python style 2.11).",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and any(
            part.lineno != part.end_lineno
            for part in (node.body, node.test, node.orelse)
        )
    ]


def check_long_lambda(source: str, path: str) -> list[check_types.Finding]:
    """Lambdas that span lines or exceed MAX_LAMBDA_LENGTH characters."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Lambda):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "\n" in segment or len(segment) > MAX_LAMBDA_LENGTH:
            findings.append(
                _finding(
                    path,
                    node,
                    LONG_LAMBDA,
                    f"a lambda is one line of at most {MAX_LAMBDA_LENGTH} characters: define a function "
                    "(Python style 2.10).",
                )
            )
    return findings


def _is_length_test(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    left, right = node.left, node.comparators[0]
    if isinstance(right, ast.Call) and isinstance(left, ast.Constant):
        left, right = right, left
    return (
        isinstance(left, ast.Call)
        and _name(left.func) == "len"
        and isinstance(right, ast.Constant)
        and right.value == 0
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Gt, ast.Lt))
    )


def check_length_test(source: str, path: str) -> list[check_types.Finding]:
    """`len(x) == 0`, `len(x) != 0` and `len(x) > 0` used as a condition."""
    tree = _parse(source)
    if tree is None:
        return []
    conditions: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            conditions.append(node.test)
        elif isinstance(node, ast.BoolOp):
            conditions.extend(node.values)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            conditions.append(node.operand)
        elif isinstance(node, ast.comprehension):
            conditions.extend(node.ifs)
    return [
        _finding(
            path,
            condition,
            LENGTH_TEST,
            "a length compared with 0 as a condition: test the sequence itself, `if seq:` / `if not seq:` "
            "(Python style 2.14).",
        )
        for condition in conditions
        if _is_length_test(condition)
    ]


def check_assert(source: str, path: str) -> list[check_types.Finding]:
    """`assert` statements."""
    tree = _parse(source)
    if tree is None:
        return []
    return [
        _finding(
            path,
            node,
            ASSERT,
            "`assert` outside a test: raise an exception (ValueError, ...), since `python -O` drops asserts "
            "(Python style 2.4).",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]


_EXCEPTION_BASE = re.compile(r"(^|\.)(\w*Error|\w*Exception|BaseException)$")


def check_exception_name(source: str, path: str) -> list[check_types.Finding]:
    """Exception classes whose name does not end in `Error`."""
    tree = _parse(source)
    if tree is None:
        return []
    return [
        _finding(
            path,
            node,
            EXCEPTION_NAME,
            f"exception class `{node.name}`: its name ends in `Error` (Python style 2.4).",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(_EXCEPTION_BASE.search(_name(base)) for base in node.bases)
        and not node.name.endswith("Error")
    ]


def check_power_feature(source: str, path: str) -> list[check_types.Finding]:
    """`__del__`, metaclasses, `exec` and `eval`."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        what = None
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__del__"
        ):
            what = "`__del__`"
        elif isinstance(node, ast.ClassDef) and any(
            k.arg == "metaclass" for k in node.keywords
        ):
            what = "a metaclass"
        elif isinstance(node, ast.Call) and _name(node.func) in ("exec", "eval"):
            what = f"`{_name(node.func)}`"
        if what:
            findings.append(
                _finding(
                    path,
                    node,
                    POWER_FEATURE,
                    f"{what} is a power feature this code does not use (Python style 2.19).",
                )
            )
    return findings


def check_backslash(source: str, path: str) -> list[check_types.Finding]:
    """Backslash line continuations outside strings."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    inside: set[int] = set()
    fstring_starts: list[int] = []
    for token in tokens:
        if token.type == tokenize.STRING:
            inside.update(range(token.start[0], token.end[0]))
        elif token.type == getattr(tokenize, "FSTRING_START", -1):
            fstring_starts.append(token.start[0])
        elif token.type == getattr(tokenize, "FSTRING_END", -1) and fstring_starts:
            inside.update(range(fstring_starts.pop(), token.end[0]))
        elif token.type == tokenize.COMMENT:
            inside.add(-token.start[0])
    findings = []
    for number, line in enumerate(source.split("\n"), start=1):
        if (
            line.rstrip("\r").endswith("\\")
            and number not in inside
            and -number not in inside
        ):
            findings.append(
                check_types.Finding(
                    path,
                    number,
                    len(line.rstrip("\r")),
                    BACKSLASH,
                    "backslash line continuation: wrap in parentheses (Python style 3.2).",
                )
            )
    return findings


_TYPE_COMMENT = re.compile(r"#\s*type:(?!\s*ignore\[)")


def check_type_comment(source: str, path: str) -> list[check_types.Finding]:
    """`# type:` comments, other than `# type: ignore[code]`."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return [
        check_types.Finding(
            path,
            token.start[0],
            token.start[1] + 1,
            TYPE_COMMENT,
            "`# type:` comment: write an annotation; only `# type: ignore[<code>]` is used (Python style 3.19).",
        )
        for token in tokens
        if token.type == tokenize.COMMENT and _TYPE_COMMENT.match(token.string)
    ]


_PYLINT_PRAGMA = re.compile(
    r"\bpylint:\s*(?P<kind>disable-next|disable|skip-file)\b(?:\s*=\s*(?P<messages>[\w\s,-]*[\w-]))?(?P<rest>.*)$"
)
_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore\[[^\]]*\](?P<rest>.*)$")
# A reason after a pragma: a comment of its own, not a NOLINT marker (which needs a reason too).
_PRAGMA_REASON = re.compile(r"^\s+#\s*(?!NOLINT)\S")


def _reason_above(lines: list[str], line: int) -> bool:
    """True when the line above `line` (1-based) is a comment that is not itself a pragma: a block pragma's reason."""
    if line < 2:
        return False
    above = lines[line - 2].strip()
    return (
        above.startswith("#")
        and bool(above.lstrip("#").strip())
        and not _PYLINT_PRAGMA.search(above)
    )


def check_pragma_reason(source: str, path: str) -> list[check_types.Finding]:
    """pylint pragmas and `# type: ignore[code]` comments without their reason, and `# pylint: skip-file`."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    lines = source.split("\n")
    findings = []

    def add(token: tokenize.TokenInfo, message: str) -> None:
        findings.append(
            check_types.Finding(
                path,
                token.start[0],
                token.start[1] + 1,
                PRAGMA_REASON,
                f"{message} (AGENTS.md).",
            )
        )

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        ignore = _TYPE_IGNORE.search(token.string)
        if ignore and not _PRAGMA_REASON.match(ignore.group("rest")):
            add(
                token,
                "`# type: ignore[<code>]` without its reason: append `  # <why the type checker is wrong here>`",
            )
        pragma = _PYLINT_PRAGMA.search(token.string)
        if pragma is None:
            continue
        if pragma.group("kind") == "skip-file":
            add(
                token,
                "`# pylint: skip-file` silences every message of the file: disable each one where it occurs, with "
                "its reason",
            )
            continue
        if _PRAGMA_REASON.match(pragma.group("rest")):
            continue
        standalone = not token.line[: token.start[1]].strip()
        if standalone and _reason_above(lines, token.start[0]):
            continue
        add(
            token,
            f"`# pylint: {pragma.group('kind')}=...` without its reason: append `  # <reason>`, as "
            "`# pylint: disable=<message>  # <reason>`",
        )
    return findings


def check_shebang(source: str, path: str) -> list[check_types.Finding]:
    """A shebang on a file that is not executable, or none (or another one) on a file that is."""
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return []
    executable = os.access(full, os.X_OK)
    first = source.split("\n", 1)[0]
    message = None
    if first.startswith("#!") and not executable:
        message = "a shebang on a file that is not executable: delete it, or `chmod +x` the file"
    elif executable and first != SHEBANG_LINE:
        message = f"an executable file starts with exactly `{SHEBANG_LINE}`"
    if message is None:
        return []
    return [check_types.Finding(path, 1, 1, SHEBANG, f"{message} (Python style 3.7).")]


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    parts = [node.test.left] + list(node.test.comparators)
    return any(isinstance(p, ast.Name) and p.id == "__name__" for p in parts) and any(
        isinstance(p, ast.Constant) and p.value == "__main__" for p in parts
    )


def check_main_guard(source: str, path: str) -> list[check_types.Finding]:
    """A `__main__` guard that does not call `main()`, and calls at import time."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in tree.body:
        if _is_main_guard(node):
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if not any(
                _name(c.func).split(".")[-1] == "main"
                or any(_name(a) == "main" for a in c.args)
                for c in calls
            ):
                findings.append(
                    _finding(
                        path,
                        node,
                        MAIN_GUARD,
                        "the `__main__` guard calls `main()` (or `app.run(main)`), which holds the program "
                        "(Python style 3.17).",
                    )
                )
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = _name(node.value.func)
            if name not in TOP_LEVEL_CALLS:
                findings.append(
                    _finding(
                        path,
                        node,
                        MAIN_GUARD,
                        f"`{name or 'a call'}(...)` runs at import time: move it into `main()` (Python style 3.17).",
                    )
                )
    return findings


def _check(
    name: str,
    function: check_types.CheckFunction,
    description: str,
    non_test: bool = False,
) -> check_types.Check:
    return check_types.Check(
        name=name,
        languages=PYTHON,
        scope=(
            lint_files.Scope.FIRST_PARTY_NON_TEST
            if non_test
            else lint_files.Scope.FIRST_PARTY
        ),
        check_source=function,
        description=description,
    )


CHECKS = [
    _check(
        STATICMETHOD,
        check_staticmethod,
        "no @staticmethod; a module-level function (Python style 2.17).",
    ),
    _check(
        COMPLEX_COMPREHENSION,
        check_complex_comprehension,
        "comprehensions have one for and one condition (2.7).",
    ),
    _check(
        LONG_TERNARY,
        check_long_ternary,
        "each part of a conditional expression fits on one line (2.11).",
    ),
    _check(LONG_LAMBDA, check_long_lambda, "lambdas are one short line (2.10)."),
    _check(
        LENGTH_TEST, check_length_test, "`if not seq:`, not `if len(seq) == 0:` (2.14)."
    ),
    _check(ASSERT, check_assert, "no assert outside tests (2.4).", non_test=True),
    _check(
        EXCEPTION_NAME,
        check_exception_name,
        "exception class names end in Error (2.4).",
    ),
    _check(
        POWER_FEATURE,
        check_power_feature,
        "no __del__, metaclasses, exec or eval outside tests (2.19).",
        non_test=True,
    ),
    _check(BACKSLASH, check_backslash, "no backslash line continuations (3.2)."),
    _check(
        TYPE_COMMENT, check_type_comment, "annotations, not # type: comments (3.19)."
    ),
    _check(
        PRAGMA_REASON,
        check_pragma_reason,
        "pylint pragmas and type: ignore comments say why; no pylint: skip-file (AGENTS.md).",
    ),
    _check(SHEBANG, check_shebang, "a shebang exactly on executable files (3.7)."),
    _check(
        MAIN_GUARD,
        check_main_guard,
        "the __main__ guard calls main(); nothing else runs at import (3.17).",
    ),
]
