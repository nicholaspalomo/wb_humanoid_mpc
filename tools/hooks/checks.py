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

"""The registry of the repository's token-level lint checks, and the engine that runs them on one file.

Each check lives in its own module of tools/hooks, which lists it in its `CHECKS`; REGISTRY is their concatenation. A
check's name is its NOLINT category (`// NOLINT(<name>): <reason>`) and its name for `lint_code --only`. The engine
(`run_file()`) runs the checks that read a file, applies the file's NOLINT markers, and reports the markers that are
themselves wrong (tools/hooks/nolint.py).

Every check is enforced: `make lint`, the pre-commit hook and CI run them all, and there is no list of checks that are
skipped until their findings are fixed (tools/hooks/test_lint_enforcement.py keeps it that way).

To add a check: write its module (`check_source(source, path) -> list[Finding]`, and `fix_source` where a rewrite cannot
change behavior), list it in the module's CHECKS, add the module to _MODULE_CHECKS below, give it a test_<module>.py,
and fix every finding it reports in the same change.
"""

import functools
import os
import re

from tools.hooks import american_spelling
from tools.hooks import argument_comments
from tools.hooks import boost_usage
from tools.hooks import check_types
from tools.hooks import cpp_google_style
from tools.hooks import cpp_totw
from tools.hooks import include_style
from tools.hooks import inclusive_language
from tools.hooks import license_header
from tools.hooks import lint_files
from tools.hooks import no_auto
from tools.hooks import nolint
from tools.hooks import pointer_nullability
from tools.hooks import proto_file_layout
from tools.hooks import proto_next_id
from tools.hooks import python_docstrings
from tools.hooks import python_imports
from tools.hooks import python_language
from tools.hooks import textproto_headers
from tools.hooks import todo_format
from tools.hooks import yaml_usage

Check = check_types.Check
Finding = check_types.Finding
Language = check_types.Language

# The languages whose files have a comment syntax that NOLINT markers are read from.
_CODE_LANGUAGES = nolint.CODE_LANGUAGES

NOLINT_UNUSED = "nolint-unused"


def _no_findings(source: str, path: str) -> list[Finding]:
    """The check_source of the marker checks, which the engine evaluates itself."""
    del source, path  # Unused.
    return []


def _marker_check(name: str, description: str) -> Check:
    return Check(
        name=name,
        languages=_CODE_LANGUAGES,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_no_findings,
        description=description,
        hint="A NOLINT marker names the checks it silences and says why: `// NOLINT(<check>): <reason>` "
        "(tools/hooks/nolint.py).",
    )


_NOLINT_CHECKS = [
    _marker_check(
        "nolint-category",
        "a NOLINT marker names the checks it silences: a bare NOLINT silences every cpplint and clang-tidy check.",
    ),
    _marker_check("nolint-reason", "a NOLINT marker gives its reason after a colon."),
    _marker_check(
        "nolint-unknown",
        "a NOLINT marker names a check of the registry, a cpplint category or a clang-tidy check of .clang-tidy.",
    ),
    _marker_check(
        NOLINT_UNUSED,
        "a NOLINT marker for one of the repository's own checks suppresses a finding.",
    ),
    _marker_check(
        "nolint-unbalanced",
        "every NOLINTBEGIN is closed by a NOLINTEND of the same check, and every NOLINTEND closes one.",
    ),
]

# The check modules. AGENTS.md lists what each enforces ("Enforced by" and the ToTW table): a module added or removed
# here, or a check added to or removed from a module's CHECKS, changes those lists.
# LINT.IfChange(registry)
_MODULE_CHECKS: list[Check] = (
    argument_comments.CHECKS
    + include_style.CHECKS
    + boost_usage.CHECKS
    + american_spelling.CHECKS
    + proto_file_layout.CHECKS
    + proto_next_id.CHECKS
    + textproto_headers.CHECKS
    + yaml_usage.CHECKS
    + pointer_nullability.CHECKS
    + no_auto.CHECKS
    + cpp_google_style.CHECKS
    + cpp_totw.CHECKS
    + todo_format.CHECKS
    + inclusive_language.CHECKS
    + license_header.CHECKS
    + python_imports.CHECKS
    + python_docstrings.CHECKS
    + python_language.CHECKS
)
# LINT.ThenChange(//AGENTS.md:cpp_style, //AGENTS.md:nullability, //AGENTS.md:totw_rules, //AGENTS.md:python_style, //AGENTS.md:protobuf_rules)

REGISTRY: tuple[Check, ...] = tuple(_NOLINT_CHECKS + _MODULE_CHECKS)

# The cpplint categories (cpplint 2.0's _ERROR_CATEGORIES and _LEGACY_ERROR_CATEGORIES), which NOLINT markers may name.
CPPLINT_CATEGORIES = frozenset(
    {
        "build/c++11",
        "build/c++17",
        "build/class",
        "build/deprecated",
        "build/endif_comment",
        "build/explicit_make_pair",
        "build/forward_decl",
        "build/header_guard",
        "build/include",
        "build/include_alpha",
        "build/include_order",
        "build/include_subdir",
        "build/include_what_you_use",
        "build/namespaces",
        "build/namespaces_headers",
        "build/namespaces_literals",
        "build/namespaces/header/block/literals",
        "build/namespaces/header/block/nonliterals",
        "build/namespaces/header/namespace/literals",
        "build/namespaces/header/namespace/nonliterals",
        "build/namespaces/source/block/literals",
        "build/namespaces/source/block/nonliterals",
        "build/namespaces/source/namespace/literals",
        "build/namespaces/source/namespace/nonliterals",
        "build/printf_format",
        "build/storage_class",
        "legal/copyright",
        "readability/alt_tokens",
        "readability/braces",
        "readability/casting",
        "readability/check",
        "readability/constructors",
        "readability/fn_size",
        "readability/function",
        "readability/inheritance",
        "readability/multiline_comment",
        "readability/multiline_string",
        "readability/namespace",
        "readability/nolint",
        "readability/nul",
        "readability/streams",
        "readability/todo",
        "readability/utf8",
        "runtime/arrays",
        "runtime/casting",
        "runtime/explicit",
        "runtime/init",
        "runtime/int",
        "runtime/invalid_increment",
        "runtime/member_string_references",
        "runtime/memset",
        "runtime/operator",
        "runtime/printf",
        "runtime/printf_format",
        "runtime/references",
        "runtime/string",
        "runtime/threadsafe_fn",
        "runtime/vlog",
        "whitespace/blank_line",
        "whitespace/braces",
        "whitespace/comma",
        "whitespace/comments",
        "whitespace/empty_conditional_body",
        "whitespace/empty_if_body",
        "whitespace/empty_loop_body",
        "whitespace/end_of_line",
        "whitespace/ending_newline",
        "whitespace/forcolon",
        "whitespace/indent",
        "whitespace/indent_namespace",
        "whitespace/line_length",
        "whitespace/newline",
        "whitespace/operators",
        "whitespace/parens",
        "whitespace/semicolon",
        "whitespace/tab",
        "whitespace/todo",
    }
)

# The clang-tidy configuration whose checks NOLINT markers may name. A marker for a check it does not enable would
# suppress nothing, so `nolint-unknown` rejects it.
CLANG_TIDY_CONFIGS = (".clang-tidy",)
_CLANG_TIDY_CHECK = re.compile(r"^\s*-\s*[\"']?(-?[a-z0-9.*-]+)[\"']?\s*(?:#.*)?$")
_CLANG_TIDY_CHECKS_STRING = re.compile(r"^Checks:\s*[\"']([^\"']*)[\"']", re.MULTILINE)


def by_name(name: str) -> Check:
    """The registry check named `name`; KeyError when there is none."""
    for check in REGISTRY:
        if check.name == name:
            return check
    raise KeyError(name)


def names() -> frozenset[str]:
    """The names of every registry check, which `make lint` runs."""
    return frozenset(check.name for check in REGISTRY)


def clang_tidy_checks(root: str) -> frozenset[str]:
    """The clang-tidy checks CLANG_TIDY_CONFIGS enable (globs kept as written), read without running clang-tidy.

    Args:
      root: The repository root.

    Returns:
      The check names and globs; removals (`-check`) are left out.
    """
    found: set[str] = set()
    for config in CLANG_TIDY_CONFIGS:
        path = os.path.join(root, config)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        in_checks = False
        for line in text.splitlines():
            if line.startswith("Checks:"):
                in_checks = True
                inline = _CLANG_TIDY_CHECKS_STRING.match(line)
                if inline:
                    found.update(c.strip() for c in inline.group(1).split(","))
                    in_checks = False
                continue
            if in_checks:
                item = _CLANG_TIDY_CHECK.match(line)
                if item:
                    found.add(item.group(1))
                elif line.strip() and not line.lstrip().startswith("#"):
                    in_checks = False
    return frozenset(c for c in found if c and not c.startswith("-"))


@functools.lru_cache(maxsize=4)
def _clang_tidy_checks_cached(root: str) -> frozenset[str]:
    return clang_tidy_checks(root)


def is_known_category(category: str, root: str) -> bool:
    """True when a NOLINT category names a registry check, a cpplint category or a clang-tidy check of the configs."""
    if category in names() or category in CPPLINT_CATEGORIES:
        return True
    tidy = _clang_tidy_checks_cached(root)
    return category in tidy or any(
        glob.endswith("*") and category.startswith(glob[:-1]) for glob in tidy
    )


def comment_style(path: str) -> str:
    """How the NOLINT markers of the file at `path` are read (tools/hooks/nolint.py)."""
    return nolint.comment_style(path)


def run_file(
    source: str, path: str, selected: frozenset[str], root: str
) -> list[Finding]:
    """The findings of the `selected` checks in one file, after its NOLINT markers.

    Args:
      source: The content of the file.
      path: Its path, relative to the repository root.
      selected: The names of the checks to report.
      root: The repository root (for the clang-tidy configurations NOLINT markers may name).

    Returns:
      The findings, sorted.
    """
    applicable = [check for check in REGISTRY if check.applies_to(path)]
    if not any(check.name in selected for check in applicable):
        return []
    style = comment_style(path)
    markers = nolint.parse(source, style)
    suppressions = nolint.Suppressions(markers)
    marker_categories: set[str] = set()
    for marker in markers:
        marker_categories.update(marker.categories)
    module_names = frozenset(check.name for check in _MODULE_CHECKS)
    check_unused = NOLINT_UNUSED in selected and any(
        c.name == NOLINT_UNUSED for c in applicable
    )
    findings: list[Finding] = []
    for check in applicable:
        if check.name not in module_names:
            continue
        # A check that is not reported still runs where a marker names it, to tell whether the marker is used.
        if check.name not in selected and not (
            check_unused and check.name in marker_categories
        ):
            continue
        for finding in check.check_source(source, path):
            if suppressions.suppresses(check.name, finding.line):
                continue
            if check.name in selected:
                findings.append(finding)
    applicable_names = frozenset(check.name for check in applicable)
    if style != nolint.WHOLE_LINE:
        problems = nolint.marker_problems(
            markers, lambda category: is_known_category(category, root)
        )
        problems += suppressions.unbalanced
        if check_unused:
            for marker in suppressions.unused(module_names):
                unused = [c for c in marker.categories if c in module_names]
                where = (
                    "suppresses no finding"
                    if all(c in applicable_names for c in unused)
                    else "names a check that does not read this file"
                )
                problems.append(
                    nolint.Problem(
                        NOLINT_UNUSED,
                        marker.line,
                        marker.column,
                        f"`{marker.text}` {where}: delete it.",
                    )
                )
        for problem in problems:
            if problem.check in selected and problem.check in applicable_names:
                findings.append(
                    Finding(
                        path,
                        problem.line,
                        problem.column,
                        problem.check,
                        problem.message,
                    )
                )
    return sorted(findings)


def fix_file(source: str, path: str, selected: frozenset[str]) -> str:
    """`source` rewritten by the fix of every `selected` check that reads `path` and has one, in registry order."""
    for check in REGISTRY:
        if (
            check.name in selected
            and check.fix_source is not None
            and check.applies_to(path)
        ):
            source = check.fix_source(source, path)
    return source


def format_fixes() -> frozenset[str]:
    """The checks whose fix `make format` applies: fixable, and unable to change behavior."""
    return frozenset(
        check.name
        for check in REGISTRY
        if check.fixed_by_format and check.fix_source is not None
    )
