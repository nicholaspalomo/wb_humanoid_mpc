#!/usr/bin/env python3
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

"""The repository's linter: every check that needs no compiler, in one run (make lint).

Run it from the repository root as a module:

    python3 -m tools.hooks.lint_code                       # everything (make lint)
    python3 -m tools.hooks.lint_code --git-staged          # the files staged for commit (the pre-commit hook)
    python3 -m tools.hooks.lint_code --only no-auto,boost  # some checks or steps only, pending ones included
    python3 -m tools.hooks.lint_code --paths robot_runtime --only pointer-nullability --summary
    python3 -m tools.hooks.lint_code --fix --only float-literal --paths humanoid_nmpc/humanoid_wb_mpc

The steps, in order: IFTTT directives; trailing whitespace and EOF newlines; clang-format; black; isort; the token
checks of the registry (tools/hooks/checks.py: argument comments, Boost, spelling, the protobuf layout, NOLINT markers,
...); cpplint (CPPLINT.cfg); pylint on sources and on tests (.pylintrc); mypy on sources and on tests (mypy.ini). Every
step runs, then the linter fails if any did, so one run shows every finding. Checks and steps that still have findings
in the tree are PENDING (tools/hooks/checks.py) and run only when --only names them.

cpplint, pylint and mypy take turns through a lint lock (.lint_machine.lock in the checkout): one run of them needs
about 1.5 GB next to a build that is already sized to the machine (AGENTS.md "Builds share one machine's memory"). A
second run waits and says why. The token checks take no lock. Those tools come from the pinned, hashed lock
tools/hooks/lint_requirements_lock.txt (the dev image's /opt/wb-lint venv, CI's format job); a missing or different
version is an error when CI=true and otherwise a warning that skips the step.
"""

import argparse
import collections
from collections.abc import Callable, Iterator, Sequence
import concurrent.futures
import configparser
import contextlib
import dataclasses
import fcntl
import os
import re
import shutil
import subprocess
import sys

from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import lint_files

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# LINT.IfChange(lint_lock)
LINT_LOCK = ".lint_machine.lock"
LINT_LOCK_ENVIRONMENT = "WB_LINT_MACHINE_LOCK"
# LINT.ThenChange(//.gitignore:lint_lock, //AGENTS.md:style_commands)

LOCK_FILE = "tools/hooks/lint_requirements_lock.txt"
CPPLINT_CONFIG = "CPPLINT.cfg"
PYLINT_CONFIG = ".pylintrc"
MYPY_CONFIG = "mypy.ini"
ISORT_CONFIG = ".isort.cfg"

# cpplint reads these; the others (.inl, .tpp, ...) are textual includes it would misjudge.
CPPLINT_EXTENSIONS = (".h", ".hpp", ".cpp", ".cc")
CPPLINT_PROCESSES = 4

_REBUILD = (
    "rebuild the dev container (docker/Dockerfile installs tools/hooks/lint_requirements_lock.txt into /opt/wb-lint), "
    "or install the lock into a venv on PATH: python3 -m venv ~/.cache/wb-lint && ~/.cache/wb-lint/bin/pip install "
    "--require-hashes -r tools/hooks/lint_requirements_lock.txt"
)


@dataclasses.dataclass
class Context:
    """What one run lints, and how.

    Attributes:
      root: The repository root.
      files: The files to lint, relative to the root (after --paths).
      staged: Whether the token checks read the index (--git-staged) rather than the working tree.
      selected: The registry checks the token step reports.
      steps: The steps that run.
      summary: Whether the token step prints counts instead of findings.
      ci: Whether a missing or mismatched tool is an error (CI=true).
      narrowed: Whether --paths narrowed the files.
    """

    root: str
    files: list[str]
    staged: bool
    selected: frozenset[str]
    steps: frozenset[str]
    summary: bool
    ci: bool
    narrowed: bool = False

    def read(self, path: str) -> str:
        """The content a check judges: the staged blob in --git-staged mode, the working-tree file otherwise."""
        if self.staged:
            return lint_files.staged_source(self.root, path)
        with open(
            os.path.join(self.root, path), encoding="utf-8", errors="ignore"
        ) as f:
            return f.read()

    def absolute(self, paths: Sequence[str]) -> list[str]:
        return [os.path.join(self.root, path) for path in paths]


@dataclasses.dataclass
class Result:
    """The outcome of one step: "passed", "failed" or "skipped", what it printed, and what to do about a failure."""

    status: str
    lines: list[str] = dataclasses.field(default_factory=list)
    hint: str = ""


@dataclasses.dataclass(frozen=True)
class Step:
    """One step of the linter.

    Attributes:
      name: Its name for --only.
      title: What it checks, printed when it starts.
      run: Runs it.
      heavy: It runs under the lint lock (a tool that needs hundreds of MB).
    """

    name: str
    title: str
    run: Callable[[Context], Result]
    heavy: bool = False


def _passed() -> Result:
    return Result("passed")


# ----------------------------------------------------------------------------------------------------------------------
# File sets
# ----------------------------------------------------------------------------------------------------------------------
def _cpp_files(context: Context) -> list[str]:
    return [
        path
        for path in context.files
        if path.endswith(lint_files.CPP_EXTENSIONS) and lint_files.is_first_party(path)
    ]


def _python_files(context: Context) -> list[str]:
    return [
        path
        for path in context.files
        if path.endswith(".py") and lint_files.is_first_party(path)
    ]


def _python_sources(context: Context) -> list[str]:
    return [
        path for path in _python_files(context) if not lint_files.is_test_path(path)
    ]


def _python_tests(context: Context) -> list[str]:
    return [path for path in _python_files(context) if lint_files.is_test_path(path)]


# ----------------------------------------------------------------------------------------------------------------------
# Tool versions
# ----------------------------------------------------------------------------------------------------------------------
def pinned_versions(root: str) -> dict[str, str]:
    """The `name==version` pins of the lint lock, by lowercase package name."""
    pins = {}
    path = os.path.join(root, LOCK_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\;]+)", line)
                if match:
                    pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


_VERSION_PATTERNS = {
    "cpplint": re.compile(r"^cpplint\s+(\S+)", re.MULTILINE),
    "pylint": re.compile(r"^pylint\s+(\S+)", re.MULTILINE),
    "mypy": re.compile(r"^mypy\s+(\S+)", re.MULTILINE),
    "isort": re.compile(r"VERSION\s+(\S+)"),
}


def installed_version(tool: str) -> str | None:
    """The version of `tool` on PATH, or None when it is not installed or does not say."""
    executable = shutil.which(tool)
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_PATTERNS[tool].search(output.stdout + output.stderr)
    return match.group(1) if match else None


def _tool_problem(context: Context, tool: str) -> Result | None:
    """None when `tool` is installed at the version of the lock; otherwise the result of a step that cannot run it."""
    pinned = pinned_versions(context.root).get(tool)
    installed = installed_version(tool)
    if installed is not None and installed == pinned:
        return None
    if installed is None:
        problem = f"{tool} is not installed (the lock pins {tool}=={pinned})"
    else:
        problem = f"{tool} {installed} is installed, but the lock pins {tool}=={pinned}"
    if context.ci:
        return Result("failed", [f"❌ {problem}."], hint=_REBUILD)
    return Result(
        "skipped", [f"⚠️ Warning: {problem}; skipping {tool}. To run it, {_REBUILD}."]
    )


def _python_environment(root: str) -> dict[str, str]:
    """The environment pylint and mypy run in: the Bazel import roots on the path, and nothing of the container's."""
    environment = dict(os.environ)
    roots = [root] + [
        os.path.join(root, r) for r in lint_files.python_import_roots(root)
    ]
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    environment["MYPYPATH"] = os.pathsep.join(roots[1:])
    environment.pop("PYTHONSTARTUP", None)
    return environment


# ----------------------------------------------------------------------------------------------------------------------
# The lint lock
# ----------------------------------------------------------------------------------------------------------------------
def lint_lock_path(root: str) -> str:
    """The lint lock file: WB_LINT_MACHINE_LOCK, or .lint_machine.lock at the checkout root."""
    return os.environ.get(LINT_LOCK_ENVIRONMENT) or os.path.join(root, LINT_LOCK)


@contextlib.contextmanager
def lint_lock(root: str) -> Iterator[None]:
    """Holds the lint lock: one cpplint, pylint and mypy run at a time on the machine, as tools/bazel does for builds."""
    with open(lint_lock_path(root), "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "⏳ waiting for the lint lock (another make lint is running): cpplint, pylint and mypy take turns so "
                "that they fit next to a build (AGENTS.md, 'Builds share one machine's memory').",
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------------------------------------
def _run_ifttt(context: Context) -> Result:
    """Checks the LINT.IfChange / LINT.ThenChange directives (tools/hooks/check_ifttt.py)."""
    if context.staged:
        return Result(
            "skipped", ["(the pre-commit hook checks IFTTT before formatting)"]
        )
    paths = context.files if context.narrowed else []
    result = subprocess.run(
        [sys.executable, "-m", "tools.hooks.check_ifttt", *paths],
        cwd=context.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return _passed()
    return Result("failed", (result.stdout + result.stderr).rstrip().splitlines())


def whitespace_problem(content: str) -> str | None:
    """Why `content` is not exactly one newline-terminated block without trailing whitespace, or None."""
    if not content:
        return None
    expected = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    if content != expected:
        return "Trailing whitespace or missing/extra EOF newline"
    return None


def _whitespace_files(context: Context) -> list[str]:
    return [
        path
        for path in context.files
        if check_types.is_text(path)
        and lint_files.in_scope(path, lint_files.Scope.TEXT)
        and not lint_files.is_fixture(path)
    ]


def _run_whitespace(context: Context) -> Result:
    """Checks that every text file has no trailing whitespace and ends in one newline."""
    errors = []
    for path in _whitespace_files(context):
        problem = whitespace_problem(context.read(path))
        if problem:
            errors.append(f"  {path}: {problem}")
    if errors:
        return Result(
            "failed",
            ["❌ Whitespace/Newline Errors:"] + errors,
            hint="Run 'make format' to fix whitespace and newlines.",
        )
    return _passed()


def _run_clang_format(context: Context) -> Result:
    """Checks the C++ layout with clang-format (.clang-format)."""
    files = _cpp_files(context)
    if not files:
        return _passed()
    if shutil.which("clang-format") is None:
        return Result(
            "skipped", ["⚠️ Warning: clang-format not found, skipping C++ format lint."]
        )
    result = subprocess.run(
        ["clang-format", "--dry-run", "--Werror", *files],
        cwd=context.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return _passed()
    return Result(
        "failed",
        result.stderr.rstrip().splitlines() + ["clang-format violations detected."],
        hint="Run 'make format' to format C++ code.",
    )


def _run_black(context: Context) -> Result:
    """Checks the Python layout with black."""
    files = _python_files(context)
    if not files:
        return _passed()
    if shutil.which("black") is None:
        return Result(
            "skipped", ["⚠️ Warning: black not found, skipping Python format lint."]
        )
    result = subprocess.run(
        ["black", "--check", *files],
        cwd=context.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return _passed()
    return Result(
        "failed",
        (result.stdout + result.stderr).rstrip().splitlines()
        + ["black format violations detected."],
        hint="Run 'make format' to format Python code.",
    )


def _run_isort(context: Context) -> Result:
    """Checks the order of the Python imports with isort (.isort.cfg)."""
    files = _python_files(context)
    if not files:
        return _passed()
    problem = _tool_problem(context, "isort")
    if problem:
        return problem
    result = subprocess.run(
        [
            "isort",
            "--check-only",
            "--diff",
            "--settings-path",
            os.path.join(context.root, ISORT_CONFIG),
            *lint_files.isort_package_arguments(context.root),
            *files,
        ],
        cwd=context.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return _passed()
    return Result(
        "failed",
        (result.stdout + result.stderr).rstrip().splitlines(),
        hint="Run 'make format' to order the imports.",
    )


def _token_findings(context: Context) -> list[check_types.Finding]:
    findings: list[check_types.Finding] = []
    for path in context.files:
        if not any(checks.by_name(name).applies_to(path) for name in context.selected):
            continue
        findings += checks.run_file(
            context.read(path), path, context.selected, context.root
        )
    return findings


def _summary_lines(findings: list[check_types.Finding], root: str) -> list[str]:
    """The number of findings per check, and per Bazel package under each check."""
    per_check = collections.Counter(f.check for f in findings)
    per_package: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for finding in findings:
        per_package[finding.check][lint_files.bazel_package(finding.path, root)] += 1
    lines = []
    for check, count in sorted(per_check.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{count:6d}  {check}")
        for package, package_count in sorted(
            per_package[check].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"          {package_count:6d}  {package}")
    return lines


def _run_token_checks(context: Context) -> Result:
    """Runs the selected checks of the registry (tools/hooks/checks.py) over the files."""
    findings = _token_findings(context)
    if context.summary:
        lines = _summary_lines(findings, context.root)
        return Result("failed" if findings else "passed", lines)
    if not findings:
        return _passed()
    lines = [str(finding) for finding in findings]
    hints = []
    for name in sorted({finding.check for finding in findings}):
        check = checks.by_name(name)
        count = sum(1 for finding in findings if finding.check == name)
        hints.append(f"[{name}] {count} finding(s): {check.hint or check.description}")
    return Result("failed", lines, hint="\n".join(hints))


_CPPLINT_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):\s+(?P<message>.*?)\s+\[(?P<category>[^\]]+)\]\s+\[\d\]$"
)
_UNKNOWN_NOLINT = re.compile(r"^Unknown NOLINT error category: (?P<category>\S+)$")


def filter_cpplint_output(output: str) -> list[str]:
    """cpplint's findings, without its complaints about NOLINT markers for the repository's own checks.

    cpplint reports `NOLINT(argument-comment)` as an unknown category; markers for other unknown names still fail.

    Args:
      output: What cpplint printed.

    Returns:
      One `path:line: message [category]` line per finding.
    """
    lines = []
    for line in output.splitlines():
        match = _CPPLINT_LINE.match(line)
        if not match:
            continue
        unknown = _UNKNOWN_NOLINT.match(match.group("message"))
        if unknown and unknown.group("category") in checks.names():
            continue
        lines.append(
            f"{match.group('path')}:{match.group('line')}: {match.group('message')} [{match.group('category')}]"
        )
    return lines


def _chunks(items: list[str], count: int) -> list[list[str]]:
    return [items[k::count] for k in range(count) if items[k::count]]


def _run_cpplint(context: Context) -> Result:
    """Runs cpplint (CPPLINT.cfg) over the first-party C++ files, in parallel chunks."""
    files = [path for path in _cpp_files(context) if path.endswith(CPPLINT_EXTENSIONS)]
    if not files:
        return _passed()
    problem = _tool_problem(context, "cpplint")
    if problem:
        return problem

    def run_chunk(chunk: list[str]) -> str:
        result = subprocess.run(
            ["cpplint", "--quiet", *chunk],
            cwd=context.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout + result.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=CPPLINT_PROCESSES) as pool:
        outputs = list(pool.map(run_chunk, _chunks(files, CPPLINT_PROCESSES)))
    lines = sorted(
        filter_cpplint_output("\n".join(outputs)),
        key=lambda line: (line.split(":", 1)[0], int(line.split(":")[1])),
    )
    if not lines:
        return _passed()
    return Result(
        "failed",
        lines,
        hint="cpplint checks the Google C++ Style Guide (CPPLINT.cfg); a justified exception is "
        "`// NOLINT(<category>): <reason>`.",
    )


def _pylint(context: Context, files: list[str], extra: list[str]) -> Result:
    """Runs pylint (.pylintrc) over `files`, with the extra arguments `extra`."""
    if not files:
        return _passed()
    problem = _tool_problem(context, "pylint")
    if problem:
        return problem
    result = subprocess.run(
        [
            "pylint",
            f"--rcfile={os.path.join(context.root, PYLINT_CONFIG)}",
            *extra,
            *files,
        ],
        cwd=context.root,
        env=_python_environment(context.root),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line for line in result.stdout.splitlines() if re.match(r"^\S+:\d+:\d+: ", line)
    ]
    if result.returncode == 0 and not lines:
        return _passed()
    if not lines:
        lines = (result.stdout + result.stderr).rstrip().splitlines()
    return Result(
        "failed",
        lines,
        hint="pylint checks the Google Python Style Guide (.pylintrc); a justified exception is "
        "`# pylint: disable=<message>  # <reason>`.",
    )


def _run_pylint_sources(context: Context) -> Result:
    return _pylint(context, _python_sources(context), [])


def _run_pylint_tests(context: Context) -> Result:
    # Tests may reach into the unit under test (Google Python Style Guide 3.16.2).
    return _pylint(context, _python_tests(context), ["--disable=protected-access"])


def _mypy(context: Context, files: list[str], extra: list[str]) -> Result:
    """Runs mypy (mypy.ini) over `files`, with the extra arguments `extra`."""
    if not files:
        return _passed()
    problem = _tool_problem(context, "mypy")
    if problem:
        return problem
    result = subprocess.run(
        [
            "mypy",
            f"--config-file={os.path.join(context.root, MYPY_CONFIG)}",
            *extra,
            *files,
        ],
        cwd=context.root,
        env=_python_environment(context.root),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if ": error:" in line or ": note:" in line
    ]
    if result.returncode == 0:
        return _passed()
    if not lines:
        lines = (result.stdout + result.stderr).rstrip().splitlines()
    return Result(
        "failed",
        lines,
        hint="mypy checks the type annotations (mypy.ini); every non-test function is fully annotated.",
    )


# What a per-module section of mypy.ini may set: it only relaxes the global options, for a module still being swept.
_MYPY_SECTION_OPTIONS = {
    "ignore_errors": "true",
    "disallow_untyped_defs": "false",
    "disallow_incomplete_defs": "false",
}


def _module_exists(root: str, module: str) -> bool:
    """True when `module` (a mypy section name, `pkg.*` included) is a file or package below a Python root."""
    parts = module.removesuffix(".*").split(".")
    for base in [root] + [
        os.path.join(root, r) for r in lint_files.python_import_roots(root)
    ]:
        candidate = os.path.join(base, *parts)
        if os.path.isfile(candidate + ".py") or os.path.isdir(candidate):
            return True
    return False


def mypy_config_problems(root: str) -> list[str]:
    """The per-module sections of mypy.ini that name no module, or that set more than they may (_MYPY_SECTION_OPTIONS).

    mypy's own warn_unused_configs cannot be used: it needs non-incremental runs and misfires on partial file lists.

    Args:
      root: The repository root.

    Returns:
      One message per problem.
    """
    parser = configparser.ConfigParser()
    parser.read(os.path.join(root, MYPY_CONFIG), encoding="utf-8")
    problems = []
    for section in parser.sections():
        if not section.startswith("mypy-"):
            continue
        module = section[len("mypy-") :]
        if not _module_exists(root, module):
            problems.append(
                f"{MYPY_CONFIG}: [{section}] names no module of the repository: delete the section."
            )
        for key, value in parser.items(section):
            if _MYPY_SECTION_OPTIONS.get(key) != value.strip().lower():
                problems.append(
                    f"{MYPY_CONFIG}: [{section}] sets {key} = {value}: a module section only relaxes "
                    f"({', '.join(f'{k} = {v}' for k, v in _MYPY_SECTION_OPTIONS.items())})."
                )
    return problems


def _run_mypy_sources(context: Context) -> Result:
    """Runs mypy over the Python sources, after checking that mypy.ini's module sections are sound."""
    problems = mypy_config_problems(context.root)
    result = _mypy(context, _python_sources(context), [])
    if not problems:
        return result
    return Result(
        "failed",
        problems + result.lines,
        hint=result.hint or "Fix the module sections of mypy.ini.",
    )


def _run_mypy_tests(context: Context) -> Result:
    return _mypy(
        context,
        _python_tests(context),
        ["--allow-untyped-defs", "--allow-incomplete-defs"],
    )


STEPS = (
    Step("ifttt", "Checking IFTTT cross-file directives", _run_ifttt),
    Step(
        "whitespace", "Checking trailing whitespace and EOF newlines", _run_whitespace
    ),
    Step("clang-format", "Checking C++ formatting (clang-format)", _run_clang_format),
    Step("black", "Checking Python formatting (black)", _run_black),
    Step("isort", "Checking the order of Python imports (isort)", _run_isort),
    Step(
        "token-checks",
        "Running the token checks of tools/hooks/checks.py",
        _run_token_checks,
    ),
    Step(
        "cpplint", "Checking C++ with cpplint (CPPLINT.cfg)", _run_cpplint, heavy=True
    ),
    Step(
        "pylint",
        "Checking Python sources with pylint (.pylintrc)",
        _run_pylint_sources,
        heavy=True,
    ),
    Step(
        "pylint-tests",
        "Checking Python tests with pylint",
        _run_pylint_tests,
        heavy=True,
    ),
    Step(
        "mypy",
        "Checking Python types with mypy (mypy.ini)",
        _run_mypy_sources,
        heavy=True,
    ),
    Step(
        "mypy-tests",
        "Checking the types of Python tests with mypy",
        _run_mypy_tests,
        heavy=True,
    ),
)
STEP_NAMES = frozenset(step.name for step in STEPS)


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------
def _all_files(root: str) -> list[str]:
    return lint_files.repository_files(root)


def _select_files(root: str, staged: bool, paths: Sequence[str]) -> list[str]:
    """The files to lint: the staged or the repository files, under `paths` when given."""
    files = lint_files.staged_files(root) if staged else _all_files(root)
    if not paths:
        return files
    prefixes = []
    for path in paths:
        relative = lint_files.normalize(
            os.path.relpath(os.path.abspath(path), root)
            if os.path.isabs(path) or os.path.exists(path)
            else path
        ).rstrip("/")
        prefixes.append(relative)
    return [
        f
        for f in files
        if any(
            f == prefix or f.startswith(prefix + "/") or prefix == "."
            for prefix in prefixes
        )
    ]


def _parse_only(only: str | None) -> tuple[frozenset[str], frozenset[str]]:
    """The registry checks and steps --only names; with no --only, every enforced check and step."""
    if not only:
        steps = frozenset(step for step in STEP_NAMES if step not in checks.PENDING)
        return checks.enforced(), steps
    requested = frozenset(name.strip() for name in only.split(",") if name.strip())
    unknown = requested - checks.names() - STEP_NAMES
    if unknown:
        raise ValueError(
            f"unknown check or step {', '.join(sorted(unknown))}: --only takes the names of tools/hooks/checks.py "
            f"and the steps {', '.join(sorted(STEP_NAMES))}"
        )
    selected = requested & checks.names()
    steps = requested & STEP_NAMES
    if selected:
        steps |= {"token-checks"}
    if "token-checks" in requested and not selected:
        selected = checks.enforced()
    return selected, steps


def _apply_fixes(root: str, files: list[str], selected: frozenset[str]) -> list[str]:
    fixed = []
    for path in files:
        full = os.path.join(root, path)
        with open(full, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        rewritten = checks.fix_file(source, path, selected)
        if rewritten != source:
            with open(full, "w", encoding="utf-8") as f:
                f.write(rewritten)
            fixed.append(path)
    return fixed


def run(context: Context) -> int:
    """Runs the steps of `context`, prints their findings and a summary, and returns the exit status."""
    steps = [step for step in STEPS if step.name in context.steps]
    results: list[tuple[Step, Result]] = []
    heavy = [step for step in steps if step.heavy]
    with contextlib.ExitStack() as stack:
        for index, step in enumerate(steps, start=1):
            if step.heavy and step is heavy[0]:
                stack.enter_context(lint_lock(context.root))
            print(f"🔍 {index}/{len(steps)} {step.title}...", flush=True)
            result = step.run(context)
            for line in result.lines:
                print(line)
            if result.status == "failed" and result.hint:
                for line in result.hint.splitlines():
                    print(f"💡 {line}")
            results.append((step, result))
            sys.stdout.flush()
    failed = [step.name for step, result in results if result.status == "failed"]
    skipped = [step.name for step, result in results if result.status == "skipped"]
    if failed:
        print(
            f"❌ Lint failed in: {', '.join(failed)}"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )
        return 1
    print(
        "✅ All lint checks passed successfully!"
        + (f" (skipped: {', '.join(skipped)})" if skipped else "")
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the command line and runs the linter."""
    parser = argparse.ArgumentParser(
        description="The repository's linter (make lint): every check that needs no compiler."
    )
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="lint the files staged for commit; the token checks read the index (the pre-commit hook)",
    )
    parser.add_argument(
        "--only",
        help="comma-separated names of checks (tools/hooks/checks.py) or steps to run, pending ones included",
    )
    parser.add_argument(
        "--paths", nargs="+", default=[], help="lint only files under these paths"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="first apply the fixes of the selected checks to the working tree",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the number of token-check findings per check and Bazel package instead of the findings",
    )
    args = parser.parse_args(argv)
    if args.fix and args.git_staged:
        parser.error(
            "--fix rewrites the working tree; it cannot be combined with --git-staged"
        )
    try:
        selected, steps = _parse_only(args.only)
    except ValueError as error:
        parser.error(str(error))
    root = REPO_ROOT
    files = _select_files(root, args.git_staged, args.paths)
    if args.fix:
        for path in _apply_fixes(root, files, selected):
            print(f"✨ fixed {path}")
    context = Context(
        root=root,
        files=files,
        staged=args.git_staged,
        selected=selected,
        steps=steps,
        summary=args.summary,
        ci=os.environ.get("CI", "").lower() == "true",
        narrowed=bool(args.paths),
    )
    return run(context)


if __name__ == "__main__":
    sys.exit(main())
