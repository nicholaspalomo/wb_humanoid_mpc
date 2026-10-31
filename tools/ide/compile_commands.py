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

"""Writes compile_commands.json, the compilation database clangd reads, from Bazel's own compile actions.

    python3 -m tools.ide.compile_commands [--build_headers] [--quiet] [-- <bazel flags, e.g. --config=dbg>]
    python3 -m tools.ide.compile_commands --after bazel <build|test|run> <arguments>

Runs in the dev container, where Bazel and its output base are (tools/ide/README.md). Elsewhere it says so in one line
and leaves the database alone: in CI, which sets CI, and wherever the checkout is not below the dev container's mount
(the container of `make ci-local`, the host). Every container that mounts the checkout there writes it, the last one
to refresh wins; WB_COMPILE_COMMANDS=off in a container's environment leaves the database to the others.

`bazel query` lists the checkout's C++ rules, manual ones included, and its proto_library targets, and `bazel aquery`
lists their CppCompile actions with the command lines Bazel runs, so the database holds the build's real flags, defines
and include directories rather than a reconstruction. Only those targets are analyzed, so a refresh fetches no
repository that only Python targets use (the RL wheels). One entry per source file:

- `directory` is the execution root, where Bazel runs the compiler, so `bazel-out/` and the checkout's directories
  resolve as in the build;
- a path into `external/` points at the repository in the output base instead. The execution root links only the
  external repositories its last build used, so after a build of a few targets its `external/` misses most of them;
  the output base holds every fetched repository;
- `file` is the source's path in the checkout, the path the editor opens. The include directories stay relative to
  the execution root, whose top-level directories link into the checkout; clangd resolves the links, so a first-party
  header is reported at its path in the checkout too;
- the dependency-file, object-file and random-seed flags, which only Bazel's action uses, are dropped.

The query only analyzes. Generated headers (protobuf, nproto) exist once a build has made them: `--build_headers`
first builds the compilation prerequisites of the same targets, which are the generated headers and the tools that
generate them (protobuf's C++ generator, the nproto generator, MuJoCo's CMake build) but no first-party source (`make
compile-commands`, and the dev container when it starts). A target that fails to build or to analyze costs a warning,
not the refresh: the database gets the rest, and keeps its entries for the sources the new result lacks, so that a
partial result never shrinks it. A query that lists no target leaves it as it was. The file is rewritten only when its
content changes, so that clangd, which may reload it whenever it changes, has nothing to reload otherwise.

`--after` runs a Bazel command and then refreshes the database, which is how every Makefile target that builds keeps it
current. The refresh runs whether the command passed or failed (a failed build's new files are indexed too),
but not after a command that was interrupted, could not start or had a command line error, and the command's exit
status is kept; a refresh that fails or is interrupted only warns, but Ctrl-C during the refresh of a `bazel run` also
cancels the program (exit status 130). `bazel run` becomes `bazel run --script_path`, which builds the program and
writes its command line to a script; the database is refreshed, and the script then runs in this process's place. The
query gets the command's configuration flags (`--config=dbg`, `-c`, `--platforms`, ...), so that it reads the
configuration the command built and Bazel keeps the command's analysis cache. Every message, the refresh's one line
included, goes to standard error, so that a program's output stays its own.
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, NamedTuple

# LINT.IfChange(compile_commands_file)
OUTPUT_NAME = "compile_commands.json"
# The temporary file a refresh writes next to the database and renames over it.
_TEMPORARY_PREFIX = ".compile_commands."
_TEMPORARY_SUFFIX = ".tmp"
# LINT.ThenChange(//.gitignore:compile_commands_file)

# LINT.IfChange(dev_container_mount)
# Where docker-compose.yaml mounts the checkout's grandparent: a checkout below it is the dev container's.
_DEV_CONTAINER_MOUNT = "/wb_humanoid_mpc_ws"
# LINT.ThenChange(//docker-compose.yaml:dev_container_mount)
# The variable that, set to "off", leaves the database alone in a container that would write it (tools/ide/README.md).
_OPT_OUT_VARIABLE = "WB_COMPILE_COMMANDS"
_OPTED_OUT = f"{_OPT_OUT_VARIABLE}=off"

# The database's targets: the C++ rules (cc_library, cc_test, nproto_cc_library, ...), manual ones included, and the
# proto_library targets, whose .pb.cc files the C++ proto aspect compiles. `bazel query` lists them without analyzing.
_TARGETS_QUERY = (
    'kind("cc_.* rule", //...) + kind("^proto_library rule", //...)'
)
_SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")
# Output paths of Bazel's own action, which clangd neither needs nor should write: the flag and its value.
_DROPPED_FLAGS_WITH_VALUE = frozenset(["-MF", "-MT", "-MQ", "-o"])
_DROPPED_FLAGS = frozenset(["-MD", "-MMD"])
_DROPPED_PREFIXES = ("-frandom-seed=",)
# The paths under the execution root that are not links into the checkout: Bazel's outputs, and the external
# repositories, which are in the output base too.
_BAZEL_OUT = "bazel-out/"
_EXTERNAL = "external/"
# The include flags whose directory may be attached to them (`-Iexternal/x`) instead of following them.
_INCLUDE_FLAGS = ("-I", "-iquote", "-isystem", "-idirafter")

# The Bazel commands `--after` refreshes the database after; it runs any other command unchanged.
_BUILDING_COMMANDS = frozenset(["build", "test", "run"])
# The flags that select the configuration a command builds in, each followed by a value (`--config=dbg` or
# `--config dbg`). Test-only flags, aspects and output groups leave the configuration alone.
_CONFIGURATION_FLAGS = frozenset(
    [
        "--config",
        "-c",
        "--compilation_mode",
        "--platforms",
        "--copt",
        "--cxxopt",
        "--conlyopt",
        "--define",
        "--features",
    ]
)
# Bazel takes `-c opt` but not `-c=opt`, so the query gets the long name.
_COMPILATION_MODE_SHORT = "-c"
_COMPILATION_MODE = "--compilation_mode"
# The .bazelrc configs that add an aspect and its output groups only (clang-tidy), which the query must not run.
# LINT.IfChange(aspect_configs)
_ASPECT_CONFIGS = frozenset(["clang-tidy", "clang-tidy-fix"])
# LINT.ThenChange(//.bazelrc:clang_tidy_config)

# Bazel's exit statuses for a command line error, an interrupted command (Ctrl-C), and a `query --keep_going` that left
# out the packages that failed to load.
_BAZEL_COMMAND_LINE_ERROR = 2
_BAZEL_INTERRUPTED = 8
_BAZEL_PARTIAL_QUERY = 3
# The statuses after which a command's output is not read: a command line error, an interruption, a local
# environment problem and an internal error. Any other failure of a `--keep_going` command, such as 1 when some targets
# failed to build or to analyze, leaves the output of the rest.
_BAZEL_FATAL = frozenset(
    [_BAZEL_COMMAND_LINE_ERROR, _BAZEL_INTERRUPTED, 36, 37]
)
# The statuses the shell reports for a command that could not start, and for one a signal ended (128 + the signal's
# number), Ctrl-C's included.
_NOT_STARTED = 127
_SIGNALED = 128
_CTRL_C = 130


def _repository_root() -> str:
    """Returns the checkout this module is in."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def skip_reason(
    environment: Mapping[str, str], repository_root: str
) -> str | None:
    """Returns why the database is left alone here, or None in a dev container that writes it.

    Args:
      environment: The process environment. CI sets CI (GitHub, and tools/ci_local.sh for `make ci-local`), and
        WB_COMPILE_COMMANDS=off opts a dev container out.
      repository_root: The checkout's absolute path, which in the dev container is below its mount.

    Returns:
      The opt-out, what gives away another machine or container, or None.
    """
    if environment.get(_OPT_OUT_VARIABLE, "") == "off":
        return _OPTED_OUT
    if environment.get("CI", "") not in ("", "0", "false"):
        return "CI is set"
    if not repository_root.startswith(f"{_DEV_CONTAINER_MOUNT}/"):
        return f"the checkout is not below {_DEV_CONTAINER_MOUNT}"
    return None


def _skip_reason() -> str | None:
    """Returns skip_reason() for this process."""
    return skip_reason(os.environ, _repository_root())


def _say_skipped(reason: str) -> None:
    """Prints the one line that says the database is left alone, and why."""
    if reason == _OPTED_OUT:
        print(
            f"compile_commands: {reason}: {OUTPUT_NAME} is left alone",
            file=sys.stderr,
        )
        return
    print(
        f"compile_commands: not the dev container ({reason}): {OUTPUT_NAME} is"
        " left alone; run make compile-commands there (tools/ide/README.md)",
        file=sys.stderr,
    )


def _warn(message: str) -> None:
    """Prints a warning of the refresh."""
    print(f"compile_commands: warning: {message}", file=sys.stderr)


def compile_arguments(arguments: Sequence[str]) -> list[str]:
    """Drops the flags of a compile command line that only Bazel's action uses.

    Args:
      arguments: The command line of a CppCompile action, the compiler first.

    Returns:
      The command line without the dependency-file, object-file and random-seed flags.
    """
    kept = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument in _DROPPED_FLAGS_WITH_VALUE:
            skip_value = True
            continue
        if argument in _DROPPED_FLAGS or argument.startswith(_DROPPED_PREFIXES):
            continue
        kept.append(argument)
    return kept


def source_of(arguments: Sequence[str]) -> str | None:
    """Returns the source file a compile command line compiles: the value of its `-c`, or None without one."""
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-c" and arguments[index + 1].endswith(_SOURCE_SUFFIXES):
            return arguments[index + 1]
    return None


def checkout_path(
    source: str, workspace: str, execution_root: str, output_base: str
) -> str:
    """Returns where the editor opens a source of the build.

    Args:
      source: The source as the command line names it, relative to the execution root.
      workspace: The checkout's absolute path.
      execution_root: The execution root's absolute path.
      output_base: The output base's absolute path.

    Returns:
      The path in the checkout for a first-party source, under the execution root for a generated one, and under the
      output base for an external one.
    """
    if os.path.isabs(source):
        return source
    if source.startswith(_BAZEL_OUT):
        return os.path.join(execution_root, source)
    if source.startswith(_EXTERNAL):
        return os.path.join(output_base, source)
    return os.path.join(workspace, source)


def with_external_repositories(
    arguments: Sequence[str], output_base: str
) -> list[str]:
    """Returns the command line with every path into `external/` made absolute in the output base.

    Args:
      arguments: A compile command line, its paths relative to the execution root.
      output_base: The output base's absolute path, whose `external/` holds every fetched repository.

    Returns:
      The command line with `external/x` and `-Iexternal/x` (and the other include flags) rewritten; the generated
      files of external repositories, under `bazel-out/`, stay where they are.
    """
    rewritten = []
    for argument in arguments:
        if argument.startswith(_EXTERNAL):
            argument = os.path.join(output_base, argument)
        else:
            for flag in _INCLUDE_FLAGS:
                if argument.startswith(flag + _EXTERNAL):
                    argument = flag + os.path.join(
                        output_base, argument[len(flag) :]
                    )
                    break
        rewritten.append(argument)
    return rewritten


def database_entries(
    aquery: Mapping[str, Any],
    workspace: str,
    execution_root: str,
    output_base: str,
) -> list[dict[str, Any]]:
    """Turns `bazel aquery --output=jsonproto` into compilation database entries.

    A source that several targets or configurations compile keeps the command of the target whose label sorts first,
    so that the database does not change between runs.

    Args:
      aquery: The parsed jsonproto output of a query for CppCompile actions.
      workspace: The checkout's absolute path.
      execution_root: The execution root's absolute path.
      output_base: The output base's absolute path.

    Returns:
      One entry per source file, sorted by file.
    """
    labels = {
        target["id"]: target["label"] for target in aquery.get("targets", [])
    }
    commands: dict[str, tuple[str, list[str]]] = {}
    for action in aquery.get("actions", []):
        arguments = action.get("arguments", [])
        source = source_of(arguments)
        if source is None:
            continue
        file = checkout_path(source, workspace, execution_root, output_base)
        label = labels.get(action.get("targetId"), "")
        if file not in commands or label < commands[file][0]:
            commands[file] = (label, arguments)
    return [
        {
            "directory": execution_root,
            "file": file,
            "arguments": with_source(
                with_external_repositories(
                    compile_arguments(commands[file][1]), output_base
                ),
                file,
            ),
        }
        for file in sorted(commands)
    ]


def with_source(arguments: Sequence[str], file: str) -> list[str]:
    """Returns the command line with the value of its `-c` replaced by `file`."""
    replaced = list(arguments)
    for index, argument in enumerate(replaced[:-1]):
        if argument == "-c":
            replaced[index + 1] = file
            break
    return replaced


def merged_entries(
    entries: Sequence[Mapping[str, Any]], previous: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Returns a partial result's entries and the previous database's entries for the files the result lacks.

    Args:
      entries: The entries of a query that left out the targets that failed.
      previous: The database's entries before the query.

    Returns:
      One entry per file, sorted by file: the result's where it has the file, the previous database's otherwise.
    """
    by_file = {entry["file"]: entry for entry in previous}
    by_file.update({entry["file"]: entry for entry in entries})
    return [by_file[file] for file in sorted(by_file)]


def read_database(path: str) -> list[dict[str, Any]]:
    """Returns the entries of the database at `path`; none when it is missing or is not a compilation database.

    Args:
      path: The database.

    Returns:
      Its entries that name their file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        entry
        for entry in parsed
        if isinstance(entry, dict) and isinstance(entry.get("file"), str)
    ]


def write_database(entries: Iterable[Mapping[str, Any]], path: str) -> bool:
    """Writes the entries to `path` at once, unless it holds them already; returns whether it wrote.

    clangd, which may read the file at any time, sees the old or the new database; a database left as it was gives it
    nothing to reload.

    Args:
      entries: The database's entries.
      path: The database to write.

    Returns:
      False when the file already held exactly these bytes and was left alone.

    Raises:
      OSError: The file could not be written. The old one is unchanged, and no temporary file is left behind.
    """
    content = (f"{json.dumps(list(entries), indent=1)}\n").encode("utf-8")
    with contextlib.suppress(OSError):
        with open(path, "rb") as f:
            if f.read() == content:
                return False
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=os.path.dirname(path) or ".",
            prefix=_TEMPORARY_PREFIX,
            suffix=_TEMPORARY_SUFFIX,
            delete=False,
        ) as f:
            temporary = f.name
            f.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
    return True


def compile_actions_query(targets: Sequence[str]) -> str:
    """Returns the aquery expression for the CppCompile actions of the targets, each label quoted."""
    quoted = " ".join(f'"{target}"' for target in targets)
    return f'mnemonic("CppCompile", set({quoted}))'


def _bazel(
    arguments: Sequence[str], workspace: str
) -> subprocess.CompletedProcess[str]:
    """Runs a Bazel command in the checkout and returns it, finished, with its standard output.

    Args:
      arguments: The command and its arguments, without `bazel`.
      workspace: The checkout to run it in.

    Returns:
      The finished command; its standard error went to this process's.

    Raises:
      OSError: Bazel could not be started.
    """
    return subprocess.run(
        ["bazel", *arguments],
        cwd=workspace,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )


def _failure(
    completed: subprocess.CompletedProcess[str],
) -> subprocess.CalledProcessError:
    """Returns the error that reports a failed Bazel command."""
    return subprocess.CalledProcessError(completed.returncode, completed.args)


class _Targets(NamedTuple):
    """The database's targets, sorted, and whether the query left out packages that failed to load."""

    labels: list[str]
    partial: bool


class _Actions(NamedTuple):
    """The parsed `bazel aquery --output=jsonproto`, and whether it left out targets that failed to analyze."""

    aquery: Mapping[str, Any]
    partial: bool


class _Query(NamedTuple):
    """The database's entries, sorted by file, and whether they leave out targets that failed."""

    entries: list[dict[str, Any]]
    partial: bool


def _query_targets(quiet_flags: Sequence[str], workspace: str) -> _Targets:
    """Returns the labels of the database's targets, sorted: the C++ rules and the proto_library targets.

    The query only loads packages. It gets no configuration flag: the build's `build:` lines do not apply to `query`, and
    .bazelrc sets what changes how packages load for every command (`common`).

    Args:
      quiet_flags: The flags that hide Bazel's progress, or none.
      workspace: The checkout.

    Raises:
      CalledProcessError: The query failed, other than by leaving out packages that failed to load.
    """
    completed = _bazel(
        [
            "query",
            "--keep_going",
            *quiet_flags,
            "--output=label",
            _TARGETS_QUERY,
        ],
        workspace,
    )
    if completed.returncode not in (0, _BAZEL_PARTIAL_QUERY):
        raise _failure(completed)
    partial = completed.returncode == _BAZEL_PARTIAL_QUERY
    if partial:
        _warn(
            "some packages failed to load; the database keeps its entries for"
            " their sources"
        )
    return _Targets(
        sorted(
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        ),
        partial,
    )


def _build_headers(
    targets_file: str,
    flags: Sequence[str],
    quiet_flags: Sequence[str],
    workspace: str,
) -> None:
    """Builds the compilation prerequisites of the targets the file lists: their generated headers and what makes them.

    Args:
      targets_file: A file with one label per line.
      flags: The build's configuration flags.
      quiet_flags: The flags that hide Bazel's progress, or none.
      workspace: The checkout.

    Raises:
      CalledProcessError: Bazel was interrupted, rejected the command line or failed in itself. Targets that failed to
        build only cost a warning.
    """
    completed = _bazel(
        [
            "build",
            "--keep_going",
            "--output_groups=compilation_prerequisites_INTERNAL_",
            *quiet_flags,
            *flags,
            f"--target_pattern_file={targets_file}",
        ],
        workspace,
    )
    if completed.returncode in _BAZEL_FATAL:
        raise _failure(completed)
    if completed.returncode != 0:
        _warn(
            "the generated headers of some targets failed to build (bazel"
            f" exited with {completed.returncode}); clangd will not resolve"
            " their includes"
        )


def _bazel_roots(
    flags: Sequence[str], quiet_flags: Sequence[str], workspace: str
) -> tuple[str, str]:
    """Returns the execution root and the output base of the configuration the flags select.

    Args:
      flags: The configuration flags.
      quiet_flags: The flags that hide Bazel's progress, or none.
      workspace: The checkout.

    Raises:
      CalledProcessError: `bazel info` failed.
      ValueError: It did not print both.
    """
    completed = _bazel(
        ["info", *quiet_flags, *flags, "execution_root", "output_base"],
        workspace,
    )
    if completed.returncode != 0:
        raise _failure(completed)
    roots = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition(": ")
        roots[key.strip()] = value.strip()
    if not roots.get("execution_root") or not roots.get("output_base"):
        raise ValueError(f"bazel info printed {completed.stdout!r}")
    return roots["execution_root"], roots["output_base"]


def _compile_actions(
    query_file: str,
    flags: Sequence[str],
    quiet_flags: Sequence[str],
    workspace: str,
) -> _Actions:
    """Returns the parsed `bazel aquery --output=jsonproto` of the expression in the query file.

    Args:
      query_file: A file with the aquery expression (compile_actions_query()).
      flags: The configuration flags.
      quiet_flags: The flags that hide Bazel's progress, or none.
      workspace: The checkout.

    Raises:
      CalledProcessError: Bazel was interrupted, rejected the command line or failed in itself, or it failed and
        printed no actions. When it printed actions, the targets that failed to analyze only cost a warning.
      ValueError: Bazel printed something other than a JSON object.
    """
    completed = _bazel(
        [
            "aquery",
            "--keep_going",
            *quiet_flags,
            *flags,
            "--output=jsonproto",
            "--include_artifacts=false",
            f"--query_file={query_file}",
        ],
        workspace,
    )
    if completed.returncode in _BAZEL_FATAL:
        raise _failure(completed)
    parsed = json.loads(completed.stdout) if completed.stdout.strip() else {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"bazel aquery printed a {type(parsed).__name__}, not an object"
        )
    if completed.returncode != 0:
        if not parsed.get("actions"):
            raise _failure(completed)
        _warn(
            "some targets failed to analyze (bazel exited with"
            f" {completed.returncode}); the database gets the rest and keeps"
            " its entries for their sources"
        )
    return _Actions(parsed, completed.returncode != 0)


def _query_database(
    flags: Sequence[str],
    quiet_flags: Sequence[str],
    build_headers: bool,
    workspace: str,
) -> _Query:
    """Asks Bazel for the database's entries, building the generated headers first when `build_headers` is set.

    Args:
      flags: Bazel flags for the build, `info` and `aquery`, e.g. the build's `--config=dbg`.
      quiet_flags: The flags that hide Bazel's progress, or none.
      build_headers: Whether to build the compilation prerequisites of the targets first.
      workspace: The checkout.

    Returns:
      One entry per source file, sorted by file, and whether targets that failed to load or to analyze are left out.

    Raises:
      OSError: Bazel could not be started, or the query files not written.
      CalledProcessError: Bazel failed beyond the targets that failed.
      ValueError: The query listed no target, or Bazel's info or aquery output did not parse.
    """
    targets = _query_targets(quiet_flags, workspace)
    if not targets.labels:
        raise ValueError("bazel query listed no C++ target")
    with tempfile.TemporaryDirectory(prefix="compile_commands.") as scratch:
        targets_file = os.path.join(scratch, "targets.txt")
        with open(targets_file, "w", encoding="utf-8") as f:
            f.write("".join(f"{target}\n" for target in targets.labels))
        query_file = os.path.join(scratch, "aquery.txt")
        with open(query_file, "w", encoding="utf-8") as f:
            f.write(compile_actions_query(targets.labels))
        if build_headers:
            _build_headers(targets_file, flags, quiet_flags, workspace)
        execution_root, output_base = _bazel_roots(
            flags, quiet_flags, workspace
        )
        actions = _compile_actions(query_file, flags, quiet_flags, workspace)
    return _Query(
        database_entries(
            actions.aquery, workspace, execution_root, output_base
        ),
        targets.partial or actions.partial,
    )


def refresh(
    flags: Sequence[str],
    quiet: bool,
    build_headers: bool = False,
    output: str | None = None,
) -> int:
    """Queries Bazel and writes the database; returns 0, or 1 with the database unchanged when Bazel or the write failed.

    Prints one line to standard error before a quiet refresh and one after every refresh. A result that leaves out
    targets that failed keeps the previous database's entries for the files it lacks (merged_entries()).

    Args:
      flags: Bazel flags for every command but the target query, e.g. the build's `--config=dbg`.
      quiet: Whether to hide Bazel's progress.
      build_headers: Whether to build the compilation prerequisites of the targets (the generated headers) first.
      output: The database to write; the checkout's compile_commands.json when None.

    Returns:
      The exit status: 0 when the database holds the query's entries, written or already there.
    """
    workspace = _repository_root()
    quiet_flags = (
        ["--ui_event_filters=-info", "--noshow_progress"] if quiet else []
    )
    path = output or os.path.join(workspace, OUTPUT_NAME)
    if quiet:
        print(f"compile_commands: refreshing {OUTPUT_NAME}", file=sys.stderr)
    try:
        query = _query_database(flags, quiet_flags, build_headers, workspace)
        entries: Sequence[Mapping[str, Any]] = query.entries
        if query.partial:
            entries = merged_entries(query.entries, read_database(path))
        written = write_database(entries, path)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(
            f"compile_commands: {path} is unchanged: {error}", file=sys.stderr
        )
        return 1
    kept = len(entries) - len(query.entries)
    previous = (
        f", {kept} of them kept from the previous database" if kept else ""
    )
    unchanged = "" if written else " (unchanged)"
    print(
        f"compile_commands: {len(entries)} source files in"
        f" {path}{previous}{unchanged}",
        file=sys.stderr,
    )
    return 0


def command_parts(command: Sequence[str]) -> tuple[str | None, list[str]]:
    """Splits a Bazel command line into its command and the arguments Bazel parses, those before `--`.

    Args:
      command: The command line, `bazel` first, without startup options.

    Returns:
      The command (None when there is none) and its arguments up to `--`.
    """
    if len(command) < 2:
        return None, []
    arguments = list(command[2:])
    if "--" in arguments:
        arguments = arguments[: arguments.index("--")]
    return command[1], arguments


def configuration_flags(arguments: Sequence[str]) -> list[str]:
    """Returns the flags of a Bazel command's arguments that select its configuration, in order.

    Bazel parses an option wherever it stands before `--`, after the target too; what is left besides the target is the
    program's arguments, and none of them is a configuration flag.

    Args:
      arguments: The command's arguments before `--`.

    Returns:
      Each configuration flag as one `--flag=value` argument under its long name; the aspect-only configs are left out.
    """
    flags = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        name, has_value, value = argument.partition("=")
        if (
            name in _CONFIGURATION_FLAGS
            and not has_value
            and index + 1 < len(arguments)
        ):
            value = arguments[index + 1]
            index += 1
            has_value = "="
        index += 1
        if name not in _CONFIGURATION_FLAGS or not has_value:
            continue
        if name == "--config" and value in _ASPECT_CONFIGS:
            continue
        if name == _COMPILATION_MODE_SHORT:
            name = _COMPILATION_MODE
        flags.append(f"{name}={value}")
    return flags


def _run(command: Sequence[str]) -> int:
    """Runs a command and returns its exit status, waiting for it through Ctrl-C, which reaches it too.

    Args:
      command: The command line, the program first.

    Returns:
      Its exit status as the shell reports it: 128 + the signal's number when a signal ended it, 127 when it could not
      start.
    """
    try:
        process = subprocess.Popen(
            list(command)
        )  # pylint: disable=consider-using-with  # the with below closes it
    except OSError as error:
        print(
            f"compile_commands: cannot run {command[0]}: {error}",
            file=sys.stderr,
        )
        return _NOT_STARTED
    with process:
        while True:
            try:
                status = process.wait()
                break
            except KeyboardInterrupt:
                continue
    return _SIGNALED - status if status < 0 else status


def _exec(command: Sequence[str]) -> int:
    """Replaces this process with the command, its own output written out first; returns 127 only when it could not start.

    Args:
      command: The command line, the program first, found on the PATH.

    Returns:
      127, when the command could not start; otherwise this function does not return.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvp(command[0], list(command))
    except OSError as error:
        print(
            f"compile_commands: cannot run {command[0]}: {error}",
            file=sys.stderr,
        )
    return _NOT_STARTED


def _built(status: int) -> bool:
    """Returns whether a Bazel command that exited with `status` got to build: not interrupted, started, command line taken."""
    return (
        status
        not in (_BAZEL_COMMAND_LINE_ERROR, _BAZEL_INTERRUPTED, _NOT_STARTED)
        and status <= _SIGNALED
    )


def _refresh_after_command(flags: Sequence[str]) -> bool:
    """Refreshes the database after a command, warning when it stays stale; returns False when Ctrl-C stopped it."""
    try:
        status = refresh(flags, quiet=True)
    except KeyboardInterrupt:
        _warn(f"{OUTPUT_NAME} was not refreshed: interrupted")
        return False
    if status != 0:
        _warn(f"{OUTPUT_NAME} was not refreshed (tools/ide/README.md)")
    return True


def _run_program(command: Sequence[str], flags: Sequence[str]) -> int:
    """Builds the program of a `bazel run` command, refreshes the database, then replaces this process with the program.

    `bazel run --script_path` builds the program and writes its command line, its arguments included, to a script
    instead of running it; the script then runs in this process's place, as the program would from `bazel run`. Being a
    `run`, the build does not wait for tools/bazel's machine lock.

    Args:
      command: The `bazel run` command line, `bazel` first.
      flags: Its configuration flags, for the refresh.

    Returns:
      The build's exit status when it failed, 130 when Ctrl-C interrupted the refresh (the program does not start), 1
      when Bazel wrote no script, 127 when the script could not start; otherwise this function does not return.
    """
    scratch = tempfile.mkdtemp(prefix="compile_commands_run.")
    script_path = os.path.join(scratch, "run.sh")
    try:
        status = _run(
            [
                command[0],
                command[1],
                f"--script_path={script_path}",
                *command[2:],
            ]
        )
        if _built(status) and not _refresh_after_command(flags):
            return _CTRL_C
        if status != 0:
            return status
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
    except OSError as error:
        print(
            f"compile_commands: bazel run wrote no script: {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _exec(["bash", "-c", script, script_path])


def run_after(command: Sequence[str]) -> int:
    """Runs a Bazel command, refreshing the database after it builds; returns the command's exit status.

    Outside the dev container, and for a command that does not build, the command replaces this process.

    Args:
      command: The Bazel command line, `bazel` first.

    Returns:
      The exit status of the command. A `bazel run` whose build passed does not return: its program replaces this
      process once the database is refreshed (see _run_program()).
    """
    name, arguments = command_parts(command)
    if name not in _BUILDING_COMMANDS:
        return _exec(command)
    reason = _skip_reason()
    if reason is not None:
        _say_skipped(reason)
        return _exec(command)
    flags = configuration_flags(arguments)
    if name == "run" and not any(
        argument.startswith("--script_path") for argument in arguments
    ):
        return _run_program(command, flags)
    status = _run(command)
    if _built(status):
        _refresh_after_command(flags)
    return status


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parses the command line; the arguments after `--` are Bazel flags."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build_headers",
        action="store_true",
        help=(
            "first build the generated headers of the C++ targets and the tools"
            " that generate them, no first-party source"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"the database to write (default: <checkout>/{OUTPUT_NAME})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print one line instead of Bazel's progress",
    )
    parser.add_argument(
        "bazel_flags",
        nargs="*",
        help="flags for Bazel after --, e.g. --config=dbg",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Writes the database, or runs a Bazel command and then writes it (`--after`); returns the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--after"]:
        return run_after(argv[1:])
    arguments = _parse_arguments(argv)
    reason = _skip_reason()
    if reason is not None:
        _say_skipped(reason)
        return 1
    try:
        return refresh(
            arguments.bazel_flags,
            quiet=arguments.quiet,
            build_headers=arguments.build_headers,
            output=arguments.output,
        )
    except KeyboardInterrupt:
        print(
            f"compile_commands: interrupted; {OUTPUT_NAME} is unchanged",
            file=sys.stderr,
        )
        return _CTRL_C


if __name__ == "__main__":
    sys.exit(main())
