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

"""The command line of tools/launch: resolves the launch file, then prints it (--dry_run) or runs it.

    bazel run //tools/launch -- <launch file> [--machine robot|laptop] [--set name=value ...] [--dry_run]
        [--sigint_grace_period S] [--sigterm_grace_period S] [--color auto|always|never]
        [--terminal_command "x-terminal-emulator -e"] [--repo_root DIR]

The launch file is a textproto of launch_proto.LaunchFile (proto/launch_file.proto, launch_file.py).

The launcher changes into the repository root before it starts anything, so that the paths in commands are relative
to it. The root is --repo_root, else the workspace of `bazel run` (BUILD_WORKSPACE_DIRECTORY), else the nearest
directory above the launch file, or above the working directory, that holds MODULE.bazel.

SIGINT (Ctrl-C), SIGTERM and SIGHUP stop every process (process_supervisor.py); a second one escalates.

Exit status: that of a required process that failed, 0 otherwise, and 2 for an invalid launch file or command line.
"""

import argparse
from collections.abc import Mapping, Sequence
import os
import shlex
import signal
import sys
import types
from typing import TextIO

import launch_file
import process_supervisor

EXIT_USAGE = 2
# The file that marks the repository root.
REPO_ROOT_MARKER = "MODULE.bazel"
SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def resolve_launch_file_path(path: str, environment: Mapping[str, str]) -> str:
    """The launch file named on the command line, as an absolute path.

    A relative path is relative to the directory the launcher was started from (also under `bazel run`, which changes
    into the runfiles tree), else relative to the workspace of `bazel run`.

    Args:
        path: The path as the command line gives it.
        environment: The environment, for BUILD_WORKING_DIRECTORY and BUILD_WORKSPACE_DIRECTORY.

    Returns:
        `path` when it is absolute, else the first candidate that exists, or the first candidate when none exists, so
        that the error names the path the user meant.
    """
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(environment.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path)
    ]
    workspace = environment.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        candidates.append(os.path.join(workspace, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def _directory_with_marker(start: str) -> str | None:
    """The nearest directory at or above `start` that holds REPO_ROOT_MARKER, or None."""
    directory = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(directory, REPO_ROOT_MARKER)):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def find_repo_root(
    explicit: str | None, launch_file_path: str, environment: Mapping[str, str]
) -> str:
    """The directory the launcher runs every process in (see the module comment for the order of the candidates)."""
    if explicit:
        if not os.path.isdir(explicit):
            raise launch_file.LaunchFileError(
                f"--repo_root {explicit} is not a directory"
            )
        return os.path.abspath(explicit)
    workspace = environment.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        return workspace
    for start in (
        os.path.dirname(launch_file_path),
        environment.get("BUILD_WORKING_DIRECTORY", os.getcwd()),
    ):
        root = _directory_with_marker(start)
        if root is not None:
            return root
    raise launch_file.LaunchFileError(
        f"cannot find the repository root (a directory with {REPO_ROOT_MARKER}) above {launch_file_path} or the "
        "working directory; pass --repo_root"
    )


def format_dry_run(
    processes: Sequence[launch_file.ProcessSpec],
    repo_root: str,
    shutdown: launch_file.ShutdownPolicy,
    terminal_command: Sequence[str],
) -> str:
    """The resolved launch as shell lines: a comment with each process's settings, then its command."""
    lines = [
        f"# repository root (every command runs here): {repo_root}",
        f"# teardown: SIGINT, {shutdown.sigint_grace_period:g} s, SIGTERM, {shutdown.sigterm_grace_period:g} s, "
        "SIGKILL",
    ]
    for spec in processes:
        lines.append(
            f"# {spec.name}: machine={spec.machine} required={str(spec.required).lower()} "
            f"terminal={str(spec.terminal).lower()} delay={spec.delay:g}s"
        )
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in spec.env)
        command = shlex.join(process_supervisor.process_argv(spec, terminal_command))
        lines.append(f"{environment} {command}" if environment else command)
    return "\n".join(lines) + "\n"


def _non_negative_seconds(text: str) -> float:
    """An argparse type: a number of seconds that is not negative; NaN is rejected too (AGENTS.md, Python)."""
    value = float(text)
    if not value >= 0.0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative number of seconds, got {text}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """The launcher's command line."""
    parser = argparse.ArgumentParser(
        prog="launch",
        description="Start the processes of a launch file and stop them together.",
    )
    parser.add_argument(
        "launch_file", help="the launch file, a textproto of launch_proto.LaunchFile"
    )
    parser.add_argument(
        "--machine",
        choices=launch_file.MACHINES,
        default=None,
        help="start only the processes of this machine (default: all of them)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a variable of the launch file; repeatable",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the resolved commands instead of running them",
    )
    parser.add_argument(
        "--sigint_grace_period",
        type=_non_negative_seconds,
        default=None,
        help="seconds the teardown waits after SIGINT before SIGTERM (default: the launch file's, else "
        f"{launch_file.DEFAULT_SIGINT_GRACE_PERIOD_S:g})",
    )
    parser.add_argument(
        "--sigterm_grace_period",
        type=_non_negative_seconds,
        default=None,
        help="seconds the teardown waits after SIGTERM before SIGKILL (default: the launch file's, else "
        f"{launch_file.DEFAULT_SIGTERM_GRACE_PERIOD_S:g})",
    )
    parser.add_argument(
        "--color",
        choices=process_supervisor.COLOR_MODES,
        default="auto",
        help="color the [name] prefixes: auto (on a terminal, unless NO_COLOR is set), always or never",
    )
    parser.add_argument(
        "--terminal_command",
        default=shlex.join(process_supervisor.DEFAULT_TERMINAL_COMMAND),
        help="the command a process with `terminal: true` runs inside, its arguments appended (default: %(default)s)",
    )
    parser.add_argument(
        "--repo_root",
        default=None,
        help="the directory every process runs in (default: the workspace of `bazel run`, else the nearest "
        f"directory with {REPO_ROOT_MARKER} above the launch file or the working directory)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Resolves the launch file of the command line `argv` and prints it or runs it; returns the exit status."""
    args = build_parser().parse_args(argv)
    environment = dict(os.environ if environment is None else environment)
    try:
        path = resolve_launch_file_path(args.launch_file, environment)
        repo_root = find_repo_root(args.repo_root, path, environment)
        launch = launch_file.load_launch_file(
            path,
            overrides=launch_file.parse_overrides(args.overrides),
            builtins={"repo_root": repo_root},
        )
        processes = launch_file.select_machine(launch.processes, args.machine)
    except launch_file.LaunchFileError as error:
        err.write(f"launch: {error}\n")
        return EXIT_USAGE
    if not processes:
        err.write(f"launch: no process of {path} runs on machine '{args.machine}'\n")
        return EXIT_USAGE
    terminal_command: list[str] = shlex.split(args.terminal_command)
    if not terminal_command:
        err.write("launch: --terminal_command must not be empty\n")
        return EXIT_USAGE

    shutdown = launch_file.ShutdownPolicy(
        sigint_grace_period=(
            launch.shutdown.sigint_grace_period
            if args.sigint_grace_period is None
            else args.sigint_grace_period
        ),
        sigterm_grace_period=(
            launch.shutdown.sigterm_grace_period
            if args.sigterm_grace_period is None
            else args.sigterm_grace_period
        ),
    )
    if args.dry_run:
        out.write(format_dry_run(processes, repo_root, shutdown, terminal_command))
        out.flush()
        return 0

    os.chdir(repo_root)
    supervisor = process_supervisor.Supervisor(
        processes,
        repo_root=repo_root,
        shutdown=shutdown,
        output=out,
        color_mode=args.color,
        terminal_command=terminal_command,
        environment=environment,
    )

    def forward(signum: int, frame: types.FrameType | None) -> None:
        del frame
        supervisor.request_shutdown(signum)

    previous_handlers = {
        signum: signal.signal(signum, forward) for signum in SHUTDOWN_SIGNALS
    }
    try:
        return supervisor.run()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
