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

"""Runs the processes of a launch file: starts them, prefixes their output, and stops all of them together.

Every process starts in its own session, hence its own process group, so that

- a Ctrl-C in the terminal reaches only the launcher, which then stops the processes in a defined order instead of
  every process racing the terminal's SIGINT;
- a signal sent to the group also reaches whatever the process started itself (a shell's children, a terminal
  emulator's program), so a teardown leaves no grandchildren behind.

Each child also asks the kernel for a death signal (PR_SET_PDEATHSIG, SIGKILL by default), so it dies with the
launcher even when the launcher is killed without a chance to tear down. The kernel sends it when the THREAD that
started the child exits, which is why processes are only ever started from the thread that calls run(). The death
signal reaches the direct child only; its own children are reached by the teardown's group signals.

Teardown sends SIGINT to every process group, waits up to the SIGINT grace period for the groups to empty, then sends
SIGTERM and waits the SIGTERM grace period, then SIGKILL. A second shutdown request during the teardown skips to the
next stage.
"""

from collections.abc import Callable, Mapping, Sequence
import ctypes
import dataclasses
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import BinaryIO, TextIO

import launch_file

# How often the supervisor checks for exits, starts and shutdown requests.
POLL_PERIOD_S = 0.02
# How long the last stage waits for SIGKILL to take effect before reporting the processes it could not stop (a process
# in uninterruptible sleep, or a member that left its group).
SIGKILL_WAIT_S = 2.0
# How long run() waits for the output of an exited process to drain.
OUTPUT_DRAIN_TIMEOUT_S = 1.0

# The launcher's own messages use this name in place of a process name.
LAUNCHER_NAME = "launch"
# LINT.IfChange(color_modes)
COLOR_MODES = ("auto", "always", "never")
# LINT.ThenChange(//tools/launch/README.md:color_modes)
# ANSI foreground colors of the process prefixes, assigned in launch-file order.
PREFIX_COLORS = (32, 36, 33, 35, 34, 92, 96, 93, 95, 94)

# What a `terminal: true` process runs inside, its command appended; the ROS 2 launch files used the same.
DEFAULT_TERMINAL_COMMAND = ("x-terminal-emulator", "-e")

# Shell conventions for a program that could not be started.
EXIT_CANNOT_EXECUTE = 126
EXIT_NOT_FOUND = 127

# Environment variables through which a Bazel binary finds its runfiles. Under `bazel run //tools/launch` they name the
# launcher's runfiles; a child that is a Bazel binary too must find its own, so they are not passed on.
RUNFILES_ENVIRONMENT = (
    "RUNFILES_DIR",
    "RUNFILES_MANIFEST_FILE",
    "RUNFILES_MANIFEST_ONLY",
    "JAVA_RUNFILES",
    "PYTHON_RUNFILES",
)

_PR_SET_PDEATHSIG = 1


def shell_exit_code(returncode: int) -> int:
    """A Popen return code as a shell reports it: the exit status, or 128 + N for a process killed by signal N."""
    return returncode if returncode >= 0 else 128 - returncode


def describe_exit(code: int) -> str:
    """How the console reports the shell exit code `code`: `was killed by SIGINT`, or `exited with code 3`."""
    if code > 128:
        try:
            return f"was killed by {signal.Signals(code - 128).name}"
        except ValueError:
            pass
    return f"exited with code {code}"


def _is_runfiles_path(entry: str) -> bool:
    """Whether the path `entry` is inside a Bazel runfiles tree."""
    return any(part.endswith(".runfiles") for part in entry.split(os.sep))


def child_environment(
    base: Mapping[str, str], overrides: Sequence[tuple[str, str]]
) -> dict[str, str]:
    """The environment of a child: the launcher's, without its Bazel runfiles, plus the process's `env`.

    PYTHONUNBUFFERED is set unless the environment already has it, so that the output of Python children reaches the
    prefixed console line by line instead of in 8 KiB blocks.

    Args:
        base: The launcher's environment.
        overrides: The process's `env`, which wins over `base`.

    Returns:
        The environment, without the runfiles variables and the runfiles entries of PYTHONPATH.
    """
    environment = {
        key: value for key, value in base.items() if key not in RUNFILES_ENVIRONMENT
    }
    if "PYTHONPATH" in environment:
        entries = [
            entry
            for entry in environment["PYTHONPATH"].split(os.pathsep)
            if entry and not _is_runfiles_path(entry)
        ]
        if entries:
            environment["PYTHONPATH"] = os.pathsep.join(entries)
        else:
            del environment["PYTHONPATH"]
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.update(overrides)
    return environment


def process_argv(
    spec: launch_file.ProcessSpec, terminal_command: Sequence[str]
) -> list[str]:
    """The argument vector a process is started with: its command, inside the terminal command when `terminal`."""
    if spec.terminal:
        return list(terminal_command) + list(spec.command)
    return list(spec.command)


def _death_signal_preexec(death_signal: int) -> Callable[[], None] | None:
    """A preexec_fn that makes the child receive `death_signal` when the launcher dies; None off Linux."""
    if not sys.platform.startswith("linux"):
        return None
    # The running program's symbols include libc's. (ctypes.util.find_library would shell out to ldconfig.)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
    except (OSError, AttributeError):
        return None
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    launcher_pid = os.getpid()

    def preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, death_signal, 0, 0, 0)
        # The launcher may have died between the fork and the prctl, and then no death signal would ever come.
        if os.getppid() != launcher_pid:
            os.kill(os.getpid(), death_signal)

    return preexec


def _live_group_members(process_group: int) -> list[int] | None:
    """The processes of a group that are not zombies, from /proc; None where /proc cannot be read.

    A zombie still counts for kill(), so a group whose last members died but are not reaped yet (an orphaned
    grandchild waiting for init) would otherwise look alive until the teardown gave up on it.

    Args:
        process_group: The process group ID.

    Returns:
        The PIDs of its members that are neither zombies nor dead, or None when /proc cannot be listed.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    members = []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(
                f"/proc/{entry}/stat", "r", encoding="utf-8", errors="replace"
            ) as stream:
                stat = stream.read()
        except OSError:
            continue
        # The command name in parentheses may contain spaces; the fields after it are: state ppid pgrp ...
        fields = stat[stat.rfind(")") + 2 :].split()
        if (
            len(fields) > 2
            and int(fields[2]) == process_group
            and fields[0] not in ("Z", "X")
        ):
            members.append(int(entry))
    return members


class OutputSink:
    """Writes whole lines, each behind the colored [name] of the process it came from, from any thread."""

    def __init__(
        self, stream: TextIO, color_mode: str, environment: Mapping[str, str]
    ) -> None:
        if color_mode not in COLOR_MODES:
            raise ValueError(
                f"unknown color mode {color_mode!r}; the modes are: {', '.join(COLOR_MODES)}"
            )
        if color_mode == "auto":
            is_tty = getattr(stream, "isatty", lambda: False)()
            self._use_color = is_tty and "NO_COLOR" not in environment
        else:
            self._use_color = color_mode == "always"
        self._stream = stream
        self._lock = threading.Lock()
        self._broken = False

    def prefix(self, name: str, index: int | None) -> str:
        """`[name]`, colored by the process's position in the launch file; bold for the launcher (index None)."""
        if not self._use_color:
            return f"[{name}]"
        code = "1" if index is None else str(PREFIX_COLORS[index % len(PREFIX_COLORS)])
        return f"\x1b[{code}m[{name}]\x1b[0m"

    def write_line(self, prefix: str, line: str) -> None:
        """Writes `line` behind `prefix`; once the console has gone away, writes nothing."""
        with self._lock:
            if self._broken:
                return
            try:
                self._stream.write(f"{prefix} {line}\n")
                self._stream.flush()
            except (OSError, ValueError):
                # The console went away (a closed pipe): keep supervising, silently.
                self._broken = True


@dataclasses.dataclass
class _ManagedProcess:
    """A process of the launch file and what the supervisor knows of it."""

    spec: launch_file.ProcessSpec
    index: int
    prefix: str
    popen: subprocess.Popen[bytes] | None = None
    reader: threading.Thread | None = None
    # Shell convention (shell_exit_code) once the process has exited or failed to start.
    exit_code: int | None = None
    reported: bool = False


def _pump_output(stream: BinaryIO, prefix: str, sink: OutputSink) -> None:
    """Writes every line of `stream` to `sink` behind `prefix` until the stream ends, then closes it."""
    try:
        for raw in iter(stream.readline, b""):
            sink.write_line(
                prefix, raw.decode("utf-8", errors="replace").rstrip("\r\n")
            )
    except (OSError, ValueError):
        pass
    finally:
        stream.close()


def _teardown_stage(signum: int) -> int:
    """The teardown stage a shutdown request starts at: SIGINT first, SIGTERM (and SIGHUP) skip straight to SIGTERM."""
    return 0 if signum == signal.SIGINT else 1


class Supervisor:
    """Starts processes, relays their output, and stops all of them when one that is required exits.

    run() returns the launcher's exit status: the shell exit code of the first required process that failed (exited
    non-zero or could not be started) before the teardown began, and 0 otherwise. Processes that are not required
    may fail without changing it; their exit is reported on the console.
    """

    def __init__(
        self,
        processes: Sequence[launch_file.ProcessSpec],
        repo_root: str,
        shutdown: launch_file.ShutdownPolicy,
        output: TextIO,
        color_mode: str = "auto",
        terminal_command: Sequence[str] = DEFAULT_TERMINAL_COMMAND,
        environment: Mapping[str, str] | None = None,
        death_signal: int = signal.SIGKILL,
    ) -> None:
        self._repo_root = repo_root
        self._shutdown = shutdown
        self._terminal_command = list(terminal_command)
        self._environment = dict(os.environ if environment is None else environment)
        self._sink = OutputSink(output, color_mode, self._environment)
        self._launcher_prefix = self._sink.prefix(LAUNCHER_NAME, None)
        self._processes = [
            _ManagedProcess(
                spec=spec, index=index, prefix=self._sink.prefix(spec.name, index)
            )
            for index, spec in enumerate(processes)
        ]
        self._preexec = _death_signal_preexec(death_signal)
        # SimpleQueue.put is reentrant, so a signal handler may call request_shutdown() at any point of run().
        self._requests: "queue.SimpleQueue[int]" = queue.SimpleQueue()

    def request_shutdown(self, signum: int = signal.SIGINT) -> None:
        """Asks run() to stop everything, as if the launcher had received `signum`.

        Safe from signal handlers and other threads; a second request during the teardown skips to the next, stronger
        signal.
        """
        self._requests.put(signum)

    def log(self, message: str) -> None:
        """Writes one of the launcher's own messages to the console."""
        self._sink.write_line(self._launcher_prefix, message)

    def _next_request(self) -> int | None:
        """The signal of the oldest shutdown request not yet handled, or None."""
        try:
            return self._requests.get_nowait()
        except queue.Empty:
            return None

    def _start(self, process: _ManagedProcess) -> None:
        """Starts `process` in a session of its own and relays its output; a start that fails sets its exit code."""
        argv = process_argv(process.spec, self._terminal_command)
        try:
            # preexec_fn only sets the parent-death signal (PR_SET_PDEATHSIG), which start_new_session cannot do; the
            # function calls prctl, getppid and kill, nothing that could take a lock another thread holds. The process
            # outlives this call: the teardown stops it and run() reaps it.
            # pylint: disable-next=consider-using-with,subprocess-popen-preexec-fn  # The reasons are above.
            process.popen = subprocess.Popen(
                argv,
                cwd=self._repo_root,
                env=child_environment(self._environment, process.spec.env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=self._preexec,
            )
        except OSError as error:
            process.exit_code = (
                EXIT_NOT_FOUND
                if isinstance(error, FileNotFoundError)
                else EXIT_CANNOT_EXECUTE
            )
            self.log(f"cannot start {process.spec.name}: {argv[0]}: {error.strerror}")
            return
        self.log(f"started {process.spec.name} (pid {process.popen.pid})")
        process.reader = threading.Thread(
            target=_pump_output,
            args=(process.popen.stdout, process.prefix, self._sink),
            name=f"output of {process.spec.name}",
            daemon=True,
        )
        process.reader.start()

    def _collect_exits(self) -> list[_ManagedProcess]:
        """The processes that exited since the last call, reported on the console."""
        exited = []
        for process in self._processes:
            if process.exit_code is None and process.popen is not None:
                returncode = process.popen.poll()
                if returncode is not None:
                    process.exit_code = shell_exit_code(returncode)
            if process.exit_code is not None and not process.reported:
                process.reported = True
                if process.popen is not None:
                    self.log(f"{process.spec.name} {describe_exit(process.exit_code)}")
                exited.append(process)
        return exited

    def _group_alive(self, process: _ManagedProcess) -> bool:
        """Whether the process, or any member of its process group but a zombie, is still running."""
        if process.popen is None:
            return False
        if process.popen.poll() is None:
            return True
        try:
            os.killpg(process.popen.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        members = _live_group_members(process.popen.pid)
        return True if members is None else bool(members)

    def _signal_group(self, process: _ManagedProcess, signum: int) -> None:
        """Sends `signum` to the process group of `process`, unless it never started or the group is gone."""
        if process.popen is None:
            return
        try:
            os.killpg(process.popen.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    def _teardown(self, first_stage: int) -> None:
        """Signals every live process group, stage by stage from `first_stage`: SIGINT, SIGTERM, then SIGKILL."""
        stages = (
            (signal.SIGINT, self._shutdown.sigint_grace_period),
            (signal.SIGTERM, self._shutdown.sigterm_grace_period),
            (signal.SIGKILL, SIGKILL_WAIT_S),
        )
        for signum, grace_period in stages[first_stage:]:
            alive = [
                process for process in self._processes if self._group_alive(process)
            ]
            if not alive:
                break
            self.log(
                f"sending {signal.Signals(signum).name} to {', '.join(process.spec.name for process in alive)}"
            )
            for process in alive:
                self._signal_group(process, signum)
            deadline = time.monotonic() + grace_period
            while time.monotonic() < deadline:
                self._collect_exits()
                if not any(self._group_alive(process) for process in alive):
                    break
                if self._next_request() is not None:
                    self.log("shutdown requested again; escalating")
                    break
                time.sleep(POLL_PERIOD_S)
        self._collect_exits()
        survivors = [
            process.spec.name
            for process in self._processes
            if self._group_alive(process)
        ]
        if survivors:
            self.log(f"could not stop the process group of {', '.join(survivors)}")

    def _drain_output(self) -> None:
        """Waits up to OUTPUT_DRAIN_TIMEOUT_S, in all, for the output of every process to reach the console."""
        deadline = time.monotonic() + OUTPUT_DRAIN_TIMEOUT_S
        for process in self._processes:
            if process.reader is not None:
                process.reader.join(timeout=max(deadline - time.monotonic(), 0.0))

    def run(self) -> int:
        """Starts every process after its delay, supervises them, and stops them all.

        Supervision ends when every process has exited, a required one has exited, or a shutdown was requested; then
        the teardown stops the rest.

        Returns:
            The launcher's exit status (see the class comment).
        """
        start = time.monotonic()
        pending = sorted(self._processes, key=lambda process: process.spec.delay)
        status = 0
        first_stage: int | None = None
        try:
            while first_stage is None:
                signum = self._next_request()
                if signum is not None:
                    self.log(f"received {signal.Signals(signum).name}; stopping")
                    first_stage = _teardown_stage(signum)
                    break
                while pending and start + pending[0].spec.delay <= time.monotonic():
                    self._start(pending.pop(0))
                for process in self._collect_exits():
                    if process.spec.required and first_stage is None:
                        # Not None: _collect_exits() returns exited processes only.
                        if process.exit_code is not None and process.exit_code != 0:
                            status = process.exit_code
                        self.log(f"{process.spec.name} is required; stopping")
                        first_stage = 0
                if (
                    first_stage is None
                    and not pending
                    and all(
                        process.exit_code is not None for process in self._processes
                    )
                ):
                    self.log("every process has exited")
                    break
                if first_stage is None:
                    time.sleep(POLL_PERIOD_S)
        except KeyboardInterrupt:
            # Only without a SIGINT handler (the command line installs one); stop as for a Ctrl-C.
            first_stage = 0
        # Also when every process exited on its own: what they left running in their groups is stopped too.
        self._teardown(0 if first_stage is None else first_stage)
        self._drain_output()
        return status
