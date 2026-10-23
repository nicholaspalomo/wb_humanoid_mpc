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

"""Tests of the launcher's command line, in this process and as a subprocess of launch.py.

In this process for --dry_run and the errors, and as a subprocess of launch.py, the real entry point, for signal
forwarding, exit codes and the death signal.
"""

from collections.abc import Sequence
import contextlib
import io
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import launch_cli
import launch_file

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "examples", "example.launch.textproto")
LAUNCH_SCRIPT = os.path.join(HERE, "launch.py")
# Upper bound on anything a test waits for; reached only when the test fails.
TIMEOUT_S = 15.0


def process_is_gone(pid: int) -> bool:
    """Whether the process `pid` has exited: it no longer exists, or is a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as stream:
            stat = stream.read()
    except OSError:
        return True
    return stat[stat.rfind(")") + 2 :].split()[0] in ("Z", "X")


class LauncherProcess:
    """launch.py running as a subprocess, its output collected line by line on a thread."""

    def __init__(self, argv: Sequence[str]) -> None:
        environment = dict(os.environ)
        # The launcher needs protobuf and the generated launch_proto module, which this test's runfiles provide.
        environment["PYTHONPATH"] = os.pathsep.join(
            entry for entry in sys.path if entry
        )
        self.lines: list[str] = []
        self._changed = threading.Condition()
        # pylint: disable-next=consider-using-with  # finish() reaps it, and the test's cleanup kills it.
        self.popen = subprocess.Popen(
            [sys.executable, LAUNCH_SCRIPT, *argv, "--color", "never"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        stdout = self.popen.stdout
        assert stdout is not None
        for line in stdout:
            with self._changed:
                self.lines.append(line.rstrip("\n"))
                self._changed.notify_all()

    def wait_for_line(self, pattern: str) -> re.Match[str] | None:
        """The first output line matching `pattern`, waiting for it up to TIMEOUT_S."""
        regex = re.compile(pattern)
        deadline = time.monotonic() + TIMEOUT_S
        with self._changed:
            while True:
                for line in self.lines:
                    match = regex.search(line)
                    if match:
                        return match
                remaining = deadline - time.monotonic()
                if (
                    remaining <= 0.0
                    or self.popen.poll() is not None
                    and not self._reader.is_alive()
                ):
                    return None
                self._changed.wait(min(remaining, 0.1))

    def matched_line(self, pattern: str) -> re.Match[str]:
        """The match of the first output line matching `pattern`; fails the test when none comes within TIMEOUT_S."""
        match = self.wait_for_line(pattern)
        assert match is not None, f"no output line matches {pattern!r}: {self.lines}"
        return match

    def finish(self) -> tuple[int, str]:
        """The launcher's exit status and its output, once it exits (within TIMEOUT_S, else it is killed)."""
        try:
            code = self.popen.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            raise
        self._reader.join(timeout=TIMEOUT_S)
        if self.popen.stdout is not None:
            self.popen.stdout.close()
        return code, "\n".join(self.lines)


class LaunchTestCase(unittest.TestCase):
    """A temporary repository root, launch_cli.main() run on captured output, and launcher subprocesses."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # The cleanup deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.repo_root = os.path.join(self._directory.name, "repo")
        os.makedirs(self.repo_root)

    def write_launch_file(
        self, text: str, file_name: str = "test.launch.textproto"
    ) -> str:
        path = os.path.join(self._directory.name, file_name)
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        return path

    def run_main(
        self, *argv: str, environment: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        code = launch_cli.main(
            list(argv), out=out, err=err, environment=environment or {}
        )
        return code, out.getvalue(), err.getvalue()

    def launcher(self, *argv: str) -> LauncherProcess:
        launcher = LauncherProcess(list(argv) + ["--repo_root", self.repo_root])

        def kill_if_running() -> None:
            if launcher.popen.poll() is None:
                launcher.popen.kill()

        self.addCleanup(kill_if_running)
        return launcher


class DryRunTest(LaunchTestCase):

    def test_prints_every_resolved_command_and_its_settings(self) -> None:
        code, out, err = self.run_main(
            EXAMPLE, "--dry_run", "--set", "cycles=7", "--repo_root", self.repo_root
        )
        self.assertEqual(code, 0, err)
        lines = out.splitlines()
        self.assertIn(
            f"# repository root (every command runs here): {self.repo_root}", lines
        )
        self.assertIn("# teardown: SIGINT, 2 s, SIGTERM, 1 s, SIGKILL", lines)
        self.assertIn(
            "# robot_loop: machine=robot required=true terminal=false delay=0s", lines
        )
        self.assertIn(
            "# mpc: machine=laptop required=false terminal=false delay=0.1s", lines
        )
        self.assertIn(
            "GREETING=hello sh -c 'echo \"$GREETING from the laptop\"; exec sleep 60'",
            lines,
        )
        self.assertTrue(any("-lt 7 ]" in line for line in lines), out)
        self.assertTrue(
            any(f"hello from {self.repo_root}" in line for line in lines), out
        )

    def test_machine_filters_the_processes(self) -> None:
        code, out, _ = self.run_main(
            EXAMPLE, "--dry_run", "--machine", "robot", "--repo_root", self.repo_root
        )
        self.assertEqual(code, 0)
        self.assertIn("# robot_loop:", out)
        self.assertNotIn("# mpc:", out)
        self.assertNotIn("# viewer:", out)

    def test_grace_periods_on_the_command_line_override_the_file(self) -> None:
        code, out, _ = self.run_main(
            EXAMPLE,
            "--dry_run",
            "--sigint_grace_period",
            "0.5",
            "--repo_root",
            self.repo_root,
        )
        self.assertEqual(code, 0)
        self.assertIn("# teardown: SIGINT, 0.5 s, SIGTERM, 1 s, SIGKILL", out)

    def test_terminal_processes_show_the_terminal_command(self) -> None:
        path = self.write_launch_file(
            'processes { name: "teleop" machine: MACHINE_LAPTOP command: "teleop" terminal: true }'
        )
        code, out, _ = self.run_main(
            path,
            "--dry_run",
            "--terminal_command",
            "xterm -hold -e",
            "--repo_root",
            self.repo_root,
        )
        self.assertEqual(code, 0)
        self.assertIn("xterm -hold -e teleop", out.splitlines())
        code, out, _ = self.run_main(path, "--dry_run", "--repo_root", self.repo_root)
        self.assertEqual(code, 0)
        self.assertIn("x-terminal-emulator -e teleop", out.splitlines())


class ErrorTest(LaunchTestCase):

    def test_invalid_input_exits_with_status_two_and_says_why(self) -> None:
        invalid = self.write_launch_file('processes { name: "a" command: "p" }')
        cases = {
            "invalid file": ([invalid], "process 'a': 'machine' must be one of"),
            "unparsable file": (
                [
                    self.write_launch_file(
                        'processes { name: "a" machine: DESKTOP }',
                        file_name="unparsable.launch.textproto",
                    )
                ],
                # The file, the line and the column of the value the schema does not have.
                "unparsable.launch.textproto:1:32: ",
            ),
            "missing file": (
                [os.path.join(self._directory.name, "absent.textproto")],
                "cannot read the launch file",
            ),
            "unknown variable": (
                [EXAMPLE, "--set", "cylces=2"],
                "undeclared variable(s) cylces",
            ),
            "malformed --set": ([EXAMPLE, "--set", "cycles"], "expected name=value"),
            "empty terminal command": (
                [EXAMPLE, "--terminal_command", ""],
                "must not be empty",
            ),
            "bad repository root": (
                [EXAMPLE, "--repo_root", os.path.join(self._directory.name, "no")],
                "not a dir",
            ),
        }
        for case, (argv, expected) in cases.items():
            with self.subTest(case=case):
                if "--repo_root" not in argv:
                    argv = argv + ["--repo_root", self.repo_root]
                code, out, err = self.run_main(*argv)
                self.assertEqual(code, launch_cli.EXIT_USAGE)
                self.assertEqual(out, "")
                self.assertIn(expected, err)

    def test_grace_periods_reject_nan_and_negative_values(self) -> None:
        # The validator is written `not value >= 0.0`, which rejects NaN, where `value < 0.0` would let it through.
        for flag in ("--sigint_grace_period", "--sigterm_grace_period"):
            for value in ("nan", "-1"):
                with self.subTest(flag=flag, value=value):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(
                        SystemExit
                    ) as raised:
                        launch_cli.build_parser().parse_args([EXAMPLE, flag, value])
                    self.assertEqual(raised.exception.code, launch_cli.EXIT_USAGE)
                    self.assertIn(
                        f"non-negative number of seconds, got {value}",
                        stderr.getvalue(),
                    )
        args = launch_cli.build_parser().parse_args(
            [EXAMPLE, "--sigint_grace_period", "0"]
        )
        self.assertEqual(args.sigint_grace_period, 0.0)

    def test_a_machine_without_processes_is_an_error(self) -> None:
        path = self.write_launch_file(
            'processes { name: "a" machine: MACHINE_LAPTOP command: "p" }'
        )
        code, _, err = self.run_main(
            path, "--machine", "robot", "--repo_root", self.repo_root
        )
        self.assertEqual(code, launch_cli.EXIT_USAGE)
        self.assertIn("runs on machine 'robot'", err)


class RepoRootTest(LaunchTestCase):

    def test_the_root_is_explicit_then_bazels_workspace_then_the_nearest_module_file(
        self,
    ) -> None:
        nested = os.path.join(self.repo_root, "robot_models", "launch")
        os.makedirs(nested)
        pathlib.Path(self.repo_root, launch_cli.REPO_ROOT_MARKER).touch()
        launch_path = os.path.join(nested, "sim.launch.textproto")
        elsewhere = os.path.join(self._directory.name, "elsewhere")
        os.makedirs(elsewhere)
        no_bazel = {"BUILD_WORKING_DIRECTORY": elsewhere}

        self.assertEqual(
            launch_cli.find_repo_root(elsewhere, launch_path, no_bazel), elsewhere
        )
        self.assertEqual(
            launch_cli.find_repo_root(
                None, launch_path, {"BUILD_WORKSPACE_DIRECTORY": "/workspace"}
            ),
            "/workspace",
        )
        self.assertEqual(
            launch_cli.find_repo_root(None, launch_path, no_bazel), self.repo_root
        )
        # A launch file outside the checkout, started from inside it.
        self.assertEqual(
            launch_cli.find_repo_root(
                None,
                os.path.join(elsewhere, "x.textproto"),
                {"BUILD_WORKING_DIRECTORY": nested},
            ),
            self.repo_root,
        )
        with self.assertRaises(launch_file.LaunchFileError):
            launch_cli.find_repo_root(
                None, os.path.join(elsewhere, "x.textproto"), no_bazel
            )

    def test_relative_launch_files_resolve_from_where_bazel_run_was_started(
        self,
    ) -> None:
        pathlib.Path(self.repo_root, "a.textproto").touch()
        self.assertEqual(
            launch_cli.resolve_launch_file_path(
                "a.textproto", {"BUILD_WORKING_DIRECTORY": self.repo_root}
            ),
            os.path.join(self.repo_root, "a.textproto"),
        )
        self.assertEqual(
            launch_cli.resolve_launch_file_path(
                "a.textproto",
                {
                    "BUILD_WORKING_DIRECTORY": self._directory.name,
                    "BUILD_WORKSPACE_DIRECTORY": self.repo_root,
                },
            ),
            os.path.join(self.repo_root, "a.textproto"),
        )


class RunTest(LaunchTestCase):

    def test_the_example_runs_until_its_required_process_finishes(self) -> None:
        code, output = self.launcher(EXAMPLE, "--set", "cycles=2").finish()
        self.assertEqual(code, 0, output)
        lines = output.splitlines()
        self.assertIn("[robot_loop] hello from the robot, cycle 1", lines)
        self.assertIn("[robot_loop] hello from the robot, cycle 2", lines)
        self.assertIn("[launch] robot_loop is required; stopping", lines)

    def test_the_launcher_runs_processes_in_the_repository_root(self) -> None:
        path = self.write_launch_file(
            'processes { name: "where" machine: MACHINE_ROBOT command: ["sh", "-c", "pwd -P"] required: true }'
        )
        code, output = self.launcher(path).finish()
        self.assertEqual(code, 0, output)
        self.assertIn(
            f"[where] {os.path.realpath(self.repo_root)}", output.splitlines()
        )

    def test_sigint_and_sigterm_are_forwarded_and_end_with_status_zero(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum.name):
                launcher = self.launcher(EXAMPLE, "--machine", "laptop")
                self.assertIsNotNone(
                    launcher.wait_for_line(r"^\[mpc\] hello from the laptop$")
                )
                self.assertIsNotNone(launcher.wait_for_line(r"^\[viewer\] hello from"))
                pids = [
                    int(
                        launcher.matched_line(rf"started {name} \(pid (\d+)\)").group(1)
                    )
                    for name in ("mpc", "viewer")
                ]
                launcher.popen.send_signal(signum)
                code, output = launcher.finish()
                self.assertEqual(code, 0, output)
                self.assertIn(f"[launch] received {signum.name}; stopping", output)
                self.assertIn(f"[launch] mpc was killed by {signum.name}", output)
                self.assertTrue(all(process_is_gone(pid) for pid in pids), output)

    def test_a_failing_required_process_sets_the_exit_status(self) -> None:
        path = self.write_launch_file(
            'processes { name: "sleeper" machine: MACHINE_LAPTOP command: ["sh", "-c", "exec sleep 60"] }\n'
            "processes {\n"
            '  name: "failing"\n'
            "  machine: MACHINE_ROBOT\n"
            '  command: ["sh", "-c", "sleep 0.2; exit 5"]\n'
            "  required: true\n"
            "}\n"
        )
        code, output = self.launcher(path).finish()
        self.assertEqual(code, 5, output)
        self.assertIn("[launch] sleeper was killed by SIGINT", output)

    def test_children_die_with_a_launcher_that_is_killed(self) -> None:
        path = self.write_launch_file(
            'processes { name: "orphan" machine: MACHINE_ROBOT command: ["sh", "-c", "echo ready; exec sleep 60"] }'
        )
        launcher = self.launcher(path)
        self.assertIsNotNone(launcher.wait_for_line(r"^\[orphan\] ready$"))
        child = int(launcher.matched_line(r"started orphan \(pid (\d+)\)").group(1))
        # SIGKILL leaves the launcher no chance to tear down: only the death signal stops the child.
        launcher.popen.kill()
        launcher.finish()
        deadline = time.monotonic() + TIMEOUT_S
        while not process_is_gone(child) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(process_is_gone(child), f"child {child} outlived its launcher")


if __name__ == "__main__":
    unittest.main()
