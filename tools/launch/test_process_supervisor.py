"""Tests for process_supervisor.py with small sh and Python children: output, exits, delays and the teardown.

The supervisor runs in this process (run() on the main thread, as the command line does); shutdown requests come from
a helper thread once a child has printed that it is ready, so no test depends on how fast a child starts.
"""

import io
import os
import re
import signal
import sys
import tempfile
import threading
import time
import unittest
from typing import List, Optional

import launch_file
import process_supervisor

# Short grace periods keep the escalation tests fast; a passing teardown finishes as soon as its groups are empty.
FAST_SHUTDOWN = launch_file.ShutdownPolicy(
    sigint_grace_period=0.3, sigterm_grace_period=0.3
)
# Upper bound on anything a test waits for; reached only when the test fails.
TIMEOUT_S = 15.0

IGNORE_SIGINT = """
import signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
def on_term(signum, frame):
    print("got SIGTERM", flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
print("ready", flush=True)
time.sleep(60)
"""

IGNORE_SIGINT_AND_SIGTERM = """
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
time.sleep(60)
"""


def sh(name: str, script: str, **settings: object) -> launch_file.ProcessSpec:
    return launch_file.ProcessSpec(
        name=name, machine="robot", command=("sh", "-c", script), **settings
    )


def python(name: str, script: str, **settings: object) -> launch_file.ProcessSpec:
    return launch_file.ProcessSpec(
        name=name, machine="robot", command=(sys.executable, "-c", script), **settings
    )


def process_is_gone(pid: int) -> bool:
    """True when no process has the pid, or only a zombie waiting for whoever adopted it to reap it."""
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


def wait_until_gone(pid: int, timeout_s: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process_is_gone(pid):
            return True
        time.sleep(0.02)
    return process_is_gone(pid)


class SupervisorTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.output = io.StringIO()

    def supervisor(
        self,
        processes: List[launch_file.ProcessSpec],
        shutdown: launch_file.ShutdownPolicy = FAST_SHUTDOWN,
        **settings: object,
    ) -> process_supervisor.Supervisor:
        return process_supervisor.Supervisor(
            processes,
            repo_root=self._directory.name,
            shutdown=shutdown,
            output=self.output,
            color_mode="never",
            **settings,
        )

    def request_shutdown_after(
        self,
        supervisor: process_supervisor.Supervisor,
        text: str,
        signals: List[int],
        spacing_s: float = 0.0,
    ) -> threading.Thread:
        """Requests a shutdown for each of `signals` once `text` was printed (or after TIMEOUT_S regardless)."""

        def watch() -> None:
            deadline = time.monotonic() + TIMEOUT_S
            while text not in self.output.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            for signum in signals:
                supervisor.request_shutdown(signum)
                time.sleep(spacing_s)

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        return thread

    def run_supervisor(self, supervisor: process_supervisor.Supervisor) -> int:
        start = time.monotonic()
        status = supervisor.run()
        self.elapsed = time.monotonic() - start
        self.assertLess(self.elapsed, TIMEOUT_S, self.output.getvalue())
        return status

    def pid_of(self, name: str) -> int:
        match = re.search(
            rf"started {re.escape(name)} \(pid (\d+)\)", self.output.getvalue()
        )
        self.assertIsNotNone(match, self.output.getvalue())
        return int(match.group(1))


class OutputTest(SupervisorTestCase):

    def test_every_line_carries_the_name_of_its_process(self) -> None:
        status = self.run_supervisor(
            self.supervisor(
                [
                    sh("talker", "echo one; echo two >&2; printf 'no newline'"),
                    python("pythonic", "print('three')"),
                ]
            )
        )
        self.assertEqual(status, 0)
        lines = self.output.getvalue().splitlines()
        for expected in (
            "[talker] one",
            "[talker] two",
            "[talker] no newline",
            "[pythonic] three",
        ):
            with self.subTest(line=expected):
                self.assertIn(expected, lines)
        self.assertIn("[launch] every process has exited", lines)

    def test_colors_are_ansi_codes_around_the_prefix_only_when_asked(self) -> None:
        colored = process_supervisor.OutputSink(io.StringIO(), "always", {})
        self.assertRegex(colored.prefix("mpc", 0), r"^\x1b\[\d+m\[mpc\]\x1b\[0m$")
        self.assertNotEqual(colored.prefix("mpc", 0), colored.prefix("robot", 1))
        plain = process_supervisor.OutputSink(io.StringIO(), "never", {})
        self.assertEqual(plain.prefix("mpc", 0), "[mpc]")
        # auto: a StringIO is not a terminal.
        self.assertEqual(
            process_supervisor.OutputSink(io.StringIO(), "auto", {}).prefix("mpc", 0),
            "[mpc]",
        )
        with self.assertRaises(ValueError):
            process_supervisor.OutputSink(io.StringIO(), "sometimes", {})

    def test_processes_run_in_the_repository_root_with_their_environment(self) -> None:
        status = self.run_supervisor(
            self.supervisor(
                [
                    sh(
                        "env",
                        'echo "cwd=$(pwd -P)"; echo "greeting=$GREETING"; echo "runfiles=${RUNFILES_DIR:-unset}"',
                        env=(("GREETING", "hello"),),
                    )
                ],
                environment={
                    "PATH": os.environ["PATH"],
                    "RUNFILES_DIR": "/launcher.runfiles",
                },
            )
        )
        self.assertEqual(status, 0)
        output = self.output.getvalue()
        self.assertIn(f"[env] cwd={os.path.realpath(self._directory.name)}", output)
        self.assertIn("[env] greeting=hello", output)
        self.assertIn("[env] runfiles=unset", output)


class ExitTest(SupervisorTestCase):

    def test_a_required_process_that_fails_stops_the_others_and_sets_the_status(
        self,
    ) -> None:
        supervisor = self.supervisor(
            [
                sh("sleeper", "echo ready; exec sleep 60"),
                sh("planner", "sleep 0.2; exit 3", required=True),
            ]
        )
        status = self.run_supervisor(supervisor)
        self.assertEqual(status, 3)
        output = self.output.getvalue()
        self.assertIn("[launch] planner exited with code 3", output)
        self.assertIn("[launch] planner is required; stopping", output)
        self.assertIn("[launch] sleeper was killed by SIGINT", output)
        self.assertTrue(process_is_gone(self.pid_of("sleeper")))

    def test_a_required_process_that_succeeds_stops_the_others_with_status_zero(
        self,
    ) -> None:
        status = self.run_supervisor(
            self.supervisor(
                [sh("sleeper", "exec sleep 60"), sh("job", "exit 0", required=True)]
            )
        )
        self.assertEqual(status, 0)
        self.assertIn("[launch] sleeper was killed by SIGINT", self.output.getvalue())

    def test_a_process_that_is_not_required_may_fail(self) -> None:
        status = self.run_supervisor(
            self.supervisor(
                [sh("flaky", "exit 4"), sh("job", "sleep 0.3; exit 0", required=True)]
            )
        )
        self.assertEqual(status, 0)
        self.assertIn("[launch] flaky exited with code 4", self.output.getvalue())

    def test_a_required_process_killed_by_a_signal_reports_128_plus_the_signal(
        self,
    ) -> None:
        # SIGUSR1 terminates without a core dump.
        status = self.run_supervisor(
            self.supervisor([sh("crasher", "kill -USR1 $$", required=True)])
        )
        self.assertEqual(status, 128 + signal.SIGUSR1)
        self.assertIn("[launch] crasher was killed by SIGUSR1", self.output.getvalue())

    def test_a_required_program_that_does_not_exist_fails_with_127(self) -> None:
        missing = launch_file.ProcessSpec(
            name="ghost",
            machine="robot",
            command=("/nonexistent/program",),
            required=True,
        )
        status = self.run_supervisor(
            self.supervisor([missing, sh("sleeper", "exec sleep 60")])
        )
        self.assertEqual(status, process_supervisor.EXIT_NOT_FOUND)
        self.assertIn(
            "[launch] cannot start ghost: /nonexistent/program", self.output.getvalue()
        )


class DelayTest(SupervisorTestCase):

    def test_processes_start_after_their_delay_and_not_at_all_after_a_shutdown(
        self,
    ) -> None:
        status = self.run_supervisor(
            self.supervisor(
                [
                    sh("late", "echo late", delay=30.0),
                    sh("delayed", "echo delayed", delay=0.3, required=True),
                ]
            )
        )
        self.assertEqual(status, 0)
        self.assertGreaterEqual(self.elapsed, 0.3)
        output = self.output.getvalue()
        self.assertIn("[delayed] delayed", output)
        self.assertNotIn("started late", output)


class TeardownTest(SupervisorTestCase):

    def test_a_child_that_ignores_sigint_gets_sigterm(self) -> None:
        supervisor = self.supervisor([python("stubborn", IGNORE_SIGINT)])
        self.request_shutdown_after(supervisor, "[stubborn] ready", [signal.SIGINT])
        status = self.run_supervisor(supervisor)
        self.assertEqual(status, 0)
        output = self.output.getvalue()
        self.assertIn("[launch] sending SIGINT to stubborn", output)
        self.assertIn("[launch] sending SIGTERM to stubborn", output)
        self.assertIn("[stubborn] got SIGTERM", output)
        self.assertGreaterEqual(self.elapsed, FAST_SHUTDOWN.sigint_grace_period)

    def test_a_child_that_ignores_sigint_and_sigterm_gets_sigkill(self) -> None:
        supervisor = self.supervisor([python("immovable", IGNORE_SIGINT_AND_SIGTERM)])
        self.request_shutdown_after(supervisor, "[immovable] ready", [signal.SIGINT])
        status = self.run_supervisor(supervisor)
        self.assertEqual(status, 0)
        output = self.output.getvalue()
        self.assertIn("[launch] sending SIGKILL to immovable", output)
        self.assertIn("[launch] immovable was killed by SIGKILL", output)
        self.assertTrue(process_is_gone(self.pid_of("immovable")))

    def test_sigterm_skips_the_sigint_stage(self) -> None:
        supervisor = self.supervisor([sh("sleeper", "echo ready; exec sleep 60")])
        self.request_shutdown_after(supervisor, "[sleeper] ready", [signal.SIGTERM])
        self.run_supervisor(supervisor)
        output = self.output.getvalue()
        self.assertIn("[launch] received SIGTERM; stopping", output)
        self.assertNotIn("sending SIGINT", output)
        self.assertIn("[launch] sleeper was killed by SIGTERM", output)

    def test_a_second_request_escalates_without_waiting_for_the_grace_period(
        self,
    ) -> None:
        patient = launch_file.ShutdownPolicy(
            sigint_grace_period=60.0, sigterm_grace_period=60.0
        )
        supervisor = self.supervisor(
            [python("stubborn", IGNORE_SIGINT)], shutdown=patient
        )
        self.request_shutdown_after(
            supervisor,
            "[stubborn] ready",
            [signal.SIGINT, signal.SIGINT],
            spacing_s=0.2,
        )
        self.run_supervisor(supervisor)
        output = self.output.getvalue()
        self.assertIn("[launch] shutdown requested again; escalating", output)
        self.assertIn("[stubborn] got SIGTERM", output)
        self.assertLess(self.elapsed, 10.0)

    def test_grandchildren_die_with_their_group_after_the_leader_is_gone(self) -> None:
        # A non-interactive shell starts background jobs with SIGINT ignored, so the grandchild survives the SIGINT
        # that ends its parent: the teardown has to see the group is not empty and go on to SIGTERM.
        pid_file = os.path.join(self._directory.name, "grandchild.pid")
        supervisor = self.supervisor(
            [sh("parent", f'sleep 60 & echo $! > "{pid_file}"; echo ready; wait')]
        )
        self.request_shutdown_after(supervisor, "[parent] ready", [signal.SIGINT])
        self.run_supervisor(supervisor)
        with open(pid_file, "r", encoding="utf-8") as stream:
            grandchild = int(stream.read())
        output = self.output.getvalue()
        self.assertIn("[launch] parent was killed by SIGINT", output)
        self.assertIn("[launch] sending SIGTERM to parent", output)
        self.assertTrue(
            wait_until_gone(grandchild, timeout_s=1.0),
            f"grandchild {grandchild} survived",
        )
        self.assertNotIn("could not stop", output)


class LeftoverTest(SupervisorTestCase):

    def test_what_an_exited_process_left_in_its_group_is_stopped_at_the_end(
        self,
    ) -> None:
        pid_file = os.path.join(self._directory.name, "leftover.pid")
        status = self.run_supervisor(
            self.supervisor([sh("daemonizer", f'sleep 60 & echo $! > "{pid_file}"')])
        )
        self.assertEqual(status, 0)
        with open(pid_file, "r", encoding="utf-8") as stream:
            leftover = int(stream.read())
        output = self.output.getvalue()
        self.assertIn("[launch] every process has exited", output)
        self.assertIn("[launch] sending SIGINT to daemonizer", output)
        self.assertTrue(wait_until_gone(leftover, timeout_s=1.0), output)


class HelpersTest(unittest.TestCase):

    def test_shell_exit_codes(self) -> None:
        self.assertEqual(process_supervisor.shell_exit_code(0), 0)
        self.assertEqual(process_supervisor.shell_exit_code(3), 3)
        self.assertEqual(process_supervisor.shell_exit_code(-signal.SIGKILL), 137)
        self.assertEqual(process_supervisor.describe_exit(137), "was killed by SIGKILL")
        self.assertEqual(process_supervisor.describe_exit(2), "exited with code 2")

    def test_the_child_environment_drops_the_launchers_runfiles(self) -> None:
        base = {
            "PATH": "/usr/bin",
            "RUNFILES_DIR": "/x/launch.runfiles",
            "RUNFILES_MANIFEST_FILE": "/x/launch.runfiles/MANIFEST",
            "PYTHONPATH": os.pathsep.join(
                ["/x/launch.runfiles/_main", "/usr/local/lib/python3/site-packages"]
            ),
        }
        environment = process_supervisor.child_environment(base, [("DISPLAY", ":99")])
        self.assertNotIn("RUNFILES_DIR", environment)
        self.assertNotIn("RUNFILES_MANIFEST_FILE", environment)
        self.assertEqual(
            environment["PYTHONPATH"], "/usr/local/lib/python3/site-packages"
        )
        self.assertEqual(environment["DISPLAY"], ":99")
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
        # Nothing but runfiles: no PYTHONPATH at all, rather than an empty one (which means the working directory).
        only_runfiles = process_supervisor.child_environment(
            {"PYTHONPATH": "/x/a.runfiles/_main"}, []
        )
        self.assertNotIn("PYTHONPATH", only_runfiles)
        # A process's env wins.
        self.assertEqual(
            process_supervisor.child_environment({}, [("PYTHONUNBUFFERED", "0")])[
                "PYTHONUNBUFFERED"
            ],
            "0",
        )

    def test_terminal_processes_run_inside_the_terminal_command(self) -> None:
        spec = launch_file.ProcessSpec(
            name="teleop",
            machine="laptop",
            command=("teleop", "--rate=25"),
            terminal=True,
        )
        self.assertEqual(
            process_supervisor.process_argv(spec, ["xterm", "-e"]),
            ["xterm", "-e", "teleop", "--rate=25"],
        )
        plain = launch_file.ProcessSpec(name="mpc", machine="laptop", command=("mpc",))
        self.assertEqual(
            process_supervisor.process_argv(plain, ["xterm", "-e"]), ["mpc"]
        )


if __name__ == "__main__":
    unittest.main()
