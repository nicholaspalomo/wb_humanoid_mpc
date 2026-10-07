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

"""The "Open Rerun viewer" button's launcher, without the Rerun bridge.

RerunViewerLauncher runs a stand-in bridge (a shell script in a temporary checkout) or a fake Popen; it only ever stops
the process it started itself.
"""

from collections.abc import Callable
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from remote_control import rerun_viewer


def _wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _alive(pid: int) -> bool:
    """Whether `pid` runs: a zombie, which only waits for its new parent to reap it, does not."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as stat_file:
            return stat_file.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _FakeProcess:
    """A Popen that records the signals it is sent and exits on the first."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.signals: list[int] = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        self.returncode = -signum

    def wait(self, timeout=None):
        del timeout  # Unused.
        return self.returncode

    def kill(self):
        self.returncode = -9


class TestRerunViewerLauncher(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # The cleanup below deletes it.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = self.tmpdir.name
        self.network = os.path.join(
            self.repo_root, "config", "ipc", "network.textproto"
        )
        self.calls = []

    def _fake_popen(self, command, **kwargs):
        self.calls.append((command, kwargs))
        self.process = _FakeProcess()
        return self.process

    def _install_bridge(self, script="#!/bin/sh\nexec sleep 60\n"):
        binary = os.path.join(self.repo_root, rerun_viewer.RERUN_BRIDGE_BINARY)
        os.makedirs(os.path.dirname(binary))
        with open(binary, "w", encoding="utf-8") as handle:
            handle.write(script)
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
        return binary

    def test_the_command_is_the_built_bridge_with_the_network_file(self):
        launcher = rerun_viewer.RerunViewerLauncher(self.repo_root, self.network)
        self.assertEqual(
            launcher.command(),
            [
                os.path.join(self.repo_root, rerun_viewer.RERUN_BRIDGE_BINARY),
                f"--network_config={self.network}",
            ],
        )
        # Bazel's output directory of the bridge's target.
        self.assertTrue(
            rerun_viewer.RERUN_BRIDGE_BINARY.startswith(
                os.path.join(".bazel", "bin", "")
            )
        )
        self.assertTrue(
            rerun_viewer.RERUN_BRIDGE_BINARY.endswith(
                rerun_viewer.RERUN_BRIDGE_TARGET.lstrip("/").rsplit("/", 1)[1]
            )
        )

    def test_the_robots_urdf_is_handed_to_the_bridge_when_known(self):
        launcher = rerun_viewer.RerunViewerLauncher(
            self.repo_root, self.network, urdf_file="/robots/g1.urdf"
        )
        self.assertEqual(
            launcher.command()[1:],
            [f"--network_config={self.network}", "--urdf=/robots/g1.urdf"],
        )

    def test_a_bridge_that_is_not_built_is_not_started_and_says_how_to_build_it(self):
        launcher = rerun_viewer.RerunViewerLauncher(
            self.repo_root, self.network, popen=self._fake_popen
        )
        result = launcher.open()
        self.assertFalse(result.started)
        self.assertIn(f"bazel build {rerun_viewer.RERUN_BRIDGE_TARGET}", result.message)
        self.assertEqual(self.calls, [])

    def test_a_built_bridge_is_started_once_in_its_own_session(self):
        self._install_bridge()
        launcher = rerun_viewer.RerunViewerLauncher(
            self.repo_root, self.network, popen=self._fake_popen
        )
        self.assertTrue(launcher.open().started)
        command, options = self.calls[0]
        self.assertEqual(command, launcher.command())
        self.assertTrue(options["start_new_session"])
        self.assertTrue(callable(options["preexec_fn"]), "no parent-death signal")
        self.assertEqual(options["cwd"], self.repo_root)
        # While it runs, the button does not start a second one.
        second = launcher.open()
        self.assertFalse(second.started)
        self.assertEqual(len(self.calls), 1)

    def test_close_stops_the_bridge_it_started_and_only_that(self):
        self._install_bridge()
        launcher = rerun_viewer.RerunViewerLauncher(
            self.repo_root, self.network, popen=self._fake_popen
        )
        launcher.close()  # Nothing started: nothing to stop.
        launcher.open()
        launcher.close()
        self.assertEqual(len(self.process.signals), 1)
        self.assertFalse(launcher.is_running())
        # A bridge that exited can be started again.
        self.assertTrue(launcher.open().started)

    def test_a_real_child_is_started_and_stopped(self):
        self._install_bridge()
        launcher = rerun_viewer.RerunViewerLauncher(self.repo_root, self.network)
        self.addCleanup(launcher.close)
        self.assertTrue(launcher.open().started)
        self.assertTrue(launcher.is_running())
        launcher.close()
        self.assertFalse(launcher.is_running())

    def test_the_bridge_ends_with_a_gui_that_is_killed(self):
        # A GUI killed outright runs no cleanup, and the bridge is in a session of its own that no process-group
        # teardown reaches: the kernel's parent-death signal is what ends it.
        pid_file = os.path.join(self.repo_root, "bridge.pid")
        self._install_bridge(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 60\n')
        gui_script = textwrap.dedent(
            f"""
            import time
            from remote_control.rerun_viewer import RerunViewerLauncher

            launcher = RerunViewerLauncher({self.repo_root!r}, {self.network!r})
            assert launcher.open().started
            print("started", flush=True)
            time.sleep(60)
            """
        )
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        # pylint: disable-next=consider-using-with  # The cleanups below kill and reap it.
        gui = subprocess.Popen(
            [sys.executable, "-c", gui_script],
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
        )
        self.addCleanup(gui.wait)

        def kill_gui() -> None:
            if gui.poll() is None:
                gui.kill()

        self.addCleanup(kill_gui)
        assert gui.stdout is not None
        self.assertEqual(gui.stdout.readline().strip(), "started")

        def bridge_pid() -> int:
            try:
                with open(pid_file, encoding="utf-8") as handle:
                    return int(handle.read().strip() or 0)
            except (OSError, ValueError):
                return 0

        self.assertTrue(_wait_for(lambda: bridge_pid() > 0))
        bridge = bridge_pid()
        self.addCleanup(_kill_quietly, bridge)
        self.assertTrue(_alive(bridge))

        gui.kill()
        gui.wait()
        self.assertTrue(_wait_for(lambda: not _alive(bridge)))

    def test_a_binary_that_cannot_run_is_reported(self):
        self._install_bridge()

        def failing_popen(command, **kwargs):
            raise OSError("exec format error")

        result = rerun_viewer.RerunViewerLauncher(
            self.repo_root, self.network, popen=failing_popen
        ).open()
        self.assertFalse(result.started)
        self.assertIn("exec format error", result.message)


if __name__ == "__main__":
    unittest.main()
