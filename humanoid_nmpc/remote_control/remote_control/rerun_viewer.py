"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""The GUI's "Open Rerun viewer" button: starts the Rerun bridge, which opens the native Rerun viewer.

The bridge (humanoid_nmpc/humanoid_rerun_viewer; humanoid_nmpc/docs/distributed_runtime/README.md, "Visualization with
Rerun") subscribes to viz/scene and viz/telemetry on the bus and spawns the viewer. The launch files start it next to
the MPC; the button is for a viewer that was closed, or a session started without one. It runs the binary Bazel has
built in the checkout, and says how to build it when it is missing: the GUI never runs Bazel itself, which would
compete with the operator's own builds for the machine's memory (AGENTS.md).

Free of Tk, so that test/test_rerun_viewer.py covers it headless.
"""

import ctypes
import dataclasses
import logging
import os
import signal
import subprocess
from typing import Callable, List, Optional

_LOGGER = logging.getLogger(__name__)

# The bridge's Bazel target, and where `bazel build` puts its binary in the checkout (.bazelrc's --symlink_prefix).
# LINT.IfChange(rerun_bridge_binary)
RERUN_BRIDGE_TARGET = "//humanoid_nmpc/humanoid_rerun_viewer"
RERUN_BRIDGE_BINARY = os.path.join(
    ".bazel", "bin", "humanoid_nmpc", "humanoid_rerun_viewer", "humanoid_rerun_viewer"
)
# LINT.ThenChange(//.bazelrc:symlink_prefix)

# How long close() waits for the bridge to exit after SIGTERM before it kills it [s].
STOP_GRACE_PERIOD = 2.0

# prctl(2)'s option that asks the kernel for a signal when the parent ends (linux/prctl.h).
_PR_SET_PDEATHSIG = 1


def _parent_death_signal() -> Optional[Callable[[], None]]:
    """A preexec_fn for the bridge: the kernel sends it SIGTERM when the GUI ends, however it ends.

    close() stops the bridge when the GUI shuts down; this covers a GUI that is killed (SIGKILL, a crash), which runs
    no cleanup. The signal follows the thread that started the bridge, which is the GUI's main (Tk) thread. Everything
    the child calls is looked up here, before the fork: between fork and exec it only makes two system calls. None
    where libc has no prctl.
    """
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()
    option = ctypes.c_int(_PR_SET_PDEATHSIG)
    signum = ctypes.c_ulong(int(signal.SIGTERM))
    unused = ctypes.c_ulong(0)

    def set_parent_death_signal() -> None:
        prctl(option, signum, unused, unused, unused)
        if os.getppid() != parent:
            # The GUI ended before the request took effect: no signal will come.
            os._exit(1)  # pylint: disable=protected-access

    return set_parent_death_signal


@dataclasses.dataclass(frozen=True)
class OpenResult:
    """What open() did: `started` is False when the bridge was already running or could not be started."""

    started: bool
    message: str


class RerunViewerLauncher:
    """Starts the Rerun bridge as a child process of the GUI, and stops it when the GUI closes.

    Args:
        repo_root: the source checkout whose build output holds the bridge.
        network_config: the network file the bridge subscribes with (an absolute path).
        urdf_file: the robot's URDF, whose meshes the bridge draws; empty: the bridge draws no robot.
        popen: starts the process (subprocess.Popen); the tests pass a fake one.
    """

    def __init__(
        self,
        repo_root: str,
        network_config: str,
        urdf_file: str = "",
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self._repo_root = repo_root
        self._network_config = network_config
        self._urdf_file = urdf_file
        self._popen = popen
        self._process: Optional[subprocess.Popen] = None

    @property
    def binary(self) -> str:
        return os.path.join(self._repo_root, RERUN_BRIDGE_BINARY)

    def command(self) -> List[str]:
        """The bridge's command line: its binary, the network file and, when known, the robot's URDF."""
        # LINT.IfChange(rerun_bridge_flags)
        command = [self.binary, f"--network_config={self._network_config}"]
        if self._urdf_file:
            command.append(f"--urdf={self._urdf_file}")
        # LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/cli.py)
        return command

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def open(self) -> OpenResult:
        """Starts the bridge unless it is already running or is not built."""
        if self.is_running():
            return OpenResult(
                started=False,
                message="The Rerun bridge this GUI started is still running; its viewer is open.",
            )
        if not os.access(self.binary, os.X_OK):
            return OpenResult(
                started=False,
                message=(
                    f"The Rerun bridge is not built ({self.binary} is missing). Build it with "
                    f"'bazel build {RERUN_BRIDGE_TARGET}' and press the button again."
                ),
            )
        try:
            self._process = self._popen(
                self.command(),
                cwd=self._repo_root,
                stdin=subprocess.DEVNULL,
                # Its own process group, so that a Ctrl-C in the GUI's terminal does not reach it twice. The launcher's
                # teardown of the GUI's group does not reach it either, hence the parent-death signal.
                start_new_session=True,
                preexec_fn=_parent_death_signal(),
            )
        except OSError as error:
            self._process = None
            return OpenResult(
                started=False, message=f"Could not start the Rerun bridge: {error}"
            )
        return OpenResult(started=True, message="Started the Rerun bridge.")

    def close(self) -> None:
        """Stops the bridge this launcher started, if it still runs: SIGTERM, then SIGKILL after a grace period."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=STOP_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            _LOGGER.warning(
                "The Rerun bridge did not stop within %.1f s; killing it.",
                STOP_GRACE_PERIOD,
            )
            process.kill()
            process.wait()
        except OSError:
            # Exited between poll() and the signal.
            pass
