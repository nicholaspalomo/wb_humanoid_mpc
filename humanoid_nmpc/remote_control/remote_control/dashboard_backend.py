"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

"""The backend of an interactive dashboard (a Jupyter notebook, a script): simulations and a virtual joystick.

Provides:
1. SimProcessManager: launches, follows and stops one simulation target (a Makefile launch target) in the background.
2. VirtualJoystick: walking commands from code, published on the IPC bus like the GUI's sticks.

Free of Tk. VirtualJoystick publishes operator/walking_velocity_command as the `teleop` node of the network file, or
through a publisher it is given (the tests give it one that records).
"""

import os
import queue
import signal
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import walking_velocity_command_pb2

from remote_control import config_files
from remote_control.humanoid_finite_state_machine import load_robot_config
from remote_control.operator_bus import (
    DEFAULT_NETWORK_CONFIG,
    TELEOP_NODE,
    TopicPublisher,
    walking_velocity_command,
)
from remote_control.teleop import (
    WALKING_COMMAND_RATE_HZ,
    connect_walking_command_publisher,
)

# How long stop() waits for a simulation's processes to exit after SIGTERM before it kills them [s]. The launcher
# (tools/launch) tears its processes down on SIGTERM, with grace periods of its own.
STOP_GRACE_PERIOD = 10.0


class SimProcessManager:
    """Runs one simulation target at a time in the background, in a process group of its own.

    TARGETS is the one table of the targets a dashboard offers. Its commands are the Makefile's launch targets, which
    a later stage renames when the launch files move to tools/launch: change them here, and nowhere else.
    """

    TARGETS: Dict[str, Dict[str, str]] = {
        "g1_centroidal_dummy": {
            "name": "Unitree G1 Centroidal — Dummy Sim",
            "command": "make launch-g1-dummy-sim-vnc",
            "type": "dummy",
            "robot": "g1",
        },
        "g1_centroidal_sim": {
            "name": "Unitree G1 Centroidal — MuJoCo Physics Sim",
            "command": "make launch-g1-sim-vnc",
            "type": "mujoco",
            "robot": "g1",
        },
        "g1_wb_dummy": {
            "name": "Unitree G1 Whole-Body — Dummy Sim",
            "command": "make launch-wb-g1-dummy-sim-vnc",
            "type": "dummy",
            "robot": "g1",
        },
        "g1_wb_sim": {
            "name": "Unitree G1 Whole-Body — MuJoCo Physics Sim",
            "command": "make launch-wb-g1-sim-vnc",
            "type": "mujoco",
            "robot": "g1",
        },
        "atlas_centroidal_dummy": {
            "name": "DRC Atlas Centroidal — Dummy Sim",
            "command": "make launch-drc-atlas-dummy-sim-vnc",
            "type": "dummy",
            "robot": "atlas",
        },
        "atlas_centroidal_sim": {
            "name": "DRC Atlas Centroidal — MuJoCo Ground Sim",
            "command": "make launch-drc-atlas-sim-vnc",
            "type": "mujoco",
            "robot": "atlas",
        },
        "r1_centroidal_dummy": {
            "name": "Unitree R1 Centroidal — Dummy Sim",
            "command": "make launch-r1-dummy-sim-vnc",
            "type": "dummy",
            "robot": "r1",
        },
        "r1_centroidal_sim": {
            "name": "Unitree R1 Centroidal — MuJoCo Physics Sim",
            "command": "make launch-r1-sim-vnc",
            "type": "mujoco",
            "robot": "r1",
        },
        "sa01_centroidal_dummy": {
            "name": "EngineAI SA01 Centroidal — Dummy Sim",
            "command": "make launch-sa01-dummy-sim-vnc",
            "type": "dummy",
            "robot": "sa01",
        },
        "sa01_centroidal_sim": {
            "name": "EngineAI SA01 Centroidal — MuJoCo Physics Sim",
            "command": "make launch-sa01-sim-vnc",
            "type": "mujoco",
            "robot": "sa01",
        },
    }

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        stop_grace_period: float = STOP_GRACE_PERIOD,
    ) -> None:
        self.workspace_dir = (
            workspace_dir or config_files.find_repo_root() or os.getcwd()
        )
        self.stop_grace_period = stop_grace_period
        self.process: Optional[subprocess.Popen] = None
        self.active_target_key: Optional[str] = None
        self.log_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def command_of(self, target_key: str) -> List[str]:
        """The command line of a target.

        Raises:
            ValueError: an unknown target; the message lists the known ones.
        """
        if target_key not in self.TARGETS:
            raise ValueError(
                f"Unknown target '{target_key}'. Available: {list(self.TARGETS)}"
            )
        return ["bash", "-c", self.TARGETS[target_key]["command"]]

    def launch(
        self,
        target_key: str,
        on_output: Optional[Callable[[str], None]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Stops the running target and launches `target_key` in the background; returns whether it started.

        Its output goes, line by line, to `log_queue` (the newest 1000 lines) and, in chunks, to `on_output`.

        Raises:
            ValueError: an unknown target.
        """
        command = self.command_of(target_key)
        self.stop()
        target_info = self.TARGETS[target_key]

        run_env = os.environ.copy()
        if run_env.get("DISPLAY") in (":1", "", None):
            run_env["DISPLAY"] = ":99"
        run_env["PYTHONUNBUFFERED"] = "1"
        if env_vars:
            run_env.update(env_vars)

        if on_output:
            on_output(f"🚀 Initializing target: {target_info['name']}...\n")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.workspace_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=run_env,
                start_new_session=True,
            )
        except OSError as error:
            if on_output:
                on_output(f"❌ Failed to launch {target_key}: {error}\n")
            self.process = None
            return False
        self.active_target_key = target_key
        self._reader_thread = threading.Thread(
            target=self._read_output, args=(self.process, on_output), daemon=True
        )
        self._reader_thread.start()
        return True

    def _read_output(
        self, process: subprocess.Popen, on_output: Optional[Callable[[str], None]]
    ) -> None:
        """Forwards the target's output until it closes its stdout (it exited, or stop() killed it)."""
        buffer: List[str] = []
        last_flush = time.monotonic()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                buffer.append(line)
                try:
                    self.log_queue.put_nowait(line)
                except queue.Full:
                    try:
                        self.log_queue.get_nowait()
                        self.log_queue.put_nowait(line)
                    except (queue.Empty, queue.Full):
                        pass
                now = time.monotonic()
                if on_output and (now - last_flush >= 0.1 or len(buffer) >= 15):
                    on_output("".join(buffer))
                    buffer.clear()
                    last_flush = now
        except (OSError, ValueError):
            # The pipe broke under the reader.
            pass
        if on_output and buffer:
            on_output("".join(buffer))

    def stop(self) -> bool:
        """Stops the running target and every process of its group: SIGTERM, then SIGKILL after the grace period."""
        process = self.process
        self.process = None
        self.active_target_key = None
        if process is None:
            return True
        # The group, not only its leader: a `bash -c make ...` that has exited may have left processes in it, which the
        # launcher (tools/launch) stops in the same case.
        if process.poll() is None or _process_group_alive(process.pid):
            _stop_process_group(process, self.stop_grace_period)
        try:
            process.wait(timeout=self.stop_grace_period)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader = self._reader_thread
        self._reader_thread = None
        if reader is not None:
            # The pipe ends when the last process holding it exits; one that left the group may keep it open, and its
            # reader then stays behind as a daemon thread rather than stalling the caller.
            reader.join(timeout=1.0)
        if process.stdout is not None and (reader is None or not reader.is_alive()):
            process.stdout.close()
        return True

    def get_status(self) -> Dict[str, str]:
        """The state of the target this manager launched: RUNNING while its process runs, STOPPED otherwise."""
        if self.is_running and self.active_target_key is not None:
            return {
                "status": "RUNNING",
                "target": self.active_target_key,
                "name": self.TARGETS[self.active_target_key]["name"],
                "pid": str(self.process.pid),
            }
        return {"status": "STOPPED", "target": "None", "name": "None", "pid": "None"}


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(process: subprocess.Popen, grace_period: float) -> None:
    """SIGTERM to the process group `process` leads, then SIGKILL to what is left of it after `grace_period` seconds.

    Only the group this manager created (start_new_session, so its id is the leader's pid) is signaled: never a
    process found by name.
    """
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        # Reaps the leader once it exits: a zombie still counts as a member of its group.
        process.poll()
        if not _process_group_alive(pgid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class VirtualJoystick:
    """Walking commands from code, published on the bus at the topic's rate while streaming.

    Args:
        publish_rate: the streaming rate [Hz].
        robot_name: the robot whose nominal pelvis height is the initial height command.
        workspace_dir: where to look for the robot's files (default: the checkout).
        network_config: the network file of the bus, used by connect() when no publisher is given.
        node_name: the node connect() publishes as.
        publisher: a walking command publisher to use instead of connecting (the tests pass one that records).
        auto_connect: connect (and, with auto_stream, start streaming) on construction.
        auto_stream: start streaming on the first command.
    """

    _ACTIVE_INSTANCES: List["VirtualJoystick"] = []

    def __init__(
        self,
        publish_rate: float = WALKING_COMMAND_RATE_HZ,
        robot_name: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        network_config: str = DEFAULT_NETWORK_CONFIG,
        node_name: str = TELEOP_NODE,
        publisher: Optional[TopicPublisher] = None,
        auto_connect: bool = False,
        auto_stream: bool = True,
    ) -> None:
        # One joystick per process: a notebook cell run twice must not leave the first one streaming, nor bound to the
        # node's port.
        while VirtualJoystick._ACTIVE_INSTANCES:
            previous = VirtualJoystick._ACTIVE_INSTANCES.pop()
            previous.shutdown()

        if (
            publisher is not None
            and publisher.topic != topics.OPERATOR_WALKING_VELOCITY_COMMAND
        ):
            raise ValueError(
                f"the publisher publishes '{publisher.topic}', not the walking command"
            )
        self.publish_rate = publish_rate
        self.robot_name = robot_name
        self.workspace_dir = workspace_dir
        self.network_config = network_config
        self.node_name = node_name
        self.auto_stream = auto_stream
        try:
            config = load_robot_config(
                robot_name=robot_name, workspace_dir=workspace_dir
            )
            self.desired_height = float(config.get("nominal_pelvis_height_bent", 0.70))
        except (
            Exception
        ):  # pylint: disable=broad-except - any unreadable robot falls back to the default
            self.desired_height = 0.70

        self.v_x = 0.0
        self.v_y = 0.0
        self.v_yaw = 0.0
        self._lock = threading.Lock()
        self._publisher = publisher
        self._owns_bus = False
        self._init_error: Optional[str] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_streaming = threading.Event()

        if auto_connect:
            self.connect()
            if self.auto_stream:
                self.start_streaming()
        VirtualJoystick._ACTIVE_INSTANCES.append(self)

    def connect(self) -> bool:
        """Connects to the bus as `node_name`, unless a publisher was given; returns whether it is connected."""
        if self._publisher is not None:
            return True
        try:
            self._publisher = connect_walking_command_publisher(
                self.network_config, self.node_name
            )
        except (
            Exception
        ) as error:  # pylint: disable=broad-except - reported through init_error
            self._init_error = str(error)
            return False
        self._owns_bus = True
        self._init_error = None
        return True

    @property
    def is_connected(self) -> bool:
        return self._publisher is not None

    @property
    def init_error(self) -> Optional[str]:
        """Why the last connect() failed, or None."""
        return self._init_error

    def command(self) -> walking_velocity_command_pb2.WalkingVelocityCommand:
        """The command the joystick holds now."""
        with self._lock:
            return walking_velocity_command(
                self.v_x, self.v_y, self.v_yaw, self.desired_height
            )

    def start_streaming(self) -> None:
        """Publishes the current command at `publish_rate` on a background thread until stop_streaming()."""
        if not self.connect():
            return
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stop_streaming.clear()
        period = 1.0 / max(1.0, float(self.publish_rate))

        def stream() -> None:
            while not self._stop_streaming.is_set():
                self._publish(self.command())
                self._stop_streaming.wait(period)

        self._stream_thread = threading.Thread(target=stream, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self) -> None:
        self._stop_streaming.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None

    def publish_now(self) -> None:
        """Publishes the current command once, and starts streaming when auto_stream is set."""
        if not self.connect():
            return
        if self.auto_stream:
            self.start_streaming()
        self._publish(self.command())

    def _publish(
        self, command: walking_velocity_command_pb2.WalkingVelocityCommand
    ) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            publisher.publish(command)
        except robot_ipc.BusError:
            # shutdown() closed the bus while the stream thread, which it waits for only a second, was still running:
            # there is nothing to send to any more.
            pass

    def set_velocity(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        desired_height: Optional[float] = None,
    ) -> None:
        """Sets the commanded walking velocity (normalized, in [-1, 1]) and publishes it."""
        with self._lock:
            self.v_x = float(linear_x)
            self.v_y = float(linear_y)
            self.v_yaw = float(angular_z)
            if desired_height is not None:
                self.desired_height = float(desired_height)
        self.publish_now()

    def stop(self) -> None:
        """Emergency stop: zero velocity, published at once."""
        self.set_velocity(0.0, 0.0, 0.0)

    def step(
        self, direction: str, delta_v: float = 0.2, delta_yaw: float = 0.2
    ) -> None:
        """Changes the velocity by one step: forward, backward, left, right, turn_left, turn_right or stop."""
        if direction == "stop":
            self.stop()
            return
        with self._lock:
            if direction == "forward":
                self.v_x = min(1.0, self.v_x + delta_v)
            elif direction == "backward":
                self.v_x = max(-1.0, self.v_x - delta_v)
            elif direction == "left":
                self.v_y = min(0.5, self.v_y + delta_v)
            elif direction == "right":
                self.v_y = max(-0.5, self.v_y - delta_v)
            elif direction == "turn_left":
                self.v_yaw = min(1.0, self.v_yaw + delta_yaw)
            elif direction == "turn_right":
                self.v_yaw = max(-1.0, self.v_yaw - delta_yaw)
            else:
                raise ValueError(
                    f"unknown direction '{direction}' (valid: forward, backward, left, right, turn_left, "
                    "turn_right, stop)"
                )
        self.publish_now()

    def shutdown(self) -> None:
        """Stops streaming and closes the bus this joystick connected (not one it was given)."""
        self.stop_streaming()
        publisher = self._publisher
        if self._owns_bus and publisher is not None:
            publisher.bus.close()
            self._publisher = None
            self._owns_bus = False
