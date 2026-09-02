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

"""Interactive Dashboard Backend for Jupyter Notebook & Colab.

Provides:
1. SimProcessManager: Subprocess manager to launch, monitor, and stop Dummy and MuJoCo simulations.
2. VirtualJoystickROS2: ROS2 teleoperation interface publishing WalkingVelocityCommand.
3. Interactive UI widget builders using ipywidgets and HTML/CSS.
"""

import glob
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional


import ctypes

from .humanoid_finite_state_machine import (
    ControlMode,
    HumanoidFSM,
    VirtualGantry,
    MODE_METADATA,
    CYCLE_MODES,
    load_robot_config,
    get_available_robots,
)


def ensure_ros2_paths():
    """Auto-detects and adds ROS2 and Bazel-generated message bindings to sys.path."""
    # 1. Base ROS2 distribution site-packages from ROS_DISTRO or /opt/ros
    ros_distro = os.environ.get("ROS_DISTRO")
    ros_paths = []
    if ros_distro and os.path.exists(f"/opt/ros/{ros_distro}"):
        ros_paths.append(
            f"/opt/ros/{ros_distro}/lib/python3.{sys.version_info.minor}/site-packages"
        )
    else:
        for candidate in sorted(glob.glob("/opt/ros/*")):
            sp = f"{candidate}/lib/python3.{sys.version_info.minor}/site-packages"
            if os.path.isdir(sp):
                ros_paths.append(sp)

    for ros_path in ros_paths:
        if os.path.exists(ros_path) and ros_path not in sys.path:
            sys.path.append(ros_path)

    # 2. Bazel-generated message packages (humanoid_mpc_msgs, ocs2_ros2_msgs)
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.expanduser("~/.cache/bazel"),
        os.path.join(os.getcwd(), ".bazel"),
        "/root/.cache/bazel",
        "/home/ubuntu/.cache/bazel",
    ]
    for c_dir in candidates:
        if os.path.exists(c_dir):
            for match in glob.glob(
                f"{c_dir}/**/install/*/lib/{py_ver}/site-packages",
                recursive=True,
            ):
                if match not in sys.path:
                    sys.path.append(match)


ensure_ros2_paths()


class SimProcessManager:
    """Manages background simulation subprocesses (Dummy Sim & MuJoCo Sim)."""

    TARGETS: Dict[str, Dict[str, str]] = {
        "g1_centroidal_dummy": {
            "name": "Unitree G1 Centroidal — Dummy Sim (RViz)",
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
            "name": "Unitree G1 Whole-Body — Dummy Sim (RViz)",
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
            "name": "DRC Atlas Centroidal — Dummy Sim (RViz)",
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
            "name": "Unitree R1 Centroidal — Dummy Sim (RViz)",
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
    }

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.process: Optional[subprocess.Popen] = None
        self.active_target_key: Optional[str] = None
        self.log_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self._reader_thread: Optional[threading.Thread] = None

    def launch(
        self,
        target_key: str,
        on_output: Optional[Callable[[str], None]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Launches a simulation target in the background."""
        self.stop()

        if target_key not in self.TARGETS:
            raise ValueError(
                f"Unknown target '{target_key}'. Available: {list(self.TARGETS.keys())}"
            )

        target_info = self.TARGETS[target_key]
        cmd = target_info["command"]

        # Prepare environment
        run_env = os.environ.copy()
        if run_env.get("DISPLAY") in (":1", "", None):
            run_env["DISPLAY"] = ":99"
        run_env["PYTHONUNBUFFERED"] = "1"
        run_env["RCUTILS_LOGGING_BUFFERED_STREAM"] = "0"
        if env_vars:
            run_env.update(env_vars)

        # Build execution string sourcing setup_env.sh
        setup_script = os.path.join(self.workspace_dir, "setup_env.sh")
        if os.path.exists(setup_script):
            full_cmd = f"source {setup_script} && {cmd}"
        else:
            full_cmd = cmd

        if on_output:
            on_output(f"🚀 Initializing target: {target_info['name']}...\n")

        try:
            self.process = subprocess.Popen(
                ["bash", "-c", full_cmd],
                cwd=self.workspace_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=run_env,
                preexec_fn=os.setsid,
            )
            self.is_running = True
            self.active_target_key = target_key

            def _reader():
                buffer = []
                last_flush = time.time()
                try:
                    while self.is_running:
                        proc = self.process
                        if proc is None or proc.stdout is None:
                            break
                        try:
                            line = proc.stdout.readline()
                        except Exception:
                            break
                        if not line:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.01)
                            # Periodically flush if idle
                            now = time.time()
                            if buffer and (now - last_flush >= 0.1):
                                chunk = "".join(buffer)
                                buffer.clear()
                                last_flush = now
                                if on_output:
                                    on_output(chunk)
                            continue

                        buffer.append(line)
                        try:
                            self.log_queue.put_nowait(line)
                        except queue.Full:
                            try:
                                self.log_queue.get_nowait()
                                self.log_queue.put_nowait(line)
                            except Exception:
                                pass

                        now = time.time()
                        if now - last_flush >= 0.1 or len(buffer) >= 15:
                            chunk = "".join(buffer)
                            buffer.clear()
                            last_flush = now
                            if on_output:
                                on_output(chunk)

                    if buffer:
                        chunk = "".join(buffer)
                        if on_output:
                            on_output(chunk)
                except Exception:
                    pass
                finally:
                    self.is_running = False

            self._reader_thread = threading.Thread(target=_reader, daemon=True)
            self._reader_thread.start()
            return True

        except Exception as e:
            if on_output:
                on_output(f"❌ Failed to launch {target_key}: {e}\n")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """Stops the active simulation process cleanly."""
        if self.process is not None:
            try:
                import signal

                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.1)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    self.process.kill()
            except Exception:
                pass
            self.process = None

        # Clean up spawned ROS nodes specifically by binary name or bazel install path
        # (Targeted commands ensure Jupyter/Python kernels are never matched)
        try:
            cleanup_cmds = [
                ["pkill", "-9", "-f", "/tmp/.bazel_ros_install"],
                ["pkill", "-9", "-f", "base_velocity_controller_gui"],
                ["pkill", "-9", "-f", "ros2 launch"],
                ["pkill", "-9", "-x", "rviz2"],
                ["pkill", "-9", "-x", "robot_state_publisher"],
                ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sqp_node"],
                ["pkill", "-9", "-x", "humanoid_centroidal_mpc_dummy_sim_node"],
                ["pkill", "-9", "-x", "humanoid_centroidal_mpc_sim"],
                ["pkill", "-9", "-x", "humanoid_wb_mpc_sqp_node"],
                ["pkill", "-9", "-x", "humanoid_wb_mpc_sim"],
            ]
            for c in cleanup_cmds:
                subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Allow processes to actually terminate before launching a new sim
            time.sleep(0.5)
        except Exception:
            pass

        self.is_running = False
        self.active_target_key = None
        return True

    def get_status(self) -> Dict[str, str]:
        """Returns the current simulation execution status."""
        active_pids = []
        try:
            # Use pgrep -a to get full command lines so we can filter out
            # build wrappers (bash -c "bazel build ... && ros2 launch ...")
            # that mention sim target names but aren't the actual running sim.
            pcheck = subprocess.run(
                [
                    "pgrep",
                    "-a",
                    "-f",
                    r"/tmp/\.bazel_ros_install/(humanoid_centroidal_mpc_ros2|humanoid_wb_mpc_ros2)/lib/|/opt/ros/jazzy/lib/rviz2/rviz2",
                ],
                capture_output=True,
                text=True,
            )
            if pcheck.returncode == 0 and pcheck.stdout.strip():
                current_pid = str(os.getpid())
                for line in pcheck.stdout.strip().splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) < 2:
                        continue
                    pid_s, cmdline = parts
                    if pid_s == current_pid:
                        continue
                    # Skip wrapper commands that contain "bazel build" or "make launch"
                    if "bazel build" in cmdline or "make launch" in cmdline:
                        continue
                    active_pids.append(pid_s)
        except Exception:
            pass

        if active_pids:
            self.is_running = True
            pid_str = (
                str(self.process.pid)
                if (self.process and self.process.poll() is None)
                else active_pids[0]
            )
            target_key = self.active_target_key or "atlas_centroidal_sim"
            target_name = self.TARGETS.get(target_key, {}).get(
                "name", "Humanoid Simulation & Visualizer"
            )
            return {
                "status": "RUNNING",
                "target": target_key,
                "name": target_name,
                "pid": pid_str,
            }
        elif self.is_running and self.process and self.process.poll() is None:
            return {
                "status": "BUILDING",
                "target": self.active_target_key or "Unknown",
                "name": self.TARGETS.get(self.active_target_key, {}).get(
                    "name", "Compiling targets..."
                ),
                "pid": str(self.process.pid),
            }
        else:
            self.is_running = False
            return {
                "status": "STOPPED",
                "target": "None",
                "name": "None",
                "pid": "None",
            }


class VirtualJoystickROS2:
    """Virtual Joystick Teleoperation publisher over ROS2."""

    _ACTIVE_INSTANCES = []

    def __init__(
        self,
        topic: str = "/humanoid/walking_velocity_command",
        publish_rate: float = 25.0,
        robot_name: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        auto_connect: bool = False,
        auto_stream: bool = True,
    ):
        # Shutdown any previous active joystick instances in the same process
        while VirtualJoystickROS2._ACTIVE_INSTANCES:
            prev = VirtualJoystickROS2._ACTIVE_INSTANCES.pop()
            try:
                prev.shutdown()
            except Exception:
                pass

        self.topic = topic
        self.publish_rate = publish_rate
        self.robot_name = robot_name
        self.workspace_dir = workspace_dir

        # Determine nominal height from robot config
        try:
            cfg = load_robot_config(robot_name=robot_name, workspace_dir=workspace_dir)
            self.desired_height = float(cfg.get("nominal_pelvis_height_bent", 0.70))
        except Exception:
            self.desired_height = 0.70

        self.v_x = 0.0
        self.v_y = 0.0
        self.v_yaw = 0.0

        self._node = None
        self._publisher = None
        self._stream_thread = None
        self._is_streaming = False
        self._is_active = False
        self._ros_available = False
        self._init_error = None
        self._publish_active = False
        self.auto_stream = auto_stream

        if auto_connect:
            self.connect()
            if self.auto_stream:
                self.start_streaming()
        VirtualJoystickROS2._ACTIVE_INSTANCES.append(self)

    def connect(self) -> bool:
        """Initializes ROS2 node and publisher safely without background spin."""
        if self._is_active and self._publisher is not None:
            return True
        try:
            ensure_ros2_paths()
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from humanoid_mpc_msgs.msg import WalkingVelocityCommand

            if not rclpy.ok():
                try:
                    from rclpy.signals import SignalHandlerOptions

                    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
                except Exception:
                    try:
                        rclpy.init()
                    except Exception:
                        pass

            node_name = f"jupyter_virtual_joystick_{int(time.time()*1000) % 100000}"
            self._node = rclpy.create_node(node_name)
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
            self._publisher = self._node.create_publisher(
                WalkingVelocityCommand, self.topic, qos
            )
            self._ros_available = True
            self._is_active = True
            return True
        except Exception as e:
            self._ros_available = False
            self._is_active = False
            self._init_error = str(e)
            return False

    @property
    def is_ros_connected(self) -> bool:
        """Returns True if the ROS2 node and publisher are actively connected."""
        return self._ros_available and self._is_active

    def start_streaming(self):
        """Starts 25 Hz background streaming of WalkingVelocityCommand to ROS2."""
        if not self.is_ros_connected:
            self.connect()
        if not self.is_ros_connected:
            return
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._is_streaming = True

        def _stream_loop():
            period = 1.0 / max(1.0, float(self.publish_rate))
            while self._is_streaming and self._is_active:
                if self._publisher is not None:
                    try:
                        from humanoid_mpc_msgs.msg import WalkingVelocityCommand

                        msg = WalkingVelocityCommand()
                        msg.linear_velocity_x = float(self.v_x)
                        msg.linear_velocity_y = float(self.v_y)
                        msg.angular_velocity_z = float(self.v_yaw)
                        msg.desired_pelvis_height = float(self.desired_height)
                        self._publisher.publish(msg)
                    except Exception:
                        pass
                time.sleep(period)

        self._stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self):
        """Stops background streaming."""
        self._is_streaming = False

    def publish_now(self):
        """Publishes the current velocity command immediately to ROS2."""
        if not self.is_ros_connected:
            self.connect()
        if self.auto_stream and not self._is_streaming:
            self.start_streaming()
        if self._publisher is not None:
            try:
                from humanoid_mpc_msgs.msg import WalkingVelocityCommand

                msg = WalkingVelocityCommand()
                msg.linear_velocity_x = float(self.v_x)
                msg.linear_velocity_y = float(self.v_y)
                msg.angular_velocity_z = float(self.v_yaw)
                msg.desired_pelvis_height = float(self.desired_height)
                self._publisher.publish(msg)
            except Exception:
                pass

    def set_velocity(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        desired_height: Optional[float] = None,
    ):
        """Sets the active commanded walking velocity."""
        self.v_x = float(linear_x)
        self.v_y = float(linear_y)
        self.v_yaw = float(angular_z)
        if desired_height is not None:
            self.desired_height = float(desired_height)
        self._publish_active = True
        self.publish_now()

    def stop(self):
        """Emergency stop: zero out commanded velocities."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_yaw = 0.0
        self._publish_active = True
        self.publish_now()

    def step(self, direction: str, delta_v: float = 0.2, delta_yaw: float = 0.2):
        """Applies directional incremental velocity steps."""
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
        elif direction == "stop":
            self.stop()
            return
        self._publish_active = True
        self.publish_now()

    def shutdown(self):
        """Destroys the ROS2 node and releases resources."""
        self.stop_streaming()
        self._is_active = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
            self._publisher = None

    @property
    def is_ros_connected(self) -> bool:
        return self._ros_available
