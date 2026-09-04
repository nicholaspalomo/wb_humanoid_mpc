"""****************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

import glob
import os
import shutil
import subprocess
import yaml
import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from rclpy.node import Node
from humanoid_mpc_msgs.msg import WalkingVelocityCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from remote_control import XBoxControllerInterface
from remote_control.tk_app import (
    JoystickGui,
    LEDIndicatorGui,
    JointPdGainsTab,
    MpcParamsTab,
)


class App(tk.Tk):
    def __init__(
        self,
        pd_gains_file: str = "",
        task_file: str = "",
        enable_online_tuning: bool = True,
        enable_telemetry: bool = True,
    ):
        super().__init__()
        self.title("Robot Base Controller & Tuning")
        self.pd_gains_file = pd_gains_file
        self.task_file = task_file
        self.enable_online_tuning = enable_online_tuning
        self.enable_telemetry = enable_telemetry
        self._fsm_command_callback = None

        # Position window on the right side of the screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        gui_width = 960
        gui_height = 700
        pos_x = max(0, screen_width - gui_width - 30)
        pos_y = 50
        self.geometry(f"{gui_width}x{gui_height}+{pos_x}+{pos_y}")
        self.minsize(800, 520)

        # Set window background color
        self.configure(bg="#1e1e1e")

        # Add padding around the window
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Style configuration
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Configure dark theme colors
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TNotebook", background="#181818", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#2d2d2d",
            foreground="#cccccc",
            padding=[16, 8],
            font=("Helvetica", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#007acc")],
            foreground=[("selected", "#ffffff")],
        )

        style.configure(
            "TLabel", background="#1e1e1e", foreground="#ffffff", font=("Helvetica", 11)
        )

        # Regular button style
        style.configure(
            "TButton",
            padding=6,
            background="#007acc",
            foreground="#ffffff",
            font=("Helvetica", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#005a9e")],
            foreground=[("active", "#ffffff")],
        )

        # Disabled button style
        style.configure(
            "Disabled.TButton",
            padding=6,
            background="#444444",
            foreground="#888888",
            font=("Helvetica", 10, "bold"),
        )

        # Modern checkbox style
        style.configure(
            "TCheckbutton",
            background="#1e1e1e",
            foreground="#ffffff",
            font=("Helvetica", 10),
        )

        # Modern scale (slider) style
        style.configure(
            "Vertical.TScale",
            background="#1e1e1e",
            troughcolor="#2d2d2d",
            bordercolor="#007acc",
        )

        # Top-level Notebook tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # Tab 1: Base Controller
        tab_base = ttk.Frame(self.notebook)
        self.notebook.add(tab_base, text="🕹️ Base Controller")

        # Tab 2: Joint PD Gains
        tab_pd = ttk.Frame(self.notebook)
        self.notebook.add(tab_pd, text="⚙️ Joint PD Gains")
        self.joint_pd_tab = JointPdGainsTab(
            tab_pd,
            pd_gains_file=self.pd_gains_file,
            enable_online_tuning=self.enable_online_tuning,
        )
        self.joint_pd_tab.pack(fill="both", expand=True)

        # Tab 3: MPC Parameters
        tab_mpc = ttk.Frame(self.notebook)
        self.notebook.add(tab_mpc, text="📈 MPC Parameters")
        self.mpc_params_tab = MpcParamsTab(
            tab_mpc,
            task_file=self.task_file,
            enable_online_tuning=self.enable_online_tuning,
        )
        self.mpc_params_tab.pack(fill="both", expand=True)

        # Build Tab 1: Base Controller contents
        self.auto_center_var = tk.BooleanVar(value=False)

        main_frame = ttk.Frame(tab_base)
        main_frame.pack(padx=15, pady=15, fill="both", expand=True)

        # Left Joystick (Linear Velocity)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=15, pady=15)

        left_label = ttk.Label(left_frame, text="Linear Velocity (LS)")
        left_label.pack()

        self.joystick_left = JoystickGui(
            left_frame, auto_center_var=self.auto_center_var, fix_y_axis=False
        )
        self.joystick_left.pack(pady=(5, 5))

        # Right Joystick (Angular Velocity Yaw)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=15, pady=15)

        right_label = ttk.Label(right_frame, text="Angular Velocity Yaw (RS)")
        right_label.pack()

        self.joystick_right = JoystickGui(
            right_frame, auto_center_var=self.auto_center_var, fix_y_axis=True
        )
        self.joystick_right.pack(pady=(5, 5))

        # Slider frame
        self.min_height = 0.2
        self.max_height = 1.3
        self.height_scale = (self.max_height - self.min_height) / 100.0
        self.slider_default_value = (0.8 - self.min_height) / self.height_scale
        self.slider_frame = ttk.Frame(main_frame)
        self.slider_frame.grid(row=0, column=2, padx=15, pady=10, sticky="ns")

        self.slider_label = ttk.Label(self.slider_frame, text="Root Height (LT + RT)")
        self.slider_label.pack(pady=(0, 5))

        self.slider = ttk.Scale(
            self.slider_frame,
            from_=100,
            to=0,
            orient="vertical",
            command=self.slider_callback,
        )
        self.slider.set(self.slider_default_value)
        self.slider.pack(expand=True, fill="y")
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))

        # --- FSM Mode Selector ---
        fsm_frame = ttk.Frame(control_frame)
        fsm_frame.pack(side="left", padx=5)

        ttk.Label(
            fsm_frame,
            text="FSM Mode:",
            font=("Helvetica", 9, "bold"),
        ).pack(side="left", padx=(0, 4))

        self.fsm_mode_var = tk.StringVar(value="ZERO_TORQUE")
        self.fsm_dropdown = ttk.Combobox(
            fsm_frame,
            textvariable=self.fsm_mode_var,
            values=["ZERO_TORQUE", "JOINT_PD", "GRAVITY_COMP", "WB_MPC", "SAFETY"],
            state="readonly",
            width=14,
        )
        self.fsm_dropdown.pack(side="left")
        self.fsm_dropdown.bind("<<ComboboxSelected>>", self._on_fsm_change)

        # --- Gantry Lock Toggle ---
        self.gantry_var = tk.BooleanVar(value=True)  # Locked by default
        self.gantry_toggle = ttk.Checkbutton(
            control_frame,
            text="Gantry Lock",
            variable=self.gantry_var,
            command=self._on_gantry_toggle,
        )
        self.gantry_toggle.pack(side="left", padx=10)

        # Separator
        ttk.Separator(control_frame, orient="vertical").pack(
            side="left", fill="y", padx=5
        )

        # Create LED
        self.joystick_connected_indicator = LEDIndicatorGui(
            control_frame, "Joystick Connection", size=30
        )
        self.joystick_connected_indicator.pack(side="left", padx=10)

        self.center_button = ttk.Button(
            control_frame, text="Center", command=self.center_all
        )
        self.center_button.pack(side="left", padx=10)

        self.auto_center_checkbox = ttk.Checkbutton(
            control_frame,
            text="Auto Center",
            variable=self.auto_center_var,
            command=self.auto_center_callback,
        )
        self.auto_center_checkbox.pack(side="left")

        # PlotJuggler launch button
        self.plotjuggler_btn = ttk.Button(
            control_frame,
            text="📊 PlotJuggler",
            command=self._launch_plotjuggler,
        )
        if not self.enable_telemetry:
            self.plotjuggler_btn.configure(
                state="disabled",
                text="📊 PlotJuggler (Disabled)",
            )
        self.plotjuggler_btn.pack(side="left", padx=10)

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def _launch_plotjuggler(self):
        """Launch PlotJuggler with the humanoid telemetry layout."""
        if not self.enable_telemetry:
            import tkinter.messagebox as mb

            mb.showinfo(
                "Telemetry Disabled",
                "Telemetry is disabled in task.yaml (enableTelemetry: false).",
            )
            return

        try:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../..")
            )
            layout_candidate = os.path.join(
                repo_root, "tools", "plotjuggler", "humanoid_telemetry.xml"
            )

            # Resolve PlotJuggler binary or fallback to ROS 2 package runner
            ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
            direct_bin = f"/opt/ros/{ros_distro}/lib/plotjuggler/plotjuggler"

            if shutil.which("plotjuggler"):
                cmd = ["plotjuggler"]
            elif os.path.isfile(direct_bin) and os.access(direct_bin, os.X_OK):
                cmd = [direct_bin]
            elif shutil.which("ros2"):
                cmd = ["ros2", "run", "plotjuggler", "plotjuggler"]
            else:
                alt_bins = glob.glob("/opt/ros/*/lib/plotjuggler/plotjuggler")
                if alt_bins and os.access(alt_bins[0], os.X_OK):
                    cmd = [alt_bins[0]]
                else:
                    cmd = ["plotjuggler"]

            cmd.extend(["--buffer_size", "60"])
            if os.path.exists(layout_candidate):
                cmd.extend(["--layout", layout_candidate])

            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            import tkinter.messagebox as mb

            mb.showwarning(
                "PlotJuggler",
                "plotjuggler executable was not found on PATH. Run 'make plotjuggler' from the terminal.",
            )
        except Exception as e:
            print(f"Error launching PlotJuggler: {e}")

    def slider_callback(self, value):
        pass

    def _on_fsm_change(self, event):
        """Handle FSM dropdown selection change."""
        mode = self.fsm_mode_var.get()
        if self._fsm_command_callback:
            self._fsm_command_callback(mode)

    def _on_gantry_toggle(self):
        """Handle gantry lock checkbox toggle."""
        cmd = "LOCK_GANTRY" if self.gantry_var.get() else "UNLOCK_GANTRY"
        if self._fsm_command_callback:
            self._fsm_command_callback(cmd)

    def update_fsm_state(self, state_str: str):
        """Update GUI from ROS 2 state message: 'MODE,GANTRY_STATE'."""
        try:
            parts = [p.strip() for p in state_str.split(",")]
            if len(parts) >= 1:
                fsm_state = parts[0]
                valid_modes = (
                    "ZERO_TORQUE",
                    "JOINT_PD",
                    "GRAVITY_COMP",
                    "WB_MPC",
                    "SAFETY",
                )
                if fsm_state in valid_modes:
                    self.fsm_mode_var.set(fsm_state)
            if len(parts) >= 2:
                gantry_state = parts[1]
                self.gantry_var.set(gantry_state == "GANTRY_LOCKED")
        except Exception:
            pass

    def set_joystick_connected(self, is_connected):
        self.joystick_connected_indicator.set_state(is_connected)
        if is_connected:
            self.center_button.configure(state="disabled")
            self.auto_center_checkbox.configure(state="disabled")
            self.center_button["style"] = "Disabled.TButton"

        else:
            self.center_button.configure(state="normal")
            self.auto_center_checkbox.configure(state="normal")
            self.center_button["style"] = "TButton"

    def auto_center_callback(self):
        if self.auto_center_var.get():
            self.center_all()

    def on_slider_release(self, event):
        if self.auto_center_var.get():
            self.slider.set(self.slider_default_value)

    def center_all(self):
        self.joystick_left.set_position()
        self.joystick_right.set_position()
        self.slider.set(self.slider_default_value)

    def set_default_pelvis_height(self, height: float):
        self.max_height = max(1.3, height + 0.3)
        self.height_scale = (self.max_height - self.min_height) / 100.0
        self.slider_default_value = (height - self.min_height) / self.height_scale
        self.slider.set(self.slider_default_value)

    def set_knob_positions(self, msg: WalkingVelocityCommand):
        self.joystick_left.set_position(msg.linear_velocity_x, msg.linear_velocity_y)
        self.joystick_right.set_position(0.0, msg.angular_velocity_z)
        self.slider.set(
            (msg.desired_pelvis_height - self.min_height) / self.height_scale
        )

    def get_walking_command_msg(self):
        msg = WalkingVelocityCommand()

        msg.linear_velocity_x = self.joystick_left.x_norm
        msg.linear_velocity_y = self.joystick_left.y_norm
        msg.angular_velocity_z = self.joystick_right.y_norm

        msg.desired_pelvis_height = (
            self.slider.get() * self.height_scale + self.min_height
        )
        return msg


class RosJoystickApp(Node):
    def __init__(self):
        super().__init__("xbox_walking_command_publisher")

        self.publisher_rate = 25  # Hz
        self.xbox_controller_interface = XBoxControllerInterface(self.publisher_rate)

        # Declare parameters for robot-specific configuration
        self.declare_parameter("default_pelvis_height", 0.8)
        self.declare_parameter("target_command_file", "")
        self.declare_parameter("task_file", "")
        self.declare_parameter("pd_gains_file", "")
        self.declare_parameter("robot_name", "")

        default_height = float(self.get_parameter("default_pelvis_height").value)
        cmd_file = str(self.get_parameter("target_command_file").value)
        task_file = str(self.get_parameter("task_file").value)
        pd_gains_file = str(self.get_parameter("pd_gains_file").value)
        robot_name = str(self.get_parameter("robot_name").value)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

        # 1. Load default base height from target_command_file if provided
        if cmd_file and os.path.exists(cmd_file):
            try:
                with open(cmd_file, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "defaultBaseHeight" in data:
                        default_height = float(data["defaultBaseHeight"])
                        self.get_logger().info(
                            f"Loaded defaultBaseHeight={default_height} from {cmd_file}"
                        )
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to read defaultBaseHeight from {cmd_file}: {e}"
                )

        # 2. Autodetect task_file if empty
        if not task_file or not os.path.exists(task_file):
            if cmd_file and os.path.exists(cmd_file):
                cand = os.path.join(os.path.dirname(cmd_file), "task.yaml")
                if os.path.exists(cand):
                    task_file = cand
            if not task_file or not os.path.exists(task_file):
                # Search default candidates
                candidates = [
                    os.path.join(
                        repo_root,
                        "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
                    ),
                    os.path.join(
                        repo_root,
                        "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
                    ),
                    os.path.join(
                        repo_root,
                        "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
                    ),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        task_file = c
                        break

        # 3. Autodetect pd_gains_file if empty
        if not pd_gains_file or not os.path.exists(pd_gains_file):
            if task_file and os.path.exists(task_file):
                cand = os.path.abspath(
                    os.path.join(
                        os.path.dirname(task_file),
                        "..",
                        "controller",
                        "joint_pd_gains.yaml",
                    )
                )
                if os.path.exists(cand):
                    pd_gains_file = cand
            if not pd_gains_file or not os.path.exists(pd_gains_file):
                candidates = [
                    os.path.join(
                        repo_root,
                        "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.yaml",
                    ),
                    os.path.join(
                        repo_root,
                        "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.yaml",
                    ),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        pd_gains_file = c
                        break

        if task_file:
            self.get_logger().info(f"Using MPC task file: {task_file}")
        if pd_gains_file:
            self.get_logger().info(f"Using Joint PD gains file: {pd_gains_file}")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=25)

        self.publisher_ = self.create_publisher(
            WalkingVelocityCommand, "/humanoid/walking_velocity_command", qos_profile
        )

        # FSM command publisher & state subscriber
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.fsm_cmd_pub = self.create_publisher(
            String, "/humanoid/fsm_command", cmd_qos
        )

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.fsm_state_sub = self.create_subscription(
            String, "/humanoid/fsm_state", self._fsm_state_callback, state_qos
        )

        enable_online_tuning = True
        enable_telemetry = True
        if task_file and os.path.exists(task_file):
            try:
                from remote_control.tk_app.yaml_editor_utils import load_yaml_safe

                cfg = load_yaml_safe(task_file)
                if "enableOnlineTuning" in cfg:
                    enable_online_tuning = bool(cfg["enableOnlineTuning"])
                elif "enable_online_tuning" in cfg:
                    enable_online_tuning = bool(cfg["enable_online_tuning"])

                if "enableTelemetry" in cfg:
                    enable_telemetry = bool(cfg["enableTelemetry"])
                elif "enable_telemetry" in cfg:
                    enable_telemetry = bool(cfg["enable_telemetry"])
            except Exception as e:
                self.get_logger().warning(
                    f"Could not parse online tuning/telemetry flags from {task_file}: {e}"
                )

        self.get_logger().info(
            f"Online tuning enabled: {enable_online_tuning}, Telemetry enabled: {enable_telemetry}"
        )

        self.app = App(
            pd_gains_file=pd_gains_file,
            task_file=task_file,
            enable_online_tuning=enable_online_tuning,
            enable_telemetry=enable_telemetry,
        )
        self.app.set_default_pelvis_height(default_height)
        self.app._fsm_command_callback = self._send_fsm_command

        self.timer = self.create_timer(1 / self.publisher_rate, self.timer_callback)

        self.ros_thread = threading.Thread(target=self.ros_spin)
        self.ros_thread.daemon = True
        self.ros_thread.start()

        self.counter = 0
        self._gui_active = False

    def _send_fsm_command(self, cmd_text: str):
        msg = String()
        msg.data = cmd_text
        self.fsm_cmd_pub.publish(msg)

    def _fsm_state_callback(self, msg: String):
        self.app.after(0, self.app.update_fsm_state, msg.data)

    def timer_callback(self):

        if self.xbox_controller_interface.joystick_connected:
            success, msg = self.xbox_controller_interface.get_walking_command_msg()
            if success:
                self.app.set_knob_positions(msg)
                self.publisher_.publish(msg)
                self.app.set_joystick_connected(True)

        else:
            self.app.set_joystick_connected(False)
            cmd_msg = self.app.get_walking_command_msg()
            # Always publish so the height slider value is transmitted for gantry control,
            # even when all velocity components are zero.
            self.publisher_.publish(cmd_msg)
            # check for connection every 2 seconds
            if self.counter >= (2 * self.publisher_rate):
                self.xbox_controller_interface.get_joystick_connection()
                self.counter = 0
            self.counter = self.counter + 1

    def ros_spin(self):
        rclpy.spin(self)

    def run(self):
        try:
            self.app.mainloop()
        finally:
            self.destroy_node()
            rclpy.shutdown()


def main():
    rclpy.init()
    app = RosJoystickApp()
    app.run()


if __name__ == "__main__":
    main()
