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

import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from rclpy.node import Node
from humanoid_mpc_msgs.msg import WalkingVelocityCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from remote_control import XBoxControllerInterface
from remote_control.tk_app import JoystickGui, LEDIndicatorGui


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Robot Base Controller")

        # Position window on the right side of the screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        gui_width = 850
        gui_height = 420
        pos_x = max(0, screen_width - gui_width - 30)
        pos_y = 50
        self.geometry(f"{gui_width}x{gui_height}+{pos_x}+{pos_y}")
        self.minsize(750, 380)

        # Set window background color
        self.configure(bg="#2c2c2c")

        # Add padding around the window
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Style configuration
        # Configure modern dark theme styles
        style = ttk.Style()

        # Configure dark theme colors
        style.configure("TFrame", background="#2c2c2c")
        style.configure(
            "TLabel", background="#2c2c2c", foreground="#ffffff", font=("Helvetica", 12)
        )

        # Regular button style
        style.configure(
            "TButton",
            padding=8,
            background="#4a90e2",
            foreground="#ffffff",
            font=("Helvetica", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#357abd")],
            foreground=[("active", "#ffffff")],
        )

        # Disabled button style
        style.configure(
            "Disabled.TButton",
            padding=8,
            background="#cccccc",
            foreground="#888888",
            font=("Helvetica", 10, "bold"),
        )

        # Modern checkbox style
        style.configure(
            "TCheckbutton",
            background="#2c2c2c",
            foreground="#ffffff",
            font=("Helvetica", 10),
        )

        # Modern scale (slider) style
        style.configure(
            "Vertical.TScale",
            background="#2c2c2c",
            troughcolor="#363636",
            bordercolor="#4a90e2",
        )

        self.auto_center_var = tk.BooleanVar(value=False)

        # Main container frame
        main_frame = ttk.Frame(self)
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
        self.slider_default_value = 75
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

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

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

    def set_knob_positions(self, msg: WalkingVelocityCommand):
        self.joystick_left.set_position(msg.linear_velocity_x, msg.linear_velocity_y)
        self.joystick_right.set_position(0.0, msg.angular_velocity_z)
        self.slider.set((msg.desired_pelvis_height - 0.2) / 0.008)

    def get_walking_command_msg(self):
        msg = WalkingVelocityCommand()

        msg.linear_velocity_x = self.joystick_left.x_norm
        msg.linear_velocity_y = self.joystick_left.y_norm
        msg.angular_velocity_z = self.joystick_right.y_norm

        msg.desired_pelvis_height = self.slider.get() * 0.008 + 0.2
        return msg


class RosJoystickApp(Node):
    def __init__(self):
        super().__init__("xbox_walking_command_publisher")

        self.publisher_rate = 25  # Hz
        self.xbox_controller_interface = XBoxControllerInterface(self.publisher_rate)

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

        self.app = App()
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
