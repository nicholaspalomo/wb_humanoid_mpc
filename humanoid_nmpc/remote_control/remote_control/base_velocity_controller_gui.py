"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
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

"""The operator GUI: walking velocity, FSM mode and gantry, PD gains, joint targets, MPC parameters, dodgeball.

    bazel run //humanoid_nmpc/remote_control:base_velocity_controller_gui -- \\
        --task_file=robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml

App is the Tk window; OperatorGui runs it against the IPC bus (operator_bus.OperatorBus): every 25 Hz tick it
publishes the walking command (always, so that the height slider reaches the gantry in simulation), applies the FSM
states the robot published, and reads the Xbox controller when one is connected. Everything runs on Tk's thread
except the bus's own receive thread, which only fills the FSM state mailbox.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import tkinter as tk
from tkinter import ttk
from types import FrameType
from typing import Any, Callable, Dict, List, Optional

import robot_ipc
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2
from remote_control import config_files
from remote_control import fsm_state
from remote_control import teleop
from remote_control.operator_bus import (
    OPERATOR_NODE,
    OperatorBus,
    walking_velocity_command,
)
from remote_control.rerun_viewer import RerunViewerLauncher
from remote_control.tk_app import (
    JoystickGui,
    LEDIndicatorGui,
    JointPdGainsTab,
    JointTargetsTab,
    MpcParamsTab,
    CommandLimitsTab,
    DodgeballTab,
)

_LOGGER = logging.getLogger(__name__)


class App(tk.Tk):
    """The GUI window. It knows nothing of the bus: it is given a publisher per tuning topic (anything with
    publish(message), operator_bus.TopicPublisher in the GUI) and a callback for FSM commands, and OperatorGui feeds it
    the robot's FSM state.

    Args:
        pd_gains_file: joint_pd_gains.yaml of the Joint PD Gains and Joint Targets tabs.
        task_file: task.yaml of the MPC Parameters tab.
        reference_file: reference.yaml of the Joint Targets and Command Limits tabs.
        enable_online_tuning: whether the tuning tabs publish and save (the task file's enableOnlineTuning).
        param_publisher: the publisher of operator/mpc_parameters.
        pd_gains_publisher: the publisher of operator/pd_gains.
        joint_targets_publisher: the publisher of operator/joint_targets.
        dodgeball_publisher: the publisher of operator/dodgeball_throw.
        rerun_viewer: starts the Rerun bridge for the "Open Rerun viewer" button (None: the button is disabled).
        fsm_command_callback: sends an FSM command (a mode name, LOCK_GANTRY or UNLOCK_GANTRY).
    """

    def __init__(
        self,
        pd_gains_file: str = "",
        task_file: str = "",
        reference_file: str = "",
        enable_online_tuning: bool = True,
        param_publisher=None,
        pd_gains_publisher=None,
        joint_targets_publisher=None,
        dodgeball_publisher=None,
        rerun_viewer: Optional[RerunViewerLauncher] = None,
        fsm_command_callback: Optional[Callable[[str], object]] = None,
    ):
        super().__init__()
        self.title("Robot Base Controller & Tuning")
        self.pd_gains_file = pd_gains_file
        self.task_file = task_file
        self.reference_file = reference_file
        self.enable_online_tuning = enable_online_tuning
        self.rerun_viewer = rerun_viewer
        self._fsm_command_callback = fsm_command_callback
        self.fsm_mode_var = tk.StringVar(value="ZERO_TORQUE")
        # The last robot/fsm_state message, which update_fsm_state() compares the next one with.
        self._last_fsm_state: Optional[fsm_state.FsmState] = None
        self.param_publisher = param_publisher
        self.pd_gains_publisher = pd_gains_publisher
        self.dodgeball_publisher = dodgeball_publisher
        self.joint_targets_publisher = joint_targets_publisher

        # Position window on the left side of the screen, on top of the Rerun viewer
        gui_width = 960
        gui_height = 700
        pos_x = 30
        pos_y = 50
        self.geometry(f"{gui_width}x{gui_height}+{pos_x}+{pos_y}")
        self.minsize(800, 520)

        # Raise window on top of other windows (e.g. the Rerun viewer)
        self.lift()
        self.attributes("-topmost", True)
        self.after(1500, lambda: self.attributes("-topmost", False))

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

        # Ensure Combobox popdown list has proper theme colors and readable font
        self.option_add("*TCombobox*Listbox.background", "#2d2d2d")
        self.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        self.option_add("*TCombobox*Listbox.selectBackground", "#007acc")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.font", ("Helvetica", 10))

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
            param_publisher=self.pd_gains_publisher,
        )
        self.joint_pd_tab.pack(fill="both", expand=True)

        # Tab 3: MPC Parameters
        tab_mpc = ttk.Frame(self.notebook)
        self.notebook.add(tab_mpc, text="📈 MPC Parameters")
        self.mpc_params_tab = MpcParamsTab(
            tab_mpc,
            task_file=self.task_file,
            enable_online_tuning=self.enable_online_tuning,
            param_publisher=self.param_publisher,
        )
        self.mpc_params_tab.pack(fill="both", expand=True)

        # Tab 4: Joint Targets (for JOINT_PD mode)
        tab_targets = ttk.Frame(self.notebook)
        self.notebook.add(tab_targets, text="🎯 Joint Targets")
        self.joint_targets_tab = JointTargetsTab(
            tab_targets,
            pd_gains_file=self.pd_gains_file,
            reference_file=self.reference_file,
            fsm_mode_var=self.fsm_mode_var,
            param_publisher=self.joint_targets_publisher,
        )
        self.joint_targets_tab.pack(fill="both", expand=True)

        # Tab 5: Command Limits (reference.yaml). Not carried on the parameter topic: this tab saves the file, and the
        # parameter updater of either MPC node (makeCentroidalMpcParameterUpdater) watches it and reloads the command
        # limits of the running controller about a second later.
        tab_limits = ttk.Frame(self.notebook)
        self.notebook.add(tab_limits, text="🎚️ Command Limits")
        self.command_limits_tab = CommandLimitsTab(
            tab_limits, reference_file=self.reference_file
        )
        self.command_limits_tab.pack(fill="both", expand=True)

        # Tab 6: Dodgeball. A disturbance generator for push-recovery testing: the throw is computed here and applied
        # by the MuJoCo simulator, so the tab does nothing on hardware or against the dummy sim.
        tab_dodgeball = ttk.Frame(self.notebook)
        self.notebook.add(tab_dodgeball, text="🏐 Dodgeball")
        self.dodgeball_tab = DodgeballTab(
            tab_dodgeball, throw_publisher=self.dodgeball_publisher
        )
        self.dodgeball_tab.pack(fill="both", expand=True)

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

        self.gantry_touch_btn = ttk.Button(
            control_frame,
            text="👇 Touch Ground",
            command=self._on_gantry_touch,
        )
        self.gantry_touch_btn.pack(side="left", padx=5)

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

        # Starts the Rerun bridge, which opens the native Rerun viewer (rerun_viewer.py).
        self.rerun_viewer_btn = ttk.Button(
            control_frame,
            text="📊 Open Rerun viewer",
            command=self._open_rerun_viewer,
        )
        if self.rerun_viewer is None:
            self.rerun_viewer_btn.configure(state="disabled")
        self.rerun_viewer_btn.pack(side="left", padx=10)

        # --- Simulation row: measured contact state of the controller ---
        # The task file selects the contact estimator by name (contactEstimator, robot_model/ContactEstimatorRegistry.h);
        # the checkbox switches between the MuJoCo ground truth (cheater_sim) and every contact point touching
        # (always_in_contact). The MPC Parameters tab owns the selection and publishes it on the parameter topic.
        sim_frame = ttk.Frame(main_frame)
        sim_frame.grid(row=2, column=0, columnspan=5, pady=(8, 0))
        self.cheater_contacts_var = tk.BooleanVar(value=False)
        self.cheater_contacts_checkbox = ttk.Checkbutton(
            sim_frame,
            text="Cheater contact estimator",
            variable=self.cheater_contacts_var,
            command=self._on_cheater_contacts_toggle,
        )
        self.cheater_contacts_checkbox.pack(side="left", padx=10)
        self.mpc_params_tab.on_contact_estimator_changed = (
            self._sync_cheater_contacts_checkbox
        )
        self._sync_cheater_contacts_checkbox()

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def _open_rerun_viewer(self):
        """Starts the Rerun bridge, or says why it did not."""
        if self.rerun_viewer is None:
            return
        result = self.rerun_viewer.open()
        if not result.started:
            import tkinter.messagebox as mb

            mb.showinfo("Rerun viewer", result.message)

    def set_fsm_command_callback(self, callback: Optional[Callable[[str], object]]):
        """Sets what the FSM mode selector and the gantry checkbox call with their command."""
        self._fsm_command_callback = callback

    def slider_callback(self, value):
        pass

    def _on_fsm_change(self, event):
        """Handle FSM dropdown selection change."""
        mode = self.fsm_mode_var.get()
        if self._fsm_command_callback:
            self._fsm_command_callback(mode)
        # Notify Joint Targets tab of mode change
        if hasattr(self, "joint_targets_tab"):
            self.joint_targets_tab.on_mode_changed()

    def _on_cheater_contacts_toggle(self):
        """Selects the contact estimator through the MPC Parameters tab, which publishes it live."""
        self.mpc_params_tab.set_cheater_contact_estimator(
            self.cheater_contacts_var.get()
        )

    def _sync_cheater_contacts_checkbox(self):
        """Mirrors the tab's selection (file load, reset, save) and disables the checkbox when it cannot apply."""
        tab = self.mpc_params_tab
        self.cheater_contacts_var.set(tab.is_cheater_contact_estimator_selected())
        usable = tab.enable_online_tuning and tab.has_contact_estimator_selection()
        self.cheater_contacts_checkbox.configure(
            state="normal" if usable else "disabled"
        )

    def _on_gantry_toggle(self):
        """Handle gantry lock checkbox toggle."""
        # LINT.IfChange(gantry_commands)
        cmd = "LOCK_GANTRY" if self.gantry_var.get() else "UNLOCK_GANTRY"
        # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/FsmCommand.cpp:fsm_command_names)
        if self._fsm_command_callback:
            self._fsm_command_callback(cmd)
        if self.gantry_var.get():
            # Auto-center joysticks when re-locking the gantry so no residual
            # velocity command is applied while the robot is suspended.
            self.joystick_left.set_position()
            self.joystick_right.set_position()

    def _on_gantry_touch(self):
        """Smoothly lower the gantry height slider to the default pelvis height over 2 seconds."""
        steps = 50
        delay_ms = int(2000 / steps)
        start_val = self.slider.get()
        target_val = self.slider_default_value

        def step_slider(step=0):
            if step <= steps:
                alpha = step / steps
                current_val = start_val + alpha * (target_val - start_val)
                self.slider.set(current_val)
                self.after(delay_ms, lambda: step_slider(step + 1))

        step_slider(0)

    def update_fsm_state(self, message: fsm_state_pb2.FsmState):
        """Follows one robot/fsm_state message (fsm_state.py); a message whose mode is not a ControlMode is ignored.

        The mode selector and the gantry checkbox mirror the message. The joysticks are re-centered whenever
        fsm_state.should_recenter says so: on a transition into a passive mode, on a new gantry lock and on a controller
        reset - the simulator's fall recovery catching the robot, or the simulator putting it back in its initial state
        even while the gantry was already locked. A stick left forward would otherwise keep commanding a walk the
        operator never meant to give, and re-entering WB_MPC would execute it.
        """
        state = fsm_state.from_message(message)
        if state is None:
            return
        previous = self._last_fsm_state
        self._last_fsm_state = state
        if self.fsm_mode_var.get() != state.mode:
            self.fsm_mode_var.set(state.mode)
            # Notify Joint Targets tab of mode change
            if hasattr(self, "joint_targets_tab"):
                self.joint_targets_tab.on_mode_changed()
        if self.gantry_var.get() != state.gantry_locked:
            self.gantry_var.set(state.gantry_locked)
        if fsm_state.should_recenter(previous, state):
            self.joystick_left.set_position()
            self.joystick_right.set_position()

    def set_joystick_connected(self, is_connected):
        if (
            hasattr(self, "_last_joystick_connected")
            and self._last_joystick_connected == is_connected
        ):
            return
        self._last_joystick_connected = is_connected

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

    def set_knob_positions(
        self, msg: walking_velocity_command_pb2.WalkingVelocityCommand
    ):
        self.joystick_left.set_position(msg.linear_velocity_x, msg.linear_velocity_y)
        self.joystick_right.set_position(0.0, msg.angular_velocity_z)
        self.slider.set(
            (msg.desired_pelvis_height - self.min_height) / self.height_scale
        )

    def get_walking_command_msg(
        self,
    ) -> walking_velocity_command_pb2.WalkingVelocityCommand:
        """The command of the on-screen sticks and the height slider."""
        return walking_velocity_command(
            linear_velocity_x=self.joystick_left.x_norm,
            linear_velocity_y=self.joystick_left.y_norm,
            angular_velocity_z=self.joystick_right.y_norm,
            desired_pelvis_height=self.slider.get() * self.height_scale
            + self.min_height,
        )


class TerminationRequested(SystemExit):
    """SIGTERM or SIGHUP, raised on the main thread by the handlers OperatorGui.run() installs.

    A SystemExit because that is the one exception Tkinter lets out of a callback that runs when the signal arrives;
    run() catches it and ends the GUI as closing its window does.
    """


# tools/launch stops its processes with SIGTERM, and with SIGHUP when its terminal closes.
_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _request_termination(signum: int, frame: Optional[FrameType]) -> None:
    del frame  # Unused.
    raise TerminationRequested(128 + signum)


def _install_termination_handlers() -> Dict[int, Any]:
    """Makes SIGTERM and SIGHUP raise TerminationRequested; returns the handlers it replaced.

    Only on the main thread, the only one Python runs signal handlers on; elsewhere it installs nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {
        signum: signal.signal(signum, _request_termination)
        for signum in _TERMINATION_SIGNALS
    }


def _restore_signal_handlers(handlers: Dict[int, Any]) -> None:
    for signum, handler in handlers.items():
        signal.signal(signum, handler)


class OperatorGui:
    """Runs an App against the operator bus at the walking command's rate.

    Every tick, on Tk's thread: the FSM states the robot published since the last tick go to the App, then one walking
    command is published - the Xbox controller's while one is connected (the on-screen sticks follow it), the App's
    otherwise. The App's command is published even when it is all zero, so that the height slider reaches the gantry
    in simulation.

    Args:
        app: the window.
        operator_bus: the bus (not started; run() starts it).
        gamepad: the Xbox controller (xbox_controller_interface.GamepadPoller), or None.
        rate_hz: the tick rate [Hz].
        on_close: called after the window closes (stops the Rerun bridge the GUI started).
    """

    def __init__(
        self,
        app: App,
        operator_bus: OperatorBus,
        gamepad=None,
        rate_hz: float = teleop.WALKING_COMMAND_RATE_HZ,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"the rate must be positive, got {rate_hz}")
        self._app = app
        self._operator_bus = operator_bus
        self._gamepad = gamepad
        self._period_ms = max(1, int(round(1000.0 / rate_hz)))
        self._on_close = on_close
        app.set_fsm_command_callback(operator_bus.fsm_command.send)

    def tick(self) -> None:
        """One period: apply the received FSM states, then publish one walking command."""
        for message in self._operator_bus.take_fsm_states():
            self._app.update_fsm_state(message)

        command = None
        if self._gamepad is not None and self._gamepad.connected:
            command = self._gamepad.tick()
            if command is not None:
                self._app.set_knob_positions(command)
                self._app.set_joystick_connected(True)
        else:
            self._app.set_joystick_connected(False)
            command = self._app.get_walking_command_msg()
            if self._gamepad is not None:
                # Counts towards the next scan for a controller.
                self._gamepad.tick()
        if command is not None:
            self._operator_bus.walking_velocity_command.publish(command)

    def _run_tick(self) -> None:
        try:
            self.tick()
        except (
            Exception
        ):  # pylint: disable=broad-except - one bad tick must not stop the commands
            _LOGGER.exception("A GUI tick failed.")
        self._app.after(self._period_ms, self._run_tick)

    def run(self) -> None:
        """Starts the bus, runs the window until it closes, then closes the bus and calls on_close.

        Closing the window, Ctrl-C (SIGINT), SIGTERM and SIGHUP all end it this way. A GUI that a signal ended without
        this cleanup would leave the Rerun bridge it started running, in a session of its own that the launcher's
        teardown of the GUI's process group does not reach.
        """
        self._operator_bus.start()
        self._app.after(self._period_ms, self._run_tick)
        previous_handlers = _install_termination_handlers()
        try:
            self._app.mainloop()
        except (KeyboardInterrupt, TerminationRequested):
            pass
        finally:
            # A second signal during the cleanup ends the process as it would have before.
            _restore_signal_handlers(previous_handlers)
            self._operator_bus.close()
            if self._on_close is not None:
                self._on_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The operator GUI of the humanoid MPC, on the IPC bus."
    )
    parser.add_argument(
        "--task_file",
        default="",
        help="task.yaml of the MPC Parameters tab (default: found next to --reference_file, then a shipped robot's)",
    )
    parser.add_argument(
        "--reference_file",
        default="",
        help="reference.yaml: the default pelvis height and joint targets (default: next to the task file)",
    )
    parser.add_argument(
        "--pd_gains_file",
        default="",
        help="joint_pd_gains.yaml of the Joint PD Gains tab (default: next to the task file)",
    )
    parser.add_argument(
        "--urdf_file",
        default="",
        help="the robot's URDF, which the Rerun viewer button hands the Rerun bridge (default: none, no robot drawn)",
    )
    parser.add_argument(
        "--default_pelvis_height",
        type=float,
        default=None,
        help="the height slider's default [m] (default: reference.yaml's defaultBaseHeight)",
    )
    teleop.add_bus_flags(parser, default_node=OPERATOR_NODE)
    return parser


def _make_gamepad():
    """The Xbox controller's poller, or None when pygame cannot run here."""
    try:
        from remote_control.xbox_controller_interface import (  # pylint: disable=import-outside-toplevel
            GamepadPoller,
            XBoxControllerInterface,
        )

        controller = XBoxControllerInterface(teleop.WALKING_COMMAND_RATE_HZ)
    except (
        Exception
    ) as error:  # pylint: disable=broad-except - the GUI works without a controller
        _LOGGER.warning("No Xbox controller support: %s", error)
        return None
    return GamepadPoller(controller, teleop.WALKING_COMMAND_RATE_HZ)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    repo_root = config_files.find_repo_root()
    # Paths on the command line are relative to where the GUI was started (bazel run changes into the runfiles).
    network_config = config_files.resolve_input_path(args.network_config, repo_root)
    urdf_file = config_files.resolve_input_path(args.urdf_file, repo_root)
    files = config_files.resolve_config_files(
        task_file=config_files.resolve_input_path(args.task_file, repo_root),
        reference_file=config_files.resolve_input_path(args.reference_file, repo_root),
        pd_gains_file=config_files.resolve_input_path(args.pd_gains_file, repo_root),
        repo_root=repo_root,
    )
    if repo_root:
        # The tabs' robot presets are paths relative to the checkout, and "Save to YAML" must edit its files.
        os.chdir(repo_root)
    for label, path in (
        ("MPC task file", files.task_file),
        ("Joint PD gains file", files.pd_gains_file),
        ("reference file", files.reference_file),
    ):
        _LOGGER.info("Using %s: %s", label, path or "(none found)")
    enable_online_tuning = config_files.read_online_tuning(files.task_file)
    _LOGGER.info("Online tuning enabled: %s", enable_online_tuning)

    try:
        operator_bus = OperatorBus.connect(network_config, args.ipc_node)
    except (
        OSError,
        robot_ipc.NetworkConfigError,
        robot_ipc.BusError,
        ValueError,
    ) as error:
        print(f"base_velocity_controller_gui: {error}", file=sys.stderr)
        return 1
    rerun_viewer = (
        RerunViewerLauncher(repo_root, network_config, urdf_file=urdf_file)
        if repo_root
        else None
    )
    app = App(
        pd_gains_file=files.pd_gains_file,
        task_file=files.task_file,
        reference_file=files.reference_file,
        enable_online_tuning=enable_online_tuning,
        param_publisher=operator_bus.mpc_parameters,
        pd_gains_publisher=operator_bus.pd_gains,
        joint_targets_publisher=operator_bus.joint_targets,
        dodgeball_publisher=operator_bus.dodgeball_throw,
        rerun_viewer=rerun_viewer,
    )
    default_height = args.default_pelvis_height
    if default_height is None:
        default_height = config_files.read_default_pelvis_height(files.reference_file)
    app.set_default_pelvis_height(default_height)

    gui = OperatorGui(
        app,
        operator_bus,
        gamepad=_make_gamepad(),
        on_close=rerun_viewer.close if rerun_viewer is not None else None,
    )
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
