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

"""The Joint PD Gains tab: a row per gain of the joint PD gains file, a global scale, search, published and saved.

The rows are the parameters of the file's schema (humanoid_mpc_config/joint_pd_gains_file.proto, through
config_schema): the default gains, and each joint's, grouped by limb. Every change is published, debounced, as the
whole file (humanoid_mpc_config.JointPdGainsFile on operator/pd_gains) after a strict parse of the edited text; "Save"
writes the changes into the file and keeps every other byte of it, then sends the saved text to the robot's store
(robot_config_save.py), whose copy the robot's controllers read.
"""

from collections.abc import Callable
import functools
import os
import re
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
from typing import Any

from humanoid_mpc_config import joint_pd_gains_file_pb2

from remote_control import config_schema
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import tuned_file
from remote_control.tk_app import combobox
from remote_control.tk_app import parameter_rows
from remote_control.tk_app import robot_save_status
from remote_control.tk_app import scrollable_frame

# The gains the global scale buttons scale, by their field names in the schema's Gains and JointGains (a test checks
# that both messages have them).
KP_FIELD = "kp"
KD_FIELD = "kd"
SCALE_FACTORS = (0.001, 0.01, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)

# The limb sections the joints' rows are grouped in, in the order they are shown.
SECTION_ORDER = (
    "Torso & Spine",
    "Left Leg",
    "Right Leg",
    "Left Arm",
    "Right Arm",
    "Head & Neck",
    "Other Joints",
)
DEFAULT_GAINS_SECTION = "⚙ Default Gains"

_PUBLISH_DEBOUNCE_MS = 300
# How long a passing status stays on the status line [ms]; a Save's stays until the next status.
_STATUS_DISPLAY_MS = 5000


def classify_joint_section(joint_name: str) -> str:
    """The limb section a joint's rows are grouped under, read from its name."""
    name_lower = joint_name.lower()
    if re.search(r"waist|spine|torso|back", name_lower):
        return "Torso & Spine"
    if re.search(r"left.*leg|l_leg|left.*hip|left.*knee|left.*ankle", name_lower):
        return "Left Leg"
    if re.search(r"right.*leg|r_leg|right.*hip|right.*knee|right.*ankle", name_lower):
        return "Right Leg"
    if re.search(r"left.*arm|l_arm|left.*shoulder|left.*elbow|left.*wrist", name_lower):
        return "Left Arm"
    if re.search(
        r"right.*arm|r_arm|right.*shoulder|right.*elbow|right.*wrist", name_lower
    ):
        return "Right Arm"
    if re.search(r"neck|head", name_lower):
        return "Head & Neck"
    return "Other Joints"


def section_of(spec: config_schema.ParameterSpec) -> str:
    """The section of a gain's row: the default gains, or the limb of the joint whose element holds it."""
    if spec.row is None:
        return DEFAULT_GAINS_SECTION
    return classify_joint_section(spec.row)


def _disabled_banner(reason: str) -> str:
    """The banner of a tab whose online tuning is off, for `reason` (empty: the task file's enable_online_tuning)."""
    return f"🔒 Online Tuning Disabled ({reason or 'enable_online_tuning: false in the task file'})"


class JointPdGainsTab(ttk.Frame):
    """The Joint PD Gains tab: a row per gain of the file, published live and saved to it.

    Args:
        parent: the notebook the tab is added to.
        pd_gains_file: the joint PD gains file to load (one that does not exist is reported, and nothing is loaded);
            None: the first preset, when it exists.
        on_gains_updated: called with the file's path after every save; None: nothing is called.
        enable_online_tuning: whether the rows, the scale buttons and saving are enabled.
        param_publisher: the publisher of operator/pd_gains; None: nothing is published.
        robot_saver: the GUI's saver of the robot's copies (robot_config_save.RobotConfigSaver); None: Save writes
            the laptop's file only, and says so.
        **kwargs: the options of the tab's ttk.Frame.
    """

    KNOWN_PRESETS = {
        "Unitree G1 (WB)": "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto",
        "DRC Atlas (Centroidal)": "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto",
        "Unitree R1 (Centroidal)": "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
        "Unitree G1 (Centroidal)": "robot_models/unitree_g1/g1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
        "EngineAI SA01 (Centroidal)": "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    }

    def __init__(
        self,
        parent: tk.Misc,
        pd_gains_file: str | None = None,
        on_gains_updated: Callable[[str], None] | None = None,
        enable_online_tuning: bool = True,
        param_publisher: operator_bus.TopicPublisher | None = None,
        robot_saver: robot_config_save.RobotConfigSaver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")

        self.pd_gains_file = pd_gains_file
        self.on_gains_updated = on_gains_updated
        self.enable_online_tuning = enable_online_tuning
        # The publisher of operator/pd_gains (operator_bus.TopicPublisher): publish(JointPdGainsFile).
        self.param_publisher = param_publisher

        # The file, as loaded or last saved, with the operator's changes.
        self.gains: tuned_file.TunedFile | None = None
        # The rows, by parameter path ("default_gains.kp", "joint_gains[joint=l_leg_kny].kd").
        self.slider_rows: dict[str, parameter_rows.ParameterRow] = {}
        self.section_frames: dict[str, ttk.LabelFrame] = {}
        self.scale_buttons: list[ttk.Button] = []
        # tkinter after() ID for debounced publish
        self._debounce_publish_id: str | None = None
        # The tkinter after() ID that clears a passing status; None when none is pending.
        self._status_clear_id: str | None = None
        # The robot's copy of the file after a Save.
        self.robot_save = robot_save_status.RobotSaveStatus(
            self, robot_saver, self._show_save_status
        )

        self._build_header_ui()
        self._build_global_scale_ui()
        self._build_search_ui()

        # Scrollable container for joint gains
        self.scroll_container = scrollable_frame.ScrollableFrame(
            self, bg_color="#2c2c2c"
        )
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # A file given that is not there is reported, never replaced by another robot's: the fallback is for none.
        if self.pd_gains_file:
            self.load_file(self.pd_gains_file)
        else:
            default_path = list(self.KNOWN_PRESETS.values())[0]
            if os.path.exists(default_path):
                self.load_file(default_path)

    def _build_header_ui(self) -> None:
        """The toolbar (preset, file, reload, reset, save), the status line and the online tuning banner."""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Label(toolbar, text="Robot Preset:", font=("Helvetica", 9, "bold")).pack(
            side="left", padx=(0, 4)
        )
        self.preset_var = tk.StringVar(value="Select Preset...")
        preset_cb = ttk.Combobox(
            toolbar,
            textvariable=self.preset_var,
            values=list(self.KNOWN_PRESETS.keys()),
            state="readonly",
            width=22,
        )
        preset_cb.pack(side="left", padx=(0, 10))
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)
        preset_cb.bind("<Button-1>", combobox.open_dropdown_on_click)

        self.path_var = tk.StringVar(value=self.pd_gains_file or "")
        path_entry = ttk.Entry(toolbar, textvariable=self.path_var, width=32)
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        browse_btn = ttk.Button(toolbar, text="Browse...", command=self._browse_file)
        browse_btn.pack(side="left", padx=2)

        reload_btn = ttk.Button(toolbar, text="⟳ Reload", command=self.reload_file)
        reload_btn.pack(side="left", padx=2)

        self.reset_btn = ttk.Button(
            toolbar, text="↺ Reset All", command=self.reset_all_defaults
        )
        self.reset_btn.pack(side="left", padx=2)

        self.save_btn = ttk.Button(toolbar, text="💾 Save", command=self.save)
        self.save_btn.pack(side="left", padx=(4, 0))

        self.status_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 9, "italic"),
            foreground="#27ae60",
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 2))

        self.warning_banner = ttk.Label(
            self,
            text=_disabled_banner(""),
            font=("Helvetica", 9, "bold"),
            foreground="#e67e22",
        )
        if not self.enable_online_tuning:
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
            self.save_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")

    def _build_global_scale_ui(self) -> None:
        """The buttons that scale every joint's Kp or Kd by one factor."""
        scale_frame = ttk.LabelFrame(self, text="⚡ Global Quick Scaling (All Joints)")
        scale_frame.pack(fill="x", padx=10, pady=(0, 6))
        btn_state = "normal" if self.enable_online_tuning else "disabled"
        for field_name, title in ((KP_FIELD, "Scale Kp:"), (KD_FIELD, "Scale Kd:")):
            row = ttk.Frame(scale_frame)
            row.pack(fill="x", padx=6, pady=3)
            ttk.Label(row, text=title, font=("Helvetica", 9, "bold"), width=10).pack(
                side="left"
            )
            for factor in SCALE_FACTORS:
                btn = ttk.Button(
                    row,
                    text=f"{factor}x",
                    width=6,
                    command=functools.partial(self._scale_all, field_name, factor),
                    state=btn_state,
                )
                btn.pack(side="left", padx=2)
                self.scale_buttons.append(btn)

    def _build_search_ui(self) -> None:
        """The filter box: only the rows whose path contains its text are shown."""
        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=10, pady=(0, 6))

        ttk.Label(search_frame, text="🔍 Filter:", font=("Helvetica", 9, "bold")).pack(
            side="left", padx=(0, 4)
        )
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_filter)
        search_entry = ttk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=("Helvetica", 9),
        )
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        clear_btn = ttk.Button(
            search_frame, text="Clear", width=6, command=lambda: self.search_var.set("")
        )
        clear_btn.pack(side="left")

    def _on_search_filter(self, *args: object) -> None:
        del args  # Unused: the arguments of Tk's variable trace.
        query = self.search_var.get().strip().lower()
        for key, row in self.slider_rows.items():
            if not query or query in key.lower():
                row.pack(fill="x", padx=4, pady=1)
            else:
                row.pack_forget()

    def _on_preset_selected(self, event: tk.Event | None = None) -> None:
        del event  # Unused.
        preset_name = self.preset_var.get()
        if preset_name in self.KNOWN_PRESETS:
            path = self.KNOWN_PRESETS[preset_name]
            if os.path.exists(path):
                self.load_file(path)
            else:
                self._show_status(f"Preset path not found: {path}", error=True)

    def _browse_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select joint_pd_gains.textproto",
            filetypes=[("Textproto files", "*.textproto"), ("All files", "*.*")],
        )
        if selected:
            self.load_file(selected)

    def load_file(self, file_path: str) -> None:
        """Loads a joint PD gains file and renders its rows; one that does not parse is reported and shows none."""
        self.pd_gains_file = os.path.abspath(file_path)
        self.path_var.set(self.pd_gains_file)
        try:
            self.gains = tuned_file.TunedFile(
                self.pd_gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
            )
        except (OSError, tuned_file.TunedFileError) as error:
            self.gains = None
            self._populate_rows()
            self._show_status(f"Cannot load {self.pd_gains_file}: {error}", error=True)
            return
        self._populate_rows()
        self._show_status(f"Loaded: {os.path.basename(self.pd_gains_file)}")

    def reload_file(self) -> None:
        """Reads the file again, dropping every change, and publishes it, so that the robot drops the published gains."""
        if self.pd_gains_file and os.path.exists(self.pd_gains_file):
            self.load_file(self.pd_gains_file)
            if self.gains is not None:
                self._publish_to_topic()

    def _populate_rows(self) -> None:
        """Rebuilds the rows from the loaded file: the default gains, then each joint's under its limb."""
        for widget in self.scroll_container.scrollable_content.winfo_children():
            widget.destroy()
        self.slider_rows.clear()
        self.section_frames.clear()
        if self.gains is None:
            return

        for section in (DEFAULT_GAINS_SECTION, *SECTION_ORDER):
            self.section_frames[section] = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text=section if section == DEFAULT_GAINS_SECTION else f"• {section}",
            )
        for spec in self.gains.rendered():
            row = parameter_rows.make_row(
                self.section_frames[section_of(spec)],
                spec,
                config_schema.display_label(spec, self.gains.message),
                self.gains.value(spec.path),
                on_change=functools.partial(self._on_row_change, spec.path),
                label_width=34,
            )
            if row is None:
                continue
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[spec.path] = row
        for frame in self.section_frames.values():
            if frame.winfo_children():
                frame.pack(fill="x", padx=6, pady=4)

        if not self.enable_online_tuning:
            for row in self.slider_rows.values():
                row.set_state("disabled")

    def set_online_tuning_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enables or disables online tuning; `reason` is why it is off, for the banner (empty: the task file says so)."""
        self.enable_online_tuning = bool(enabled)
        state = "normal" if self.enable_online_tuning else "disabled"
        if not self.enable_online_tuning:
            self.warning_banner.configure(text=_disabled_banner(reason))
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
        else:
            self.warning_banner.pack_forget()
        self.save_btn.configure(state=state)
        self.reset_btn.configure(state=state)
        for btn in self.scale_buttons:
            btn.configure(state=state)
        for row in self.slider_rows.values():
            row.set_state(state)

    def _scale_all(self, field_name: str, factor: float) -> None:
        """Scales the gain `field_name` of the defaults and of every joint by `factor`, from its saved value.

        A joint that leaves the gain out inherits default_gains' (the schema's rule), so it is not given one of its own:
        it follows the scaled default.

        Args:
            field_name: KP_FIELD or KD_FIELD.
            factor: The factor, one of SCALE_FACTORS.
        """
        if not self.enable_online_tuning or self.gains is None:
            return
        for path, row in self.slider_rows.items():
            spec = self.gains.spec(path)
            if (
                spec.field.name != field_name
                or spec.value is None
                or not isinstance(row.default_value, float)
            ):
                continue
            row.set_value(row.default_value * factor)
        self._show_status(f"Scaled all {field_name.upper()} gains by {factor}x")

    def reset_all_defaults(self) -> None:
        """Returns every gain to the file as loaded or last saved, and publishes it."""
        if not self.enable_online_tuning or self.gains is None:
            return
        self.gains.reset()
        for row in self.slider_rows.values():
            callback, row.on_change = row.on_change, None
            row.reset_to_default()
            row.on_change = callback
        self._schedule_publish()
        self._show_status("All gains reset to the saved file")

    def _on_row_change(self, path: str, label: str, value: Any) -> None:
        """A row's on_change, bound to its parameter path; the label it reports is for display only."""
        del label  # Unused: for display only.
        if self.gains is None:
            return
        try:
            self.gains.set(path, value)
        except (KeyError, ValueError) as error:
            self._show_status(f"Cannot change {path}: {error}", error=True)
            return
        self._schedule_publish()

    def _schedule_publish(self) -> None:
        """Debounces a publish of the file on operator/pd_gains; the robot's controllers read it."""
        if self._debounce_publish_id is not None:
            self.after_cancel(self._debounce_publish_id)
        self._debounce_publish_id = self.after(
            _PUBLISH_DEBOUNCE_MS, self._publish_to_topic
        )

    def build_gains(self) -> Any:
        """The JointPdGainsFile with the operator's changes, parsed strictly from the edited text; None without a file.

        Raises:
            tuned_file.TunedFileError: the edited text would not parse into its edited message.
        """
        return None if self.gains is None else self.gains.edited().message

    def _publish_to_topic(self) -> None:
        """Publishes the file with the current gains, as a JointPdGainsFile on operator/pd_gains."""
        self._debounce_publish_id = None
        if not self.enable_online_tuning or not self.param_publisher:
            return
        try:
            gains = self.build_gains()
            if gains is not None:
                self.param_publisher.publish(gains)
        except tuned_file.TunedFileError as error:
            self._show_status(f"Not published: {error}", error=True)
        # pylint: disable-next=broad-exception-caught  # Shown to the operator: a Tk callback must not raise.
        except Exception as error:
            self._show_status(f"Error publishing to topic: {error}", error=True)

    def save(self) -> bool:
        """Writes every change into the file on the laptop, then sends the saved file to the robot.

        The laptop's file keeps its first version as .bak, and the saved values become the defaults. Then its exact
        text goes to the robot's store, also when nothing changed (it re-synchronizes a robot that refused an earlier
        save or missed an editor's); the status line follows the robot's answer. A laptop save that fails sends nothing.

        Returns:
            Whether the laptop's file was saved.
        """
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return False
        if self.gains is None or not self.pd_gains_file:
            self._show_status("No file loaded to save.", error=True)
            return False
        try:
            text = self.gains.save().text
        except (OSError, tuned_file.TunedFileError) as error:
            self._show_status(f"Error saving: {error}", error=True)
            return False
        for row in self.slider_rows.values():
            row.mark_saved()
        self.robot_save.send(
            robot_config_save.KIND_JOINT_PD_GAINS, self.gains.path, text
        )
        if self.on_gains_updated:
            self.on_gains_updated(self.pd_gains_file)
        return True

    def _show_save_status(self, msg: str, error: bool) -> None:
        """Shows the outcome of a Save, which stays until the next status."""
        self._show_status(msg, error=error, transient=False)

    def _show_status(
        self, msg: str, error: bool = False, transient: bool = True
    ) -> None:
        """Shows `msg` on the status line, in red for an error; a transient one is cleared after _STATUS_DISPLAY_MS."""
        if self._status_clear_id is not None:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        if transient:
            self._status_clear_id = self.after(_STATUS_DISPLAY_MS, self._clear_status)

    def _clear_status(self) -> None:
        self._status_clear_id = None
        self.status_label.configure(text="")
