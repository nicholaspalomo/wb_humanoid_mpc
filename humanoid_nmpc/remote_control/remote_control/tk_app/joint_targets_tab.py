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

"""The Joint Targets tab: per-joint position sliders that publish operator/joint_targets in JOINT_PD mode."""

import math
import re
from tkinter import ttk
import tkinter as tk
from typing import Any

from humanoid_mpc_ipc import topics
from remote_control import config_files
from remote_control import operator_bus
from remote_control.tk_app import scrollable_frame
from remote_control.tk_app import slider_row


def _classify_joint_section(joint_name: str) -> str:
    """Classify a joint name into a limb section for grouping."""
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


class JointTargetsTab(ttk.Frame):
    """Joint target position tuning tab for JOINT_PD mode.

    Provides per-joint position sliders (in radians) that publish a JointTargets
    message on operator/joint_targets. The robot merges these into the nominal
    position vector used by the JOINT_PD controller.

    Sliders are enabled only when the FSM mode is ``JOINT_PD``.
    Default values are the reference file's ``default_joint_state``, by joint name.

    Args:
        parent: the notebook the tab is added to.
        pd_gains_file: the joint PD gains file whose joint_gains name the joints; None: no joints.
        reference_file: the reference file whose default_joint_state holds the defaults; None: every default is 0.
        fsm_mode_var: the GUI's FSM mode; the sliders are enabled while it is JOINT_PD. None: never.
        param_publisher: the publisher of operator/joint_targets; None: nothing is published.
        **kwargs: the options of the tab's ttk.Frame.
    """

    # The topic the tab's publisher publishes (operator_bus.OperatorBus.joint_targets).
    # LINT.IfChange(joint_target_topic_name)
    TOPIC_NAME = topics.OPERATOR_JOINT_TARGETS
    # LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/python/humanoid_mpc_ipc/topics.py:topics)

    def __init__(
        self,
        parent: tk.Misc,
        pd_gains_file: str | None = None,
        reference_file: str | None = None,
        fsm_mode_var: tk.StringVar | None = None,
        param_publisher: operator_bus.TopicPublisher | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")

        self.pd_gains_file = pd_gains_file if pd_gains_file else None
        self.reference_file = reference_file if reference_file else None
        self.fsm_mode_var = fsm_mode_var
        # The publisher of operator/joint_targets (operator_bus.TopicPublisher): publish(JointTargets).
        self.param_publisher = param_publisher

        self.joint_names: list[str] = []
        self.default_positions: dict[str, float] = {}
        self.slider_rows: dict[str, slider_row.SliderRow] = {}
        self.section_frames: dict[str, ttk.LabelFrame] = {}
        self._debounce_publish_id: str | None = None

        self._build_header_ui()
        self._build_search_ui()

        # Scrollable container for joint sliders
        self.scroll_container = scrollable_frame.ScrollableFrame(
            self, bg_color="#2c2c2c"
        )
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Load joint names from PD gains file and defaults from reference
        self._load_joint_data()
        self._populate_sliders()

        # Initial mode check
        self._update_enabled_state()

    def _build_header_ui(self) -> None:
        """The title, the reset button, the status line and the banner shown outside JOINT_PD."""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Label(
            toolbar,
            text="🎯 Joint Target Positions (JOINT_PD Mode)",
            font=("Helvetica", 11, "bold"),
        ).pack(side="left", padx=(0, 10))

        self.reset_btn = ttk.Button(
            toolbar, text="↺ Reset All", command=self.reset_all_defaults
        )
        self.reset_btn.pack(side="right", padx=2)

        # Status banner
        self.status_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 9, "italic"),
            foreground="#27ae60",
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 2))

        # Mode-disabled warning banner
        self.mode_banner = ttk.Label(
            self,
            text="🔒 Sliders active only in JOINT_PD mode",
            font=("Helvetica", 9, "bold"),
            foreground="#e67e22",
        )
        # Will be shown/hidden by _update_enabled_state

    def _build_search_ui(self) -> None:
        """The filter box: only the rows whose joint name contains its text are shown."""
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
        for joint_key, row in self.slider_rows.items():
            if not query or query in joint_key.lower():
                row.pack(fill="x", padx=4, pady=1)
            else:
                row.pack_forget()

    def _load_joint_data(self) -> None:
        """The joints of the PD gains file, and their default positions in the reference file."""
        self.joint_names = config_files.read_pd_gains_joint_names(self.pd_gains_file)
        self.default_positions = config_files.read_default_joint_state(
            self.reference_file
        )

    def _populate_sliders(self) -> None:
        """Create one slider per joint, grouped by limb section."""
        # Clear existing widgets
        for widget in self.scroll_container.scrollable_content.winfo_children():
            widget.destroy()

        self.slider_rows.clear()
        self.section_frames.clear()

        if not self.joint_names:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="No joints loaded. Check pd_gains_file path.",
                font=("Helvetica", 10, "italic"),
                foreground="#888888",
            ).pack(padx=20, pady=20)
            return

        # Prepare section frames in logical order
        section_order = [
            "Torso & Spine",
            "Left Leg",
            "Right Leg",
            "Left Arm",
            "Right Arm",
            "Head & Neck",
            "Other Joints",
        ]

        for sec in section_order:
            lf = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text=f"• {sec}"
            )
            self.section_frames[sec] = lf

        for jname in self.joint_names:
            sec_name = _classify_joint_section(jname)
            parent_frame = self.section_frames.get(
                sec_name, self.section_frames["Other Joints"]
            )

            default_val = self.default_positions.get(jname, 0.0)

            row = slider_row.SliderRow(
                parent_frame,
                name=jname,
                initial_value=default_val,
                min_val=-math.pi,
                max_val=math.pi,
                unit="rad",
                label_width=26,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[jname] = row

        # Pack only non-empty section frames
        for sec in section_order:
            lf = self.section_frames[sec]
            if lf.winfo_children():
                lf.pack(fill="x", padx=6, pady=4)

    def on_mode_changed(self) -> None:
        """Called when the FSM mode changes. Enable/disable sliders accordingly."""
        self._update_enabled_state()

    def _update_enabled_state(self) -> None:
        """Enable sliders only in JOINT_PD mode."""
        is_joint_pd = (
            self.fsm_mode_var is not None and self.fsm_mode_var.get() == "JOINT_PD"
        )

        if is_joint_pd:
            self.mode_banner.pack_forget()
            self.reset_btn.configure(state="normal")
            for row in self.slider_rows.values():
                row.set_state("normal")
        else:
            self.mode_banner.pack(anchor="w", padx=12, pady=(0, 2))
            self.reset_btn.configure(state="disabled")
            for row in self.slider_rows.values():
                row.set_state("disabled")

    def reset_all_defaults(self) -> None:
        """Reset all sliders to their default positions (the reference file's default_joint_state)."""
        for jname, row in self.slider_rows.items():
            default_val = self.default_positions.get(jname, 0.0)
            row.set_value(default_val)
        self._show_status("All targets reset to default joint state")

    def _on_any_slider_change(self, name: str, value: float | None) -> None:
        """Called on every slider move; debounces the publish."""
        del name, value  # Unused: the publish reads every slider.
        if self._debounce_publish_id is not None:
            self.after_cancel(self._debounce_publish_id)
        self._debounce_publish_id = self.after(100, self._publish_to_topic)

    def _publish_to_topic(self) -> None:
        """Publishes every slider's position as a JointTargets message on operator/joint_targets."""
        self._debounce_publish_id = None
        if not self.param_publisher:
            return

        # Only publish when in JOINT_PD mode
        if self.fsm_mode_var is None or self.fsm_mode_var.get() != "JOINT_PD":
            return

        try:
            targets = {
                jname: row.get_value() for jname, row in self.slider_rows.items()
            }
            self.param_publisher.publish(operator_bus.joint_targets(targets))
        # pylint: disable-next=broad-exception-caught  # Shown to the operator: a Tk callback must not raise.
        except Exception as e:
            print(f"[JointTargetsTab] ERROR in _publish_to_topic: {e}")
            self._show_status(f"Error publishing: {e}", error=True)

    def _show_status(self, msg: str, error: bool = False) -> None:
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        self.after(5000, lambda: self.status_label.configure(text=""))
