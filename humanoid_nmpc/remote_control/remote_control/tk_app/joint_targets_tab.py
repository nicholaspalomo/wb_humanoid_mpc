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

import math
import os
import re
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

import yaml

from remote_control.tk_app.scrollable_frame import ScrollableFrame
from remote_control.tk_app.slider_row import SliderRow


class JointTargetsTab(ttk.Frame):
    """
    Joint target position tuning tab for JOINT_PD mode.

    Provides per-joint position sliders (in radians) that publish to the
    ``/joint_pd_target_positions`` ROS 2 topic.  The C++ JointTargetSubscriber
    merges these into the nominal position vector used by the JOINT_PD controller.

    Sliders are enabled only when the FSM mode is ``JOINT_PD``.
    Default values are loaded from ``reference.yaml``'s ``defaultJointState``.
    """

    # LINT.IfChange(joint_target_topic_name)
    TOPIC_NAME = "/joint_pd_target_positions"
    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_ros2/include/humanoid_common_mpc_ros2/ros_comm/JointTargetSubscriber.h:joint_target_topic_name)

    def __init__(
        self,
        parent,
        pd_gains_file: Optional[str] = None,
        reference_file: Optional[str] = None,
        fsm_mode_var: Optional[tk.StringVar] = None,
        param_publisher=None,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="TFrame")

        self.pd_gains_file = pd_gains_file
        self.reference_file = reference_file
        self.fsm_mode_var = fsm_mode_var
        self.param_publisher = (
            param_publisher  # ROS publisher for /joint_pd_target_positions
        )

        self.joint_names: list = []
        self.default_positions: Dict[str, float] = {}
        self.slider_rows: Dict[str, SliderRow] = {}
        self.section_frames: Dict[str, ttk.LabelFrame] = {}
        self._debounce_publish_id = None

        self._build_header_ui()
        self._build_search_ui()

        # Scrollable container for joint sliders
        self.scroll_container = ScrollableFrame(self, bg_color="#2c2c2c")
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Load joint names from PD gains file and defaults from reference
        self._load_joint_data()
        self._populate_sliders()

        # Initial mode check
        self._update_enabled_state()

    def _build_header_ui(self):
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

    def _build_search_ui(self):
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

    def _on_search_filter(self, *args):
        query = self.search_var.get().strip().lower()
        for joint_key, row in self.slider_rows.items():
            if not query or query in joint_key.lower():
                row.pack(fill="x", padx=4, pady=1)
            else:
                row.pack_forget()

    @staticmethod
    def _classify_joint_section(joint_name: str) -> str:
        """Classify a joint name into a limb section for grouping."""
        name_lower = joint_name.lower()
        if re.search(r"waist|spine|torso|back", name_lower):
            return "Torso & Spine"
        if re.search(r"left.*leg|l_leg|left.*hip|left.*knee|left.*ankle", name_lower):
            return "Left Leg"
        if re.search(
            r"right.*leg|r_leg|right.*hip|right.*knee|right.*ankle", name_lower
        ):
            return "Right Leg"
        if re.search(
            r"left.*arm|l_arm|left.*shoulder|left.*elbow|left.*wrist", name_lower
        ):
            return "Left Arm"
        if re.search(
            r"right.*arm|r_arm|right.*shoulder|right.*elbow|right.*wrist", name_lower
        ):
            return "Right Arm"
        if re.search(r"neck|head", name_lower):
            return "Head & Neck"
        return "Other Joints"

    @staticmethod
    def _parse_reference_defaults(reference_file: str) -> Dict[str, float]:
        """Parse defaultJointState from reference.yaml, extracting joint names from comments.

        The reference.yaml format uses indexed keys like "(0,0)" with inline
        comments naming the joint:
            "(0,0)": -0.15  # left_hip_pitch_joint
        """
        defaults = {}
        if not reference_file or not os.path.exists(reference_file):
            return defaults

        try:
            with open(reference_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception:
            return defaults

        joint_state = data.get("defaultJointState", {})
        if not joint_state:
            return defaults

        # Read the raw file to extract inline comments with joint names
        try:
            with open(reference_file, "r") as f:
                lines = f.readlines()
        except Exception:
            return defaults

        in_joint_state = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("defaultJointState"):
                in_joint_state = True
                continue
            if in_joint_state:
                # End of section: non-indented non-empty line that isn't a comment
                if (
                    stripped
                    and not stripped.startswith("#")
                    and not stripped.startswith('"')
                ):
                    # Check if it looks like a new top-level key (not indented)
                    if line[0] not in (" ", "\t") and ":" in stripped:
                        break

                # Parse lines like:  "(0,0)": -0.15  # left_hip_pitch_joint
                match = re.match(r'\s*"[^"]+"\s*:\s*(-?[\d.]+)\s*#\s*(\S+)', stripped)
                if match:
                    value = float(match.group(1))
                    joint_name = match.group(2).strip()
                    defaults[joint_name] = value

        return defaults

    def _load_joint_data(self):
        """Load joint names from PD gains YAML and default positions from reference."""
        self.joint_names = []
        self.default_positions = {}

        # 1. Joint names from pd_gains_file
        if self.pd_gains_file and os.path.exists(self.pd_gains_file):
            try:
                with open(self.pd_gains_file, "r") as f:
                    data = yaml.safe_load(f)
                joint_gains = data.get("joint_gains", {})
                self.joint_names = [
                    k for k, v in joint_gains.items() if isinstance(v, dict)
                ]
            except Exception as e:
                print(f"[JointTargetsTab] Error loading PD gains file: {e}")

        # 2. Default positions from reference.yaml
        self.default_positions = self._parse_reference_defaults(self.reference_file)

    def _populate_sliders(self):
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
            sec_name = self._classify_joint_section(jname)
            parent_frame = self.section_frames.get(
                sec_name, self.section_frames["Other Joints"]
            )

            default_val = self.default_positions.get(jname, 0.0)

            row = SliderRow(
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

    def on_mode_changed(self):
        """Called when the FSM mode changes. Enable/disable sliders accordingly."""
        self._update_enabled_state()

    def _update_enabled_state(self):
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

    def reset_all_defaults(self):
        """Reset all sliders to their default (reference.yaml) positions."""
        for jname, row in self.slider_rows.items():
            default_val = self.default_positions.get(jname, 0.0)
            row.set_value(default_val)
        self._show_status("All targets reset to default joint state")

    def _on_any_slider_change(self, name: str, value: float):
        """Called on every slider move; debounces publish to ROS topic."""
        if self._debounce_publish_id is not None:
            self.after_cancel(self._debounce_publish_id)
        self._debounce_publish_id = self.after(100, self._publish_to_topic)

    def _publish_to_topic(self):
        """Publish current slider values as a YAML string to /joint_pd_target_positions."""
        self._debounce_publish_id = None
        if not self.param_publisher:
            return

        # Only publish when in JOINT_PD mode
        if self.fsm_mode_var is None or self.fsm_mode_var.get() != "JOINT_PD":
            return

        try:
            targets = {}
            for jname, row in self.slider_rows.items():
                targets[jname] = round(row.get_value(), 6)

            yaml_content = yaml.dump(targets, default_flow_style=True)

            from std_msgs.msg import String

            msg = String()
            msg.data = yaml_content
            self.param_publisher.publish(msg)
        except Exception as e:
            print(f"[JointTargetsTab] ERROR in _publish_to_topic: {e}")
            self._show_status(f"Error publishing: {e}", error=True)

    def _show_status(self, msg: str, error: bool = False):
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        self.after(5000, lambda: self.status_label.configure(text=""))
