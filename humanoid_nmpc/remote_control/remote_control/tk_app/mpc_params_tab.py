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

import logging
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, Any, Optional, List, Tuple

from remote_control.tk_app.scrollable_frame import ScrollableFrame
from remote_control.tk_app.slider_row import SliderRow
from remote_control.tk_app.yaml_editor_utils import (
    load_yaml_safe,
    update_yaml_values_in_place,
)

_LOGGER = logging.getLogger(__name__)

# Contact-constraint keys that are baked into the CppAD-compiled constraint at
# build time. MpcParameterUpdaterModule only hot-reloads the barrier's `mu` and
# `delta`, so editing these takes effect on the next launch, not immediately.
# They are still shown so the value can be saved to the task file, but the label
# says so rather than implying a live control.
# LINT.IfChange(build_time_contact_keys)
_BUILD_TIME_CONTACT_KEYS = frozenset(
    (
        "frictionCoefficient",
        "torsionalFrictionCoefficient",
        "minNormalForce",
        "gripperForce",
    )
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/mrt/MpcParameterUpdaterModule.cpp:hot_reloadable_barrier_keys)


def _contact_slider_label(prefix: str, key: str) -> str:
    """Names a contact-constraint slider, flagging the build-time-only keys."""
    if key in _BUILD_TIME_CONTACT_KEYS:
        return f"{prefix}_{key} (restart)"
    return f"{prefix}_{key}"


class MpcParamsTab(ttk.Frame):
    """
    MPC Parameters tuning tab allowing real-time slider tuning for Q/R matrices,
    task space tracking costs, foot constraints, swing trajectory, and relaxed barrier limits.
    """

    KNOWN_PRESETS = {
        "DRC Atlas (Centroidal)": "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
        "Unitree G1 (WB)": "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
        "Unitree R1 (Centroidal)": "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
        "Unitree G1 (Centroidal)": "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
    }

    MOMENTUM_LABELS = [
        "CoM Lin Mom X (h_com_x/m)",
        "CoM Lin Mom Y (h_com_y/m)",
        "CoM Lin Mom Z (h_com_z/m)",
        "CoM Ang Mom X (L_x/m)",
        "CoM Ang Mom Y (L_y/m)",
        "CoM Ang Mom Z (L_z/m)",
    ]

    BASE_POSE_LABELS = [
        "Base Pos X (p_base_x)",
        "Base Pos Y (p_base_y)",
        "Base Pos Z (p_base_z)",
        "Base Yaw (theta_base_z)",
        "Base Pitch (theta_base_y)",
        "Base Roll (theta_base_x)",
    ]

    COM_LABELS = [
        "CoM Pos X (p_com_x)",
        "CoM Pos Y (p_com_y)",
        "CoM Pos Z (p_com_z)",
    ]

    # Q_acom rows follow the centroidal state's ZYX Euler convention, matching
    # BASE_POSE_LABELS above. Row 0 is yaw.
    ACOM_LABELS = [
        "ACoM Yaw (theta_acom_z)",
        "ACoM Pitch (theta_acom_y)",
        "ACoM Roll (theta_acom_x)",
    ]

    CONTACT_FORCE_LABELS = [
        "Left Foot Force X",
        "Left Foot Force Y",
        "Left Foot Force Z",
        "Left Foot Moment X",
        "Left Foot Moment Y",
        "Left Foot Moment Z",
        "Right Foot Force X",
        "Right Foot Force Y",
        "Right Foot Force Z",
        "Right Foot Moment X",
        "Right Foot Moment Y",
        "Right Foot Moment Z",
    ]

    def __init__(
        self,
        parent,
        task_file: Optional[str] = None,
        on_params_updated=None,
        enable_online_tuning: Optional[bool] = None,
        param_publisher=None,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="TFrame")

        self.task_file = task_file
        self.on_params_updated = on_params_updated
        self._explicit_online_tuning = enable_online_tuning
        self.enable_online_tuning = (
            True if enable_online_tuning is None else enable_online_tuning
        )

        self.raw_data: Dict[str, Any] = {}
        self.slider_rows: Dict[str, SliderRow] = {}  # key_path_str -> SliderRow
        self.comment_map: Dict[str, str] = {}  # "(i,i)" -> comment description
        self._debounce_publish_id = None  # tkinter after() ID for debounced publish
        self.param_publisher = (
            param_publisher  # ROS publisher for /mpc_parameter_updates
        )
        self._live_values: Dict[str, float] = (
            {}
        )  # Persists slider values across category switches
        self._default_values: Dict[str, float] = (
            {}
        )  # Persists reset-checkpoint defaults across category switches

        self._build_header_ui()

        # Category navigation buttons / segmented bar
        self._build_category_nav_ui()

        # Scrollable container for parameters
        self.scroll_container = ScrollableFrame(self, bg_color="#2c2c2c")
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if self.task_file and os.path.exists(self.task_file):
            self.load_file(self.task_file)
        else:
            default_path = list(self.KNOWN_PRESETS.values())[0]
            if os.path.exists(default_path):
                self.load_file(default_path)

    def _build_header_ui(self):
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
        preset_cb.bind(
            "<Button-1>",
            lambda e: (
                e.widget.event_generate("<Down>", when="head")
                if e.widget.identify(e.x, e.y) != "downarrow"
                else None
            ),
        )

        self.path_var = tk.StringVar(value=self.task_file or "")
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

        self.save_btn = ttk.Button(
            toolbar, text="💾 Save to YAML", command=self.save_and_checkpoint
        )
        self.save_btn.pack(side="left", padx=(4, 0))

        self.status_label = ttk.Label(
            self, text="", font=("Helvetica", 9, "italic"), foreground="#27ae60"
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 2))

        # Online tuning disabled warning banner
        self.warning_banner = ttk.Label(
            self,
            text="🔒 Online Tuning Disabled (enableOnlineTuning: false in task.yaml)",
            font=("Helvetica", 9, "bold"),
            foreground="#e67e22",
        )
        if not self.enable_online_tuning:
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
            self.save_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")

    def set_online_tuning_enabled(self, enabled: bool):
        """Dynamically enable or disable online tuning in the GUI."""
        self.enable_online_tuning = bool(enabled)
        if not self.enable_online_tuning:
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
            if hasattr(self, "save_btn"):
                self.save_btn.configure(state="disabled")
            if hasattr(self, "reset_btn"):
                self.reset_btn.configure(state="disabled")
            for row in self.slider_rows.values():
                row.set_state("disabled")
        else:
            self.warning_banner.pack_forget()
            if hasattr(self, "save_btn"):
                self.save_btn.configure(state="normal")
            if hasattr(self, "reset_btn"):
                self.reset_btn.configure(state="normal")
            for row in self.slider_rows.values():
                row.set_state("normal")

    @staticmethod
    def _to_float(v):
        """Convert int, float, or numeric/scientific-notation string (e.g. '1e4', '1e0') to float."""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def _build_category_nav_ui(self):
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.active_category = tk.StringVar(value="State Cost (Q)")
        self.categories = [
            "State Cost (Q)",
            "Control Cost (R)",
            "Terminal Cost (Q_final)",
            "Task Space Costs",
            "Constraints & Barriers",
            "Solver & Horizon",
            "Contact Planning",
        ]

        for cat in self.categories:
            btn = ttk.Radiobutton(
                nav_frame,
                text=cat,
                value=cat,
                variable=self.active_category,
                command=self._render_active_category,
            )
            btn.pack(side="left", padx=6)

    def _on_preset_selected(self, event=None):
        preset_name = self.preset_var.get()
        if preset_name in self.KNOWN_PRESETS:
            path = self.KNOWN_PRESETS[preset_name]
            if os.path.exists(path):
                self.load_file(path)
            else:
                self._show_status(f"Preset path not found: {path}", error=True)

    def _browse_file(self):
        selected = filedialog.askopenfilename(
            title="Select task.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.load_file(selected)

    def load_file(self, file_path: str):
        self.task_file = os.path.abspath(file_path)
        self.path_var.set(self.task_file)
        self.raw_data = load_yaml_safe(self.task_file)

        # Create a backup of the original file at load time (before any
        # slider-driven writes) so "Reset All" can always restore it.
        bak_path = self.task_file + ".bak"
        if not os.path.exists(bak_path):
            import shutil

            shutil.copy2(self.task_file, bak_path)

        if self._explicit_online_tuning is not None:
            self.enable_online_tuning = self._explicit_online_tuning
        elif "enableOnlineTuning" in self.raw_data:
            self.enable_online_tuning = bool(self.raw_data["enableOnlineTuning"])
        elif "enable_online_tuning" in self.raw_data:
            self.enable_online_tuning = bool(self.raw_data["enable_online_tuning"])
        self.set_online_tuning_enabled(self.enable_online_tuning)
        self._parse_yaml_comments(self.task_file)
        self._render_active_category()
        self._show_status(f"Loaded: {os.path.basename(self.task_file)}")

    def reload_file(self):
        if self.task_file and os.path.exists(self.task_file):
            self.load_file(self.task_file)
            # Publish the reloaded values so the MPC syncs with the sliders
            self._publish_to_topic()

    def _parse_yaml_comments(self, file_path: str):
        """Extracts inline annotations like '# back_bkz' or '# p_base_z' from task.yaml."""
        self.comment_map.clear()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(
                        r'["\']\((\d+),(\d+)\)["\']\s*:\s*[0-9eE\.\+\-]+\s*#\s*(.*)',
                        line,
                    )
                    if match:
                        key = f"({match.group(1)},{match.group(2)})"
                        comment = match.group(3).strip()
                        self.comment_map[key] = comment
        except Exception:
            pass

    def _render_active_category(self):
        cat = self.active_category.get()

        # Save current slider values before destroying them
        for key, row in self.slider_rows.items():
            self._live_values[key] = row.get_value()

        # Clear content container
        for child in self.scroll_container.scrollable_content.winfo_children():
            child.destroy()

        self.slider_rows.clear()

        if cat == "State Cost (Q)":
            self._render_q_matrix()
        elif cat == "Control Cost (R)":
            self._render_r_matrix()
        elif cat == "Terminal Cost (Q_final)":
            self._render_q_final_matrix()
        elif cat == "Task Space Costs":
            self._render_task_space_costs()
        elif cat == "Constraints & Barriers":
            self._render_constraints_and_barriers()
        elif cat == "Solver & Horizon":
            self._render_solver_and_horizon()
        elif cat == "Contact Planning":
            self._render_contact_planning()

        # Restore saved slider values and defaults (from previous edits on this tab)
        for key, row in self.slider_rows.items():
            if key in self._live_values:
                row.set_value(self._live_values[key])
            if key in self._default_values:
                row.default_value = self._default_values[key]
                row._update_highlight()

        if not self.enable_online_tuning:
            for row in self.slider_rows.values():
                row.set_state("disabled")

    def _render_q_matrix(self):
        q_data = self.raw_data.get("Q", {})
        if not q_data:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="Q matrix not found in task.yaml",
            ).pack(padx=10, pady=10)
            return

        # Scaling
        scaling_val = float(q_data.get("scaling", 1.0))
        scale_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content, text="⚙ Scaling Factor"
        )
        scale_frame.pack(fill="x", padx=6, pady=4)
        row = SliderRow(
            scale_frame,
            name="Q.scaling",
            initial_value=scaling_val,
            min_val=0.01,
            max_val=max(scaling_val * 5.0, 10.0),
            label_width=22,
            on_change=self._on_any_slider_change,
        )
        row.pack(fill="x", padx=4, pady=2)
        self.slider_rows["Q.scaling"] = row

        # Momentum Weights (0..5)
        mom_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Centroidal Momentum Tracking (0..5)",
        )
        mom_frame.pack(fill="x", padx=6, pady=4)
        for i in range(6):
            key = f"({i},{i})"
            if key in q_data:
                val = float(q_data[key])
                name = self.MOMENTUM_LABELS[i]
                row = SliderRow(
                    mom_frame,
                    name=name,
                    initial_value=val,
                    min_val=0.0,
                    max_val=max(val * 4.0, 50.0),
                    label_width=28,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'Q."{key}"'] = row

        # CoM Position Tracking (Q_com)
        q_com_data = self.raw_data.get("Q_com", {})
        if q_com_data:
            com_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• CoM Position Tracking (Q_com)",
            )
            com_frame.pack(fill="x", padx=6, pady=4)
            com_scaling_val = float(q_com_data.get("scaling", 1.0))
            row = SliderRow(
                com_frame,
                name="Q_com.scaling",
                initial_value=com_scaling_val,
                min_val=0.01,
                max_val=max(com_scaling_val * 5.0, 100.0),
                label_width=28,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows["Q_com.scaling"] = row

            for i in range(3):
                key = f"({i},{i})"
                if key in q_com_data:
                    val = float(q_com_data[key])
                    name = self.COM_LABELS[i]
                    row = SliderRow(
                        com_frame,
                        name=name,
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 4.0, 50.0),
                        label_width=28,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f'Q_com."{key}"'] = row

        # Angular CoM Tracking (Q_acom)
        q_acom_data = self.raw_data.get("Q_acom", {})
        if q_acom_data:
            acom_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Angular CoM Tracking (Q_acom)",
            )
            acom_frame.pack(fill="x", padx=6, pady=4)
            acom_scaling_val = float(q_acom_data.get("scaling", 1.0))
            row = SliderRow(
                acom_frame,
                name="Q_acom.scaling",
                initial_value=acom_scaling_val,
                min_val=0.01,
                max_val=max(acom_scaling_val * 5.0, 100.0),
                label_width=28,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows["Q_acom.scaling"] = row

            for i in range(3):
                key = f"({i},{i})"
                if key in q_acom_data:
                    val = float(q_acom_data[key])
                    name = self.ACOM_LABELS[i]
                    row = SliderRow(
                        acom_frame,
                        name=name,
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 4.0, 50.0),
                        label_width=28,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f'Q_acom."{key}"'] = row

        # Base Pose Weights (6..11)
        use_com_acom = bool(self.raw_data.get("useComAndAcomTracking", False)) or bool(
            q_com_data
        )
        base_title = (
            "• Base Pose Tracking (6..11) [⚡ Overridden by CoM + ACoM Tracking]"
            if use_com_acom
            else "• Base Pose Tracking (6..11)"
        )
        base_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text=base_title,
        )
        base_frame.pack(fill="x", padx=6, pady=4)
        for i in range(6, 12):
            key = f"({i},{i})"
            if key in q_data:
                val = float(q_data[key])
                name = self.BASE_POSE_LABELS[i - 6]
                row = SliderRow(
                    base_frame,
                    name=name,
                    initial_value=val,
                    min_val=0.0,
                    max_val=max(val * 4.0, 100.0),
                    label_width=28,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'Q."{key}"'] = row

        # Joint Position Weights (12+)
        joint_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Joint Position Tracking (12+)",
        )
        joint_frame.pack(fill="x", padx=6, pady=4)
        i = 12
        while True:
            key = f"({i},{i})"
            if key not in q_data:
                break
            val = float(q_data[key])
            comment = self.comment_map.get(key, "")
            name = f"Joint {i}: {comment}" if comment else f"Joint {key}"
            row = SliderRow(
                joint_frame,
                name=name,
                initial_value=val,
                min_val=0.0,
                max_val=max(val * 5.0, 5.0),
                label_width=28,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[f'Q."{key}"'] = row
            i += 1

    def _render_r_matrix(self):
        r_data = self.raw_data.get("R", {})
        if not r_data:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="R matrix not found in task.yaml",
            ).pack(padx=10, pady=10)
            return

        # Scaling
        scaling_val = float(r_data.get("scaling", 1.0))
        scale_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content, text="⚙ Scaling Factor"
        )
        scale_frame.pack(fill="x", padx=6, pady=4)
        row = SliderRow(
            scale_frame,
            name="R.scaling",
            initial_value=scaling_val,
            min_val=0.01,
            max_val=max(scaling_val * 5.0, 10.0),
            label_width=22,
            on_change=self._on_any_slider_change,
        )
        row.pack(fill="x", padx=4, pady=2)
        self.slider_rows["R.scaling"] = row

        # Contact Force/Moment Weights (0..11)
        force_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Foot Contact Wrenches (0..11)",
        )
        force_frame.pack(fill="x", padx=6, pady=4)
        for i in range(12):
            key = f"({i},{i})"
            if key in r_data:
                val = float(r_data[key])
                name = self.CONTACT_FORCE_LABELS[i]
                row = SliderRow(
                    force_frame,
                    name=name,
                    initial_value=val,
                    min_val=0.0,
                    max_val=max(val * 4.0, 50.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'R."{key}"'] = row

        # Joint Velocity Weights (12+)
        vel_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content, text="• Joint Velocities (12+)"
        )
        vel_frame.pack(fill="x", padx=6, pady=4)
        i = 12
        while True:
            key = f"({i},{i})"
            if key not in r_data:
                break
            val = float(r_data[key])
            comment = self.comment_map.get(key, "")
            name = f"Vel {i}: {comment}" if comment else f"Vel {key}"
            row = SliderRow(
                vel_frame,
                name=name,
                initial_value=val,
                min_val=0.0,
                max_val=max(val * 4.0, 100.0),
                label_width=26,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[f'R."{key}"'] = row
            i += 1

    def _render_q_final_matrix(self):
        # Terminal cost scaling
        term_scaling = float(self.raw_data.get("terminalCostScaling", 4.0))
        scale_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content, text="⚙ Terminal Cost Scaling"
        )
        scale_frame.pack(fill="x", padx=6, pady=4)
        row = SliderRow(
            scale_frame,
            name="terminalCostScaling",
            initial_value=term_scaling,
            min_val=0.1,
            max_val=max(term_scaling * 4.0, 20.0),
            label_width=24,
            on_change=self._on_any_slider_change,
        )
        row.pack(fill="x", padx=4, pady=2)
        self.slider_rows["terminalCostScaling"] = row

        # DCM (capture point) terminal cost. When useDcmTerminalCost is on it replaces Q_final, whose sliders are then
        # still shown but have no effect on the running controller.
        dcm_data = self.raw_data.get("dcm_terminal_cost", {})
        if dcm_data:
            use_dcm = bool(self.raw_data.get("useDcmTerminalCost", False))
            dcm_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• DCM Terminal Cost (dcm_terminal_cost)"
                + (
                    "  [active, Q_final ignored]"
                    if use_dcm
                    else "  [inactive: useDcmTerminalCost is false]"
                ),
            )
            dcm_frame.pack(fill="x", padx=6, pady=4)
            for key, (
                min_val,
                max_scale,
                min_max,
            ) in self.DCM_TERMINAL_COST_RANGES.items():
                if key not in dcm_data:
                    continue
                val = self._to_float(dcm_data[key])
                if val is None:
                    continue
                row = SliderRow(
                    dcm_frame,
                    name=key,
                    initial_value=val,
                    min_val=min_val,
                    max_val=max(val * max_scale, min_max),
                    label_width=24,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"dcm_terminal_cost.{key}"] = row

        qf_data = self.raw_data.get("Q_final", {})
        if not qf_data:
            return

        qf_scaling = float(qf_data.get("scaling", 1.0))
        row_qf_s = SliderRow(
            scale_frame,
            name="Q_final.scaling",
            initial_value=qf_scaling,
            min_val=0.01,
            max_val=max(qf_scaling * 5.0, 10.0),
            label_width=24,
        )
        row_qf_s.pack(fill="x", padx=4, pady=2)
        self.slider_rows["Q_final.scaling"] = row_qf_s

        # Sync Q_final from Q button
        sync_btn = ttk.Button(
            scale_frame,
            text="⟳ Sync from Q",
            command=self._sync_q_final_from_q,
        )
        sync_btn.pack(fill="x", padx=4, pady=(4, 2))

        # Terminal Momentum & Pose
        term_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Terminal State Weights (0..11)",
        )
        term_frame.pack(fill="x", padx=6, pady=4)
        for i in range(12):
            key = f"({i},{i})"
            if key in qf_data:
                val = float(qf_data[key])
                name = (
                    self.MOMENTUM_LABELS[i] if i < 6 else self.BASE_POSE_LABELS[i - 6]
                )
                row = SliderRow(
                    term_frame,
                    name=f"Final {name}",
                    initial_value=val,
                    min_val=0.0,
                    max_val=max(val * 4.0, 50.0),
                    label_width=28,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'Q_final."{key}"'] = row

        # Terminal Joint Position Weights (12+)
        joint_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Terminal Joint Position Tracking (12+)",
        )
        joint_frame.pack(fill="x", padx=6, pady=4)
        i = 12
        while True:
            key = f"({i},{i})"
            if key not in qf_data:
                break
            val = float(qf_data[key])
            comment = self.comment_map.get(key, "")
            name = f"Final Joint {i}: {comment}" if comment else f"Final Joint {key}"
            row = SliderRow(
                joint_frame,
                name=name,
                initial_value=val,
                min_val=0.0,
                max_val=max(val * 5.0, 5.0),
                label_width=28,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[f'Q_final."{key}"'] = row
            i += 1

    def _sync_q_final_from_q(self):
        """Copy ALL Q diagonal values and scaling into Q_final sliders."""
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return

        # Use live slider values (persisted across category switches) instead
        # of re-reading the file, since we no longer auto-save to YAML.
        q_data = {}
        for key, val in self._live_values.items():
            if key.startswith("Q."):
                # Strip the "Q." prefix and any quotes
                suffix = key[2:].strip('"')
                q_data[suffix] = val

        # Fall back to raw_data if no live values for Q exist yet
        if not q_data:
            q_data = self.raw_data.get("Q", {})

        if not q_data:
            self._show_status("Q matrix not found — nothing to sync.", error=True)
            return

        synced = 0
        # Sync scaling
        qf_scaling_key = "Q_final.scaling"
        if qf_scaling_key in self.slider_rows:
            q_scale = float(q_data.get("scaling", 1.0))
            self.slider_rows[qf_scaling_key].set_value(q_scale)
            self._live_values[qf_scaling_key] = q_scale
            synced += 1

        # Sync ALL diagonal entries that exist in Q_final sliders
        for key in list(self.slider_rows.keys()):
            if not key.startswith('Q_final."('):
                continue
            # Extract the "(i,j)" part from the slider key
            diag_key = key.split("Q_final.")[1].strip('"')
            if diag_key in q_data:
                q_val = float(q_data[diag_key])
                self.slider_rows[key].set_value(q_val)
                self._live_values[key] = q_val
                synced += 1

        if synced > 0:
            self._on_any_slider_change("sync", 0.0)
            self._show_status(f"✓ Synced {synced} Q_final entries from Q")

    def _render_task_space_costs(self):
        # Foot tracking weights
        foot_costs = self.raw_data.get("task_space_foot_cost_weights", {})
        if foot_costs:
            f_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text="• Task Space Foot Costs"
            )
            f_frame.pack(fill="x", padx=6, pady=4)
            for k, v in foot_costs.items():
                val = self._to_float(v)
                if val is not None:
                    row = SliderRow(
                        f_frame,
                        name=k,
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 100.0),
                        label_width=24,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f"task_space_foot_cost_weights.{k}"] = row

        # Torso tracking weights
        torso_costs = (
            self.raw_data.get("task_space_costs", {})
            .get("torso", {})
            .get("weights", {})
        )
        if torso_costs:
            t_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Task Space Torso Costs",
            )
            t_frame.pack(fill="x", padx=6, pady=4)
            for k, v in torso_costs.items():
                val = self._to_float(v)
                if val is not None:
                    row = SliderRow(
                        t_frame,
                        name=f"torso_{k}",
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 100.0),
                        label_width=24,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f"task_space_costs.torso.weights.{k}"] = row

        # ICP cost
        icp_data = self.raw_data.get("icp_cost_weights", {})
        if "icpErrorWeight" in icp_data:
            icp_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Instantaneous Capture Point (ICP)",
            )
            icp_frame.pack(fill="x", padx=6, pady=4)
            val = float(icp_data["icpErrorWeight"])
            row = SliderRow(
                icp_frame,
                name="icpErrorWeight",
                initial_value=val,
                min_val=0.0,
                max_val=max(val * 4.0, 50.0),
                label_width=24,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=2)
            self.slider_rows["icp_cost_weights.icpErrorWeight"] = row

        # Left leg joint torque cost
        left_tc = self.raw_data.get("left_leg_torque_cost", {})
        if left_tc:
            ll_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Left Leg Joint Torque Costs",
            )
            ll_frame.pack(fill="x", padx=6, pady=4)
            ll_weights = left_tc.get("weights", {})
            ll_scaling = self._to_float(ll_weights.get("scaling", 1e-6))
            if ll_scaling is not None:
                row = SliderRow(
                    ll_frame,
                    name="left_leg_torque_cost.weights.scaling",
                    initial_value=ll_scaling,
                    min_val=1e-8,
                    max_val=max(ll_scaling * 10.0, 1e-3),
                    label_width=28,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows["left_leg_torque_cost.weights.scaling"] = row

            joint_names = left_tc.get("activeJointNames", [])
            for i, jname in enumerate(joint_names):
                key = f"({i},0)"
                if key in ll_weights:
                    val = self._to_float(ll_weights[key])
                    if val is not None:
                        row = SliderRow(
                            ll_frame,
                            name=jname,
                            initial_value=val,
                            min_val=0.0,
                            max_val=max(val * 4.0, 20.0),
                            label_width=28,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[f'left_leg_torque_cost.weights."{key}"'] = row

        # Right leg joint torque cost
        right_tc = self.raw_data.get("right_leg_torque_cost", {})
        if right_tc:
            rl_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Right Leg Joint Torque Costs",
            )
            rl_frame.pack(fill="x", padx=6, pady=4)
            rl_weights = right_tc.get("weights", {})
            rl_scaling = self._to_float(rl_weights.get("scaling", 1e-6))
            if rl_scaling is not None:
                row = SliderRow(
                    rl_frame,
                    name="right_leg_torque_cost.weights.scaling",
                    initial_value=rl_scaling,
                    min_val=1e-8,
                    max_val=max(rl_scaling * 10.0, 1e-3),
                    label_width=28,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows["right_leg_torque_cost.weights.scaling"] = row

            joint_names = right_tc.get("activeJointNames", [])
            for i, jname in enumerate(joint_names):
                key = f"({i},0)"
                if key in rl_weights:
                    val = self._to_float(rl_weights[key])
                    if val is not None:
                        row = SliderRow(
                            rl_frame,
                            name=jname,
                            initial_value=val,
                            min_val=0.0,
                            max_val=max(val * 4.0, 20.0),
                            label_width=28,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[f'right_leg_torque_cost.weights."{key}"'] = row

    def _render_constraints_and_barriers(self):
        # Foot constraint gains
        foot_cfg = self.raw_data.get("model_settings", {}).get("foot_constraint", {})
        if foot_cfg:
            fc_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text="• Foot Constraint Gains"
            )
            fc_frame.pack(fill="x", padx=6, pady=4)
            for k, v in foot_cfg.items():
                val = self._to_float(v)
                if val is not None:
                    # softConstraintWeight is a quadratic penalty weight (typical range 1-100k),
                    # not an error gain, so it needs a much larger slider range.
                    if k == "softConstraintWeight":
                        s_min, s_max = 0.0, max(val * 5.0, 100000.0)
                    else:
                        s_min, s_max = 0.0, max(val * 3.0, 50.0)
                    row = SliderRow(
                        fc_frame,
                        name=k,
                        initial_value=val,
                        min_val=s_min,
                        max_val=s_max,
                        label_width=26,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f"model_settings.foot_constraint.{k}"] = row

        # Swing trajectory config
        swing_cfg = self.raw_data.get("swing_trajectory_config", {})
        if swing_cfg:
            sw_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Swing Trajectory Parameters",
            )
            sw_frame.pack(fill="x", padx=6, pady=4)
            for k, v in swing_cfg.items():
                val = self._to_float(v)
                if val is not None:
                    # Handle signed parameters like liftOffVelocity, touchDownVelocity
                    min_val = min(val * 2.0, -0.5) if val < 0 else 0.0
                    max_val = max(val * 2.5, 0.5) if val > 0 else 0.0
                    row = SliderRow(
                        sw_frame,
                        name=k,
                        initial_value=val,
                        min_val=min_val,
                        max_val=max_val,
                        label_width=26,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[f"swing_trajectory_config.{k}"] = row

        # Relaxed log barriers & soft constraint parameters
        bar_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Barrier & Limits (mu & delta)",
        )
        bar_frame.pack(fill="x", padx=6, pady=4)

        # Friction cone barrier
        fric_cfg = self.raw_data.get("contacts", {}).get(
            "frictionForceConeSoftConstraint", {}
        )
        for k in ["frictionCoefficient", "mu", "delta"]:
            if k in fric_cfg:
                val = float(fric_cfg[k])
                row = SliderRow(
                    bar_frame,
                    name=_contact_slider_label("frictionCone", k),
                    initial_value=val,
                    min_val=0.01,
                    max_val=max(val * 4.0, 20.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contacts.frictionForceConeSoftConstraint.{k}"] = row

        # Contact moment XY barrier
        moment_cfg = self.raw_data.get("contacts", {}).get(
            "contactMomentXYSoftConstraint", {}
        )
        for k in ["mu", "delta"]:
            if k in moment_cfg:
                val = float(moment_cfg[k])
                row = SliderRow(
                    bar_frame,
                    name=f"contactMomentXY_{k}",
                    initial_value=val,
                    min_val=0.001,
                    max_val=max(val * 4.0, 5.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contacts.contactMomentXYSoftConstraint.{k}"] = row

        # Contact wrench cone barrier & soft constraint
        wrench_cfg = self.raw_data.get("contacts", {}).get(
            "contactWrenchConeSoftConstraint", {}
        )
        for k in [
            "frictionCoefficient",
            "torsionalFrictionCoefficient",
            "minNormalForce",
            "gripperForce",
            "mu",
            "delta",
        ]:
            if k in wrench_cfg:
                val = float(wrench_cfg[k])
                min_v = (
                    0.0
                    if k
                    in [
                        "minNormalForce",
                        "gripperForce",
                        "torsionalFrictionCoefficient",
                    ]
                    else 0.001
                )
                max_v = max(val * 4.0, 20.0)
                row = SliderRow(
                    bar_frame,
                    name=_contact_slider_label("wrenchCone", k),
                    initial_value=val,
                    min_val=min_v,
                    max_val=max_v,
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contacts.contactWrenchConeSoftConstraint.{k}"] = row

        # Basis non-negativity barrier (for basis-vector inputs)
        basis_bar_cfg = self.raw_data.get("contacts", {}).get(
            "basisNonNegativityBarrier", {}
        )
        for k in ["mu", "delta"]:
            if k in basis_bar_cfg:
                val = float(basis_bar_cfg[k])
                row = SliderRow(
                    bar_frame,
                    name=f"basisNonNeg_{k}",
                    initial_value=val,
                    min_val=1e-5,
                    max_val=max(val * 5.0, 1.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contacts.basisNonNegativityBarrier.{k}"] = row

        # Basis scaling regularization
        if "basisScalingRegularization" in self.raw_data.get("contacts", {}):
            val = float(self.raw_data["contacts"]["basisScalingRegularization"])
            row = SliderRow(
                bar_frame,
                name="basisScalingRegularization",
                initial_value=val,
                min_val=1e-6,
                max_val=max(val * 10.0, 0.01),
                label_width=26,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows["contacts.basisScalingRegularization"] = row

        # Joint limits barrier
        jl_cfg = self.raw_data.get("jointLimits", {})
        for k in ["mu", "delta"]:
            if k in jl_cfg:
                val = float(jl_cfg[k])
                row = SliderRow(
                    bar_frame,
                    name=f"jointLimits_{k}",
                    initial_value=val,
                    min_val=0.01,
                    max_val=max(val * 4.0, 2000.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"jointLimits.{k}"] = row

        # Collision constraint barrier
        col_cfg = self.raw_data.get("collision_constraint", {})
        for k in ["mu", "delta"]:
            if k in col_cfg:
                val = float(col_cfg[k])
                row = SliderRow(
                    bar_frame,
                    name=f"collision_{k}",
                    initial_value=val,
                    min_val=0.01,
                    max_val=max(val * 4.0, 20000.0),
                    label_width=26,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"collision_constraint.{k}"] = row

        # Foot contact frame translation (parent joint to contact frame)
        c_trans = self.raw_data.get("contacts", {}).get("contact_frame_translation", {})
        if c_trans:
            ct_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Foot Contact Frame Translation ⚠ restart",
            )
            ct_frame.pack(fill="x", padx=6, pady=4)
            for axis in ["x", "y", "z"]:
                if axis in c_trans:
                    val = self._to_float(c_trans[axis])
                    if val is not None:
                        row = SliderRow(
                            ct_frame,
                            name=f"contact_trans_{axis}",
                            initial_value=val,
                            min_val=min(val * 2.0, -0.2) if val < 0 else -0.1,
                            max_val=max(val * 2.0, 0.2) if val > 0 else 0.1,
                            label_width=26,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[
                            f"contacts.contact_frame_translation.{axis}"
                        ] = row

        # Foot support polygon bounds (contact rectangle)
        c_rect = self.raw_data.get("contacts", {}).get("contact_rectangle", {})
        if c_rect:
            cr_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Foot Support Polygon Bounds ⚠ restart",
            )
            cr_frame.pack(fill="x", padx=6, pady=4)
            for k in ["x_min", "x_max", "y_min", "y_max"]:
                if k in c_rect:
                    val = self._to_float(c_rect[k])
                    if val is not None:
                        row = SliderRow(
                            cr_frame,
                            name=k,
                            initial_value=val,
                            min_val=min(val * 2.0, -0.3) if val < 0 else 0.0,
                            max_val=max(val * 2.0, 0.3) if val > 0 else 0.0,
                            label_width=26,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[f"contacts.contact_rectangle.{k}"] = row

        # Collision sphere radii
        foot_rad = col_cfg.get("foot", {}).get("footCollisionSphereRadius")
        knee_rad = col_cfg.get("knee", {}).get("kneeCollisionSphereRadius")
        if foot_rad is not None or knee_rad is not None:
            cs_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Self-Collision Sphere Radii",
            )
            cs_frame.pack(fill="x", padx=6, pady=4)
            if foot_rad is not None:
                val = self._to_float(foot_rad)
                if val is not None:
                    row = SliderRow(
                        cs_frame,
                        name="footCollisionSphereRadius",
                        initial_value=val,
                        min_val=0.01,
                        max_val=max(val * 3.0, 0.2),
                        label_width=26,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[
                        "collision_constraint.foot.footCollisionSphereRadius"
                    ] = row
            if knee_rad is not None:
                val = self._to_float(knee_rad)
                if val is not None:
                    row = SliderRow(
                        cs_frame,
                        name="kneeCollisionSphereRadius",
                        initial_value=val,
                        min_val=0.01,
                        max_val=max(val * 3.0, 0.2),
                        label_width=26,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows[
                        "collision_constraint.knee.kneeCollisionSphereRadius"
                    ] = row

    # Slider ranges (min, max scale factor, minimum max) for the DCM terminal cost keys.
    # LINT.IfChange(dcm_terminal_cost_gui_keys)
    DCM_TERMINAL_COST_RANGES = {
        "comHeight": (0.3, 2.0, 1.5),
        "weight_x": (0.0, 4.0, 100.0),
        "weight_y": (0.0, 4.0, 100.0),
        "velocityOffsetFactor": (0.0, 2.0, 1.0),
    }
    # LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/cost/DcmTerminalCost.cpp:dcm_terminal_cost_keys)

    # Contact planning keys that are not tunable online (they change the problem structure or the threading).
    CONTACT_PLANNING_STATIC_KEYS = {
        "numNodes",
        "runInBackgroundThread",
        "verbose",
        "enforceAlternatingFeet",
    }

    def _render_contact_planning(self):
        """Render the mixed-integer contact planner parameters (contact_planning block)."""
        cp_data = self.raw_data.get("contact_planning", {})
        use_cp = bool(self.raw_data.get("useContactPlanning", False))
        header = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Mixed-Integer Contact Planner"
            + ("  [active]" if use_cp else "  [inactive: useContactPlanning is false]"),
        )
        header.pack(fill="x", padx=6, pady=4)
        if not cp_data:
            ttk.Label(header, text="No contact_planning section in task.yaml.").pack(
                anchor="w", padx=6, pady=4
            )
            return
        groups = [
            (
                "Timing & Model",
                [
                    "dt",
                    "commitTime",
                    "comHeight",
                    "gravity",
                    "minSwingDuration",
                    "maxSwingDuration",
                    "minContactDuration",
                    "maxContactDuration",
                ],
            ),
            (
                "Support & Reachability Geometry",
                [
                    "zmpHalfWidthX",
                    "zmpHalfWidthY",
                    "nominalStepWidth",
                    "minStepWidth",
                    "maxStepWidth",
                    "maxStepLength",
                    "reachX",
                    "reachYInner",
                    "reachYOuter",
                    "bigM",
                ],
            ),
            (
                "Objective Weights",
                [
                    "velocityTrackingWeight",
                    "zmpRegularizationWeight",
                    "footholdRegularizationWeight",
                    "stepWidthWeight",
                    "contactSwitchCost",
                    "terminalDcmWeight",
                    "constraintSlackWeight",
                    "constraintSlackLinearWeight",
                ],
            ),
            (
                "Solver Budget",
                [
                    "maxBranchAndBoundNodes",
                    "maxSolveTime",
                    "maxQpIterations",
                    "localSearchIterations",
                    "localSearchMaxTime",
                    "planningFrequency",
                ],
            ),
        ]
        for title, keys in groups:
            frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text=f"• {title}"
            )
            frame.pack(fill="x", padx=6, pady=4)
            for key in keys:
                if key not in cp_data or key in self.CONTACT_PLANNING_STATIC_KEYS:
                    continue
                val = self._to_float(cp_data[key])
                if val is None:
                    continue
                row = SliderRow(
                    frame,
                    name=key,
                    initial_value=val,
                    min_val=0.0,
                    max_val=max(val * 4.0, 1.0),
                    label_width=30,
                    on_change=self._on_any_slider_change,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contact_planning.{key}"] = row

    def _render_solver_and_horizon(self):
        """Render MPC loop frequencies, horizon, SQP multiple shooting, and rollout settings."""
        # MPC & MRT loop settings
        mpc_cfg = self.raw_data.get("mpc", {})
        if mpc_cfg:
            mpc_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• MPC Loop Rates & Prediction Horizon",
            )
            mpc_frame.pack(fill="x", padx=6, pady=4)
            if "timeHorizon" in mpc_cfg:
                val = self._to_float(mpc_cfg["timeHorizon"])
                if val is not None:
                    row = SliderRow(
                        mpc_frame,
                        name="timeHorizon (s) ⚠ restart",
                        initial_value=val,
                        min_val=0.1,
                        max_val=max(val * 3.0, 3.0),
                        label_width=28,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows["mpc.timeHorizon"] = row

            for freq_key, label in [
                ("mpcDesiredFrequency", "mpcDesiredFrequency (Hz) ⚠ restart"),
                ("mrtDesiredFrequency", "mrtDesiredFrequency (Hz) ⚠ restart"),
            ]:
                if freq_key in mpc_cfg:
                    val = self._to_float(mpc_cfg[freq_key])
                    if val is not None:
                        row = SliderRow(
                            mpc_frame,
                            name=label,
                            initial_value=val,
                            min_val=10.0,
                            max_val=max(val * 3.0, 200.0),
                            label_width=28,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[f"mpc.{freq_key}"] = row

        # SQP Multiple Shooting settings
        ms_cfg = self.raw_data.get("multiple_shooting", {})
        if ms_cfg:
            ms_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• SQP Multiple Shooting Algorithm Settings",
            )
            ms_frame.pack(fill="x", padx=6, pady=4)
            for k, (min_v, max_v) in [
                ("sqpIteration", (1.0, 20.0)),
                (
                    "dt",
                    (0.005, 0.1),
                ),  # ⚠ restart required — changes time discretization grid
                ("deltaTol", (1e-6, 1e-2)),
                ("g_max", (1e-4, 1.0)),
                ("g_min", (1e-8, 1e-3)),
                ("inequalityConstraintMu", (0.001, 2.0)),
                ("inequalityConstraintDelta", (0.1, 20.0)),
            ]:
                if k in ms_cfg:
                    val = self._to_float(ms_cfg[k])
                    if val is not None:
                        row = SliderRow(
                            ms_frame,
                            name=(k + " \u26a0 restart" if k == "dt" else k),
                            initial_value=val,
                            min_val=min_v,
                            max_val=max(val * 3.0, max_v),
                            label_width=28,
                            on_change=self._on_any_slider_change,
                        )
                        row.pack(fill="x", padx=4, pady=1)
                        self.slider_rows[f"multiple_shooting.{k}"] = row

        # Model & Rollout Timing
        rollout_cfg = self.raw_data.get("rollout", {})
        stance_time = self.raw_data.get("model_settings", {}).get(
            "phaseTransitionStanceTime"
        )
        if rollout_cfg or stance_time is not None:
            time_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content,
                text="• Stance & Rollout Timing",
            )
            time_frame.pack(fill="x", padx=6, pady=4)
            if stance_time is not None:
                val = self._to_float(stance_time)
                if val is not None:
                    row = SliderRow(
                        time_frame,
                        name="phaseTransitionStanceTime (s) ⚠ restart",
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 0.5),
                        label_width=28,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows["model_settings.phaseTransitionStanceTime"] = row

            if "timeStep" in rollout_cfg:
                val = self._to_float(rollout_cfg["timeStep"])
                if val is not None:
                    row = SliderRow(
                        time_frame,
                        name="rollout.timeStep (s) ⚠ restart",
                        initial_value=val,
                        min_val=0.001,
                        max_val=max(val * 3.0, 0.1),
                        label_width=28,
                        on_change=self._on_any_slider_change,
                    )
                    row.pack(fill="x", padx=4, pady=1)
                    self.slider_rows["rollout.timeStep"] = row

    def reset_all_defaults(self):
        if not self.enable_online_tuning:
            return
        for row in self.slider_rows.values():
            row.reset_to_default()
        # Publish immediately so the MPC picks up the reset values
        self._publish_to_topic()
        self._show_status("All parameters reset to loaded defaults")

    def _on_any_slider_change(self, name: str, value: float):
        """Called on every slider move; debounces publish to ROS topic.

        The C++ MpcParameterUpdaterModule subscribes to /mpc_parameter_updates
        for real-time parameter updates without touching the YAML file.
        """
        # Persist the value so it survives category tab switches
        self._live_values[name] = value
        if self._debounce_publish_id is not None:
            self.after_cancel(self._debounce_publish_id)
        self._debounce_publish_id = self.after(300, self._publish_to_topic)

    def _build_yaml_with_slider_values(self) -> str:
        """Build a complete YAML string from the original file with slider values applied.

        Reads the original task.yaml, applies ALL current slider values (from
        _live_values, which persists across category tab switches) as line
        edits, and returns the full modified YAML string (without writing to
        disk).  The C++ side writes this to a temp file for parsing.
        """
        if not self.task_file or not os.path.exists(self.task_file):
            return ""

        with open(self.task_file, "r") as f:
            lines = f.readlines()

        # Merge current slider_rows into _live_values to capture any pending changes
        for key_path_str, row in self.slider_rows.items():
            self._live_values[key_path_str] = row.get_value()

        # Build updates list from ALL live values (all categories)
        updates = []
        for key_path_str, val in self._live_values.items():
            parts = key_path_str.split(".")
            updates.append((parts, val))

        # Apply updates in-place on the lines (same logic as update_yaml_values_in_place
        # but without writing to disk)
        from remote_control.tk_app.yaml_editor_utils import _update_single_key

        for key_path, value in updates:
            lines = _update_single_key(lines, key_path, value)

        return "".join(lines)

    def _publish_to_topic(self):
        """Publish current slider values as a YAML string to /mpc_parameter_updates."""
        self._debounce_publish_id = None
        if not self.enable_online_tuning or not self.param_publisher:
            _LOGGER.debug("Skipping publish: online tuning disabled or no publisher.")
            return

        try:
            yaml_content = self._build_yaml_with_slider_values()
            if not yaml_content:
                _LOGGER.warning("Not publishing: the rebuilt task YAML is empty.")
                return

            from std_msgs.msg import String

            msg = String()
            msg.data = yaml_content
            self.param_publisher.publish(msg)
            _LOGGER.debug(
                "Published %d characters to /mpc_parameter_updates.", len(yaml_content)
            )
        except Exception as e:
            _LOGGER.exception("Failed to publish MPC parameter updates.")
            self._show_status(f"Error publishing to topic: {e}", error=True)

    def save_and_checkpoint(self):
        """Explicit save: writes to YAML AND updates the reset checkpoint.

        After this, 'Reset All' will restore sliders to these values.
        """
        self.save_to_yaml(update_defaults=True)

    def save_to_yaml(self, update_defaults: bool = False):
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return

        if not self.task_file:
            self._show_status("No file path specified to save.", error=True)
            return

        # Merge current slider_rows into _live_values to capture pending changes
        for key_path_str, row in self.slider_rows.items():
            self._live_values[key_path_str] = row.get_value()

        # Build updates from ALL live values (all tabs, not just the active one)
        updates = []
        for key_path_str, val in self._live_values.items():
            parts = key_path_str.split(".")
            updates.append((parts, val))

        try:
            # Never create .bak here — the backup was already created at
            # file-load time in load_file().
            success = update_yaml_values_in_place(
                self.task_file, updates, create_backup=False
            )
            if success:
                if update_defaults:
                    # Explicit "Save to YAML": update the reset checkpoint
                    # so "Reset All" returns to these values.
                    for key, val in self._live_values.items():
                        # Update defaults for visible sliders
                        if key in self.slider_rows:
                            self.slider_rows[key].default_value = val
                    # Also store defaults for non-visible tabs
                    self._default_values = dict(self._live_values)
                for row in self.slider_rows.values():
                    row._update_highlight()

                self._show_status(f"✓ Saved to {os.path.basename(self.task_file)}")
                if self.on_params_updated:
                    self.on_params_updated(self.task_file)
            else:
                self._show_status("Failed to save YAML file.", error=True)
        except Exception as e:
            self._show_status(f"Error saving: {e}", error=True)

    def _show_status(self, msg: str, error: bool = False):
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        self.after(5000, lambda: self.status_label.configure(text=""))
