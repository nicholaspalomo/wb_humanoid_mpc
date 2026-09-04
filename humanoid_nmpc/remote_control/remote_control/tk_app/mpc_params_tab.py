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
            toolbar, text="💾 Save to YAML", command=self.save_to_yaml
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
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'Q."{key}"'] = row

        # Base Pose Weights (6..11)
        base_frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Base Pose Tracking (6..11)",
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
        )
        row.pack(fill="x", padx=4, pady=2)
        self.slider_rows["terminalCostScaling"] = row

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
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f'Q_final."{key}"'] = row

    def _render_task_space_costs(self):
        # Foot tracking weights
        foot_costs = self.raw_data.get("task_space_foot_cost_weights", {})
        if foot_costs:
            f_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text="• Task Space Foot Costs"
            )
            f_frame.pack(fill="x", padx=6, pady=4)
            for k, v in foot_costs.items():
                if isinstance(v, (int, float)):
                    val = float(v)
                    row = SliderRow(
                        f_frame,
                        name=k,
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 100.0),
                        label_width=24,
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
                if isinstance(v, (int, float)):
                    val = float(v)
                    row = SliderRow(
                        t_frame,
                        name=f"torso_{k}",
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 100.0),
                        label_width=24,
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
            )
            row.pack(fill="x", padx=4, pady=2)
            self.slider_rows["icp_cost_weights.icpErrorWeight"] = row

    def _render_constraints_and_barriers(self):
        # Foot constraint gains
        foot_cfg = self.raw_data.get("model_settings", {}).get("foot_constraint", {})
        if foot_cfg:
            fc_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text="• Foot Constraint Gains"
            )
            fc_frame.pack(fill="x", padx=6, pady=4)
            for k, v in foot_cfg.items():
                if isinstance(v, (int, float)):
                    val = float(v)
                    row = SliderRow(
                        fc_frame,
                        name=k,
                        initial_value=val,
                        min_val=0.0,
                        max_val=max(val * 3.0, 50.0),
                        label_width=26,
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
                if isinstance(v, (int, float)):
                    val = float(v)
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
                    name=f"frictionCone_{k}",
                    initial_value=val,
                    min_val=0.01,
                    max_val=max(val * 4.0, 20.0),
                    label_width=26,
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"contacts.frictionForceConeSoftConstraint.{k}"] = row

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
                )
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[f"collision_constraint.{k}"] = row

    def reset_all_defaults(self):
        if not self.enable_online_tuning:
            return
        for row in self.slider_rows.values():
            row.reset_to_default()
        self._show_status("All parameters reset to loaded defaults")

    def save_to_yaml(self):
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return

        if not self.task_file:
            self._show_status("No file path specified to save.", error=True)
            return

        updates = []
        for key_path_str, row in self.slider_rows.items():
            parts = key_path_str.split(".")
            val = row.get_value()
            updates.append((parts, val))

        try:
            success = update_yaml_values_in_place(
                self.task_file, updates, create_backup=True
            )
            if success:
                for row in self.slider_rows.values():
                    row.default_value = row.current_value
                    row._update_highlight()

                self._show_status(
                    f"✓ Saved to {os.path.basename(self.task_file)} (backup created)"
                )
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
