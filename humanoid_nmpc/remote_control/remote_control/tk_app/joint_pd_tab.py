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
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional

from remote_control.tk_app.scrollable_frame import ScrollableFrame
from remote_control.tk_app.slider_row import SliderRow
from remote_control.tk_app.yaml_editor_utils import (
    load_yaml_safe,
    update_yaml_values_in_place,
)


class JointPdGainsTab(ttk.Frame):
    """
    Joint PD Gains tuning tab allowing real-time inspection, slider adjustment,
    master scaling, search filtering, and comment-preserving YAML saving.
    """

    KNOWN_PRESETS = {
        "Unitree G1 (WB)": "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.yaml",
        "DRC Atlas (Centroidal)": "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.yaml",
        "Unitree R1 (Centroidal)": "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/controller/joint_pd_gains.yaml",
        "Unitree G1 (Centroidal)": "robot_models/unitree_g1/g1_centroidal_mpc/config/controller/joint_pd_gains.yaml",
    }

    def __init__(
        self,
        parent,
        pd_gains_file: Optional[str] = None,
        on_gains_updated=None,
        enable_online_tuning: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="TFrame")

        self.pd_gains_file = pd_gains_file
        self.on_gains_updated = on_gains_updated
        self.enable_online_tuning = enable_online_tuning

        self.raw_data: Dict[str, Any] = {}
        self.slider_rows: Dict[str, SliderRow] = {}  # key -> SliderRow
        self.section_frames: Dict[str, ttk.LabelFrame] = {}
        self.scale_buttons: list = []

        self._build_header_ui()
        self._build_master_scale_ui()
        self._build_search_ui()

        # Scrollable container for joint gains
        self.scroll_container = ScrollableFrame(self, bg_color="#2c2c2c")
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if self.pd_gains_file and os.path.exists(self.pd_gains_file):
            self.load_file(self.pd_gains_file)
        else:
            # Try default preset
            default_path = list(self.KNOWN_PRESETS.values())[0]
            if os.path.exists(default_path):
                self.load_file(default_path)

    def _build_header_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))

        # Preset selector
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

        # File path entry & Browse
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

        self.save_btn = ttk.Button(
            toolbar, text="💾 Save to YAML", command=self.save_to_yaml
        )
        self.save_btn.pack(side="left", padx=(4, 0))

        # Status banner
        self.status_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 9, "italic"),
            foreground="#27ae60",
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

    def _build_master_scale_ui(self):
        scale_frame = ttk.LabelFrame(self, text="⚡ Master Quick Scaling (All Joints)")
        scale_frame.pack(fill="x", padx=10, pady=(0, 6))

        # Kp multiplier row
        kp_sub = ttk.Frame(scale_frame)
        kp_sub.pack(fill="x", padx=6, pady=3)

        ttk.Label(
            kp_sub, text="Scale Kp:", font=("Helvetica", 9, "bold"), width=10
        ).pack(side="left")

        btn_state = "normal" if self.enable_online_tuning else "disabled"
        for factor in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            btn = ttk.Button(
                kp_sub,
                text=f"{factor}x",
                width=5,
                command=lambda f=factor: self._scale_all("kp", f),
                state=btn_state,
            )
            btn.pack(side="left", padx=2)
            self.scale_buttons.append(btn)

        # Kd multiplier row
        kd_sub = ttk.Frame(scale_frame)
        kd_sub.pack(fill="x", padx=6, pady=(0, 4))

        ttk.Label(
            kd_sub, text="Scale Kd:", font=("Helvetica", 9, "bold"), width=10
        ).pack(side="left")

        for factor in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            btn = ttk.Button(
                kd_sub,
                text=f"{factor}x",
                width=5,
                command=lambda f=factor: self._scale_all("kd", f),
                state=btn_state,
            )
            btn.pack(side="left", padx=2)
            self.scale_buttons.append(btn)

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
            title="Select joint_pd_gains.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.load_file(selected)

    def load_file(self, file_path: str):
        self.pd_gains_file = os.path.abspath(file_path)
        self.path_var.set(self.pd_gains_file)
        self.raw_data = load_yaml_safe(self.pd_gains_file)
        self._populate_sliders()
        self._show_status(f"Loaded: {os.path.basename(self.pd_gains_file)}")

    def reload_file(self):
        if self.pd_gains_file and os.path.exists(self.pd_gains_file):
            self.load_file(self.pd_gains_file)

    def _classify_joint_section(self, joint_name: str) -> str:
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

    def _populate_sliders(self):
        # Clear existing widgets
        for widget in self.scroll_container.scrollable_content.winfo_children():
            widget.destroy()

        self.slider_rows.clear()
        self.section_frames.clear()

        # 1. Default Gains
        def_gains = self.raw_data.get("default_gains", self.raw_data.get("default", {}))
        if def_gains:
            def_frame = ttk.LabelFrame(
                self.scroll_container.scrollable_content, text="⚙ Default Gains"
            )
            def_frame.pack(fill="x", padx=6, pady=4)

            def_kp = float(def_gains.get("kp", 150.0))
            def_kd = float(def_gains.get("kd", 10.0))

            row_kp = SliderRow(
                def_frame,
                name="default_kp",
                initial_value=def_kp,
                min_val=0.0,
                max_val=max(def_kp * 3.0, 1000.0),
                unit="N·m/rad",
                label_width=20,
            )
            row_kp.pack(fill="x", padx=4, pady=2)
            self.slider_rows["default_gains.kp"] = row_kp

            row_kd = SliderRow(
                def_frame,
                name="default_kd",
                initial_value=def_kd,
                min_val=0.0,
                max_val=max(def_kd * 3.0, 100.0),
                unit="N·m·s/rad",
                label_width=20,
            )
            row_kd.pack(fill="x", padx=4, pady=2)
            self.slider_rows["default_gains.kd"] = row_kd

        # 2. Joint Gains grouped by limb
        joint_gains = self.raw_data.get("joint_gains", {})
        if not joint_gains:
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

        for jname, gdict in joint_gains.items():
            if not isinstance(gdict, dict):
                continue

            sec_name = self._classify_joint_section(jname)
            parent_frame = self.section_frames.get(
                sec_name, self.section_frames["Other Joints"]
            )

            # Container card for this joint
            joint_card = ttk.Frame(parent_frame)
            joint_card.pack(fill="x", padx=4, pady=2)

            kp_val = float(gdict.get("kp", 150.0))
            kd_val = float(gdict.get("kd", 10.0))

            kp_max = max(kp_val * 3.0, 500.0)
            kd_max = max(kd_val * 3.0, 50.0)

            # Row for Kp
            row_kp = SliderRow(
                joint_card,
                name=f"{jname} (Kp)",
                initial_value=kp_val,
                min_val=0.0,
                max_val=kp_max,
                unit="N·m/rad",
                label_width=26,
            )
            row_kp.pack(fill="x", padx=2, pady=1)
            self.slider_rows[f"joint_gains.{jname}.kp"] = row_kp

            # Row for Kd
            row_kd = SliderRow(
                joint_card,
                name=f"{jname} (Kd)",
                initial_value=kd_val,
                min_val=0.0,
                max_val=kd_max,
                unit="N·m·s/rad",
                label_width=26,
            )
            row_kd.pack(fill="x", padx=2, pady=1)
            self.slider_rows[f"joint_gains.{jname}.kd"] = row_kd

        # Pack only non-empty section frames
        for sec in section_order:
            lf = self.section_frames[sec]
            if lf.winfo_children():
                lf.pack(fill="x", padx=6, pady=4)

        if not self.enable_online_tuning:
            for row in self.slider_rows.values():
                row.set_state("disabled")

    def set_online_tuning_enabled(self, enabled: bool):
        """Dynamically enable or disable online tuning in the GUI."""
        self.enable_online_tuning = bool(enabled)
        if not self.enable_online_tuning:
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
            if hasattr(self, "save_btn"):
                self.save_btn.configure(state="disabled")
            if hasattr(self, "reset_btn"):
                self.reset_btn.configure(state="disabled")
            for btn in self.scale_buttons:
                btn.configure(state="disabled")
            for row in self.slider_rows.values():
                row.set_state("disabled")
        else:
            self.warning_banner.pack_forget()
            if hasattr(self, "save_btn"):
                self.save_btn.configure(state="normal")
            if hasattr(self, "reset_btn"):
                self.reset_btn.configure(state="normal")
            for btn in self.scale_buttons:
                btn.configure(state="normal")
            for row in self.slider_rows.values():
                row.set_state("normal")

    def _scale_all(self, gain_type: str, factor: float):
        """Scales all Kp or Kd gains across all joints by factor."""
        if not self.enable_online_tuning:
            return
        suffix = f".{gain_type}"
        for key, row in self.slider_rows.items():
            if key.endswith(suffix):
                new_val = row.default_value * factor
                row.set_value(new_val)
        self._show_status(f"Scaled all {gain_type.upper()} gains by {factor}x")

    def reset_all_defaults(self):
        if not self.enable_online_tuning:
            return
        for row in self.slider_rows.values():
            row.reset_to_default()
        self._show_status("All gains reset to loaded defaults")

    def save_to_yaml(self):
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return

        if not self.pd_gains_file:
            self._show_status("No file path specified to save.", error=True)
            return

        updates = []
        for key_path_str, row in self.slider_rows.items():
            parts = key_path_str.split(".")
            val = row.get_value()
            updates.append((parts, val))

        try:
            success = update_yaml_values_in_place(
                self.pd_gains_file, updates, create_backup=True
            )
            if success:
                # Update default values to current so modified highlights clear
                for row in self.slider_rows.values():
                    row.default_value = row.current_value
                    row._update_highlight()

                self._show_status(
                    f"✓ Saved to {os.path.basename(self.pd_gains_file)} (backup created)"
                )
                if self.on_gains_updated:
                    self.on_gains_updated(self.pd_gains_file)
            else:
                self._show_status("Failed to save YAML file.", error=True)
        except Exception as e:
            self._show_status(f"Error saving: {e}", error=True)

    def _show_status(self, msg: str, error: bool = False):
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        self.after(5000, lambda: self.status_label.configure(text=""))
