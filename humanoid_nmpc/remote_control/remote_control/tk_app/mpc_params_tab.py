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
from typing import Any, Callable, Dict, List, Optional, Tuple

from remote_control.tk_app.scrollable_frame import ScrollableFrame
from remote_control.tk_app.slider_row import SliderRow
from remote_control.tk_app.yaml_param_tree import MATRIX_KEY, tunables as read_tunables
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
        "EngineAI SA01 (Centroidal)": "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml",
    }

    # Measured contact state of the controller, selected by name in the task file (robot_model/ContactEstimatorRegistry.h)
    # and hot-reloadable through the parameter topic. The Base Controller tab's checkbox (set_cheater_contact_estimator)
    # switches between the simulator's ground truth and every contact point touching (the historical behaviour, with
    # which phase resetting must stay off).
    # LINT.IfChange(contact_estimator_gui)
    CONTACT_ESTIMATOR_KEY = "contactEstimator"
    CHEATER_SIM_CONTACT_ESTIMATOR = "cheater_sim"
    CONTACT_ESTIMATOR_WHEN_UNCHECKED = "always_in_contact"
    # LINT.ThenChange(//robot_runtime/robot_model/src/ContactEstimatorRegistry.cpp:contact_estimator_names, //robot_runtime/mujoco_sim_interface/src/CheaterSimContactEstimator.cpp:cheater_sim_contact_estimator_name)

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
        # The contact planner's own file (contact_planning.yaml next to the task file), None while the block is inline.
        self.contact_planning_file: Optional[str] = None
        self.on_params_updated = on_params_updated
        self._explicit_online_tuning = enable_online_tuning
        self.enable_online_tuning = (
            True if enable_online_tuning is None else enable_online_tuning
        )

        self.raw_data: Dict[str, Any] = {}
        self.slider_rows: Dict[str, SliderRow] = {}  # key_path_str -> SliderRow
        self.comment_map: Dict[str, str] = {}  # "(i,i)" -> comment description
        self._tunables = (
            []
        )  # every numeric leaf of the loaded configuration (yaml_param_tree.Tunable)
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
        # Called (no arguments) whenever the contact estimator selection changes: file loaded, reset, or set through
        # set_cheater_contact_estimator. The Base Controller tab keeps its checkbox in sync with it.
        self.on_contact_estimator_changed: Optional[Callable[[], None]] = None

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
        self._notify_contact_estimator_changed()

    # ── Contact estimator selection (checkbox on the Base Controller tab) ──────────────────────────────────────
    def has_contact_estimator_selection(self) -> bool:
        """True when the loaded task file selects a contact estimator by name (`contactEstimator`)."""
        return self.CONTACT_ESTIMATOR_KEY in self.raw_data

    def selected_contact_estimator(self) -> Optional[str]:
        """The live contact estimator name, or None without a selection in the file."""
        return self._live_values.get(self.CONTACT_ESTIMATOR_KEY)

    def is_cheater_contact_estimator_selected(self) -> bool:
        return self.selected_contact_estimator() == self.CHEATER_SIM_CONTACT_ESTIMATOR

    def set_cheater_contact_estimator(self, enabled: bool):
        """Selects `cheater_sim` (True) or `always_in_contact` (False) and publishes it like a slider change (the name
        reaches the simulator through the parameter topic; 'Save to YAML' writes it into the file). Ignored while
        online tuning is off or the file has no selection."""
        if not self.enable_online_tuning or not self.has_contact_estimator_selection():
            self._notify_contact_estimator_changed()
            return
        name = (
            self.CHEATER_SIM_CONTACT_ESTIMATOR
            if enabled
            else self.CONTACT_ESTIMATOR_WHEN_UNCHECKED
        )
        self._on_any_slider_change(self.CONTACT_ESTIMATOR_KEY, name)
        self._notify_contact_estimator_changed()

    def _notify_contact_estimator_changed(self):
        if self.on_contact_estimator_changed is not None:
            self.on_contact_estimator_changed()

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

        # The categories are the configuration's own top-level blocks, discovered when a file is loaded. Nothing about
        # any particular robot or cost is written down here: a block added to the YAML becomes a category, and a
        # parameter added to a block becomes a slider in it.
        self.active_category = tk.StringVar(value="")
        self.categories = []

        # The buttons flow into as many rows as the width of the tab allows. A single packed row needs more width than
        # the GUI's default window (960 px), and tkinter silently drops the buttons that do not fit, so the last
        # categories would be unreachable without resizing the window.
        self.category_buttons = []
        self._category_nav_frame = nav_frame
        self._category_nav_columns = 0
        nav_frame.bind(
            "<Configure>", lambda event: self._reflow_category_buttons(event.width)
        )
        self._reflow_category_buttons(nav_frame.winfo_reqwidth())

    def _reflow_category_buttons(self, available_width: int):
        """Kept for the window-resize hook: the drop-down holds every block whatever the width, so nothing to reflow."""
        return

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

        # The contact planner's parameters live in contact_planning.yaml next to the task file (the C++ interface
        # resolves it the same way); a task file that still carries the block inline is read as before.
        self.contact_planning_file = self._resolve_contact_planning_file(self.task_file)
        if self.contact_planning_file:
            planner_data = load_yaml_safe(self.contact_planning_file).get(
                "contact_planning"
            )
            if planner_data:
                self.raw_data["contact_planning"] = planner_data

        # Create a backup of the original files at load time (before any
        # slider-driven writes) so "Reset All" can always restore them.
        import shutil

        for original in (self.task_file, self.contact_planning_file):
            if not original:
                continue
            bak_path = original + ".bak"
            if not os.path.exists(bak_path):
                shutil.copy2(original, bak_path)

        # The file's contact estimator selection is the live value and the reset checkpoint of the checkbox.
        self._live_values.pop(self.CONTACT_ESTIMATOR_KEY, None)
        self._default_values.pop(self.CONTACT_ESTIMATOR_KEY, None)
        if self.CONTACT_ESTIMATOR_KEY in self.raw_data:
            name = str(self.raw_data[self.CONTACT_ESTIMATOR_KEY]).strip().lower()
            self._live_values[self.CONTACT_ESTIMATOR_KEY] = name
            self._default_values[self.CONTACT_ESTIMATOR_KEY] = name
        self._notify_contact_estimator_changed()

        if self._explicit_online_tuning is not None:
            self.enable_online_tuning = self._explicit_online_tuning
        elif "enableOnlineTuning" in self.raw_data:
            self.enable_online_tuning = bool(self.raw_data["enableOnlineTuning"])
        elif "enable_online_tuning" in self.raw_data:
            self.enable_online_tuning = bool(self.raw_data["enable_online_tuning"])
        self.set_online_tuning_enabled(self.enable_online_tuning)
        self._parse_yaml_comments(self.task_file)
        # Every tunable of both files, labelled from their own trailing comments. This is the whole model the GUI is
        # built from: no parameter is named anywhere in this module.
        self._tunables = read_tunables(self.task_file)
        if self.contact_planning_file:
            planner = load_yaml_safe(self.contact_planning_file).get("contact_planning")
            if planner:
                self._tunables += read_tunables(
                    self.contact_planning_file, root={"contact_planning": planner}
                )
        self._refresh_categories()
        self._render_active_category()
        self._show_status(f"Loaded: {os.path.basename(self.task_file)}")

    def reload_file(self):
        if self.task_file and os.path.exists(self.task_file):
            self.load_file(self.task_file)
            # Publish the reloaded values so the MPC syncs with the sliders
            self._publish_to_topic()

    @staticmethod
    def _resolve_contact_planning_file(task_file: str) -> Optional[str]:
        """Path of contact_planning.yaml next to the task file, or None when there is none (block inline)."""
        if not task_file:
            return None
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(task_file)), "contact_planning.yaml"
        )
        return candidate if os.path.isfile(candidate) else None

    def _split_updates(self, updates):
        """Splits (key_path, value) updates into those of the task file and those of the planner's own file."""
        if not self.contact_planning_file:
            return updates, []
        planner_updates = [u for u in updates if u[0] and u[0][0] == "contact_planning"]
        task_updates = [
            u for u in updates if not (u[0] and u[0][0] == "contact_planning")
        ]
        return task_updates, planner_updates

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

    ROOT_CATEGORY = "(top level)"

    def _refresh_categories(self):
        """Rebuilds the category list from the loaded configuration, in the order the file lists its blocks."""
        blocks = []
        for tunable in self._tunables:
            block = tunable.path[0] if len(tunable.path) > 1 else self.ROOT_CATEGORY
            if block not in blocks:
                blocks.append(block)
        self.categories = blocks
        if self.active_category.get() not in blocks:
            self.active_category.set(blocks[0] if blocks else "")
        if hasattr(self, "_category_nav_frame"):
            self._build_category_buttons()

    def _build_category_buttons(self):
        """The category selector.

        A configuration has as many blocks as it has, and the DRC Atlas file has twenty-five of them, so a row of radio
        buttons no longer fits any sensible window: tkinter silently drops the ones that do not, which would make the
        last blocks unreachable. A drop-down holds any number of them and stays the same size.
        """
        for widget in self._category_nav_frame.winfo_children():
            widget.destroy()
        self.category_buttons = []
        if not self.categories:
            return
        ttk.Label(self._category_nav_frame, text="Block:").pack(
            side="left", padx=(0, 6)
        )
        selector = ttk.Combobox(
            self._category_nav_frame,
            textvariable=self.active_category,
            values=list(self.categories),
            state="readonly",
            width=34,
        )
        selector.pack(side="left")
        selector.bind(
            "<<ComboboxSelected>>", lambda _event: self._render_active_category()
        )
        self.category_selector = selector
        # Kept so that the older reflow hook and any caller that counted buttons still see one widget per category.
        self.category_buttons = [selector]

    @staticmethod
    def _slider_key(tunable):
        """The dotted path the YAML writer expects: matrix keys are quoted, because that is how the file spells them."""
        parts = [
            (
                ('"%s"' % part)
                if MATRIX_KEY.match(part) and not part.startswith('"')
                else part
            )
            for part in tunable.path
        ]
        return ".".join(parts)

    def render_category_containing(self, slider_key: str) -> bool:
        """Renders the block a slider belongs to, so the caller can reach that slider's live row.

        The categories are the configuration's own blocks, so the block of a key is simply its first path segment.
        """
        block = slider_key.split(".")[0] if "." in slider_key else self.ROOT_CATEGORY
        if block not in self.categories:
            return False
        self.active_category.set(block)
        self._render_active_category()
        return slider_key in self.slider_rows

    def _render_block(self, category: str):
        """Sliders for every tunable of one block of the configuration, grouped by their sub-blocks, in file order.

        There is nothing here about any particular parameter. The label of a slider is the trailing comment the file
        carries next to it, its range comes from its own magnitude, and a parameter that nobody has written a line of
        code about renders exactly like one that has.
        """
        wanted = [
            tunable
            for tunable in self._tunables
            if (tunable.path[0] if len(tunable.path) > 1 else self.ROOT_CATEGORY)
            == category
        ]
        if not wanted:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="No tunable parameters in this block.",
            ).pack(anchor="w", padx=8, pady=8)
            return

        frames = {}
        for tunable in wanted:
            # One frame per sub-block, so a nested block (gait_limits, blend, a matrix) reads as a group.
            group = ".".join(tunable.path[:-1]) or category
            frame = frames.get(group)
            if frame is None:
                frame = ttk.LabelFrame(
                    self.scroll_container.scrollable_content, text="• " + group
                )
                frame.pack(fill="x", padx=6, pady=4)
                frames[group] = frame
            key = self._slider_key(tunable)
            row = SliderRow(
                frame,
                name=tunable.label,
                initial_value=tunable.value,
                min_val=tunable.minimum,
                max_val=tunable.maximum,
                label_width=46,
                on_change=self._on_any_slider_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[key] = row

    def _render_active_category(self):
        cat = self.active_category.get()

        # Save current slider values before destroying them
        for key, row in self.slider_rows.items():
            self._live_values[key] = row.get_value()

        # Clear content container
        for child in self.scroll_container.scrollable_content.winfo_children():
            child.destroy()

        self.slider_rows.clear()
        self._render_block(cat)

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

    def reset_all_defaults(self):
        if not self.enable_online_tuning:
            return
        for row in self.slider_rows.values():
            row.reset_to_default()
        if self.CONTACT_ESTIMATOR_KEY in self._default_values:
            self._live_values[self.CONTACT_ESTIMATOR_KEY] = self._default_values[
                self.CONTACT_ESTIMATOR_KEY
            ]
            self._notify_contact_estimator_changed()
        # Publish immediately so the MPC picks up the reset values
        self._publish_to_topic()
        self._show_status("All parameters reset to loaded defaults")

    def _on_any_slider_change(self, name: str, value):
        """Called on every slider move (or checkbox toggle, with the selected name); debounces publish to ROS topic.

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

        task_updates, planner_updates = self._split_updates(updates)
        for key_path, value in task_updates:
            lines = _update_single_key(lines, key_path, value)

        # The planner's file is a second YAML document with its own top-level key; appended to the task file's
        # content it parses as one document, which is what the C++ updater expects on the topic.
        if self.contact_planning_file and os.path.exists(self.contact_planning_file):
            with open(self.contact_planning_file, "r") as f:
                planner_lines = f.readlines()
            for key_path, value in planner_updates:
                planner_lines = _update_single_key(planner_lines, key_path, value)
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append("\n")
            lines.extend(planner_lines)

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
            # file-load time in load_file(). The planner's sliders go to its own file.
            task_updates, planner_updates = self._split_updates(updates)
            success = update_yaml_values_in_place(
                self.task_file, task_updates, create_backup=False
            )
            if success and planner_updates:
                success = update_yaml_values_in_place(
                    self.contact_planning_file, planner_updates, create_backup=False
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
