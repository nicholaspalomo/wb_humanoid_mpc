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

"""Command limits and reference defaults, read from the robot's ``command/reference.yaml``."""

import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Tuple

from remote_control.tk_app.scrollable_frame import ScrollableFrame
from remote_control.tk_app.slider_row import SliderRow
from remote_control.tk_app.yaml_editor_utils import (
    load_yaml_safe,
    update_yaml_values_in_place,
)


class CommandLimitsTab(ttk.Frame):
    """The scalars of ``reference.yaml``: the command limits, the command ramps and the reference defaults.

    These are the parameters that bound what the operator can ask for - the largest velocity a full stick means, the
    acceleration the reference ramps at, the nominal stance height. They do not travel on the parameter topic: the C++
    side reads them from this file, and ``MpcParameterUpdaterModule`` watches the file and reloads the consumers that
    registered for it (``TargetTrajectoriesCalculatorBase`` and ``ProceduralMpcMotionManager``). Saving is therefore
    the whole mechanism - there is no publisher here - but a running controller does pick the change up, about a
    second later. The keyboard command node is the exception: it reads the file once at start-up.

    Like the contact planner's block in the MPC parameters tab, the rendering is driven by the file: every numeric
    scalar becomes a slider, so a limit added to ``reference.yaml`` appears here without any code being written. Only
    the ranges are worth stating by hand, and only where the generic guess would be poor.
    """

    #: Slider ranges (min, max) by dotted path. Absent means the generic 0..4x of the current value.
    RANGES: Dict[str, Tuple[float, float]] = {
        "targetDisplacementVelocity": (0.0, 2.0),
        "targetRotationVelocity": (0.0, 3.0),
        "maxDisplacementVelocityX": (0.0, 2.0),
        "maxDisplacementVelocityY": (0.0, 2.0),
        "maxRotationVelocity": (0.0, 3.0),
        "maxDeltaPelvisHeight": (0.0, 1.0),
        "maxLinearAcceleration": (0.0, 5.0),
        "maxAngularAcceleration": (0.0, 10.0),
        "defaultBaseHeight": (0.3, 1.5),
        "targetJointStateInterpolationTimeConstant": (0.0, 10.0),
    }

    #: Blocks of the file another tab owns. The nominal posture is edited in the Joint Targets tab, where every entry
    #: is labelled with its joint name; a flat list of thirty numbers here would be worse than useless.
    OWNED_ELSEWHERE = {"defaultJointState": "edited in the Joint Targets tab"}

    def __init__(self, parent, reference_file: Optional[str] = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.reference_file = (
            os.path.abspath(reference_file) if reference_file else None
        )
        self.slider_rows: Dict[str, SliderRow] = {}
        self.raw_data = {}
        self.status_var = tk.StringVar(value="")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(
            header,
            text="Command Limits & Reference Defaults",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Not published on the parameter topic: saving writes this file, and the running controller "
            "reloads it about a second later.",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(2, 6))

        self.path_var = tk.StringVar(
            value=self.reference_file or "(no reference.yaml found)"
        )
        ttk.Label(header, textvariable=self.path_var, foreground="#888888").pack(
            anchor="w"
        )

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=8, pady=4)
        ttk.Button(buttons, text="💾 Save to YAML", command=self.save).pack(side="left")
        ttk.Button(buttons, text="↩ Reload", command=self.reload).pack(
            side="left", padx=6
        )
        ttk.Label(buttons, textvariable=self.status_var, foreground="#3c9a3c").pack(
            side="left", padx=10
        )

        self.scroll_container = ScrollableFrame(self)
        self.scroll_container.pack(fill="both", expand=True, padx=8, pady=8)

        self.reload()

    def reload(self):
        """Re-reads the file and rebuilds the sliders from it."""
        for child in self.scroll_container.scrollable_content.winfo_children():
            child.destroy()
        self.slider_rows.clear()
        if not self.reference_file or not os.path.exists(self.reference_file):
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="No reference.yaml was found next to the task file (config/command/reference.yaml).",
            ).pack(anchor="w", padx=6, pady=6)
            return
        self.raw_data = load_yaml_safe(self.reference_file)
        self.path_var.set(self.reference_file)
        self._render()
        self.status_var.set("")

    @staticmethod
    def _numeric_leaves(node, prefix: str = "") -> Dict[str, float]:
        """Every numeric scalar of the file, keyed by its dotted path; booleans are not sliders."""
        leaves: Dict[str, float] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                leaves.update(CommandLimitsTab._numeric_leaves(value, child))
        elif isinstance(node, bool) or node is None:
            pass
        elif isinstance(node, (int, float)):
            leaves[prefix] = node
        return leaves

    def _render(self):
        frame = ttk.LabelFrame(
            self.scroll_container.scrollable_content,
            text="• Command limits and defaults",
        )
        frame.pack(fill="x", padx=6, pady=4)
        rendered = 0
        for key, value in sorted(self._numeric_leaves(self.raw_data).items()):
            if any(
                key == owned or key.startswith(owned + ".")
                for owned in self.OWNED_ELSEWHERE
            ):
                continue
            val = float(value)
            min_val, max_val = self.RANGES.get(key, (0.0, max(abs(val) * 4.0, 1.0)))
            row = SliderRow(
                frame,
                name=key,
                initial_value=val,
                min_val=min(min_val, val),
                max_val=max(max_val, val),
                label_width=42,
                on_change=self._on_change,
            )
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[key] = row
            rendered += 1
        if rendered == 0:
            ttk.Label(frame, text="reference.yaml carries no scalar parameters.").pack(
                anchor="w", padx=6, pady=4
            )

        for owned, reason in sorted(self.OWNED_ELSEWHERE.items()):
            if owned in self.raw_data:
                ttk.Label(
                    self.scroll_container.scrollable_content,
                    text=f"{owned}: {reason}.",
                    foreground="#888888",
                ).pack(anchor="w", padx=10, pady=2)

    def _on_change(self, name: str, value):
        self.status_var.set("unsaved changes")

    def save(self) -> bool:
        """Writes every slider back into the file, preserving its comments and layout."""
        if not self.reference_file:
            return False
        updates = [
            (key.split("."), row.get_value()) for key, row in self.slider_rows.items()
        ]
        if not updates:
            return False
        if update_yaml_values_in_place(
            self.reference_file, updates, create_backup=True
        ):
            for row in self.slider_rows.values():
                row.default_value = row.get_value()
                row._update_highlight()
            self.status_var.set(
                "saved — the running controller reloads it within about a second"
            )
            return True
        self.status_var.set("save failed")
        return False
