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

import tkinter as tk
from tkinter import ttk


class SliderRow(ttk.Frame):
    """
    A single parameter row containing:
    - Parameter name label
    - Horizontal ttk.Scale slider
    - Synchronized numeric Entry box (allows precise typing)
    - Reset button (restores default value)
    - Visual indicator when modified from default value
    """

    def __init__(
        self,
        parent,
        name: str,
        initial_value: float,
        min_val: float = 0.0,
        max_val: float = 100.0,
        unit: str = "",
        on_change=None,
        label_width: int = 24,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.name = name
        self.default_value = float(initial_value)
        self.current_value = float(initial_value)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.unit = unit
        self.on_change = on_change
        self._updating = False

        # Name label
        display_name = name + (f" [{unit}]" if unit else "")
        self.label = ttk.Label(
            self,
            text=display_name,
            width=label_width,
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.label.pack(side="left", padx=(4, 6))

        # Reset button
        self.reset_btn = ttk.Button(
            self,
            text="↺",
            width=2,
            command=self.reset_to_default,
        )
        self.reset_btn.pack(side="right", padx=(4, 2))

        # Numeric entry
        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(
            self,
            textvariable=self.entry_var,
            width=9,
            font=("Courier", 9),
            justify="right",
        )
        self.entry.pack(side="right", padx=(6, 4))
        self.entry.bind("<Return>", self._on_entry_submit)
        self.entry.bind("<FocusOut>", self._on_entry_submit)

        # Scale slider
        self.scale_var = tk.DoubleVar(value=self.current_value)
        self.scale = ttk.Scale(
            self,
            from_=self.min_val,
            to=self.max_val,
            orient="horizontal",
            variable=self.scale_var,
            command=self._on_scale_change,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=4)

        # Update text entry display
        self._format_entry(self.current_value)

    def _format_entry(self, val: float):
        if abs(val) >= 1000 or (abs(val) < 0.001 and val != 0.0):
            self.entry_var.set(f"{val:.2e}")
        elif abs(val) >= 100:
            self.entry_var.set(f"{val:.1f}")
        elif abs(val) >= 10:
            self.entry_var.set(f"{val:.2f}")
        else:
            self.entry_var.set(f"{val:.3f}")

    def _on_scale_change(self, val_str):
        if self._updating:
            return
        try:
            val = float(val_str)
            self.current_value = val
            self._updating = True
            self._format_entry(val)
            self._updating = False
            self._update_highlight()
            if self.on_change:
                self.on_change(self.name, self.current_value)
        except Exception:
            pass

    def _on_entry_submit(self, event=None):
        if self._updating:
            return
        try:
            val = float(self.entry_var.get().strip())
            # Extend scale bounds if user typed beyond current min/max
            if val > self.max_val:
                self.max_val = val * 1.5
                self.scale.configure(to=self.max_val)
            elif val < self.min_val:
                self.min_val = val * 1.5 if val < 0 else val
                self.scale.configure(from_=self.min_val)

            self.current_value = val
            self._updating = True
            self.scale_var.set(val)
            self._updating = False
            self._update_highlight()
            if self.on_change:
                self.on_change(self.name, self.current_value)
        except ValueError:
            # Revert on invalid entry
            self._format_entry(self.current_value)

    def _update_highlight(self):
        is_modified = abs(self.current_value - self.default_value) > 1e-6
        if is_modified:
            self.label.configure(foreground="#4a90e2")  # Highlight modified
        else:
            self.label.configure(foreground="#ffffff")

    def reset_to_default(self):
        self.set_value(self.default_value)

    def set_value(self, val: float):
        self.current_value = float(val)
        if self.current_value > self.max_val:
            self.max_val = self.current_value * 1.5
            self.scale.configure(to=self.max_val)
        if self.current_value < self.min_val:
            self.min_val = (
                self.current_value * 1.5
                if self.current_value < 0
                else self.current_value
            )
            self.scale.configure(from_=self.min_val)

        self._updating = True
        self.scale_var.set(self.current_value)
        self._format_entry(self.current_value)
        self._updating = False
        self._update_highlight()
        if self.on_change:
            self.on_change(self.name, self.current_value)

    def get_value(self) -> float:
        return self.current_value

    def is_modified(self) -> bool:
        return abs(self.current_value - self.default_value) > 1e-6

    def set_state(self, state: str):
        """Set state ('normal' or 'disabled') for interactive elements."""
        self.scale.configure(state=state)
        self.entry.configure(state=state)
        self.reset_btn.configure(state=state)
