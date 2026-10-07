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

"""One parameter row of the tuning tabs: a label, a slider, an entry box and a reset button."""

from collections.abc import Callable
import math
from tkinter import ttk
import tkinter as tk
from typing import Any

# The lower end of a logarithmic slider without a positive one, as a fraction of its upper end.
_LOG_SCALE_FLOOR = 1e-4


class SliderRow(ttk.Frame):
    """One parameter row of a tuning tab, which the operator drags, types into or resets.

    It contains:
    - Parameter name label
    - Horizontal ttk.Scale slider
    - Synchronized numeric Entry box (allows precise typing)
    - Reset button (restores default value)
    - Visual indicator when modified from default value

    By default a value typed beyond the slider's range WIDENS the range, which suits tuning parameters whose useful range
    is not known in advance. `clamp_to_range=True` instead holds every value - typed, set or dragged - inside
    [min_val, max_val] and shows the clamped value, for rows whose range is a hard limit on the other side of the wire.
    A value that is not a finite number is rejected either way and the entry reverts. `log_scale=True` spaces the
    slider's travel by decades (a weight or a tolerance spanning several of them), and `integer=True` holds the value
    to whole numbers.

    The entry shows the value rounded. Leaving the entry, or pressing Return, without changing its text changes nothing:
    only a text the operator typed is taken, so the rounding never reaches the value.

    `unset=True` is a parameter that has no value yet (an optional field the file leaves out, whose absence means
    something of its own): its entry is empty and its slider rests at `initial_value`, a placeholder, until the
    operator drags it or types a value. Resetting it returns it to unset, which `on_change` hears as None.

    Args:
        parent: the widget the row is packed into.
        name: the parameter's name, shown on the label and passed to `on_change`.
        initial_value: the value the row starts at, and its default (the reset button restores it); with `unset`, only
            where the slider rests.
        min_val: the lower end of the slider's range.
        max_val: the upper end of the slider's range.
        unit: shown in brackets after the name; empty: no unit.
        on_change: called with (name, value) after every change of the value, and with (name, None) when the row is reset
            to unset; None: nothing is called.
        label_width: the width of the name label, in characters.
        clamp_to_range: whether [min_val, max_val] is a hard limit rather than a range a typed value widens.
        log_scale: whether the slider's travel is logarithmic (min_val must then be positive; a smaller one is raised).
        integer: whether the value is a whole number (dragged and typed values are rounded).
        unset: whether the row starts, and resets to, having no value.
        **kwargs: the options of the row's ttk.Frame.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        initial_value: float,
        min_val: float = 0.0,
        max_val: float = 100.0,
        unit: str = "",
        on_change: Callable[[str, float | None], None] | None = None,
        label_width: int = 24,
        clamp_to_range: bool = False,
        *,
        log_scale: bool = False,
        integer: bool = False,
        unset: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)

        self.name = name
        self.integer = integer
        self.log_scale = log_scale
        self.default_value = self._whole(float(initial_value))
        self.current_value = self.default_value
        self.max_val = float(max_val)
        self.min_val = float(min_val)
        if log_scale and not self.min_val > 0.0:
            self.min_val = (
                self.max_val * _LOG_SCALE_FLOOR
                if self.max_val > 0.0
                else _LOG_SCALE_FLOOR
            )
        self.unit = unit
        self.on_change = on_change
        self.clamp_to_range = clamp_to_range
        self._updating = False
        # Whether the row has no value now, and whether its default (what reset restores) is having none.
        self._unset = unset
        self._default_unset = unset
        # The text the entry was last given by the row (_format_entry()); a submit of the same text is no change.
        self._shown_text = ""

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
        self.scale_var = tk.DoubleVar(value=self._position(self.current_value))
        self.scale = ttk.Scale(
            self,
            from_=self._position(self.min_val),
            to=self._position(self.max_val),
            orient="horizontal",
            variable=self.scale_var,
            command=self._on_scale_change,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=4)

        # Update text entry display
        self._show_current()

    def _position(self, value: float) -> float:
        """Where `value` sits on the slider: the value itself, or its decade on a logarithmic slider."""
        if not self.log_scale:
            return value
        return math.log10(max(value, self.min_val))

    def _value_at(self, position: float) -> float:
        """The value at a position of the slider (the inverse of _position()), whole for an integer row."""
        value = 10.0**position if self.log_scale else position
        return self._whole(value)

    def _whole(self, value: float) -> float:
        return float(round(value)) if self.integer else value

    def _format_entry(self, val: float) -> None:
        if self.integer:
            text = f"{int(val)}"
        elif abs(val) >= 1000 or (abs(val) < 0.001 and val != 0.0):
            text = f"{val:.2e}"
        elif abs(val) >= 100:
            text = f"{val:.1f}"
        elif abs(val) >= 10:
            text = f"{val:.2f}"
        else:
            text = f"{val:.3f}"
        self._set_entry_text(text)

    def _set_entry_text(self, text: str) -> None:
        self._shown_text = text
        self.entry_var.set(text)

    def _show_current(self) -> None:
        """Shows the current value in the entry, or an empty entry while the row has none."""
        if self._unset:
            self._set_entry_text("")
        else:
            self._format_entry(self.current_value)

    def _on_scale_change(self, val_str: str) -> None:
        """The slider's callback: takes the dragged value, shows it in the entry and reports it.

        Args:
            val_str: the slider's value, as Tk passes it (a string).
        """
        if self._updating:
            return
        try:
            val = self._value_at(float(val_str))
            self.current_value = val
            self._unset = False
            self._updating = True
            self._format_entry(val)
            self._updating = False
            self._update_highlight()
            if self.on_change:
                self.on_change(self.name, self.current_value)
        # pylint: disable-next=broad-exception-caught  # A Tk callback must not raise.
        except Exception:
            pass

    def _on_entry_submit(self, event: tk.Event | None = None) -> None:
        """The entry's callback (Return, focus out): takes the typed value, or reverts the entry when it is not one.

        Args:
            event: the Tk event; None when called directly.
        """
        del event  # Unused.
        if self._updating:
            return
        text = self.entry_var.get().strip()
        if text == self._shown_text:
            return  # nothing typed: the shown text is the value rounded (or empty for an unset row), not a new value
        try:
            val = self._whole(float(text))
            if not math.isfinite(val):
                raise ValueError(f"{val} is not a finite number")
            if self.clamp_to_range:
                val = min(max(val, self.min_val), self.max_val)
            else:
                self._widen_to(val)

            self.current_value = val
            self._unset = False
            self._updating = True
            self.scale_var.set(self._position(val))
            # Shows what was accepted, which differs from what was typed when it was clamped.
            self._format_entry(val)
            self._updating = False
            self._update_highlight()
            if self.on_change:
                self.on_change(self.name, self.current_value)
        except ValueError:
            # Revert on invalid entry
            self._show_current()

    def _widen_to(self, value: float) -> None:
        """Widens the slider's range to take `value` (a typed or set value beyond it)."""
        if value > self.max_val:
            self.max_val = value * 1.5
            self.scale.configure(to=self._position(self.max_val))
        if value < self.min_val and not (self.log_scale and value <= 0.0):
            self.min_val = value * 1.5 if value < 0 else value
            self.scale.configure(from_=self._position(self.min_val))

    def _update_highlight(self) -> None:
        if self.is_modified():
            self.label.configure(foreground="#4a90e2")  # Highlight modified
        else:
            self.label.configure(foreground="#ffffff")

    def reset_to_default(self) -> None:
        """Restores the default: its value, or no value for a row whose default is unset (on_change hears None)."""
        if not self._default_unset:
            self.set_value(self.default_value)
            return
        self._unset = True
        self.current_value = self.default_value
        self._updating = True
        self.scale_var.set(self._position(self.current_value))
        self._show_current()
        self._updating = False
        self._update_highlight()
        if self.on_change:
            self.on_change(self.name, None)

    def set_default(self, value: float) -> None:
        """Makes `value` the row's default (what the reset button restores), and shows whether the row differs from it."""
        self.default_value = float(value)
        self._default_unset = False
        self._update_highlight()

    def mark_saved(self) -> None:
        """Makes what the row shows its default, as after a save: its value, or no value while it has none."""
        self.default_value = self.current_value
        self._default_unset = self._unset
        self._update_highlight()

    def set_value(self, val: float) -> None:
        """Sets the value as if the operator had typed it, and reports it to `on_change`.

        A value that is not a finite number is ignored. With `clamp_to_range` the value is clamped into the range;
        otherwise a value outside it widens the range.

        Args:
            val: the new value.
        """
        val = self._whole(float(val))
        if not math.isfinite(val):
            return
        if self.clamp_to_range:
            val = min(max(val, self.min_val), self.max_val)
        self.current_value = val
        self._unset = False
        self._widen_to(val)

        self._updating = True
        self.scale_var.set(self._position(self.current_value))
        self._format_entry(self.current_value)
        self._updating = False
        self._update_highlight()
        if self.on_change:
            self.on_change(self.name, self.current_value)

    def get_value(self) -> float:
        """The value; while the row has none (has_value()), the placeholder its slider rests at."""
        return self.current_value

    def has_value(self) -> bool:
        """False while the row is unset: the operator has given it no value yet."""
        return not self._unset

    def is_modified(self) -> bool:
        if self._unset or self._default_unset:
            return self._unset != self._default_unset
        return abs(self.current_value - self.default_value) > 1e-6

    def set_state(self, state: str) -> None:
        """Set state ('normal' or 'disabled') for interactive elements."""
        self.scale.configure(state=state)
        self.entry.configure(state=state)
        self.reset_btn.configure(state=state)
