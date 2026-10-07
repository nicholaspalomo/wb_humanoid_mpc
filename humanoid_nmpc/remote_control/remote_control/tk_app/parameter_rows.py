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


"""The widget of one parameter of a configuration file, chosen by its kind in the file's schema.

    row = parameter_rows.make_row(frame, spec, label, value, on_change)  # None for a parameter the GUI does not show

A number or an integer is a slider (slider_row.SliderRow, logarithmic where the schema says so), a bool a checkbox, a
choice (an enum, or a string naming a registry entry) a drop-down - typed into where the registry's names are not
known - and a name list (a term list) its names, read-only. Every row has the interface the tuning tabs drive:
`name` (its label), `default_value`, `current_value`, get_value(), set_value() (which reports the change), set_default(),
reset_to_default(), mark_saved() and set_state('normal' or 'disabled'). Nothing here names a parameter.

A parameter the file leaves unset (config_schema.ValueSource.UNSET: an optional field without a default, whose absence
means something of its own, such as "the model's height") gets a row that shows no value until the operator gives it
one. No widget turns it into a value by itself: leaving an entry untouched changes nothing, and resetting the row returns
it to unset, which on_change hears as None (tuned_file.TunedFile.set() then drops the change).
"""

from collections.abc import Callable, Sequence
from tkinter import ttk
import tkinter as tk
from typing import Any, TypeAlias

from remote_control import config_schema
from remote_control.tk_app import slider_row

# What a row calls with its name and its new value after every change.
OnChange: TypeAlias = Callable[[str, Any], None]

_MODIFIED_COLOR = "#4a90e2"
_PLAIN_COLOR = "#ffffff"


class _ValueRow(ttk.Frame):
    """A label, a value widget and a reset button; the subclasses hold the value, None while the row has none."""

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        initial_value: Any,
        on_change: OnChange | None,
        label_width: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.name = name
        self.default_value = initial_value
        self.current_value = initial_value
        self.on_change = on_change
        self.label = ttk.Label(
            self,
            text=name,
            width=label_width,
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.label.pack(side="left", padx=(4, 6))
        self.reset_btn = ttk.Button(
            self, text="↺", width=2, command=self.reset_to_default
        )
        self.reset_btn.pack(side="right", padx=(4, 2))

    def _show(self, value: Any) -> None:
        """Shows `value`; None: no value (an unset parameter)."""
        raise NotImplementedError

    def _changed(self, value: Any) -> None:
        """Takes a value the operator gave, and reports it."""
        self.current_value = value
        self._update_highlight()
        if self.on_change is not None:
            self.on_change(self.name, value)

    def _update_highlight(self) -> None:
        self.label.configure(
            foreground=_MODIFIED_COLOR if self.is_modified() else _PLAIN_COLOR
        )

    def get_value(self) -> Any:
        return self.current_value

    def set_value(self, value: Any) -> None:
        """Shows `value` as if the operator had given it, and reports it to `on_change`."""
        self._show(value)
        self._changed(value)

    def set_default(self, value: Any) -> None:
        """Makes `value` what the reset button restores, and shows whether the row differs from it."""
        self.default_value = value
        self._update_highlight()

    def reset_to_default(self) -> None:
        """Restores the default: its value, or no value for an unset parameter (on_change hears None)."""
        if self.default_value is None:
            self._show(None)
            self._changed(None)
            return
        self.set_value(self.default_value)

    def mark_saved(self) -> None:
        """Makes what the row shows its default, as after a save."""
        self.set_default(self.current_value)

    def has_value(self) -> bool:
        """False while the row is unset: the operator has given it no value yet."""
        return self.current_value is not None

    def is_modified(self) -> bool:
        return bool(self.current_value != self.default_value)

    def set_state(self, state: str) -> None:
        """'normal' or 'disabled' for the interactive widgets."""
        self.reset_btn.configure(state=state)


class CheckRow(_ValueRow):
    """A bool: a checkbox, in its third ("alternate") state while it has no value.

    Args:
        parent: the widget the row is packed into.
        name: its label, passed to `on_change`.
        initial_value: the value it starts at, and its default; None: unset.
        on_change: called with (name, value) after every change; None: nothing is called.
        label_width: the width of the label, in characters.
        **kwargs: the options of the row's ttk.Frame.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        initial_value: bool | None,
        on_change: OnChange | None = None,
        label_width: int = 24,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            parent,
            name,
            None if initial_value is None else bool(initial_value),
            on_change,
            label_width,
            **kwargs,
        )
        self.variable = tk.BooleanVar(value=bool(initial_value))
        self.checkbox = ttk.Checkbutton(
            self, variable=self.variable, command=self._on_toggle
        )
        self.checkbox.pack(side="left", padx=4)
        self._show(self.current_value)

    def _on_toggle(self) -> None:
        self.checkbox.state(["!alternate"])
        self._changed(bool(self.variable.get()))

    def _show(self, value: Any) -> None:
        if value is None:
            self.variable.set(False)
            self.checkbox.state(["alternate"])
        else:
            self.checkbox.state(["!alternate"])
            self.variable.set(bool(value))

    def set_value(self, value: Any) -> None:
        super().set_value(bool(value))

    def set_state(self, state: str) -> None:
        super().set_state(state)
        self.checkbox.configure(state=state)


class ChoiceRow(_ValueRow):
    """A choice among names: a drop-down of them, or an entry for a name when they are not known.

    Args:
        parent: the widget the row is packed into.
        name: its label, passed to `on_change`.
        initial_value: the name it starts at, and its default; None: unset (the drop-down shows no name).
        choices: the names it offers; None: unknown, and the operator types one (the receiver's registry checks it).
        on_change: called with (name, value) after every change; None: nothing is called.
        label_width: the width of the label, in characters.
        **kwargs: the options of the row's ttk.Frame.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        initial_value: str | None,
        choices: Sequence[str] | None = None,
        on_change: OnChange | None = None,
        label_width: int = 24,
        **kwargs: Any,
    ) -> None:
        shown = "" if initial_value is None else str(initial_value)
        super().__init__(
            parent,
            name,
            None if initial_value is None else shown,
            on_change,
            label_width,
            **kwargs,
        )
        self.choices = tuple(choices) if choices is not None else None
        self.variable = tk.StringVar(value=shown)
        values = list(self.choices) if self.choices is not None else [shown]
        self._readonly = self.choices is not None
        self.combobox = ttk.Combobox(
            self,
            textvariable=self.variable,
            values=values,
            state="readonly" if self._readonly else "normal",
            width=28,
        )
        self.combobox.pack(side="left", padx=4)
        self.combobox.bind("<<ComboboxSelected>>", self._on_selected)
        self.combobox.bind("<Return>", self._on_selected)
        self.combobox.bind("<FocusOut>", self._on_selected)

    def _on_selected(self, event: tk.Event | None = None) -> None:
        del event  # Unused.
        value = self.variable.get().strip()
        if not value or value == self.current_value:
            self._show(self.current_value)
            return
        self._changed(value)

    def _show(self, value: Any) -> None:
        self.variable.set("" if value is None else str(value))

    def set_value(self, value: Any) -> None:
        super().set_value(str(value))

    def set_state(self, state: str) -> None:
        super().set_state(state)
        self.combobox.configure(
            state=(
                "disabled"
                if state == "disabled"
                else ("readonly" if self._readonly else "normal")
            )
        )


class NameListRow(ttk.Frame):
    """A name list (the terms of a formulation): its names, shown and not edited.

    Args:
        parent: the widget the row is packed into.
        name: its label.
        names: the names.
        label_width: the width of the label, in characters.
        **kwargs: the options of the row's ttk.Frame.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        names: Sequence[str],
        label_width: int = 24,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.name = name
        self.default_value = tuple(names)
        self.current_value = tuple(names)
        # Never called: the row reports no change. Kept so that every row has one.
        self.on_change: OnChange | None = None
        ttk.Label(
            self,
            text=name,
            width=label_width,
            anchor="w",
            font=("Helvetica", 9, "bold"),
        ).pack(side="left", padx=(4, 6))
        self.names_label = ttk.Label(
            self,
            text=", ".join(names) if names else "(none)",
            anchor="w",
            wraplength=520,
            justify="left",
        )
        self.names_label.pack(side="left", fill="x", expand=True, padx=4)

    def get_value(self) -> Any:
        """The names (a tuple of str), typed as every row's value is."""
        return self.current_value

    def set_value(self, value: Any) -> None:
        del value  # Read-only: a term of a name list is not toggled from the GUI.

    def set_default(self, value: Any) -> None:
        del value  # Read-only.

    def reset_to_default(self) -> None:
        """Nothing to reset: the row is read-only."""

    def mark_saved(self) -> None:
        """Nothing to mark: the row is read-only."""

    def has_value(self) -> bool:
        return True

    def is_modified(self) -> bool:
        return False

    def set_state(self, state: str) -> None:
        del state  # Nothing interactive.


ParameterRow: TypeAlias = slider_row.SliderRow | CheckRow | ChoiceRow | NameListRow


def make_row(
    parent: tk.Misc,
    spec: config_schema.ParameterSpec,
    label: str,
    value: Any,
    on_change: OnChange | None = None,
    label_width: int = 46,
) -> ParameterRow | None:
    """The row of a parameter, by its kind; None for one the GUI does not show (text, or excluded by the schema).

    Args:
        parent: the widget the row is packed into.
        spec: the parameter.
        label: its label (config_schema.display_label()).
        value: the value it shows now (None: none, for an unset parameter the operator has not given one); its default
            (the reset button's) is the file's value, spec.value, or unset for a parameter the file leaves unset.
        on_change: called with (label, value) after every change, and with (label, None) when an unset parameter is reset
            to unset; None: nothing is called.
        label_width: the width of the label, in characters.

    Returns:
        The row, not packed yet; None when the parameter has none.
    """
    if not spec.renders:
        return None
    row: ParameterRow
    unset = spec.value is None
    if spec.kind in (config_schema.Kind.NUMBER, config_schema.Kind.INTEGER):
        low, high = spec.slider_range() or (0.0, 1.0)
        row = slider_row.SliderRow(
            parent,
            name=label,
            # An unset parameter's slider rests at the low end of its range, a placeholder that is not its value.
            initial_value=low if unset else float(spec.value),
            min_val=low,
            max_val=high,
            on_change=None,
            label_width=label_width,
            log_scale=spec.tuning.log_scale,
            integer=spec.kind == config_schema.Kind.INTEGER,
            unset=unset,
        )
        if isinstance(value, (int, float)) and (
            not row.has_value() or float(value) != row.get_value()
        ):
            row.set_value(float(value))
        row.on_change = on_change
        return row
    if spec.kind == config_schema.Kind.BOOL:
        check = CheckRow(
            parent,
            label,
            None if unset else bool(spec.value),
            on_change=None,
            label_width=label_width,
        )
        if value is not None:
            check.set_value(bool(value))
        check.on_change = on_change
        return check
    if spec.kind == config_schema.Kind.CHOICE:
        choice = ChoiceRow(
            parent,
            label,
            None if unset else str(spec.value),
            spec.choices,
            on_change=None,
            label_width=label_width,
        )
        if value is not None:
            choice.set_value(str(value))
        choice.on_change = on_change
        return choice
    if spec.kind == config_schema.Kind.NAME_LIST:
        return NameListRow(parent, label, tuple(value or ()), label_width=label_width)
    return None
