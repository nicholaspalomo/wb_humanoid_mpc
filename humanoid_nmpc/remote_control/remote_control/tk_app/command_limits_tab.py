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

"""Command limits and reference defaults, read from the robot's ``config/command/reference.textproto``."""

from collections.abc import Callable
import functools
import os
from tkinter import ttk
import tkinter as tk
from typing import Any

from humanoid_mpc_config import reference_file_pb2

from remote_control import config_schema
from remote_control import robot_config_save
from remote_control import tuned_file
from remote_control.tk_app import parameter_rows
from remote_control.tk_app import robot_save_status
from remote_control.tk_app import scrollable_frame

# The colors of the status line: a passing or good status, and a problem.
_STATUS_COLOR = "#3c9a3c"
_PROBLEM_COLOR = "#e74c3c"


class CommandLimitsTab(ttk.Frame):
    """The parameters of the reference file: the command limits, the command ramps and the reference defaults.

    These are the parameters that bound what the operator can ask for - the largest velocity a full stick means, the
    acceleration the reference ramps at, the nominal stance height. They do not travel on the parameter topic: the C++
    side reads them from this file, and the parameter updater of the running MPC, of either formulation, watches the
    file and reloads the consumers that registered for it (``TargetTrajectoriesCalculatorBase`` and
    ``ProceduralMpcMotionManager``). Saving is therefore the whole mechanism - there is no publisher here - and the
    running MPC picks the change up, about a second later, for the parameters the schema marks hot; the others are
    labeled "(restart)". Save also sends the saved file to the robot's store, which the robot process reads at its next
    start, as the keyboard command node reads the file.

    The rows are the parameters of the file's schema (humanoid_mpc_config/reference_file.proto, through
    config_schema), so a limit added to the schema appears here with no code written anywhere.

    Args:
        parent: the notebook the tab is added to.
        reference_file: the robot's reference file; None: the tab says it found none.
        robot_saver: the GUI's saver of the robot's copies (robot_config_save.RobotConfigSaver); None: Save writes
            the laptop's file only, and says so.
        on_saved: called after the laptop's file was saved, before it is sent to the robot (the GUI follows the file's
            default_base_height with it); None: nothing is.
        **kwargs: the options of the tab's ttk.Frame.
    """

    def __init__(
        self,
        parent: tk.Misc,
        reference_file: str | None = None,
        robot_saver: robot_config_save.RobotConfigSaver | None = None,
        on_saved: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._on_saved = on_saved
        self.reference_file = (
            os.path.abspath(reference_file) if reference_file else None
        )
        self.reference: tuned_file.TunedFile | None = None
        self.slider_rows: dict[str, parameter_rows.ParameterRow] = {}
        self.status_var = tk.StringVar(value="")
        # The robot's copy of the file after a Save.
        self.robot_save = robot_save_status.RobotSaveStatus(
            self, robot_saver, self._show_save_status
        )

        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(
            header,
            text="Command Limits & Reference Defaults",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Not published on the parameter topic: saving writes this file, which the running MPC reloads "
            "about a second later, and sends it to the robot, which reads it at its next start.",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(2, 6))

        self.path_var = tk.StringVar(
            value=self.reference_file or "(no reference file found)"
        )
        ttk.Label(header, textvariable=self.path_var, foreground="#888888").pack(
            anchor="w"
        )

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=8, pady=4)
        ttk.Button(buttons, text="💾 Save", command=self.save).pack(side="left")
        ttk.Button(buttons, text="↩ Reload", command=self.reload).pack(
            side="left", padx=6
        )
        self.status_label = ttk.Label(
            buttons, textvariable=self.status_var, foreground=_STATUS_COLOR
        )
        self.status_label.pack(side="left", padx=10)

        self.scroll_container = scrollable_frame.ScrollableFrame(self)
        self.scroll_container.pack(fill="both", expand=True, padx=8, pady=8)

        self.reload()

    def reload(self) -> None:
        """Re-reads the file and rebuilds the rows from it."""
        for child in self.scroll_container.scrollable_content.winfo_children():
            child.destroy()
        self.slider_rows.clear()
        self.reference = None
        if not self.reference_file or not os.path.exists(self.reference_file):
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="No reference file was found next to the task file (config/command/reference.textproto).",
            ).pack(anchor="w", padx=6, pady=6)
            return
        try:
            self.reference = tuned_file.TunedFile(
                self.reference_file, reference_file_pb2.ReferenceFile
            )
        except (OSError, tuned_file.TunedFileError) as error:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text=f"Cannot load {self.reference_file}: {error}",
                wraplength=880,
                justify="left",
            ).pack(anchor="w", padx=6, pady=6)
            return
        self.path_var.set(self.reference_file)
        self._render(self.reference)
        self.status_var.set("")

    def _render(self, reference: tuned_file.TunedFile) -> None:
        """A row for every rendered parameter of the file, grouped by the block that holds it."""
        frames: dict[str, ttk.LabelFrame] = {}
        for spec in reference.rendered():
            group = tuned_file.group_of(spec) or "Command limits and defaults"
            frame = frames.get(group)
            if frame is None:
                frame = ttk.LabelFrame(
                    self.scroll_container.scrollable_content, text="• " + group
                )
                frame.pack(fill="x", padx=6, pady=4)
                frames[group] = frame
            row = parameter_rows.make_row(
                frame,
                spec,
                config_schema.display_label(spec, reference.message),
                reference.value(spec.path),
                on_change=functools.partial(self._on_change, spec.path),
            )
            if row is None:
                continue
            row.pack(fill="x", padx=4, pady=1)
            self.slider_rows[spec.path] = row
        if not self.slider_rows:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="The reference file carries no tunable parameters.",
            ).pack(anchor="w", padx=6, pady=4)

    def _on_change(self, path: str, label: str, value: Any) -> None:
        del label  # Unused: for display only.
        if self.reference is None:
            return
        try:
            self.reference.set(path, value)
        except (KeyError, ValueError) as error:
            self._show_status(f"cannot change {path}: {error}", error=True)
            return
        self._show_status("unsaved changes", error=False)

    def save(self) -> bool:
        """Writes every change into the file on the laptop, then sends the saved file to the robot.

        The laptop's file keeps every other byte (and its first version as .bak); the running MPC reloads it within
        about a second. Then its exact text goes to the robot's store, also when nothing changed (it re-synchronizes a
        robot that refused an earlier save or missed an editor's); the status line follows the robot's answer. A laptop
        save that fails sends nothing.

        Returns:
            Whether the laptop's file was saved.
        """
        if self.reference is None:
            return False
        try:
            text = self.reference.save().text
        except (OSError, tuned_file.TunedFileError) as error:
            self._show_status(f"save failed: {error}", error=True)
            return False
        for row in self.slider_rows.values():
            row.mark_saved()
        if self._on_saved is not None:
            self._on_saved()
        self.robot_save.send(
            robot_config_save.KIND_REFERENCE, self.reference.path, text
        )
        return True

    def _show_save_status(self, msg: str, error: bool) -> None:
        """Shows the outcome of a Save on the status line."""
        self._show_status(msg, error=error)

    def _show_status(self, msg: str, error: bool) -> None:
        self.status_var.set(msg)
        self.status_label.configure(
            foreground=_PROBLEM_COLOR if error else _STATUS_COLOR
        )
