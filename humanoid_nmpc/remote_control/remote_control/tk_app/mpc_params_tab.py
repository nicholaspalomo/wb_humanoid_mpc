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

"""The MPC Parameters tab: a widget per parameter of the task file and the contact-planning file, published and saved.

Nothing here names a parameter. The tab shows what the schemas of the two files hold (config_schema): every number,
bool and registry name the schema does not exclude, labeled by its field names with what the schema adds (unit, reload
class, applicability), in the blocks of the file. A slider moved is a change of a tuned_file.TunedFile;
every change is published, debounced, as the whole of both files (humanoid_mpc_config.MpcParameterUpdate on
operator/mpc_parameters, with the task file's identity) after a strict parse of the edited text, and written into the
files only by "Save", which keeps every other byte of them. Save then sends the task file's saved text to the robot's
store (robot_config_save.py); the contact-planning file is the MPC's alone and stays on the laptop.
"""

from collections.abc import Callable
import functools
import logging
import os
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
from typing import Any

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import task_file_pb2

from remote_control import config_files
from remote_control import config_schema
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import tuned_file
from remote_control.tk_app import combobox
from remote_control.tk_app import parameter_rows
from remote_control.tk_app import robot_save_status
from remote_control.tk_app import scrollable_frame

_LOGGER = logging.getLogger(__name__)

# The block selector's entry of the contact-planning file, and the prefix of its parameters' keys in slider_rows. The
# task file has no field of this name (it is retired there, task_file.proto), so the keys of the two files never meet.
CONTACT_PLANNING_BLOCK = "contact_planning"

# The block selector's entry of the task file's top-level scalars.
ROOT_CATEGORY = "(top level)"

# The debounce of a publish after a change [ms].
_PUBLISH_DEBOUNCE_MS = 300
# How long a passing status stays on the status line [ms]; a Save's stays until the next status.
_STATUS_DISPLAY_MS = 5000


def _disabled_banner(reason: str) -> str:
    """The banner of a tab whose online tuning is off, for `reason` (empty: the task file's enable_online_tuning)."""
    return f"🔒 Online Tuning Disabled ({reason or 'enable_online_tuning: false in the task file'})"


class MpcParamsTab(ttk.Frame):
    """The MPC Parameters tab: a row per tunable parameter of the task file (and its contact-planning file).

    Args:
        parent: the notebook the tab is added to.
        task_file: the task file to load (one that does not exist is reported, and nothing is loaded); None: the first
            preset, when it exists.
        on_params_updated: called with the task file's path after every save; None: nothing is called.
        enable_online_tuning: whether the rows and saving are enabled; None: as the task file's enable_online_tuning
            says.
        param_publisher: the publisher of operator/mpc_parameters; None: nothing is published.
        robot_saver: the GUI's saver of the robot's copies (robot_config_save.RobotConfigSaver); None: Save writes
            the laptop's files only, and says so.
        **kwargs: the options of the tab's ttk.Frame.
    """

    KNOWN_PRESETS = {
        "DRC Atlas (Centroidal)": "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
        "Unitree G1 (WB)": "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
        "Unitree R1 (Centroidal)": "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
        "Unitree G1 (Centroidal)": "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
        "EngineAI SA01 (Centroidal)": "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    }

    # The measured contact state of the controller, selected by name in the task file (robot_model/ContactEstimatorRegistry.h)
    # and hot-reloaded from the parameter topic. The Base Controller tab's drop-down (set_contact_estimator) offers the
    # names of the contact_estimator registry (config_registries.textproto).
    # LINT.IfChange(contact_estimator_gui)
    CONTACT_ESTIMATOR_KEY = "contact_estimator"
    # LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:contact_estimator)

    # The task file's field that selects the contact inputs, which the labels of the contact blocks depend on.
    # LINT.IfChange(contact_input_parameterization_gui)
    CONTACT_INPUT_PARAMETERIZATION_KEY = "contact_input_parameterization"
    # LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:contact_input_parameterization)

    def __init__(
        self,
        parent: tk.Misc,
        task_file: str | None = None,
        on_params_updated: Callable[[str], None] | None = None,
        enable_online_tuning: bool | None = None,
        param_publisher: operator_bus.TopicPublisher | None = None,
        robot_saver: robot_config_save.RobotConfigSaver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")

        self.task_file = task_file
        # The contact planner's own file (contact_planning.textproto next to the task file); None without one.
        self.contact_planning_file: str | None = None
        self.on_params_updated = on_params_updated
        self._explicit_online_tuning = enable_online_tuning
        self.enable_online_tuning = (
            True if enable_online_tuning is None else enable_online_tuning
        )

        # The two files, as loaded or last saved, with the operator's changes.
        self.task: tuned_file.TunedFile | None = None
        self.contact_planning: tuned_file.TunedFile | None = None
        # The names each registry accepts (config_registries.textproto): the choices of the registry strings.
        self.registries = config_files.load_registries()
        # The rows of the block on screen, by key: a task file parameter's path, or a contact-planning parameter's
        # path behind CONTACT_PLANNING_BLOCK + ".".
        self.slider_rows: dict[str, parameter_rows.ParameterRow] = {}
        # tkinter after() ID for debounced publish
        self._debounce_publish_id: str | None = None
        # The publisher of operator/mpc_parameters (operator_bus.TopicPublisher): publish(MpcParameterUpdate).
        self.param_publisher = param_publisher
        # Called (no arguments) whenever the contact estimator selection changes: file loaded, reset, or set through
        # set_contact_estimator. The Base Controller tab keeps its drop-down in sync with it.
        self.on_contact_estimator_changed: Callable[[], None] | None = None
        # The tkinter after() ID that clears a passing status; None when none is pending.
        self._status_clear_id: str | None = None
        # The robot's copy of the task file after a Save.
        self.robot_save = robot_save_status.RobotSaveStatus(
            self, robot_saver, self._show_save_status
        )

        self._build_header_ui()
        self._build_category_nav_ui()

        # Scrollable container for parameters
        self.scroll_container = scrollable_frame.ScrollableFrame(
            self, bg_color="#2c2c2c"
        )
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # A file given that is not there is reported, never replaced by another robot's: the fallback is for none.
        if self.task_file:
            self.load_file(self.task_file)
        else:
            default_path = list(self.KNOWN_PRESETS.values())[0]
            if os.path.exists(default_path):
                self.load_file(default_path)

    def _build_header_ui(self) -> None:
        """The toolbar (preset, file, reload, reset, save), the status line and the online tuning banner."""
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
        preset_cb.bind("<Button-1>", combobox.open_dropdown_on_click)

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
            toolbar, text="💾 Save", command=self.save_and_checkpoint
        )
        self.save_btn.pack(side="left", padx=(4, 0))

        self.status_label = ttk.Label(
            self, text="", font=("Helvetica", 9, "italic"), foreground="#27ae60"
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 2))

        # Online tuning disabled warning banner
        self.warning_banner = ttk.Label(
            self,
            text=_disabled_banner(""),
            font=("Helvetica", 9, "bold"),
            foreground="#e67e22",
        )
        if not self.enable_online_tuning:
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
            self.save_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")

    def set_online_tuning_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enables or disables online tuning; `reason` is why it is off, for the banner (empty: the task file says so)."""
        self.enable_online_tuning = bool(enabled)
        state = "normal" if self.enable_online_tuning else "disabled"
        if not self.enable_online_tuning:
            self.warning_banner.configure(text=_disabled_banner(reason))
            self.warning_banner.pack(anchor="w", padx=12, pady=(0, 2))
        else:
            self.warning_banner.pack_forget()
        self.save_btn.configure(state=state)
        self.reset_btn.configure(state=state)
        for row in self.slider_rows.values():
            row.set_state(state)
        self._notify_contact_estimator_changed()

    # ── Contact estimator selection (the drop-down on the Base Controller tab) ─────────────────────────────────
    def has_contact_estimator_selection(self) -> bool:
        """True when the loaded task file selects a contact estimator by name (`contact_estimator`)."""
        return self.task is not None and self.task.message.HasField(
            self.CONTACT_ESTIMATOR_KEY
        )

    def selected_contact_estimator(self) -> str | None:
        """The live contact estimator name, or None without a selection in the file."""
        if not self.has_contact_estimator_selection() or self.task is None:
            return None
        return str(self.task.value(self.CONTACT_ESTIMATOR_KEY))

    def contact_estimator_names(self) -> tuple[str, ...]:
        """The names the contact estimator can be selected among: its registry's, and the live one if it is not."""
        names: tuple[str, ...] = ()
        if self.task is not None and self.task.has(self.CONTACT_ESTIMATOR_KEY):
            names = self.task.spec(self.CONTACT_ESTIMATOR_KEY).choices or ()
        selected = self.selected_contact_estimator()
        if selected is not None and selected not in names:
            names += (selected,)
        return names

    def set_contact_estimator(self, name: str) -> None:
        """Selects the contact estimator `name` and publishes it like a slider change.

        The name reaches the simulator through the parameter topic; "Save" writes it into the file. Ignored while
        online tuning is off or the file has no selection.

        Args:
            name: a name of the contact_estimator registry (contact_estimator_names()).
        """
        if not self.enable_online_tuning or not self.has_contact_estimator_selection():
            self._notify_contact_estimator_changed()
            return
        row = self.slider_rows.get(self.CONTACT_ESTIMATOR_KEY)
        if isinstance(row, parameter_rows.ChoiceRow):
            row.on_change, callback = None, row.on_change
            row.set_value(name)
            row.on_change = callback
        self._on_any_change(self.CONTACT_ESTIMATOR_KEY, name)
        self._notify_contact_estimator_changed()

    def _notify_contact_estimator_changed(self) -> None:
        if self.on_contact_estimator_changed is not None:
            self.on_contact_estimator_changed()

    def _build_category_nav_ui(self) -> None:
        """The frame the block selector goes into; the selector is built when a file is loaded."""
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=10, pady=(0, 6))
        # The categories are the files' own top-level blocks, as their schemas order them: a block added to a schema
        # becomes a category, and a field added to a block becomes a row in it.
        self.active_category = tk.StringVar(value="")
        self.categories: list[str] = []
        self.category_selector: ttk.Combobox | None = None
        self._category_nav_frame = nav_frame

    def _on_preset_selected(self, event: tk.Event | None = None) -> None:
        del event  # Unused.
        preset_name = self.preset_var.get()
        if preset_name in self.KNOWN_PRESETS:
            path = self.KNOWN_PRESETS[preset_name]
            if os.path.exists(path):
                self.load_file(path)
            else:
                self._show_status(f"Preset path not found: {path}", error=True)

    def _browse_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select task.textproto",
            filetypes=[("Textproto files", "*.textproto"), ("All files", "*.*")],
        )
        if selected:
            self.load_file(selected)

    def load_file(self, file_path: str) -> None:
        """Loads a task file (and the contact_planning.textproto next to it) and renders its rows.

        The file's enable_online_tuning applies unless the tab was given one explicitly. A file that does not parse
        into its schema is reported on the status line, with its line and column, and leaves the tab empty.

        Args:
            file_path: the task file.
        """
        self.task_file = os.path.abspath(file_path)
        self.path_var.set(self.task_file)
        self.contact_planning_file = config_files.contact_planning_file(self.task_file)
        try:
            self.task = tuned_file.TunedFile(
                self.task_file, task_file_pb2.TaskFile, self.registries
            )
            self.contact_planning = None
            if self.contact_planning_file:
                self.contact_planning = tuned_file.TunedFile(
                    self.contact_planning_file,
                    contact_planning_file_pb2.ContactPlanningFile,
                    self.registries,
                )
        except (OSError, tuned_file.TunedFileError) as error:
            self.task = None
            self.contact_planning = None
            self.categories = []
            self._refresh_categories()
            self._clear_rows()
            self._show_status(f"Cannot load {self.task_file}: {error}", error=True)
            self._notify_contact_estimator_changed()
            return

        if self._explicit_online_tuning is not None:
            self.enable_online_tuning = self._explicit_online_tuning
        else:
            self.enable_online_tuning = bool(self.task.message.enable_online_tuning)
        self._refresh_categories()
        self._render_active_category()
        self.set_online_tuning_enabled(self.enable_online_tuning)
        self._show_status(f"Loaded: {os.path.basename(self.task_file)}")

    def reload_file(self) -> None:
        """Reads the files again, dropping every change, and publishes them so that the MPC follows."""
        if self.task_file and os.path.exists(self.task_file):
            self.load_file(self.task_file)
            self._publish_to_topic()

    def _files(self) -> list[tuple[str, tuned_file.TunedFile]]:
        """(key prefix, file) of the loaded files: the task file, and the contact-planning file behind its block."""
        files: list[tuple[str, tuned_file.TunedFile]] = []
        if self.task is not None:
            files.append(("", self.task))
        if self.contact_planning is not None:
            files.append((CONTACT_PLANNING_BLOCK + ".", self.contact_planning))
        return files

    def _locate(self, key: str) -> tuple[tuned_file.TunedFile, str]:
        """The file and the parameter path of a row key; KeyError for a key of neither."""
        for prefix, file in reversed(self._files()):
            if prefix and key.startswith(prefix) and file.has(key[len(prefix) :]):
                return file, key[len(prefix) :]
        if self.task is not None and self.task.has(key):
            return self.task, key
        raise KeyError(key)

    def _category_of(self, prefix: str, spec: config_schema.ParameterSpec) -> str:
        if prefix:
            return CONTACT_PLANNING_BLOCK
        return spec.block or ROOT_CATEGORY

    def _refresh_categories(self) -> None:
        """Rebuilds the block selector from the loaded files, in the order their schemas list the blocks."""
        blocks: list[str] = []
        for prefix, file in self._files():
            for spec in file.rendered():
                block = self._category_of(prefix, spec)
                if block not in blocks:
                    blocks.append(block)
        self.categories = blocks
        if self.active_category.get() not in blocks:
            self.active_category.set(blocks[0] if blocks else "")
        for widget in self._category_nav_frame.winfo_children():
            widget.destroy()
        self.category_selector = None
        if not blocks:
            return
        ttk.Label(self._category_nav_frame, text="Block:").pack(
            side="left", padx=(0, 6)
        )
        # A drop-down holds any number of blocks (the DRC Atlas task file has some forty), where a row of buttons
        # would lose the ones that do not fit.
        selector = ttk.Combobox(
            self._category_nav_frame,
            textvariable=self.active_category,
            values=list(blocks),
            state="readonly",
            width=34,
        )
        selector.pack(side="left")
        selector.bind(
            "<<ComboboxSelected>>", lambda _event: self._render_active_category()
        )
        self.category_selector = selector

    def contact_input_parameterization(self) -> str:
        """The contact input parameterization of the loaded task file (its schema default when it names none)."""
        if self.task is None:
            return ""
        return str(self.task.value(self.CONTACT_INPUT_PARAMETERIZATION_KEY))

    def render_category_containing(self, key: str) -> bool:
        """Renders the block of the row `key`, so the caller can reach that row.

        Args:
            key: The row's key: a parameter path of the task file, or CONTACT_PLANNING_BLOCK + "." and one of the
                contact-planning file.

        Returns:
            Whether the row is on screen now.
        """
        for prefix, file in self._files():
            path = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if (prefix and not key.startswith(prefix)) or not file.has(path):
                continue
            block = self._category_of(prefix, file.spec(path))
            if block not in self.categories:
                return False
            self.active_category.set(block)
            self._render_active_category()
            return key in self.slider_rows
        return False

    def _clear_rows(self) -> None:
        for child in self.scroll_container.scrollable_content.winfo_children():
            child.destroy()
        self.slider_rows.clear()

    def _render_active_category(self) -> None:
        """Renders the selected block: a row per rendered parameter, grouped by the blocks under it, in schema order."""
        self._clear_rows()
        category = self.active_category.get()
        frames: dict[str, ttk.LabelFrame] = {}
        for prefix, file in self._files():
            for spec in file.rendered():
                if self._category_of(prefix, spec) != category:
                    continue
                group = tuned_file.group_of(spec) or category
                frame = frames.get(group)
                if frame is None:
                    frame = ttk.LabelFrame(
                        self.scroll_container.scrollable_content,
                        text="• " + (prefix + group if prefix else group),
                    )
                    frame.pack(fill="x", padx=6, pady=4)
                    frames[group] = frame
                key = prefix + spec.path
                row = parameter_rows.make_row(
                    frame,
                    spec,
                    config_schema.display_label(spec, file.message),
                    file.value(spec.path),
                    on_change=functools.partial(self._on_row_change, key),
                )
                if row is None:
                    continue
                row.pack(fill="x", padx=4, pady=1)
                self.slider_rows[key] = row
        if not self.slider_rows:
            ttk.Label(
                self.scroll_container.scrollable_content,
                text="No tunable parameters in this block.",
            ).pack(anchor="w", padx=8, pady=8)
        if not self.enable_online_tuning:
            for row in self.slider_rows.values():
                row.set_state("disabled")

    def reset_all_defaults(self) -> None:
        """Returns every parameter to the file as loaded or last saved, and publishes the result."""
        if not self.enable_online_tuning:
            return
        for _, file in self._files():
            file.reset()
        for row in self.slider_rows.values():
            callback, row.on_change = row.on_change, None
            row.reset_to_default()
            row.on_change = callback
        self._notify_contact_estimator_changed()
        self._publish_to_topic()
        self._show_status("All parameters reset to the saved files")

    def _on_row_change(self, key: str, label: str, value: Any) -> None:
        """A row's on_change, bound to its key; the label the row reports is for display only."""
        del label  # Unused: for display only.
        self._on_any_change(key, value)

    def _on_any_change(self, key: str, value: Any) -> None:
        """A change of the parameter `key`: held in its file, and published after the debounce.

        Args:
            key: the row key of the parameter (its path, behind CONTACT_PLANNING_BLOCK + "." for the planner's).
            value: its new value.
        """
        try:
            file, path = self._locate(key)
            file.set(path, value)
        except (KeyError, ValueError) as error:
            self._show_status(f"Cannot change {key}: {error}", error=True)
            return
        if key == self.CONTACT_ESTIMATOR_KEY:
            self._notify_contact_estimator_changed()
        self._schedule_publish()

    def _schedule_publish(self) -> None:
        """Debounces a publish of both files on operator/mpc_parameters."""
        if self._debounce_publish_id is not None:
            self.after_cancel(self._debounce_publish_id)
        self._debounce_publish_id = self.after(
            _PUBLISH_DEBOUNCE_MS, self._publish_to_topic
        )

    def build_parameter_update(self) -> Any:
        """The MpcParameterUpdate of the two files with the operator's changes, parsed strictly from their edited text.

        Returns:
            The update; None without a task file.

        Raises:
            tuned_file.TunedFileError: an edited text would not parse into its edited message.
        """
        if self.task is None:
            return None
        planner = (
            self.contact_planning.edited().message
            if self.contact_planning is not None
            else None
        )
        return operator_bus.mpc_parameter_update(
            self.task.edited().message,
            planner,
            config_path=robot_config_save.config_path_of(self.task.path),
        )

    def _publish_to_topic(self) -> None:
        """Publishes the two files with the current values, as an MpcParameterUpdate on operator/mpc_parameters."""
        self._debounce_publish_id = None
        if not self.enable_online_tuning or not self.param_publisher:
            _LOGGER.debug("Skipping publish: online tuning disabled or no publisher.")
            return
        try:
            update = self.build_parameter_update()
            if update is None:
                _LOGGER.warning("Not publishing: no task file is loaded.")
                return
            self.param_publisher.publish(update)
        except tuned_file.TunedFileError as error:
            self._show_status(f"Not published: {error}", error=True)
        # pylint: disable-next=broad-exception-caught  # Shown to the operator: a Tk callback must not raise.
        except Exception as error:
            _LOGGER.exception("Failed to publish MPC parameter updates.")
            self._show_status(f"Error publishing to topic: {error}", error=True)

    def save_and_checkpoint(self) -> None:
        """Explicit save: writes the files AND makes them the reset checkpoint ('Reset All' returns to them)."""
        self.save()

    def save(self) -> bool:
        """Writes every change into its file on the laptop, then sends the saved task file to the robot.

        The laptop's files keep every byte the changes do not touch (the planner's changes go into
        contact_planning.textproto, which is saved first: it is the MPC's alone, so that a failure to save it leaves the
        task file's two copies, the laptop's and the robot's, as they were). Then the exact text of the task file,
        saved, goes to the robot's store, also when nothing changed (it re-synchronizes a robot that refused an earlier
        save or missed an editor's); the status line follows the robot's answer. A laptop save that fails sends nothing.

        Returns:
            Whether the laptop's files were saved.
        """
        if not self.enable_online_tuning:
            self._show_status("Online tuning is disabled.", error=True)
            return False
        if self.task is None or not self.task_file:
            self._show_status("No file loaded to save.", error=True)
            return False
        if self.contact_planning is not None:
            try:
                self.contact_planning.save()
            except (OSError, tuned_file.TunedFileError) as error:
                self._show_status(
                    f"Error saving the contact planner's file: {error}; the task file was not saved",
                    error=True,
                )
                return False
        try:
            task_text = self.task.save().text
        except (OSError, tuned_file.TunedFileError) as error:
            planner = (
                " (the contact planner's file was saved)"
                if self.contact_planning is not None
                else ""
            )
            self._show_status(f"Error saving: {error}{planner}", error=True)
            return False
        for row in self.slider_rows.values():
            row.mark_saved()
        self.robot_save.send(robot_config_save.KIND_TASK, self.task.path, task_text)
        if self.on_params_updated:
            self.on_params_updated(self.task_file)
        return True

    def _show_save_status(self, msg: str, error: bool) -> None:
        """Shows the outcome of a Save, which stays until the next status."""
        self._show_status(msg, error=error, transient=False)

    def _show_status(
        self, msg: str, error: bool = False, transient: bool = True
    ) -> None:
        """Shows `msg` on the status line, in red for an error; a transient one is cleared after _STATUS_DISPLAY_MS."""
        if self._status_clear_id is not None:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        if transient:
            self._status_clear_id = self.after(_STATUS_DISPLAY_MS, self._clear_status)

    def _clear_status(self) -> None:
        self._status_clear_id = None
        self.status_label.configure(text="")
