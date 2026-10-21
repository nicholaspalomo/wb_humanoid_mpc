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

"""Configuration and filtering for 2D line graph plots in Rerun.

A plot configuration selects which topics, signals, wildcards, contract tabs or panels to display on 2D line graphs in
the Rerun viewer, either from a typed textproto file (humanoid_rerun_viewer_proto.PlotConfig) or a line-delimited text
file. A textproto also names the MuJoCo viewer of the robot process (mujoco_viewer), which the robot reads from the same
file (humanoid_nmpc/humanoid_common_mpc_app/robot/README.md, --plot_config); this module checks it as the robot does.
"""

from collections.abc import Sequence
import dataclasses
import fnmatch

from google.protobuf import text_format
from humanoid_rerun_viewer_proto import plot_config_pb2

from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract

DEFAULT_CUSTOM_TAB_TITLE = "Signals"
STATUS_TAB_NAME = "Status"

# The MuJoCo viewers of the robot process, by registry name: what mujoco_viewer may list.
# LINT.IfChange(mujoco_viewer_names)
GLFW_VIEWER = "glfw"
WEB_VIEWER = "web"
NO_VIEWER = "none"
MUJOCO_VIEWER_NAMES = (GLFW_VIEWER, WEB_VIEWER, NO_VIEWER)
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/viewer/ViewerNames.h:viewer_names)


class PlotConfigError(ValueError):
    """A plot configuration that cannot be read or parsed."""


@dataclasses.dataclass(frozen=True)
class PlotConfig:
    """The configured signals and options for 2D line graphs and 3D robot instances in Rerun.

    Attributes:
        signals: The topics, signals, entity paths, wildcards, or contract tab/panel titles to plot.
        custom_tab_title: The title of the tab that groups custom or unmatched signals (default: "Signals").
        robot_instances: The robot model instances to draw in the 3D scene (e.g. "measured", "terminal_state",
            "terminal_target"); empty: every instance the scene carries.
        draw_terminal_state: Whether to draw the end-of-trajectory robots: False draws only "measured", whatever
            robot_instances says; True or None leaves the choice to robot_instances.
        mujoco_viewer: The robot process's MuJoCo viewer by registry name, a preference list (empty: the robot
            process's built-in default, which proto/plot_config.proto names).
    """

    signals: tuple[str, ...] = ()
    custom_tab_title: str = DEFAULT_CUSTOM_TAB_TITLE
    robot_instances: tuple[str, ...] = ()
    draw_terminal_state: bool | None = None
    mujoco_viewer: tuple[str, ...] = ()


def _clean_str(value: str) -> str:
    return value.strip().strip("\"'")


# LINT.IfChange(viewer_list_rules)
# The viewers nothing may follow in a preference list, since a later entry could never be used, and why.
_LAST_VIEWERS = {NO_VIEWER: "always starts", GLFW_VIEWER: "cannot fail at start-up"}


def check_mujoco_viewer(names: Sequence[str], source: str = "") -> tuple[str, ...]:
    """Checks a mujoco_viewer preference list as the robot process checks it at start-up.

    Every name is one of MUJOCO_VIEWER_NAMES, exactly as written, none repeats, and nothing follows "none", which always
    starts, or "glfw", which cannot fail at start-up: a later entry could never be used. An empty list is the default.

    Args:
        names: The mujoco_viewer entries, in order.
        source: Optional file path or source identifier for error messages.

    Returns:
        The names, as a tuple.

    Raises:
        PlotConfigError: A name is unknown or repeated, or another follows "none" or "glfw".
    """
    prefix = f"{source}: " if source else ""
    listed = ",".join(names)
    for index, name in enumerate(names):
        if name not in MUJOCO_VIEWER_NAMES:
            raise PlotConfigError(
                f"{prefix}mujoco_viewer: unknown viewer {name!r} in {listed!r}; the viewers are "
                f"{', '.join(MUJOCO_VIEWER_NAMES)}"
            )
        if name in names[:index]:
            raise PlotConfigError(
                f"{prefix}mujoco_viewer: the viewer {name!r} is listed twice in {listed!r}"
            )
        if index + 1 < len(names) and name in _LAST_VIEWERS:
            raise PlotConfigError(
                f"{prefix}mujoco_viewer: nothing may follow {name!r} in {listed!r}, as it "
                f"{_LAST_VIEWERS[name]}"
            )
    return tuple(names)


# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/viewer/ViewerSettings.cpp:viewer_list_rules)


def parse_plot_config(text: str, source: str = "") -> PlotConfig:
    """Parses plot configuration text from either a textproto or a line-by-line signal list.

    Args:
        text: The file content.
        source: Optional file path or source identifier for error messages.

    Returns:
        The parsed PlotConfig.

    Raises:
        PlotConfigError: When protobuf parsing of a textproto fails, or its mujoco_viewer is not a valid viewer list
            (check_mujoco_viewer()).
    """
    trimmed = text.strip()
    is_declared_textproto = trimmed.startswith("# proto-")
    message = plot_config_pb2.PlotConfig()
    parsed_as_proto = False
    if is_declared_textproto:
        try:
            text_format.Parse(text, message)
            parsed_as_proto = True
        except text_format.ParseError as error:
            prefix = f"{source}: " if source else ""
            raise PlotConfigError(f"{prefix}cannot parse textproto: {error}") from error
    else:
        try:
            text_format.Parse(text, message)
            parsed_as_proto = True
        except text_format.ParseError:
            parsed_as_proto = False

    if parsed_as_proto:
        custom_title = (
            message.custom_tab_title.strip()
            if message.custom_tab_title
            else DEFAULT_CUSTOM_TAB_TITLE
        )
        draw_terminal = (
            message.draw_terminal_state
            if message.HasField("draw_terminal_state")
            else None
        )
        return PlotConfig(
            signals=tuple(s.strip() for s in message.signals if s.strip()),
            custom_tab_title=custom_title or DEFAULT_CUSTOM_TAB_TITLE,
            robot_instances=tuple(
                r.strip() for r in message.robot_instances if r.strip()
            ),
            draw_terminal_state=draw_terminal,
            mujoco_viewer=check_mujoco_viewer(tuple(message.mujoco_viewer), source),
        )

    signals: list[str] = []
    robot_instances: list[str] = []
    draw_terminal_state: bool | None = None
    custom_title = DEFAULT_CUSTOM_TAB_TITLE
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("custom_tab_title:"):
            custom_title = _clean_str(line.split(":", 1)[1]) or DEFAULT_CUSTOM_TAB_TITLE
            continue
        if line.startswith("robot_instances:"):
            val = _clean_str(line.split(":", 1)[1])
            if val:
                for item in val.split(","):
                    item_stripped = item.strip()
                    if item_stripped:
                        robot_instances.append(item_stripped)
            continue
        if line.startswith("draw_terminal_state:"):
            val = _clean_str(line.split(":", 1)[1]).lower()
            draw_terminal_state = val in ("true", "1", "yes")
            continue
        signals.append(line)
    return PlotConfig(
        signals=tuple(signals),
        custom_tab_title=custom_title,
        robot_instances=tuple(robot_instances),
        draw_terminal_state=draw_terminal_state,
    )


def load_plot_config(path: str) -> PlotConfig:
    """Reads and parses a plot configuration file (.textproto or text file).

    Args:
        path: Path to the configuration file.

    Returns:
        The parsed PlotConfig.

    Raises:
        OSError: The file cannot be read.
        PlotConfigError: The file is malformed.
    """
    with open(path, encoding="utf-8") as file:
        return parse_plot_config(file.read(), path)


def _matches(target: str, pattern: str) -> bool:
    t = target.strip().lower()
    p = pattern.strip().lower()
    return t == p or fnmatch.fnmatchcase(t, p)


def _path_matches(entity_path: str, pattern: str) -> bool:
    """True when entity_path matches pattern, with or without telemetry/ and wildcards."""
    e = entity_path.strip().lstrip("/")
    p = pattern.strip().lstrip("/")

    if _matches(e, p):
        return True

    e_telemetry = f"telemetry/{e}" if not e.startswith("telemetry/") else e
    p_telemetry = f"telemetry/{p}" if not p.startswith("telemetry/") else p
    if _matches(e_telemetry, p_telemetry):
        return True

    e_sub = e[len("telemetry/") :] if e.startswith("telemetry/") else e
    p_sub = p[len("telemetry/") :] if p.startswith("telemetry/") else p
    if _matches(e_sub, p_sub):
        return True

    if e.endswith("/**"):
        base = e[:-3]
        if p.startswith(base) or _matches(base, p):
            return True
    if e_sub.endswith("/**"):
        base_sub = e_sub[:-3]
        if p_sub.startswith(base_sub) or _matches(base_sub, p_sub):
            return True

    return False


def matches_panel(panel: telemetry_contract.Panel, pattern: str) -> bool:
    """True if pattern matches the panel's title, paths, or curve series names."""
    if _matches(panel.title, pattern):
        return True
    for path in panel.paths:
        if _path_matches(path, pattern):
            return True
    for curve in panel.curves:
        if _path_matches(curve.path, pattern):
            return True
        clean_p = pattern.strip().lstrip("/")
        if _matches(f"{curve.path}/{curve.series}", clean_p):
            return True
        if _matches(curve.series, pattern.strip()):
            return True
    return False


def matches_tab_title(tab: telemetry_contract.Tab, pattern: str) -> bool:
    """True if pattern matches the tab's title."""
    return _matches(tab.title, pattern)


def matches_status_series(series: status_contract.StatusSeries, pattern: str) -> bool:
    """True if pattern matches the status series title, path, or series names."""
    if _matches(series.title, pattern):
        return True
    if _path_matches(series.path, pattern):
        return True
    clean_p = pattern.strip().lstrip("/")
    if _matches(series.path.removeprefix(f"{status_contract.STATUS_ROOT}/"), clean_p):
        return True
    for name in series.names:
        if _matches(f"{series.path}/{name}", clean_p) or _matches(name, clean_p):
            return True
    return False


def filter_contract_tabs(
    tabs: Sequence[telemetry_contract.Tab],
    signals: Sequence[str],
) -> tuple[list[telemetry_contract.Tab], set[str]]:
    """Filters contract tabs to those containing panels matching any signal.

    Args:
        tabs: The contract tabs to filter.
        signals: The user-configured signals, patterns, or titles.

    Returns:
        A tuple of (filtered_tabs, matched_signals).
    """
    filtered_tabs: list[telemetry_contract.Tab] = []
    matched_signals: set[str] = set()

    for tab in tabs:
        tab_title_matched = False
        for signal in signals:
            if matches_tab_title(tab, signal):
                matched_signals.add(signal)
                tab_title_matched = True

        if tab_title_matched:
            filtered_tabs.append(tab)
            for panel in tab.panels():
                for signal in signals:
                    if matches_panel(panel, signal):
                        matched_signals.add(signal)
            continue

        filtered_rows: list[tuple[telemetry_contract.Panel, ...]] = []
        for row in tab.rows:
            filtered_row: list[telemetry_contract.Panel] = []
            for panel in row:
                for signal in signals:
                    if matches_panel(panel, signal):
                        filtered_row.append(panel)
                        matched_signals.add(signal)
                        break
            if filtered_row:
                filtered_rows.append(tuple(filtered_row))

        if filtered_rows:
            filtered_tabs.append(
                telemetry_contract.Tab(title=tab.title, rows=tuple(filtered_rows))
            )

    return filtered_tabs, matched_signals


def filter_status_series(
    signals: Sequence[str],
) -> tuple[list[status_contract.StatusSeries], set[str]]:
    """Filters status series to those matching any signal.

    Args:
        signals: The user-configured signals, patterns, or titles.

    Returns:
        A tuple of (filtered_series, matched_signals).
    """
    matched_signals: set[str] = set()
    status_tab_matched = False
    for signal in signals:
        if _matches(STATUS_TAB_NAME, signal):
            matched_signals.add(signal)
            status_tab_matched = True

    if status_tab_matched:
        for series in status_contract.STATUS_SERIES:
            for signal in signals:
                if matches_status_series(series, signal):
                    matched_signals.add(signal)
        return list(status_contract.STATUS_SERIES), matched_signals

    filtered_series: list[status_contract.StatusSeries] = []
    for series in status_contract.STATUS_SERIES:
        for signal in signals:
            if matches_status_series(series, signal):
                filtered_series.append(series)
                matched_signals.add(signal)
                break

    return filtered_series, matched_signals


def unmatched_signals(
    signals: Sequence[str],
    matched_signals: set[str],
) -> list[str]:
    """Returns signals that did not match any contract tab, panel, or status series."""
    return [sig for sig in signals if sig not in matched_signals]


def has_telemetry_signals(config: PlotConfig | None) -> bool:
    """True if the plot configuration requires subscribing to viz/telemetry."""
    if config is None:
        return True
    if not config.signals:
        return False
    _, status_matched = filter_status_series(config.signals)
    unmatched = unmatched_signals(config.signals, status_matched)
    if unmatched:
        return True
    contract_tabs, _ = filter_contract_tabs(telemetry_contract.TABS, config.signals)
    return bool(contract_tabs)


def resolve_allowed_instances(
    config: PlotConfig | None,
    cli_robot_instances: Sequence[str] | None = None,
    cli_terminal_state: bool | None = None,
) -> tuple[str, ...] | None:
    """Determines the robot instances to log based on CLI flags and plot config.

    Args:
        config: The parsed PlotConfig, if any.
        cli_robot_instances: Explicit list of robot instance names from CLI, if given.
        cli_terminal_state: Explicit CLI flag for drawing the terminal state (True/False/None).

    Returns:
        A tuple of instance names to log, or None to log all default contract instances.
    """
    if cli_robot_instances:
        return tuple(cli_robot_instances)
    if cli_terminal_state is False:
        return (scene_contract.MEASURED,)
    if cli_terminal_state is True:
        return None
    if config is not None:
        if config.draw_terminal_state is False:
            return (scene_contract.MEASURED,)
        if config.robot_instances:
            return config.robot_instances
        if config.draw_terminal_state is True:
            return None
    return None
