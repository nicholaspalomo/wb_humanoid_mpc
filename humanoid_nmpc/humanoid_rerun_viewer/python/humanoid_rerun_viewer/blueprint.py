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

"""The viewer's layout: the 3D scene next to the plot tabs.

    +-------------------------------+----------------------------------------------+
    | Scene (world, fixed frame)    | [Base Pose & Euler] [Base Twist] ... [Status] |
    |  robots, markers, plan, grid  |  one time-series view per panel, in the rows  |
    |                               |  and columns of telemetry_contract.TABS       |
    +-------------------------------+                                              |
    | Events (FSM, MPC, loop, bridge)|                                              |
    +-------------------------------+----------------------------------------------+

The 3D view shows the world with a 1 m line grid and hides what is clutter most of the time (the terminal target
robot, the equivalent corner forces, the collision spheres: scene_contract.hidden_by_default()); the eye follows the
measured robot's root link. Every plot shows the last PLOT_WINDOW_S seconds of the robot's clock up to the time cursor.
"""

from collections.abc import Sequence
import math

import rerun as rr
import rerun.blueprint as rrb

from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import plot_config as plot_config_module
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract

PLOT_WINDOW_S = 10.0
SCENE_VIEW_NAME = "Scene"
EVENTS_VIEW_NAME = "Events"
STATUS_TAB_NAME = "Status"

# The initial orbit camera, the one the ROS-era 3D view opened with: focal point, distance, pitch and yaw.
_CAMERA_FOCAL_POINT = (0.608772337436676, 0.09100822359323502, 0.0074924081563949585)
_CAMERA_DISTANCE = 2.4502856731414795
_CAMERA_PITCH = 0.36879807710647583
_CAMERA_YAW = 6.122678756713867


def initial_eye_position() -> list[float]:
    """Where the orbit camera starts: `distance` from the focal point at its pitch and yaw."""
    return [
        _CAMERA_FOCAL_POINT[0]
        + _CAMERA_DISTANCE * math.cos(_CAMERA_PITCH) * math.cos(_CAMERA_YAW),
        _CAMERA_FOCAL_POINT[1]
        + _CAMERA_DISTANCE * math.cos(_CAMERA_PITCH) * math.sin(_CAMERA_YAW),
        _CAMERA_FOCAL_POINT[2] + _CAMERA_DISTANCE * math.sin(_CAMERA_PITCH),
    ]


def _absolute(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def scene_view(
    tracked_link: str | None = None,
    allowed_instances: Sequence[str] | None = None,
) -> rrb.Spatial3DView:
    """The 3D view of the world; `tracked_link`, when given, is the measured robot's link the eye follows."""
    tracking_entity = (
        _absolute(scene_contract.link_path(scene_contract.MEASURED, tracked_link))
        if tracked_link
        else None
    )
    hidden_paths = list(scene_contract.hidden_by_default())
    if allowed_instances is not None:
        for style in scene_contract.ROBOT_INSTANCES:
            if style.name not in allowed_instances:
                hidden_paths.append(scene_contract.instance_path(style.name))
    return rrb.Spatial3DView(
        name=SCENE_VIEW_NAME,
        origin=_absolute(scene_contract.WORLD_ROOT),
        contents=[
            "$origin/**",
            *[f"- {_absolute(path)}/**" for path in hidden_paths],
        ],
        line_grid=rrb.LineGrid3D(
            visible=True,
            spacing=1.0,
            plane=rr.components.Plane3D.XY,
            color=palette.GRID_COLOR,
        ),
        eye_controls=rrb.EyeControls3D(
            position=initial_eye_position(),
            look_target=list(_CAMERA_FOCAL_POINT),
            eye_up=[0.0, 0.0, 1.0],
            tracking_entity=tracking_entity,
        ),
        overrides={
            _absolute(path): rrb.EntityBehavior(visible=False) for path in hidden_paths
        },
    )


def _time_axis() -> rrb.TimeAxis:
    return rrb.TimeAxis(
        view_range=rr.TimeRange(
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-PLOT_WINDOW_S),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        )
    )


def view_contents(root: str, paths: Sequence[str]) -> list[str]:
    """The query expressions of a view that plots `paths` (relative to `root`; "x/**" is the subtree of x)."""
    return [f"+ {_absolute(root)}/{path}" for path in paths]


def plot_view(panel: telemetry_contract.Panel) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        name=panel.title,
        origin=_absolute(scene_contract.TELEMETRY_ROOT),
        contents=view_contents(scene_contract.TELEMETRY_ROOT, panel.paths),
        axis_x=_time_axis(),
    )


def plot_tab(tab: telemetry_contract.Tab) -> rrb.Vertical:
    return rrb.Vertical(
        *[rrb.Horizontal(*[plot_view(panel) for panel in row]) for row in tab.rows],
        name=tab.title,
    )


def _status_view(series: status_contract.StatusSeries) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        name=series.title,
        origin=_absolute(scene_contract.STATUS_ROOT),
        contents=[f"+ {_absolute(series.path)}"],
        axis_x=_time_axis(),
    )


def status_tab() -> rrb.Vertical:
    return rrb.Vertical(
        rrb.Horizontal(
            _status_view(status_contract.MPC_SOLVE_TIME),
            _status_view(status_contract.MPC_FAILURES),
            _status_view(status_contract.MPC_OBSERVATIONS),
        ),
        rrb.Horizontal(
            _status_view(status_contract.LOOP_PERIOD),
            _status_view(status_contract.LOOP_COMPUTE_TIME),
            _status_view(status_contract.LOOP_LATENESS),
        ),
        rrb.Horizontal(
            _status_view(status_contract.LOOP_EVENTS),
            _status_view(status_contract.POLICY_AGE),
        ),
        name=STATUS_TAB_NAME,
    )


def filtered_status_tab(
    series_list: Sequence[status_contract.StatusSeries],
) -> rrb.Vertical | None:
    if not series_list:
        return None
    views = [_status_view(series) for series in series_list]
    rows = [views[i : i + 3] for i in range(0, len(views), 3)]
    return rrb.Vertical(
        *[rrb.Horizontal(*row) for row in rows],
        name=STATUS_TAB_NAME,
    )


def custom_signal_view(signal: str) -> rrb.TimeSeriesView:
    clean = signal.strip().lstrip("/")
    if clean.startswith("status/"):
        origin = _absolute(scene_contract.STATUS_ROOT)
        content_path = f"+ {origin}/{clean[len('status/'):]}"
    elif clean.startswith("telemetry/"):
        origin = _absolute(scene_contract.TELEMETRY_ROOT)
        content_path = f"+ {origin}/{clean[len('telemetry/'):]}"
    else:
        origin = _absolute(scene_contract.TELEMETRY_ROOT)
        content_path = f"+ {origin}/{clean}"
    return rrb.TimeSeriesView(
        name=clean,
        origin=origin,
        contents=[content_path],
        axis_x=_time_axis(),
    )


def custom_signals_tab(
    signals: Sequence[str],
    title: str = plot_config_module.DEFAULT_CUSTOM_TAB_TITLE,
) -> rrb.Vertical | None:
    if not signals:
        return None
    views = [custom_signal_view(sig) for sig in signals]
    rows = [views[i : i + 3] for i in range(0, len(views), 3)]
    return rrb.Vertical(
        *[rrb.Horizontal(*row) for row in rows],
        name=title,
    )


def build_configured_tabs(
    config: plot_config_module.PlotConfig,
) -> list[rrb.Vertical]:
    """The plot tabs filtered and constructed according to `config`."""
    if not config.signals:
        return []

    filtered_tabs, matched = plot_config_module.filter_contract_tabs(
        telemetry_contract.TABS, config.signals
    )
    status_series, status_matched = plot_config_module.filter_status_series(
        config.signals
    )
    matched.update(status_matched)

    tabs: list[rrb.Vertical] = [plot_tab(tab) for tab in filtered_tabs]
    if status_series:
        status = filtered_status_tab(status_series)
        if status is not None:
            tabs.append(status)

    unmatched = plot_config_module.unmatched_signals(config.signals, matched)
    if unmatched:
        custom_tab = custom_signals_tab(
            unmatched,
            title=config.custom_tab_title
            or plot_config_module.DEFAULT_CUSTOM_TAB_TITLE,
        )
        if custom_tab is not None:
            tabs.append(custom_tab)

    return tabs


def events_view() -> rrb.TextLogView:
    return rrb.TextLogView(
        name=EVENTS_VIEW_NAME,
        origin=_absolute(scene_contract.STATUS_ROOT),
        contents="$origin/**",
    )


def build_blueprint(
    tracked_link: str | None = None,
    plots: bool = True,
    config: plot_config_module.PlotConfig | None = None,
    allowed_instances: Sequence[str] | None = None,
) -> rrb.Blueprint:
    """The whole layout; `tracked_link` is the measured robot's link the 3D view follows (None: a fixed eye)."""
    if plots:
        if config is not None:
            tabs = build_configured_tabs(config)
        else:
            tabs = [plot_tab(tab) for tab in telemetry_contract.TABS] + [status_tab()]
        if tabs:
            content = rrb.Horizontal(
                rrb.Vertical(
                    scene_view(tracked_link, allowed_instances),
                    events_view(),
                    row_shares=[3, 1],
                ),
                rrb.Tabs(*tabs, active_tab=0),
                column_shares=[1, 1],
            )
        else:
            content = rrb.Vertical(
                scene_view(tracked_link, allowed_instances),
                events_view(),
                row_shares=[4, 1],
            )
    else:
        content = rrb.Vertical(
            scene_view(tracked_link, allowed_instances),
            events_view(),
            row_shares=[4, 1],
        )
    return rrb.Blueprint(
        content,
        rrb.BlueprintPanel(state=rrb.components.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.components.PanelState.Collapsed),
        rrb.TimePanel(
            timeline=scene_contract.ROBOT_TIMELINE,
            play_state=rrb.components.PlayState.Following,
        ),
        auto_layout=False,
        auto_views=False,
    )
