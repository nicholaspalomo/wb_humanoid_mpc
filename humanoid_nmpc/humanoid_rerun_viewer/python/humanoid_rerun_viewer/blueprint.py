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

import math
from typing import List, Optional, Sequence

import rerun as rr
import rerun.blueprint as rrb

from humanoid_rerun_viewer import palette
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


def initial_eye_position() -> List[float]:
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


def scene_view(tracked_link: Optional[str] = None) -> rrb.Spatial3DView:
    """The 3D view of the world; `tracked_link`, when given, is the measured robot's link the eye follows."""
    tracking_entity = (
        _absolute(scene_contract.link_path(scene_contract.MEASURED, tracked_link))
        if tracked_link
        else None
    )
    return rrb.Spatial3DView(
        name=SCENE_VIEW_NAME,
        origin=_absolute(scene_contract.WORLD_ROOT),
        contents="$origin/**",
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
            _absolute(path): rrb.EntityBehavior(visible=False)
            for path in scene_contract.hidden_by_default()
        },
    )


def _time_axis() -> rrb.TimeAxis:
    return rrb.TimeAxis(
        view_range=rr.TimeRange(
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-PLOT_WINDOW_S),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        )
    )


def view_contents(root: str, paths: Sequence[str]) -> List[str]:
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


def events_view() -> rrb.TextLogView:
    return rrb.TextLogView(
        name=EVENTS_VIEW_NAME,
        origin=_absolute(scene_contract.STATUS_ROOT),
        contents="$origin/**",
    )


def build_blueprint(tracked_link: Optional[str] = None) -> rrb.Blueprint:
    """The whole layout; `tracked_link` is the measured robot's link the 3D view follows (None: a fixed eye)."""
    tabs = [plot_tab(tab) for tab in telemetry_contract.TABS] + [status_tab()]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(scene_view(tracked_link), events_view(), row_shares=[3, 1]),
            rrb.Tabs(*tabs, active_tab=0),
            column_shares=[1, 1],
        ),
        rrb.BlueprintPanel(state=rrb.components.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.components.PanelState.Collapsed),
        rrb.TimePanel(
            timeline=scene_contract.ROBOT_TIMELINE,
            play_state=rrb.components.PlayState.Following,
        ),
        auto_layout=False,
        auto_views=False,
    )
