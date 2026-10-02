"""The entities the bridge writes from robot/fsm_state, mpc/status and robot/loop_timing, and about itself.

The text logs are what an operator reads: an FSM change, a failing solver, a loop that overran. The scalars are the few
numbers worth a plot: the solve time, the loop period, its wake-up lateness, the loop's overruns, missed periods and
drops, the policy's age.
"""

import dataclasses
from typing import Tuple

from humanoid_rerun_viewer import scene_contract

STATUS_ROOT = scene_contract.STATUS_ROOT

# Text logs (rerun.TextLog).
FSM_STATE_LOG = f"{STATUS_ROOT}/robot/fsm_state"
LOOP_TIMING_LOG = f"{STATUS_ROOT}/robot/loop_timing"
MPC_STATUS_LOG = f"{STATUS_ROOT}/mpc/status"
BRIDGE_LOG = f"{STATUS_ROOT}/bridge"


@dataclasses.dataclass(frozen=True)
class StatusSeries:
    """Scalars the bridge plots from a status message: the entity <path> with one series per name."""

    path: str
    names: Tuple[str, ...]
    unit: str
    title: str


# LINT.IfChange(status_series)
MPC_SOLVE_TIME = StatusSeries(
    f"{STATUS_ROOT}/mpc/solve_time", ("solve_time",), "ms", "MPC Solve Time [ms]"
)
MPC_FAILURES = StatusSeries(
    f"{STATUS_ROOT}/mpc/consecutive_failures",
    ("consecutive_failures",),
    "",
    "MPC Consecutive Failures",
)
MPC_OBSERVATIONS = StatusSeries(
    f"{STATUS_ROOT}/mpc/observations",
    ("received", "skipped"),
    "",
    "MPC Observations per Status",
)
LOOP_PERIOD = StatusSeries(
    f"{STATUS_ROOT}/robot/loop_period",
    ("target", "mean", "max"),
    "ms",
    "Realtime Loop Period [ms]",
)
LOOP_COMPUTE_TIME = StatusSeries(
    f"{STATUS_ROOT}/robot/compute_time",
    ("max",),
    "ms",
    "Realtime Loop Compute Time [ms]",
)
LOOP_LATENESS = StatusSeries(
    f"{STATUS_ROOT}/robot/wake_up_lateness",
    ("max",),
    "ms",
    "Realtime Loop Wake-up Lateness [ms]",
)
LOOP_EVENTS = StatusSeries(
    f"{STATUS_ROOT}/robot/loop_events",
    (
        "overruns",
        "missed_periods",
        "telemetry_samples_dropped",
        "stale_policies_dropped",
    ),
    "",
    "Overruns, Missed Periods and Drops",
)
POLICY_AGE = StatusSeries(
    f"{STATUS_ROOT}/robot/policy_age", ("policy_age",), "s", "Policy Age [s]"
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:status_series)

STATUS_SERIES: Tuple[StatusSeries, ...] = (
    MPC_SOLVE_TIME,
    MPC_FAILURES,
    MPC_OBSERVATIONS,
    LOOP_PERIOD,
    LOOP_COMPUTE_TIME,
    LOOP_LATENESS,
    LOOP_EVENTS,
    POLICY_AGE,
)
