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

"""The entities the bridge writes from robot/fsm_state, mpc/status and robot/loop_timing, and about itself.

The text logs are what an operator reads: an FSM change, a failing solver, a loop that overran. The scalars are the few
numbers worth a plot: the solve time, the loop period, its wake-up lateness, the loop's overruns, missed periods and
drops, the policy's age.
"""

import dataclasses

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
    names: tuple[str, ...]
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

STATUS_SERIES: tuple[StatusSeries, ...] = (
    MPC_SOLVE_TIME,
    MPC_FAILURES,
    MPC_OBSERVATIONS,
    LOOP_PERIOD,
    LOOP_COMPUTE_TIME,
    LOOP_LATENESS,
    LOOP_EVENTS,
    POLICY_AGE,
)
