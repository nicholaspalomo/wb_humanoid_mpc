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

"""The state the robot publishes on robot/fsm_state, and when the remote control re-centers its joysticks.

Free of Tk, so that test/test_fsm_state.py covers it headless; the Base Controller tab is a thin shell over it.

THE MESSAGE is humanoid_mpc_msgs.FsmState (fsm_state.proto): `mode` is a ControlMode name, `gantry_locked` says
whether the gantry holds the robot, and `controller_resets` counts the controller resets the robot made since it
started - every discontinuity of the plant after which the controller starts again from where the robot is
(SimFallRecovery::Cycle::discontinuity): a catch, a LOCK_GANTRY, and a reset the simulator made on its own thread. The
robot publishes it on every change and periodically, so a GUI that connects late learns the current state.

RE-CENTERING. A stick left forward is a walking command the operator is still giving, whether or not the controller is
executing it. The sticks are therefore released whenever the robot stops being walked by the MPC through no motion of
the stick: on every transition into a passive mode, on every new gantry lock and on every controller reset. The reset
counter is what covers the case the other two cannot see: a simulator reset while the gantry was already locked in
JOINT_PD changes neither the mode nor the gantry, and used to leave the stick where it was, so that re-entering WB_MPC
walked the robot on a command nobody meant to give.
"""

import dataclasses

from humanoid_mpc_msgs import fsm_state_pb2

from remote_control import humanoid_finite_state_machine

#: The mode names a message may carry.
_MODE_NAMES = frozenset(
    control_mode.value for control_mode in humanoid_finite_state_machine.ControlMode
)


@dataclasses.dataclass(frozen=True)
class FsmState:
    """One robot/fsm_state message, as the GUI follows it."""

    mode: str
    gantry_locked: bool
    controller_resets: int


# LINT.IfChange(fsm_state_format)
def from_message(message: fsm_state_pb2.FsmState) -> FsmState | None:
    """The state `message` carries, or None when its mode is not a ControlMode name (the GUI then ignores it)."""
    if message.mode not in _MODE_NAMES:
        return None
    return FsmState(
        mode=message.mode,
        gantry_locked=bool(message.gantry_locked),
        controller_resets=int(message.controller_resets),
    )


# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_msgs/fsm_state.proto)


def should_recenter(previous: FsmState | None, current: FsmState) -> bool:
    """Whether `current`, published after `previous` (None for the first message), releases the joysticks.

    True on a transition into a passive mode, on a gantry that became locked and on a change of the controller reset
    count. The first message is a transition from an unknown state: it releases the sticks when it reports a passive
    mode or a locked gantry, and its reset count is only the baseline the next ones are compared with.

    Args:
        previous: The state the GUI followed until now; None before the first message.
        current: The state just received.

    Returns:
        Whether the GUI releases its joysticks to the center.
    """
    if current.mode in humanoid_finite_state_machine.PASSIVE_MODES and (
        previous is None or previous.mode != current.mode
    ):
        return True
    if current.gantry_locked and (previous is None or not previous.gantry_locked):
        return True
    return (
        previous is not None and current.controller_resets != previous.controller_resets
    )
