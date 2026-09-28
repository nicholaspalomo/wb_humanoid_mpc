"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""The state the FSM publishes on `/humanoid/fsm_state`, and when the remote control re-centers its joysticks.

Free of Tk and ROS, so that test/test_fsm_state.py covers it headless; the Base Controller tab is a thin shell over it.

THE MESSAGE is `<mode>,<gantry>,<controller resets>`, written by SimFsmBridge (formatFsmState() in SimFsmBridge.cpp):
`<mode>` is a ControlMode name, `<gantry>` is GANTRY_LOCKED or GANTRY_UNLOCKED, and `<controller resets>` counts the
controller resets the simulation loop made since it started - every discontinuity of the plant after which the
controller starts again from where the robot is (SimFallRecovery::Cycle::discontinuity): a catch, a LOCK_GANTRY, and a
reset the simulator made on its own thread. The two trailing fields are optional here, so that a publisher of the older
two- or one-field form still drives the mode selector.

RE-CENTERING. A stick left forward is a walking command the operator is still giving, whether or not the controller is
executing it. The sticks are therefore released whenever the robot stops being walked by the MPC through no motion of
the stick: on every transition into a passive mode, on every new gantry lock and on every controller reset. The reset
counter is what covers the case the other two cannot see: a simulator reset while the gantry was already locked in
JOINT_PD changes neither the mode nor the gantry, and used to leave the stick where it was, so that re-entering WB_MPC
walked the robot on a command nobody meant to give.
"""

from dataclasses import dataclass
from typing import Optional

from remote_control.humanoid_finite_state_machine import ControlMode, PASSIVE_MODES

# LINT.IfChange(fsm_state_format)
#: The gantry field of the message.
GANTRY_LOCKED = "GANTRY_LOCKED"
GANTRY_UNLOCKED = "GANTRY_UNLOCKED"
#: The separator of the fields.
SEPARATOR = ","
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_ros2/src/fsm/SimFsmBridge.cpp:fsm_state_format)


@dataclass(frozen=True)
class FsmState:
    """One `/humanoid/fsm_state` message. A field the message does not carry is None."""

    mode: str
    gantry_locked: Optional[bool] = None
    controller_resets: Optional[int] = None


def parse_fsm_state(text: str) -> Optional[FsmState]:
    """The state `text` carries, or None when its mode is not a ControlMode name.

    A gantry field other than GANTRY_LOCKED / GANTRY_UNLOCKED, or a reset count that is not a non-negative integer, is
    read as absent rather than failing the whole message: the mode it carries is still the FSM's.
    """
    fields = [field.strip() for field in text.split(SEPARATOR)]
    mode = fields[0]
    if mode not in {control_mode.value for control_mode in ControlMode}:
        return None
    gantry_locked = None
    if len(fields) >= 2 and fields[1] in (GANTRY_LOCKED, GANTRY_UNLOCKED):
        gantry_locked = fields[1] == GANTRY_LOCKED
    controller_resets = None
    # ASCII digits only: str.isdigit() also accepts characters such as "²" that int() refuses, which would raise out of
    # the Tk callback instead of reading the count as absent.
    if len(fields) >= 3 and fields[2].isascii() and fields[2].isdigit():
        controller_resets = int(fields[2])
    return FsmState(
        mode=mode, gantry_locked=gantry_locked, controller_resets=controller_resets
    )


def should_recenter(previous: Optional[FsmState], current: FsmState) -> bool:
    """Whether `current`, published after `previous` (None for the first message), releases the joysticks.

    True on a transition into a passive mode, on a gantry that became locked and on a change of the controller reset
    count. The first message is a transition from an unknown state: it releases the sticks when it reports a passive
    mode or a locked gantry, and its reset count is only the baseline the next ones are compared with.
    """
    if current.mode in PASSIVE_MODES and (
        previous is None or previous.mode != current.mode
    ):
        return True
    if current.gantry_locked and (previous is None or not previous.gantry_locked):
        return True
    return (
        previous is not None
        and previous.controller_resets is not None
        and current.controller_resets is not None
        and current.controller_resets != previous.controller_resets
    )
