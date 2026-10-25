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

"""The control modes of the robot process's FSM, by the names operator/fsm_command and robot/fsm_state carry.

The robot process runs the FSM (its mode names: humanoid_common_mpc/mrt/ControlMode.h); the remote control names them:
- ZERO_TORQUE: de-energized, passive motors.
- JOINT_PD: joint position servoing to the nominal posture.
- GRAVITY_COMP: gravity compensation, zero-g compliance.
- WB_MPC: the MPC's policy is executed.
- SAFETY: a damped joint PD whose gains decay to zero torque.

fsm_state.py reads robot/fsm_state with these names.
"""

import enum


# LINT.IfChange(control_mode_names)
class ControlMode(str, enum.Enum):
    """Whole-body humanoid control modes."""

    ZERO_TORQUE = "ZERO_TORQUE"
    JOINT_PD = "JOINT_PD"
    GRAVITY_COMP = "GRAVITY_COMP"
    WB_MPC = "WB_MPC"
    SAFETY = "SAFETY"


#: The modes whose action is computed without the MPC (control_mode::isPassive in ControlMode.h). The remote control
#: re-centers its joysticks on every transition into one of them (fsm_state.should_recenter).
PASSIVE_MODES = frozenset(
    {
        ControlMode.ZERO_TORQUE.value,
        ControlMode.JOINT_PD.value,
        ControlMode.GRAVITY_COMP.value,
        ControlMode.SAFETY.value,
    }
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/mrt/ControlMode.h:control_mode_names, //humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/mrt/ControlMode.h:control_mode_families)
