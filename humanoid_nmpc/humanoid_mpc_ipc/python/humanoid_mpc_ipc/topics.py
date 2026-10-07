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

"""The topics of the IPC bus, the Python twin of include/humanoid_mpc_ipc/Topics.h.

A topic is the first frame of every bus message (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics"). The
comment of each constant names the message it carries and who publishes it. ZeroMQ's SUB filter matches a prefix of
the topic frame, so no topic may be a prefix of another (test/test_topics.py checks it, and that every constant here
equals its C++ twin).
"""

# LINT.IfChange(topics)
# MpcObservation, robot -> MPC, every control cycle.
ROBOT_MPC_OBSERVATION = "robot/mpc_observation"
# RobotStateSample, robot -> MPC (visualization), every telemetry period.
ROBOT_STATE = "robot/state"
# FsmState, robot -> GUI, on change and at 2 Hz.
ROBOT_FSM_STATE = "robot/fsm_state"
# LoopTiming, robot -> GUI and Rerun bridge, at 1 Hz.
ROBOT_LOOP_TIMING = "robot/loop_timing"
# MpcPolicy, MPC -> robot and dummy sim, every solve.
MPC_POLICY = "mpc/policy"
# MpcStatus, MPC -> robot (solver health), GUI and Rerun bridge, every solve attempt.
MPC_STATUS = "mpc/status"
# VisualizationScene, MPC (visualization) -> Rerun bridge.
VIZ_SCENE = "viz/scene"
# TelemetrySeries, MPC (visualization) -> Rerun bridge, one per robot/state sample.
VIZ_TELEMETRY = "viz/telemetry"
# WalkingVelocityCommand, GUI and teleop -> MPC and robot, at 25 Hz.
OPERATOR_WALKING_VELOCITY_COMMAND = "operator/walking_velocity_command"
# FsmCommand, GUI -> robot, on change.
OPERATOR_FSM_COMMAND = "operator/fsm_command"
# MpcParameterUpdate (humanoid_mpc_config: the whole task and contact planner files), GUI -> MPC and robot, on edit.
OPERATOR_MPC_PARAMETERS = "operator/mpc_parameters"
# JointPdGainsFile (humanoid_mpc_config: the whole PD gains file), GUI -> robot, on edit.
OPERATOR_PD_GAINS = "operator/pd_gains"
# JointTargets, GUI -> robot, on edit (JOINT_PD only).
OPERATOR_JOINT_TARGETS = "operator/joint_targets"
# DodgeballThrow, GUI -> robot (simulation), on button press.
OPERATOR_DODGEBALL_THROW = "operator/dodgeball_throw"
# ConfigFileSave, GUI and push_robot_config -> robot, on Save: a configuration file for the robot's persistent copy.
OPERATOR_CONFIG_SAVE = "operator/config_save"
# ConfigFileSaveStatus, robot -> GUI and push_robot_config, once per ConfigFileSave.
ROBOT_CONFIG_SAVE_STATUS = "robot/config_save_status"

# Every topic above, for tools that list or check them.
ALL_TOPICS: tuple[str, ...] = (
    ROBOT_MPC_OBSERVATION,
    ROBOT_STATE,
    ROBOT_FSM_STATE,
    ROBOT_LOOP_TIMING,
    MPC_POLICY,
    MPC_STATUS,
    VIZ_SCENE,
    VIZ_TELEMETRY,
    OPERATOR_WALKING_VELOCITY_COMMAND,
    OPERATOR_FSM_COMMAND,
    OPERATOR_MPC_PARAMETERS,
    OPERATOR_PD_GAINS,
    OPERATOR_JOINT_TARGETS,
    OPERATOR_DODGEBALL_THROW,
    OPERATOR_CONFIG_SAVE,
    ROBOT_CONFIG_SAVE_STATUS,
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/include/humanoid_mpc_ipc/Topics.h:topics)
