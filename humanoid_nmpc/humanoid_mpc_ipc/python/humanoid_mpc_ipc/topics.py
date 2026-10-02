"""The topics of the IPC bus, the Python twin of include/humanoid_mpc_ipc/Topics.h.

A topic is the first frame of every bus message (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics"). The
comment of each constant names the message it carries and who publishes it. ZeroMQ's SUB filter matches a prefix of
the topic frame, so no topic may be a prefix of another (test/test_topics.py checks it, and that every constant here
equals its C++ twin).
"""

from typing import Tuple

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
# YamlDocument, GUI -> MPC and robot, on edit.
OPERATOR_MPC_PARAMETERS = "operator/mpc_parameters"
# YamlDocument, GUI -> robot, on edit.
OPERATOR_PD_GAINS = "operator/pd_gains"
# JointTargets, GUI -> robot, on edit (JOINT_PD only).
OPERATOR_JOINT_TARGETS = "operator/joint_targets"
# YamlDocument, GUI -> robot (simulation), on button press.
OPERATOR_DODGEBALL_THROW = "operator/dodgeball_throw"

# Every topic above, for tools that list or check them.
ALL_TOPICS: Tuple[str, ...] = (
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
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/include/humanoid_mpc_ipc/Topics.h:topics)
