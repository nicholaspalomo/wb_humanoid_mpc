# humanoid_mpc_msgs

The Protocol Buffers messages of the IPC bus (`robot_runtime/robot_ipc`) that connects the robot's realtime process,
the MPC process and the operator tools. Which process publishes which message on which topic is in
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md).

## Rules for the .proto files

Both are checked by `tools/hooks/lint_code.py` and by the pre-commit hook (`tools/hooks/pre-commit`).

- **One definition per file, named after it** (google3's 1-1-1 rule, `tools/hooks/proto_file_layout.py`): every file
  defines exactly one top-level message or enum, and is named after it in snake_case: `MpcPolicy` lives in
  `mpc_policy.proto`, `ControllerType` in `controller_type.proto`. A type that only makes sense inside its parent (such
  as `TargetContactPatch.Kind`) stays nested in it.
- **The nproto struct** (`tools/hooks/proto_file_layout.py`, `tools/nproto/README.md`): the first statement of every
  top-level message names the plain C++ struct nproto generates for it (enums: `nproto.generate_enum`), a fully
  qualified name ending in the definition's name, the file importing `"nproto/options.proto"`. Nested types become
  nested C++ types of their parent's struct and carry no option:

  ```proto
  // Next ID: 4
  message Vector3 {
    option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

    reserved 4 to max;

    double x = 1;
    double y = 2;
    double z = 3;
  }
  ```
- **`// Next ID: N` and `reserved N to max;`** (`tools/hooks/proto_next_id.py`): the line directly above every
  `message Name {` names the next free field number, and the message's first statement after its options reserves every
  number from it on, so protoc itself refuses a field that takes the number before both are raised. When you add a field, take that
  number and raise both. When you delete a field, reserve its number and leave both alone, because a deleted number
  must never come back. `--fix` adds missing ones and raises stale ones.

The package is `humanoid_mpc_msgs`. Files import each other as `humanoid_mpc_msgs/<file>.proto`, C++ includes
`"humanoid_mpc_msgs/<file>.pb.h"` (target `:messages_cc_proto`), and Python imports
`from humanoid_mpc_msgs import <file>_pb2` (target `:messages_py_proto`). Every `.proto` file in this directory belongs
to `:messages_proto` (a glob), so a new file needs no BUILD change.

protoc is the prebuilt binary of the `protobuf` module in `MODULE.bazel`. The Python code links against the protobuf
wheel of the `@operator_deps` hub (`//tools/python:python_proto_toolchain`), whose version must match the Python
gencode of that protoc; the two are tied together with a `LINT.IfChange` directive. `:messages_cc_test` and
`:messages_py_test` check that the generated code loads, round-trips an `MpcPolicy` and travels over ZeroMQ.

`:messages_nproto` holds the plain C++ structs of the messages (`tools/nproto/README.md`): `#include
"humanoid_mpc_msgs/<file>.nproto.h"` for the struct `ocs2::humanoid::msgs::<Name>`, with Eigen vectors for the
`repeated double` fields and no protobuf include, and `"humanoid_mpc_msgs/<file>.nproto.pb.h"` for its `ToProto()` /
`FromProto()`. `:messages_nproto_test` round-trips every message of the package through its struct both ways, and
`:messages_nproto_allocations_test` checks that the messages of the realtime loop and the MPC link (`RobotStateSample`,
`MpcObservation`, `SystemObservation`, `LoopTiming`, `FsmState`, `MpcPolicy`) convert without allocating once the
objects they convert into have the value's shape.

## The messages

### The MPC link (robot <-> MPC)

```text
robot  --MpcObservation-->  MPC   (topic robot/mpc_observation, every control cycle, latest value wins)
robot  <--MpcPolicy-------  MPC   (topic mpc/policy, every solve, latest value wins)
```

Time is the robot's clock throughout: the MPC solves from `observation.time` and plans in it, so the two machines need
no clock synchronization.

| File | Message | Role |
|---|---|---|
| `system_observation.proto` | `SystemObservation` | `ocs2::SystemObservation` |
| `reset_requests.proto` | `ResetRequests` | the robot's reset counters, carried by every observation |
| `mpc_observation.proto` | `MpcObservation` | observation + reset counters |
| `mode_schedule.proto` | `ModeSchedule` | `ocs2::ModeSchedule` |
| `target_trajectories.proto` | `TargetTrajectories` | `ocs2::TargetTrajectories` |
| `performance_index.proto` | `PerformanceIndex` | `ocs2::PerformanceIndex` |
| `controller_type.proto` | `ControllerType` (enum) | feedforward or linear controller |
| `mpc_solver_status.proto` | `MpcSolverStatus` | the MPC side's solver health |
| `target_contact_patch.proto` | `TargetContactPatch` | a planned foot placement, for the MuJoCo viewer |
| `viewer_annotations.proto` | `ViewerAnnotations` | display-only data the simulator draws about the plan |
| `mpc_policy.proto` | `MpcPolicy` | one MPC solution (primal solution, command data, performance) |
| `mpc_status.proto` | `MpcStatus` | published after every solve attempt |

### Operator commands and state

| File | Message | Topic |
|---|---|---|
| `walking_velocity_command.proto` | `WalkingVelocityCommand` | `operator/walking_velocity_command` |
| `fsm_command.proto` | `FsmCommand` | `operator/fsm_command` |
| `fsm_state.proto` | `FsmState` | `robot/fsm_state` |
| `joint_targets.proto` | `JointTargets` | `operator/joint_targets` |
| `dodgeball_throw.proto` | `DodgeballThrow` | `operator/dodgeball_throw` |
| `config_file_kind.proto` | `ConfigFileKind` (enum) | which of the robot's files a save is: task, reference or PD gains |
| `config_file_save.proto` | `ConfigFileSave` | `operator/config_save` |
| `config_file_save_status.proto` | `ConfigFileSaveStatus` | `robot/config_save_status` |

`operator/mpc_parameters` and `operator/pd_gains` carry whole configuration files, `MpcParameterUpdate` and
`JointPdGainsFile` of `humanoid_nmpc/humanoid_mpc_config`, whose protos stay in their own package
(humanoid_nmpc/docs/distributed_runtime/README.md, "Topics").

`ConfigFileSave` is the GUI's Save of a file the robot process reads, for the robot's persistent copy
(humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the configuration"). It carries the file's text - exactly
the bytes written into the laptop's copy - rather than a typed file, so the robot stores what the laptop has byte for
byte; it names the file's kind, the robot (`robot_name`), the configuration (`config_path`, the path from
`robot_models/` on) and the schema fingerprint of the kind's file message, which the robot checks before it parses
the text as strictly as at start-up. The robot answers every save with a `ConfigFileSaveStatus` of the same
`sequence`: `SAVED`, `UNCHANGED`, `NOT_STORED` (a robot process without a store; `stored_path` names the file it
reads in place, which the GUI compares with the laptop's), `REFUSED` or `FAILED`.

### Telemetry

What the realtime process measured and commanded, and how its loop kept time. The realtime thread only copies a sample
into a preallocated ring buffer; the robot process's communication thread serializes and publishes it. Everything
derived from it with the robot model (frame kinematics, generalized coordinates, the MPC references) is computed on
the MPC machine by the visualization publisher, never on the robot.

| File | Message | Topic |
|---|---|---|
| `robot_state_sample.proto` | `RobotStateSample` | `robot/state` |
| `loop_timing.proto` | `LoopTiming` | `robot/loop_timing` |

### Visualization (Rerun)

What the Rerun viewer draws. The visualization publisher (C++, on the machine that runs the MPC) computes every
position with the robot model and the MPC's model of the state. The Rerun bridge (Python,
`humanoid_nmpc/humanoid_rerun_viewer`) maps these messages onto Rerun archetypes and knows nothing about the robot
beyond its URDF. Entity paths are relative to the bridge's roots ("world" for the 3D scene, "telemetry" for the plots)
and are the contract with the bridge's blueprint.

| File | Message | Role |
|---|---|---|
| `color.proto` | `Color` | RGBA in [0, 1] |
| `robot_model_instance.proto` | `RobotModelInstance` | link poses of one copy of the robot model |
| `arrows.proto` | `Arrows` | Rerun `Arrows3D` |
| `spheres.proto` | `Spheres` | Rerun `Points3D` with radii |
| `line_strip.proto`, `line_strips.proto` | `LineStrip`, `LineStrips` | Rerun `LineStrips3D` |
| `visualization_scene.proto` | `VisualizationScene` | topic `viz/scene` |
| `scalar_group.proto` | `ScalarGroup` | named scalars plotted together |
| `telemetry_series.proto` | `TelemetrySeries` | topic `viz/telemetry` |

### Shared geometry

`vector.proto` (`Vector`, a dense `vector_t`), `vector3.proto`, `quaternion.proto` (Hamilton, `Eigen::Quaterniond`),
`pose.proto`, `wrench.proto` (world-frame force and torque).
