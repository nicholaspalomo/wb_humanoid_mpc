# humanoid_common_mpc_app/visualization

The visualization publisher of the MPC process. It computes, with the MPC's robot model and Pinocchio, everything the
Rerun bridge ([`humanoid_rerun_viewer`](../../humanoid_rerun_viewer/README.md)) draws, and publishes it on the bus:

- `viz/scene` (`VisualizationScene`): the robot model instances as link world poses, and the markers;
- `viz/telemetry` (`TelemetrySeries`): the plots, one message per `robot/state` sample.

It runs in the MPC process, on a low-priority thread of its own, so the robot's realtime control thread never draws
anything. It works for both formulations (it only
uses `MpcRobotModelBase`) and for every robot: the frame names come from the task file and the model. The entity paths
and series names are the bridge's contract, tabulated in its README; `SceneContract.h` and the group table of
`TelemetryBuilder.cpp` are tied to the bridge's contract modules with `LINT.IfChange`, and `:test_end_to_end` checks the
two against each other through an `.rrd` file.

## Threads

```text
MPC solver thread ---- setPolicy(command, solution) ----> [triple buffer]  --+
  (MpcServer::Hooks::postSolveObserver)                                       |
                  ---- setObservation(observation) -----> [triple buffer]  --+--> visualization thread (SCHED_OTHER,
bus IO thread -------- pushRobotState(sample) ----------> [bounded SPSC      |    nice +10), every pollPeriod (5 ms):
  (subscribeRobotState: robot/state, every sample)          queue, 64]      --+    one viz/telemetry per sample,
                                                                                   viz/scene at rerun_scene_frequency
                                                                                   -> Bus::publish()
```

Feeding never blocks and never waits for the visualization: each mailbox has one producer and copies into storage it
already holds (no allocation once the shapes are stable). When the visualization falls behind, `robot/state` samples
that do not fit in the queue are dropped and counted (`Statistics::robotStatesDropped`), never queued without bound; a
newer observation or policy replaces one not yet drawn. The visualization thread runs on the time-sharing scheduler with
its nice value raised (`Options::niceIncrement`), so it yields to the solver; `Options::cores` can pin it away from the
solver's cores. Nothing of it runs on the robot's realtime thread.

A scene is published when something new arrived and its period has passed (on the monotonic clock, so a robot clock
that goes back, as when a simulation restarts, does not stop it; the old visualizer froze until the clock caught up).

## Attaching it in the MPC node

```cpp
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"

using ocs2::humanoid::visualization::VisualizationModel;
using ocs2::humanoid::visualization::VisualizationPublisher;

// Before the solver thread starts using the model: the publisher copies the Pinocchio model and clones the robot model.
const VisualizationModel model{taskFile, urdfFile, &interface.getPinocchioInterface(), &mpcRobotModel};
ASSIGN_OR_RETURN(std::unique_ptr<VisualizationPublisher> visualization, VisualizationPublisher::Create(model, *bus));
RETURN_IF_ERROR(visualization->subscribeRobotState(*bus));  // before bus->start()
RETURN_IF_ERROR(visualization->start());                     // publishes once the bus runs

// setPolicy() with every published solution, setObservation() with the observation it was solved from.
ocs2::humanoid::ipc::MpcServer::Hooks hooks;
hooks.postSolveObserver = visualization->postSolveObserver();
// MpcServer::Create(*bus, mpc, resetTarget, config, hooks); bus->start(); server->start();
// Shutdown: server->stop(); visualization->stop(); bus->stop(). The publisher outlives the server.
```

With `MpcNodeRuntime` (`humanoid_common_mpc_app/node`) the steps are its `VisualizationAttacher`, which is called
with the node's bus before it starts: `VisualizationPublisher::MakeBusAttacher(model, options, &publisher)` creates the
publisher on that bus, subscribes it to `robot/state` and returns `postSolveObserver()`. `CentroidalMpcNode` and
`WBMpcNode` always attach it; they keep the publisher as a member declared before the runtime, so that it outlives the
server, start it in their `start()` before the runtime and stop it in their `stop()` before the runtime, which owns
the bus.

`mpcRobotModel` is the model the MPC's dynamics use (for the centroidal MPC the effective one, the
`BasisInputsModelDecorator` when the task file selects `basis_vectors`), and the Pinocchio model the MPC's
(`loadCustomPinocchioInterface()`). `Create()` takes a `PublishFunction` instead of a bus for tests. Any process
that runs a solver feeds it the same three calls.

## What it computes

**Scene** (`SceneBuilder`), from the latest observation, policy and `robot/state` sample:

- `measured`: every link of the URDF with every joint of the sample, from a Pinocchio model of the whole URDF (the
  MPC's model fixes the joints it does not plan for). Without a sample in the last `Options::robotStateTimeout`
  (500 ms; the dummy sim sends none), the observation's state on the MPC's model.
- `terminal_state` (the plan's last node) and `terminal_target` (the reference at the plan's last time), on the MPC's
  model.
- At the observation: the contact force of every stance foot as an arrow ending at its center of pressure, the net
  center of pressure, the four equivalent corner forces of every stance foot (`ContactWrenchMapper<4>` on the task
  file's contact rectangle), and the collision spheres (`collision_constraint` of the task file). The wrenches are the
  policy's input at the observation's time through the state-aware world-frame accessors, so they are right for the
  basis-vector inputs too.
- The plan: the paths of `rerun_plan_frames`, of the base and of the CoM projected to the ground (the stance feet's mean
  height at the observation), and the footholds where a foot lands inside the horizon, in the contact colors.

**Telemetry** (`TelemetryBuilder`): every group of the bridge's telemetry contract, at the sample's time, for three
sources: measured (the sample, `RobotStateDecoder`), reference (the target trajectories of the latest policy's command)
and plan (the latest policy). See the bridge's README for the paths, the names and the units.

## Defects of the ROS 2 visualizer it does not copy

| Before | Now |
|---|---|
| `/joint_states` had two producers: the visualizer (MPC joints only, the others 0) and the telemetry | One measured robot, drawn from `robot/state` with every joint of the URDF |
| The force, CoP and corner-force markers used `observation.input`, which the MuJoCo sims leave zero, and a stance foot without force gave a CoP of 0/0 = NaN | The policy's input at the observation's time; below `kMinNormalForceForCop` (1 N) the CoP is the contact frame |
| `mpc/desired/*` was the reference, not the plan | `reference` groups, and new `plan` groups (`dofs/{position,velocity}/plan`, `frames/*/<frame>/plan`); the `mpc` wrenches are the plan's |
| `quaternionToEulerZYX` returned (roll, pitch, yaw) under that name | `eulerAnglesZyxFromRotation()` returns (yaw, pitch, roll), `rollPitchYawFromRotation()` the plots' order |
| A function-static frame list fixed by the first model, hard-coded to the G1's frames; a function-static ground height | `rerun_plan_frames` from the task file, checked against the model at start-up; the ground height is the builder's |
| The rate gate (`lastTime_`) was never reset on a clock rewind | The scene's rate is kept on the monotonic clock; the measured acceleration restarts after a rewind |
| Inline on the realtime thread: a copy of the Pinocchio model per policy, FK per node, DDS publishes | A thread of its own in the MPC process, fed through lock-free mailboxes |

## Task-file fields

Read at start-up from the robot's typed task file (`config/mpc/task.textproto`, `humanoid_mpc_config.TaskFile`, parsed
strictly: an unknown field or a value of the wrong type is refused with its file, line and column), together with the
contact polygons (`contacts`) and the collision spheres (`collision_constraint`) the scene draws. The tuning GUI's MPC
Parameters tab shows `rerun_scene_frequency` as a slider like every number of the schema; a change takes effect when the
MPC node restarts. The start-up log names the values.

<!-- LINT.IfChange(task_keys) -->
| Field | Absent | Meaning |
|---|---|---|
| `rerun_scene_frequency` | 30 | [Hz] the most `viz/scene` messages per second; a positive number |
| `telemetry_frames` | the contact frames | the frames of the `frames/<kind>/<frame>/<source>` plots, one entry per frame |
| `rerun_plan_frames` | the contact frames | the frames whose planned paths `world/plan/end_effectors` draws, one strip each (the G1 adds its lidar and palms) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:visualization_task_keys, //humanoid_nmpc/humanoid_common_mpc_app/visualization/include/humanoid_common_mpc_app/visualization/VisualizationConfig.h:visualization_task_fields) -->

A frame a list names that the MPC's model does not have, a duplicate, or a name that is not a valid entity path is an
error at start-up that names the field.

## The `robot/state` sample

`RobotStateDecoder` maps the sample's joints by name. The five joint-action arrays are each empty or aligned with
`joint_names`; a joint without an action has a NaN position target (`robot_state_sample.proto`). The effort applied is
`feed-forward + kp (q_des - q) + kd (qd_des - qd)`, 0 without an action. `measured_contact_wrenches` are in contact
order, in the world frame.

## Tests

| Target | What it checks |
|---|---|
| `:test_visualization_config` | the fields' defaults, values and refusals, read strictly from a task file; a frame the model lacks is refused; every shipped task file sets the fields and its publisher starts |
| `:test_robot_state_decoder` | the Euler angle conventions; the base in the world frame and as Euler rates; joints by name in any order; the PD law of the action; refusals |
| `:test_scene_builder` | for the G1 centroidal, the Atlas with basis-vector inputs and the G1 whole-body MPC: every path once; link poses, plan points, footholds and collision spheres against an independently built Pinocchio model; forces ending at the hand-computed CoP; corner forces equivalent to the wrench; no NaN without force; sample-only and observation-only scenes; the plan recomputed only for a new policy |
| `:test_telemetry_builder` | the group layout against the contract; base, Euler angles, twists, joints, wrenches, the reference and the plan against hand-computed values; the measured acceleration and its reset on a rewind; every value finite; every formulation |
| `:test_visualization_publisher` | one series per sample in order; a held publisher makes the queue drop and never blocks the feeders; the scene's rate bound; no scene without news; rejected samples; start and stop; `robot/state` to `viz/telemetry` over loopback buses; the MPC node's attacher publishing `viz/telemetry` and `viz/scene` on the node's bus from the robot's samples and its observer, and refusing a null publisher |
| `:test_end_to_end` | the driver (`:visualization_publisher_driver`) -> the bus -> the Rerun bridge's save sink: every link with visuals of all three instances, every marker, exactly the contract's telemetry groups and names, every series in the `.rrd` |
