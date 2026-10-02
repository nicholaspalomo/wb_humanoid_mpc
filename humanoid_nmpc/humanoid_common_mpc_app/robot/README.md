# humanoid_common_mpc_app/robot

The robot process of the distributed runtime ([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../docs/distributed_runtime/README.md)):
the binary that runs on the robot's realtime computer, and - with the MuJoCo backend - in simulation, in the same
topology. It is the realtime loop over a **robot backend** chosen by name and an **MRT joint controller** of either
formulation, and the mailboxes, telemetry and FSM around it on the bus. It replaces the ROS-era MuJoCo simulation
nodes (deleted with ROS) and does what they did, in the same order, without ROS.

<!-- LINT.IfChange(robot_binaries) -->
| Binary | Bus node | What |
|---|---|---|
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_robot` | `robot` | the centroidal MRT joint controller on a backend |
| `//humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_robot` | `robot` | the whole-body MRT joint controller on a backend |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/BUILD.bazel:robot_binaries, //humanoid_nmpc/humanoid_wb_mpc_app/BUILD.bazel:robot_binaries) -->

```bash
ROBOT=robot_models/drc_atlas
bazel run //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_robot -- --robot_name=drc_atlas \
    --task_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/mpc/task.yaml \
    --reference_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/command/reference.yaml \
    --urdf_file=$PWD/$ROBOT/drc_atlas_description/urdf/atlas.urdf \
    --mjcf_file=$PWD/$ROBOT/drc_atlas_description/urdf/atlas.xml \
    --network_config=$PWD/config/ipc/network.textproto --realtime_priority=80
```

and the MPC node ([`humanoid_centroidal_mpc_app`](../../humanoid_centroidal_mpc_app/README.md)) with the same files, on
this machine or another one of the network file. Start them in either order: the robot comes up in ZERO_TORQUE, held by
the gantry in simulation, and streams `robot/mpc_observation` until the MPC answers.

## The command line

The robot binaries take the flags they share with the MPC node (`--robot_name`, `--task_file`, `--reference_file`,
`--urdf_file`, `--network_config`; [`../node`](../node/README.md#the-command-line); `--gait_file` is the MPC node's and
not read) and these:

<!-- LINT.IfChange(robot_flags) -->
| Flag | Default | What |
|---|---|---|
| `--mjcf_file` | | the MuJoCo scene; required by `--backend=mujoco` |
| `--ipc_node` | `robot` | the bus node the process publishes as |
| `--backend` | `mujoco` | the robot backend, by name (`RobotBackendRegistry`) |
| `--realtime_priority` | `0` | `SCHED_FIFO` priority of the realtime thread, 1-99, with the memory locked; 0 = the time-sharing scheduler |
| `--realtime_cores` | `default` | cores of the realtime thread: `default` (`ThreadAffinity.h`'s MRT cores of the process's CPU set, a container's cpuset included), `none`, or a list (`4,5`) |
| `--backend_cores` | `default` | cores of the backend's threads (the MuJoCo physics): `default` (the simulation cores of that set), `none`, or a list |
| `--mpc_link` | | retired: the robot reaches its MPC node over the bus only; any value is refused at start-up, naming the MPC node |
| `--headless` | `false` | the MuJoCo backend without its viewer (the robot-runtime image has no GL; the tests have no display) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:robot_flags) -->

A setting the process may not take - `SCHED_FIFO` without `CAP_SYS_NICE` or an rtprio limit, `mlockall` without
`CAP_IPC_LOCK` or a memlock limit, a core outside the container's cpuset - is logged and skipped; the loop runs
regardless. `default` cores are taken from the CPUs the process may use (`sched_getaffinity`), so that a container's
cpuset (`deploy_robot.sh --cpuset`) gets cores inside it.

**No optimal control problem on the robot.** The binary builds only what the controller needs:
`CentroidalMpcInterface::CreateControllerModels()` (the model settings, the Pinocchio model, the centroidal model info,
the MPC robot model and the effective one - the basis-vector decorator under `contactInputParameterization:
basis_vectors` - and the initial state) or `WBMpcInterface::CreateControllerModels()`. Nothing is taped, generated or
loaded with CppAD; the binary still links the formulation library. Its MPC is the MPC node, over the bus; the robot
binaries have no in-process MPC (`--mpc_link=in_process` was removed, and `InProcessMpcLink` serves the controller unit
tests and the lockstep closed loop of `humanoid_nmpc/humanoid_mpc_validation`, which mirrors this process's cycle).

## The backends

<!-- LINT.IfChange(backend_names) -->
| Name | Class | What |
|---|---|---|
| `mujoco` | `MujocoRobotBackend` | `robot_runtime/mujoco_sim_interface`'s `MujocoSimInterface`: the scene of `--mjcf_file`, the task file's simulator keys, the viewer (unless `--headless`), and the `cheater_sim` contact estimator |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotBackendRegistry.h:backend_names) -->

A backend is a `RobotBackend` (`RobotBackend.h`): the `robot::model::RobotHWInterfaceBase` the loop reads and writes,
`initialize()` and `start()`, whether the actuators take the action (`acceptsJointAction()`: false while the torques are
off), the measured foot forces, its safe state (`enterSafeState()`: MuJoCo's zero torque; the loop's last act when a
cycle throws or the process stops), the contact estimators only it can provide, and its simulator, if it is one. A new
backend is one `RobotBackendRegistry::add()`; an unknown `--backend` is refused with the list of the available names.

**No hardware backend exists yet.** `RobotProcess` drives the simulator's gantry and torque switch through the
`SimFsmBridge` and the `SimFallRecovery`, so `RobotProcess::Create()` refuses a backend without a simulator
(Unimplemented). A hardware backend brings its own FSM bridge (the torque switch of its drives, no gantry); until then the
robot deployment runs the MuJoCo backend headless on the robot computer, which is a true split-machine simulation.

## The realtime thread

`RealtimeLoopRunner` runs the loop on a thread of its own: `robot::realtime::configureCurrentThread()` from
`--realtime_priority` and `--realtime_cores` (name `robot_rt`; with a priority: `mlockall`, a prefaulted stack and
`SCHED_FIFO`), then the cycle once per period of `mpc.mrtDesiredFrequency`, paced by `robot::realtime::PeriodicTimer`
on absolute deadlines (overruns skip the missed periods: a controller acts on the newest state). Every cycle goes into
`robot::realtime::LoopTimingStats`; when its one-second window closes, the snapshot goes through a `robot::TripleBuffer`
to the communication thread, which publishes `robot/loop_timing`. Its `overruns` count only cycles whose compute time
exceeded the period; `missed_periods` counts the periods the timer skipped, also those lost to a late wake-up (a
scheduler latency, a blocked write), and `max_lateness_s` the latest wake-up of the window, so that a loop that loses
its rate without computing too long shows on the topic and in the Rerun bridge's status plots.

The cycle (`RobotProcess::cycle()`):

1. read the robot state from the backend;
2. take the operator's joint targets into the JOINT_PD posture;
3. hand the controller the mode and the posture (the centroidal controller the mode first, the whole-body controller
   the posture first, as their sims did) and run `computeJointControlAction()`, which writes the observation into the
   MPC link;
4. tell the viewer the contact state the policy in use plans for now (the contact timeline);
5. every telemetry decimation, copy the cycle into the telemetry ring;
6. apply a contact estimator or a contact wrench gate that changed;
7. the FSM bridge: a dodgeball throw, then one FSM command, or the gantry height slider;
8. the fall recovery, and the MPC resets and FSM states it calls for;
9. send the joint action to the backend, unless the torques are off;
10. the MPC's health into the FSM state.

The ROS sims sent the action (9) before the FSM command and the fall recovery (7, 8). A cycle that switched the
torques back on then left the backend the action latched when they went off - in WB_MPC, high gains and the inverse
dynamics' feedforward - for a whole control period, against a robot that had slumped since. Now that cycle hands it the
action it computed, ZERO_TORQUE's (no gain, no torque), and the next cycle the new mode's; and the simulator executes no
action applied before its torques came back on (`RobotHWInterfaceBase::discardAppliedJointAction()`).

A cycle that throws (a contact estimator that reports the wrong number of flags, Pinocchio) ends the loop on the
realtime thread instead of the process: `RealtimeLoopRunner` catches it, the backend goes to its safe state, and
`RobotProcess::runUntilShutdown()` returns the error, so that the binary exits with a failure and the container's
restart policy brings the robot back in ZERO_TORQUE.

### Every hand-over of the realtime thread

| From | To | What | Mechanism | Realtime side |
|---|---|---|---|---|
| realtime | IO | the MPC observation | `RemoteMpcLink`'s `robot::TripleBuffer<ObservationSlot>` (published on `robot/mpc_observation`) | copy into a preallocated slot |
| IO | realtime | the policy | `MRT_BASE` buffer, `moveToBuffer()` / `updatePolicy()` | try-lock swap; evaluated by `RealtimePolicyEvaluator` |
| IO | realtime | the policy deadline (link loss) | `std::atomic<double>` | one load per cycle |
| IO | realtime | the MPC's health | `MpcResetSupervisor` atomics | loads |
| realtime | IO | reset requests | `MpcResetSupervisor` counters (carried by the observations) | two atomic increments, no lock (a solver waiting out a back-off polls them) |
| IO | realtime | FSM commands | `robot::realtime::SpscQueue<FsmCommandEvent>`, parsed to an enum on the IO thread | `tryPop` of plain data |
| IO | realtime | joint targets | `robot::TripleBuffer<std::vector<double>>` of the whole posture, merged on the IO thread | copy into the preallocated posture |
| IO | realtime | dodgeball throws | `SpscQueue<DodgeballThrow>`, parsed with `parseDodgeballThrow()` on the IO thread | `tryPop` of plain data, then the simulator's `throwDodgeball()` (a `robot::TripleBuffer` the physics thread takes it from) |
| realtime | backend | the joint action | `RobotHWInterfaceBase`: a `robot::TripleBuffer` of the action, with the generation of the last torque switch-on | a copy into a preallocated slot |
| backend | realtime | the robot state, the measured foot forces | `RobotHWInterfaceBase` and `MujocoSimInterface`: `robot::TripleBuffer`s the physics thread publishes after every step | a copy out of the newest slot; no mjData read |
| realtime | backend | the torque switch | an atomic; the physics thread writes `dof_damping` before its next step | one store |
| IO | realtime | gantry height slider | `std::atomic<double>` (NaN until a walking command arrives) | one load |
| IO / task-file watcher | realtime | `contactEstimator`, `contact_wrench_gate` | `SpscQueue<ControllerSettingsUpdate>`: the estimator built on the IO thread and kept alive by the mailbox | pop in place; a `shared_ptr` copy, never the last owner |
| IO | controller | PD gains | the controller's `setPdGainsYaml()` (parsed on the IO thread) into its `JointPdGainsMailbox` | `receive()`: a copy between preallocated vectors |
| realtime | IO | the FSM state | `FsmStateMailbox`: `robot::TripleBuffer<msgs::FsmState>` | assignment of short strings (small-string buffer) |
| realtime | IO | telemetry | `TelemetrySampler`: `SpscQueue<msgs::RobotStateSample>` with preallocated slots | numbers written in place; joint names written once, at construction |
| realtime | IO | loop timing | `robot::TripleBuffer<LoopTimingSnapshot>` | a trivially copyable struct |
| realtime | IO | reports (log lines) of the process and of the controller | `RealtimeEventLog`: `SpscQueue<RealtimeEvent>`, the controller's `ControllerEventSink` | plain data; the IO thread formats and logs |
| IO | viewer | target patches, scaled velocity | `RemoteMpcLink::takeAnnotations()` (`robot::TripleBuffer<msgs::ViewerAnnotations>` filled when a policy is accepted), drawn by `MujocoViewerAnnotator` | not on the realtime thread |

The realtime thread calls no bus, protobuf, YAML or file function, takes no lock another thread holds, and writes no
log line: the controllers report through their `ControllerEventSink`, which `RobotProcess::Create()` sets to the
process's `RealtimeEventLog` (a diverged policy, a clock rewind, SAFETY, a new contact estimator or gate). Their
per-cycle diagnostics of the ramp into WB_MPC and of JOINT_PD are gone; the telemetry (`robot/state`: targets, gains,
feedforward and measured state of every joint) carries what they printed. The robot process adds no allocation, nor
does the backend hand-over (`:test_robot_realtime_allocations`), and the controllers' own code allocates nothing in the
observation, the passive modes and the hold of WB_MPC (`test_centroidal_mpc_mrt_joint_controller_allocations`,
`testWBMpcMrtJointControllerAllocations`). What remains:

- Pinocchio's passes over the composite base joint of the MPC models (the Euler-angle floating base) make one
  temporary each, of the joint's dynamic-size motion subspace: the centroidal map of the centroidal observation, the
  nonlinear effects of the gravity compensation (JOINT_PD, GRAVITY_COMP, the WB_MPC hold). The tests measure those
  passes and expect exactly them;

- executing a policy in WB_MPC allocates: the model accessors return vectors by value (`getJointAngles()`,
  `getGeneralizedVelocities()` through OCS2's centroidal mapping, the basis-vector decorator's wrench reconstruction)
  and the inverse dynamics builds its Jacobians per call. Its policy evaluation does not (`RealtimePolicyEvaluator`,
  bit-identical by its parity test), and a policy of a controller type the evaluator does not take falls back to
  `MRT_BASE::evaluatePolicy()`. These are temporaries freed in the cycle that made them, so in steady state the
  allocator reuses the same blocks rather than growing the heap;
- the simulator's viewer setters (`setTargetContactFlags()`, atomics).

## The communication thread

The bus's IO thread. Its handlers fill the `OperatorCommandMailbox`; its periodic callbacks empty the realtime
thread's mailboxes every 5 ms: the event log to the log, the telemetry ring to the telemetry sinks, the FSM state to
`robot/fsm_state` (on change, and again at 2 Hz), the loop timing to `robot/loop_timing`, and the newest viewer
annotations to the MuJoCo viewer. It polls the PD gains file every 100 (centroidal) or 500 (whole-body) control periods,
about 1 Hz as before, and the task file's controller-side keys once a second.

<!-- LINT.IfChange(operator_mailbox) -->
| Topic | Delivery | Into |
|---|---|---|
| `operator/fsm_command` | all | `parseFsmCommand()`; a repeat (the sequence of the command accepted last) and an unknown name are dropped |
| `operator/joint_targets` | latest | merged into the JOINT_PD posture; a joint of another robot is skipped, a value that is not finite refused |
| `operator/dodgeball_throw` | all | `parseDodgeballThrow()`; the newest throw is handed over |
| `operator/walking_velocity_command` | latest | `desired_pelvis_height`, clamped to 0.2-1.5 m: the gantry height while it is locked |
| `operator/mpc_parameters` | latest | `contactEstimator` and `contact_wrench_gate` (`parseControllerSideSettings()`, as the MPC's parameter updater reads them) |
| `operator/pd_gains` | latest | the controller's `setPdGainsYaml()` |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/OperatorCommandMailbox.cpp:gantry_height_limits, //humanoid_nmpc/docs/distributed_runtime/README.md:topic_table) -->

The walking commands' and the parameter documents' other consumer is the MPC node, on its own subscriptions.

## The task file

<!-- LINT.IfChange(robot_task_keys) -->
| Key | Default | What |
|---|---|---|
| `contactEstimator` | `cheater_sim` | the controller's measured contact state (`ContactEstimatorRegistry`; the backend adds `cheater_sim`); hot-reloaded from the file the robot reads (in `make launch-<robot>-sim` the checkout's, mounted; on a deployed robot its image's copy) and from the GUI over the bus |
| `contact_wrench_gate` | instantaneous | `debounceTime`, `rampTime` [s] of the touch-down shaping; hot-reloaded as `contactEstimator` |
| `simContactForceThreshold`, `simContactTimelineWindow`, `simVisualizations`, `gantryHold`, `simProjectile` | the simulator's | the MuJoCo backend |
| `simMaxBaseTiltAngle`, `simGantryCatchLift` | 0 | the fall recovery ([`docs/mpc_reset`](../../docs/mpc_reset/README.md)) |
| `mpcEntryBlendTime` | the controller's | [s] the ramp into WB_MPC (centroidal) |
| `safetyDecayTimeConstant` | the controller's | [s] the SAFETY decay |
| `wbMpcFeedforward` | `inverse_dynamics` | the feedforward of WB_MPC: `inverse_dynamics` or `gravity_compensation` (centroidal only; a debugging aid) |
| `telemetrySinks` | `[bus]` | where the telemetry goes, by name (`TelemetrySinkRegistry`); `[]` samples nothing |
| `telemetryFrequency` | min(100 Hz, the control rate) | [Hz] the telemetry rate |
| `mpcLink.policyTimeout` | 0.5 | [s, robot clock] the remote link's link-loss timeout (`RemoteMpcLink::Config`): longer than one TCP retransmission (Linux: 200 ms at least), so that a lost packet is not a lost link |

Retired, and refused at start-up with their replacement: `enableTelemetry` and `enable_telemetry` (now
`telemetrySinks`), `useGravityCompFeedforward` (now `wbMpcFeedforward`). A value of the wrong type is refused by its
key, where the ROS sims logged a warning and dropped the block.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotProcessSettings.cpp:robot_task_keys) -->

<!-- LINT.IfChange(telemetry_sinks) -->
| Telemetry sink | What |
|---|---|
| `bus` | `BusTelemetrySink`: every sample on `robot/state` (`RobotStateSample`, every joint by name) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h:telemetry_sink_names) -->

## Files

| File | What |
|---|---|
| `RobotProcess.h` | the process: the loop, the mailboxes, the bus callbacks |
| `RealtimeLoopRunner.h`, `RealtimeEventLog.h`, `FallRecoveryTypes.h` | the realtime thread's own building blocks (`:realtime_loop`, Abseil and `robot_runtime` only) |
| `RobotBackend.h`, `RobotBackendRegistry.h`, `MujocoRobotBackend.h` | the backends by name |
| `RobotController.h`, `MrtRobotController.h` | the MRT joint controller of either formulation, as the loop drives it |
| `RemoteMpcLinkAdapter.h` | the controllers' `MpcLink` over `ipc::RemoteMpcLink` |
| `OperatorCommandMailbox.h`, `FsmCommand.h`, `ControllerSideSettings.h`, `DodgeballThrowParser.h` | the operator's commands |
| `SimFsmBridge.h`, `FsmStateMailbox.h`, `SimFallRecovery.h`, `InitialSimState.h` | the simulator's FSM and fall recovery |
| `TelemetrySampler.h`, `TelemetrySink.h`, `BusTelemetrySink.h`, `TelemetrySinkRegistry.h` | the telemetry |
| `MujocoViewerAnnotator.h` | the MPC's viewer annotations in the MuJoCo viewer |
| `RobotProcessSettings.h`, `TaskFileWatcher.h` | the task file |
| `JointNamesByIndex.h` | the joint names in joint-index order (`RobotDescription::getJointNames()` is in hash-map order) |
| `RobotAppOptions.h`, `RobotAppFlags.h` | the command line (`:robot_app_flags` defines the flags: binaries only) |
| `test/ChildProcess.h`, `test/LoopbackNetwork.h`, `test/ScriptedOperator.h` | `:end_to_end_test_support` (testonly): a binary started as a process and stopped with SIGTERM, a loopback network file, and the operator the end-to-end tests script |

## Tests

| Target | What |
|---|---|
| `:test_realtime_loop_runner` | pacing on absolute deadlines (no drift with half the period computing), overruns counted, a snapshot per window, a clean stop, a set-up step the process may not take reported and skipped, a cycle that throws ending the loop and running the fault function once |
| `:test_robot_realtime_allocations` | the realtime side of the mailboxes, the event log, the FSM state, the telemetry sampler (a full ring included) and the loop runner allocate nothing; the realtime side does not wait while the IO thread holds the mailbox's lock |
| `:test_operator_command_mailbox` | FSM commands in order and once, aliases, unknown names; joint targets merged, the newest wins; dodgeballs; the gantry clamp; controller settings resolved, estimators kept alive, unknown names; the handlers on a loopback bus |
| `:test_sim_fsm_bridge` | the FSM against the headless simulator: torques, gantry, one command per cycle, the slider, controller resets and health in the state, joint targets, the dodgeball |
| `:test_sim_fall_recovery`, `:test_base_tilt_angle` | the fall recovery against the real simulator (ported), and its reports |
| `:test_dodgeball_throw_parser` | the GUI's golden payload (shared with remote_control's test) and every refusal (ported) |
| `:test_telemetry_sampler` | decimation, the sample round-tripped through `robot/state`'s message, the joint names, drops |
| `:test_robot_backend_registry` | `mujoco` by name (headless Atlas), an unknown name listing the available ones, the configuration from the options |
| `:test_robot_process_settings` | defaults, every key, wrong types, the retired booleans, every shipped task file |
| `:test_robot_app_options` | the retired `--mpc_link` refused with its replacement, the core lists, the `MpcLink` over the remote link handed to the controller once |
| `:test_realtime_event_log` | the reports' order, wording and drops, the controller's reports through it |
| `:test_robot_process` | the process without an MPC: a scripted controller, the headless simulator, a loopback bus and an operator - modes, `robot/state`, `robot/fsm_state`, `robot/loop_timing` (skipped periods and lateness included), the period, settings from the GUI and the task file, PD gains, the gantry's resets, health, `stop()`; the cycle that switches the torques on handing the backend its own action; a cycle that throws stopping the loop with the backend in its safe state |
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:test_centroidal_mpc_robot_end_to_end` | the robot binary as a process against an `MpcServer` and an operator: WB_MPC, a policy applied, the period held, the JOINT_PD hold on link loss, SIGTERM |
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:test_centroidal_mpc_node` (its last case) | the robot binary and the MPC node binary as two processes, as every simulation launch runs them: WB_MPC, the gantry released, standing, walking forward and stopping without a fall, the period held, fresh policies, `viz/scene` on the bus, SIGTERM |
