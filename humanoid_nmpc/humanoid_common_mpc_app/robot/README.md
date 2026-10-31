# humanoid_common_mpc_app/robot

The robot process of the distributed runtime ([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../docs/distributed_runtime/README.md)):
the binary that runs on the robot's realtime computer, and (with the MuJoCo backend) in simulation, in the same
topology. It is the realtime loop over a **robot backend** chosen by name and an **MRT joint controller** of either
formulation, and the mailboxes, telemetry and FSM around it on the bus.

<!-- LINT.IfChange(robot_binaries) -->
| Binary | Bus node | What |
|---|---|---|
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_robot` | `robot` | the centroidal MRT joint controller on a backend |
| `//humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_robot` | `robot` | the whole-body MRT joint controller on a backend |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/BUILD.bazel:robot_binaries, //humanoid_nmpc/humanoid_wb_mpc_app/BUILD.bazel:robot_binaries) -->

```bash
ROBOT=robot_models/drc_atlas
bazel run //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_robot -- --robot_name=drc_atlas \
    --task_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/mpc/task.textproto \
    --reference_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/command/reference.textproto \
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
| `--viewer` | | the MuJoCo backend's viewer, by name (`MujocoViewerRegistry`), or a comma-separated list tried in order: `glfw` (a window on `DISPLAY`), `web` (offscreen, served to a browser at `--web_viewer_address`:`--web_viewer_port`), `none` (the robot-runtime image, the tests); `web,none` falls back to none when the web viewer cannot start, `web` alone refuses to start then. An unknown name is refused with the list. Empty: the `mujoco_viewer` of `--plot_config`, else `web` (`kDefaultViewerName`) |
| `--plot_config` | | a plot configuration (`humanoid_rerun_viewer_proto.PlotConfig`, the Rerun bridge's `--plot_config`; the launch files pass `humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto`) whose `mujoco_viewer` names the viewer while `--viewer` is empty, `web` when it lists none. Read only then, strictly (`nproto::LoadTextprotoFile()`: a syntax error or an unknown field at its `<file>:<line>:<column>`), and a bad file or viewer name is refused at start-up |
| `--web_viewer_address` | `127.0.0.1` | the address the web viewer binds: loopback, so only this machine reaches it (an SSH tunnel or VS Code's port forwarding from another); `0.0.0.0` or an interface's address serves every client that reaches it, without authentication (a bridge-networked container needs it) |
| `--web_viewer_port` | `6090` | the web viewer's TCP port (`0`: any free one, logged) |
| `--web_viewer_allowed_hosts` | | host names the web viewer answers besides `localhost`, `127.0.0.1`, `::1` and the address it binds, comma-separated and without ports (a machine's LAN name with `--web_viewer_address=0.0.0.0`; `name:6090` is refused at start-up); any other `Host` is refused |
| `--viewer_cores` | `default` | cores of the web viewer's threads (rendering, the server): `default` (`ThreadAffinity.h`'s viewer cores: those no other partition takes, else the simulation cores, never the MPC's), `none`, or a list; the threads run under `SCHED_IDLE`, so the physics preempts them at once |
| `--config_store_dir` | | the persistent directory of this robot configuration ([The configuration store](#the-configuration-store)), seeded from `--task_file`, `--reference_file` and the PD gains beside them; empty: read those files in place |
| `--config_seed` | `when_bundle_changes` | when a stored copy is replaced by the bundled file, by name: `when_bundle_changes`, `every_start`, `never` |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:robot_flags) -->

A setting the process may not take (`SCHED_FIFO` without `CAP_SYS_NICE` or an rtprio limit, `mlockall` without
`CAP_IPC_LOCK` or a memlock limit, a core outside the container's cpuset) is logged and skipped; the loop runs
regardless. `default` cores are taken from the CPUs the process may use (`sched_getaffinity`), so that a container's
cpuset (`deploy_robot.sh --cpuset`) gets cores inside it.

**The MuJoCo viewer** is chosen by name, never by a boolean: `--viewer` when it is given (the Makefile's `VIEWER=`,
`WB_ROBOT_VIEWER`), else the `mujoco_viewer` of the plot configuration `--plot_config` names, else `web`
(`viewerNamesFromFlags()`, which logs where the choice came from). So `make launch-<robot>-sim` opens the web viewer at
`http://localhost:6090` from the shipped plot configuration alone, and the robot-runtime image, whose entry point passes
`--viewer=none`, never reads the file's choice unless `WB_ROBOT_PLOT_CONFIG` asks for it. `--viewer=web` renders
offscreen through EGL on threads of its own (the `--viewer_cores`, under `SCHED_IDLE`; never the realtime or the physics thread) and serves a page at
`http://localhost:6090` that shows the simulation and takes the viewer's mouse and keys (orbit, pan, zoom, the
visualizations' hotkeys, camera reset and tracking). The page moves the viewer's camera and markers only: nothing it
sends reaches the physics, the controller, the gantry and its weld, or the bus, which stay the operator GUI's. It binds
loopback by default; another machine reaches it through `ssh -L 6090:127.0.0.1:6090 <host>` (or VS Code's port
forwarding), and an `--web_viewer_address` other than loopback is an explicit opt-in that the viewer warns about, with
its `Host` checks still on (`robot_runtime/mujoco_sim_interface/README.md`, "Viewers", has the trust model). Without
EGL (`robot-runtime`), or when its port is taken, `--viewer=web` refuses to start, naming the fix; `--viewer=web,none`
runs without a viewer instead. `MUJOCO_EGL_DEVICE_ID` picks the EGL device (a GPU) to render on, read once at start-up.

**No optimal control problem on the robot.** The binary builds only what the controller needs:
`CentroidalMpcInterface::CreateControllerModels()` (the model settings, the Pinocchio model, the centroidal model info,
the MPC robot model and the effective one, the basis-vector decorator under `contact_input_parameterization:
"basis_vectors"`, and the initial state) or `WBMpcInterface::CreateControllerModels()`. Nothing is taped, generated or
loaded with CppAD; the binary still links the formulation library. Its MPC is the MPC node, over the bus; the robot
binaries have no in-process MPC (`InProcessMpcLink` serves the controller unit tests and the lockstep closed loop of
`humanoid_nmpc/humanoid_mpc_validation`, which mirrors this process's cycle).

## The backends

<!-- LINT.IfChange(backend_names) -->
| Name | Class | What |
|---|---|---|
| `mujoco` | `MujocoRobotBackend` | `robot_runtime/mujoco_sim_interface`'s `MujocoSimInterface`: the scene of `--mjcf_file`, the task file's simulator keys, the viewer of `--viewer`, and the `cheater_sim` contact estimator |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotBackendRegistry.h:backend_names) -->

A backend is a `RobotBackend` (`RobotBackend.h`): the `robot::model::RobotHWInterfaceBase` the loop reads and writes,
`initialize()` and `start()`, whether the actuators take the action (`acceptsJointAction()`: false while the torques are
off), the measured foot forces, its safe state (`enterSafeState()`: MuJoCo's zero torque; the loop's last act when a
cycle throws or the process stops), the switch back on (`enableTorques()`), the contact estimators only it can provide,
the supervisor that drives it (`supervisorName()`) and its simulator, if it is one. A new backend is one
`RobotBackendRegistry::add()`; an unknown `--backend` is refused with the list of the available names.

### The supervisors

The FSM bridge (`FsmBridge`: the operator's FSM commands, the JOINT_PD posture, the FSM state) and the fall recovery
(`FallRecovery`: what a fall or a jump of the plant does) are interfaces; the backend names the pair that drives it,
and `RobotSupervisorRegistry` builds it:

<!-- LINT.IfChange(supervisor_names) -->
| Supervisor | FSM bridge | Fall recovery | Backends |
|---|---|---|---|
| `sim` | `SimFsmBridge`: the simulator's torque switch, its virtual gantry and height slider, the dodgeballs | `SimFallRecovery`: the catch on the gantry, the simulator's own resets, the settle sequence ([`docs/mpc_reset`](../../docs/mpc_reset/README.md)) | `mujoco` |
| `hardware` | `HardwareFsmBridge`: the backend's `enableTorques()` / `enterSafeState()`; the gantry commands are reported and ignored, the FSM state says the gantry is unlocked | `HardwareFallRecovery`: past `sim_max_base_tilt_angle` under torque, the safe state and ZERO_TORQUE, once; no discontinuity, no relocated base | a robot's own; `test/FakeRobotBackend.h` in the tests |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotSupervisorRegistry.h:supervisor_names) -->

The `sim` supervisor of a backend that is no simulator is refused at `RobotProcess::Create()`, naming `hardware`.

### A hardware backend

No robot's backend ships yet; one plugs in like this, and `:test_hardware_supervisor` runs the whole process on the
test-only `FakeRobotBackend`, which does exactly it with a thread in place of the drives:

1. A `robot::model::RobotHWInterfaceBase` subclass whose driver thread (the robot's EtherCAT or vendor SDK loop) calls
   `publishRobotState()` with every new reading and `takeJointAction()` for every command it sends: the joints, the
   contact flags of its foot sensors, the IMU's orientation and angular velocity as the base's, the IMU's specific force
   (`setSpecforceBaseRtWorldEwrtBase()`, at the base's origin and in its frame: an IMU off the origin compensates
   its lever arm), and for the base's position and linear velocity its best guess (the initial pose, a motion capture).
   It executes an action only while `takeJointAction()` says it is fresh.
2. A `RobotBackend` around it: `acceptsJointAction()` / `enableTorques()` / `enterSafeState()` are the drives' torque
   switch (the safe state is the drives' damping), `readMeasuredContactForces()` its foot force sensors (zero without),
   `registerContactEstimators()` any estimator only it provides, `supervisorName()` `kHardwareSupervisorName`, and
   `simulator()` stays nullptr. Everything the realtime thread calls is lock-free and allocates nothing.
3. One `RobotBackendRegistry::add()` under the robot's name, with an `OptionsCheck` of its options, and the name in the
   `backend_names` table above; `--backend=<name>` (`WB_ROBOT_BACKEND`,
   `make deploy-robot BACKEND=`) selects it.
4. The task file's `state_estimator: "kalman"`, whose base position and velocity come from the IMU and the leg
   kinematics instead of the backend's guess (`kalman_state_estimator` its noise model), and `contact_estimator:
   "robot_state"` for the foot sensors' flags.

## The realtime thread

`RealtimeLoopRunner` runs the loop on a thread of its own: `robot::realtime::configureCurrentThread()` from
`--realtime_priority` and `--realtime_cores` (name `robot_rt`; with a priority: `mlockall`, a prefaulted stack and
`SCHED_FIFO`), then the cycle once per period of `mpc.mrt_desired_frequency`, paced by `robot::realtime::PeriodicTimer`
on absolute deadlines (overruns skip the missed periods: a controller acts on the newest state). Every cycle goes into
`robot::realtime::LoopTimingStats`; when its one-second window closes, the snapshot goes through a `robot::TripleBuffer`
to the communication thread, which publishes `robot/loop_timing`. Its `overruns` count only cycles whose compute time
exceeded the period; `missed_periods` counts the periods the timer skipped, also those lost to a late wake-up (a
scheduler latency, a blocked write), and `max_lateness_s` the latest wake-up of the window, so that a loop that loses
its rate without computing too long shows on the topic and in the Rerun bridge's status plots.

The cycle (`RobotProcess::cycle()`):

1. read the robot state from the backend, through the task file's state estimator (`state_estimator`): `ground_truth`
   hands it on as it is, `kalman` replaces the base position and linear velocity with its estimate
   (`humanoid_state_estimation`); what every later step reads;
2. take the operator's joint targets into the JOINT_PD posture;
3. hand the controller the mode and the posture (the centroidal controller the mode first, the whole-body controller
   the posture first: `CycleInputOrder`) and run `computeJointControlAction()`, which writes the observation into the
   MPC link;
4. tell the viewer the contact state the policy in use plans for now (the contact timeline);
5. every telemetry decimation, copy the cycle into the telemetry ring;
6. apply a contact estimator or a contact wrench gate that changed;
7. the FSM bridge: a dodgeball throw, then one FSM command, or the gantry height slider (the `sim` supervisor's; the
   `hardware` one takes the FSM command only);
8. the fall recovery, and the MPC resets and FSM states it calls for; a base the backend relocated (the simulator's
   reset or catch) starts the state estimator again from the backend's state;
9. send the joint action to the backend, unless the torques are off;
10. the MPC's health into the FSM state.

The action (9) goes to the backend after the FSM command and the fall recovery (7, 8). Sent before them, a cycle that
switched the torques back on would leave the backend the action latched when they went off - in WB_MPC, high gains and
the inverse dynamics' feedforward - for a whole control period, against a robot that had slumped since. That cycle
hands it the action it computed, ZERO_TORQUE's (no gain, no torque), and the next cycle the new mode's; and the
simulator executes no action applied before its torques came back on (`RobotHWInterfaceBase::discardAppliedJointAction()`).

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
| IO | realtime | dodgeball throws | `SpscQueue<DodgeballThrow>`, converted with `dodgeballThrowFromMessage()` on the IO thread | `tryPop` of plain data, then the simulator's `throwDodgeball()` (a `robot::TripleBuffer` the physics thread takes it from) |
| realtime | backend | the joint action | `RobotHWInterfaceBase`: a `robot::TripleBuffer` of the action, with the generation of the last torque switch-on | a copy into a preallocated slot |
| backend | realtime | the robot state, the measured foot forces | `RobotHWInterfaceBase` and `MujocoSimInterface`: `robot::TripleBuffer`s the physics thread publishes after every step | a copy out of the newest slot; no mjData read |
| realtime | backend | the torque switch | an atomic; the physics thread writes `dof_damping` before its next step | one store |
| IO | realtime | gantry height slider | `std::atomic<double>` (NaN until a walking command arrives) | one load |
| IO / task-file watcher | realtime | `contact_estimator`, `contact_wrench_gate` | `SpscQueue<ControllerSettingsUpdate>`: the estimator built on the IO thread and kept alive by the mailbox | pop in place; a `shared_ptr` copy, never the last owner |
| IO | controller | PD gains | the controller's `setPdGains()` (resolved on the IO thread) into its `JointPdGainsMailbox` | `receive()`: a copy between preallocated vectors |
| realtime | IO | the FSM state | `FsmStateMailbox`: `robot::TripleBuffer<msgs::FsmState>` | assignment of short strings (small-string buffer) |
| realtime | IO | telemetry | `TelemetrySampler`: `SpscQueue<msgs::RobotStateSample>` with preallocated slots | numbers written in place; joint names written once, at construction |
| realtime | IO | loop timing | `robot::TripleBuffer<LoopTimingSnapshot>` | a trivially copyable struct |
| realtime | IO | reports (log lines) of the process and of the controller | `RealtimeEventLog`: `SpscQueue<RealtimeEvent>`, the controller's `ControllerEventSink` | plain data; the IO thread formats and logs |
| IO | viewer | target patches, scaled velocity | `RemoteMpcLink::takeAnnotations()` (`robot::TripleBuffer<msgs::ViewerAnnotations>` filled when a policy is accepted), drawn by `MujocoViewerAnnotator` | not on the realtime thread |

The realtime thread calls no bus, protobuf, configuration or file function, takes no lock another thread holds, and writes no
log line: the controllers report through their `ControllerEventSink`, which `RobotProcess::Create()` sets to the
process's `RealtimeEventLog` (a diverged policy, a clock rewind, SAFETY, a new contact estimator or gate, a policy buffer
whose try-lock keeps failing). Their per-cycle diagnostics of the ramp into WB_MPC and of JOINT_PD are gone; the
telemetry (`robot/state`: targets, gains, feedforward and measured state of every joint) carries what they printed.
OCS2's `MRT_BASE::updatePolicy()` does not log either: it never waits for the policy buffer, and a failed try-lock is
counted (`getConsecutiveFailedPolicyUpdateLocks()`, an atomic), which the controllers post once per run longer than
`MRT_BASE::kPolicyUpdateLockContentionThreshold` as `kPolicyUpdateLockContended`; the solver side destroys the policies
a reset or a discard drops after it has released the buffer's lock, so freeing them never holds that try-lock off
(`//lib/ocs2:test_mrt_policy_buffer`). The robot process adds no allocation, nor does the backend hand-over, nor the
`kalman` state estimator with its leg kinematics (`:test_robot_realtime_allocations`, which counts the realtime thread's
allocations over hundreds of whole cycles of the process on the fake backend), and neither controller allocates: the
observation, the passive modes, the hold of WB_MPC, a contact estimator swap (whose name is a view of a literal,
`ContactEstimator::getName()`), the weight compensation of WB_MPC before its first policy and the execution of a policy
in WB_MPC (`test_centroidal_mpc_mrt_joint_controller_allocations` on the wrench inputs of the G1 and the basis-vector
inputs of the Atlas, `testWBMpcMrtJointControllerAllocations`; zero allocations each). The policy is evaluated at the
lookahead and at the measurement on preallocated buffers (`LookaheadPolicyEvaluator` over `RealtimePolicyEvaluator`,
bit-identical to MRT_BASE's by its parity test), the targets are read into workspaces sized at construction, and the
rigid-body dynamics - the gravity torques, the centroidal momentum of the observation, the sole frames' orientations
that rotate basis-vector wrenches into the world, and the inverse dynamics of a policy (base free) and of the weight
compensation (base held) - run on `RealtimeRigidBodyDynamics` (humanoid_common_mpc/pinocchio_model). Pinocchio's passes
over the MPC models' composite base joint, `Composite(Translation, Spherical)` (the quaternion floating base), make a
heap temporary of the joint's dynamic-size motion subspace in every call (`crba()`, `rnea()`, `nonLinearEffects()`,
`computeJointJacobians()`, `computeCentroidalMap()`); `RealtimeRigidBodyDynamics` runs them on a twin of the model whose
root is a FreeFlyer, with the same q and the base velocity `[R' dp_W, w_B]` in place of `[dp_W, w_B]`, whose subspace
is the fixed-size identity, and takes and returns the MPC model's coordinates (`testRealtimeRigidBodyDynamics` holds
it to the MPC model's own passes and to RNEA, and its functions to zero allocations, against the composite root's
positive count). A policy the evaluator refuses (a controller type it does not take on the state, or a policy of other
dimensions) is never left to `MRT_BASE::evaluatePolicy()`, which allocates and aborts on a malformed policy: the robot
is held with the JOINT_PD action and `kPolicyRefused` is posted once, until a policy the evaluator takes is in use.
What remains:

- the simulator's viewer setters (`setTargetContactFlags()`, atomics).

## The communication thread

The bus's IO thread. Its handlers fill the `OperatorCommandMailbox`; its periodic callbacks empty the realtime
thread's mailboxes every 5 ms: the event log to the log, the telemetry ring to the telemetry sinks, the FSM state to
`robot/fsm_state` (on change, and again at 2 Hz), the loop timing to `robot/loop_timing`, and the newest viewer
annotations to the MuJoCo viewer. It polls the PD gains file every 100 (centroidal) or 500 (whole-body) control periods,
about 1 Hz, and the task file's controller-side settings once a second. Both are the typed files of
humanoid_nmpc/humanoid_mpc_config (`joint_pd_gains.textproto`, `task.textproto`; [The configuration
files](#the-configuration-files)).

<!-- LINT.IfChange(operator_mailbox) -->
| Topic | Delivery | Into |
|---|---|---|
| `operator/fsm_command` | all | `parseFsmCommand()`; a repeat (the sequence of the command accepted last) and an unknown name are dropped |
| `operator/joint_targets` | latest | merged into the JOINT_PD posture; a joint of another robot is skipped, a value that is not finite refused |
| `operator/dodgeball_throw` | all | `DodgeballThrow`, `dodgeballThrowFromMessage()`; the newest throw is handed over |
| `operator/walking_velocity_command` | latest | `cmd_pos_base_rt_world_ewrt_world_z`, clamped to 0.2-1.5 m: the gantry height while it is locked |
| `operator/mpc_parameters` | latest | `MpcParameterUpdate`: the task file's `contact_estimator` and `contact_wrench_gate` (`controllerSideSettingsFromConfig()`, as the robot's task-file watcher reads them), as a whole file: no gate block is the instantaneous gate |
| `operator/pd_gains` | latest | `JointPdGainsUpdate`, the controller's `setPdGains()` |
| `operator/config_save` | all | `ConfigFileSave`, the configuration store (`ConfigFileStore`): checked and written on its writer thread, answered on `robot/config_save_status` |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/OperatorCommandMailbox.cpp:gantry_height_limits, //humanoid_nmpc/docs/distributed_runtime/README.md:topic_table) -->

The walking commands' and the parameter updates' other consumer is the MPC node, on its own subscriptions. An
`MpcParameterUpdate` or `JointPdGainsUpdate` of another configuration - its `config_path` is not the bundled file's
(`configFileIdentity()`, `RobotProcess::Config::taskFileIdentity`, `RobotProcess::Config::pdGainsFileIdentity`) - is refused
before anything of it is taken, so that a GUI opened on another configuration of the robot (the centroidal G1's against
the whole-body G1, whose `robot_name` and joints are the same) changes nothing here.

## The configuration files

The robot process reads the robot's typed configuration files, each a textproto of one message of
[`humanoid_mpc_config`](../../humanoid_mpc_config/README.md) in the robot's MPC package
(`robot_models/<robot>/<package>/config/`):

| File | Message | What the robot process takes from it |
|---|---|---|
| `mpc/task.textproto` (`--task_file`) | `TaskFile` | the fields below, the model settings and the initial state of the controller's models |
| `command/reference.textproto` (`--reference_file`) | `ReferenceFile` | the default joint state of the controller's models |
| `controller/joint_pd_gains.textproto` (found from the task file's path) | `JointPdGainsFile` | the joint PD gains of the MRT joint controller |

A file is parsed strictly: an unknown field, a value of the wrong type or a field given twice stops the start-up with
the file, line and column, and a retired field is refused with what replaced it. Tune a file by editing it, or in the
tuning GUI ([`remote_control`](../../remote_control/README.md#tuning-the-configuration)), whose controls are the fields of the
file's schema: a change is published on the bus as the edited file (`operator/mpc_parameters`, `operator/pd_gains`),
and **Save** writes it into the file, keeping every comment and every value it did not change. The robot process
applies the task file's `contact_estimator` and `contact_wrench_gate` and the PD gains while it runs, from the bus and
from the files it watches (its stored copies, below, which the GUI's **Save** writes); every other field takes effect
at the next start.

### The configuration store

With `--config_store_dir` (the robot images set it: `/var/lib/wb-humanoid-robot/config/<ROBOT>`, a directory of the
robot's host, tools/deploy/README.md) the robot reads its files from persistent copies, which the tuning GUI's **Save**
writes over the bus (humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the configuration"). The bundled files
(`--task_file`, `--reference_file` and the PD gains beside the task file) seed them (`RobotConfigDirectory`):

<!-- LINT.IfChange(config_store_layout) -->
| In the store | What |
|---|---|
| `mpc/task.textproto`, `command/reference.textproto`, `controller/joint_pd_gains.textproto` | the files the robot reads, at their paths below `config/` |
| `<file>.seed` | the bundled bytes the copy was last seeded from |
| `<file>.bak` | the copy a save or a reseeding replaced |
| `<file>.rejected` | a copy the robot did not start with where the bundle's started, replaced by the bundle's |
| `mpc/contact_planning.textproto` | the bundle's contact planner's file, mirrored at every start (removed when the bundle has none); the MPC's file, which no save writes |
| `.booting` | written at every start, removed once the start is confirmed (below) or the process stopped cleanly |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotConfigDirectory.cpp:config_store_layout, //humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/StoredConfigFileKinds.h:stored_config_backup_suffix) -->

A missing copy is seeded at every start; `--config_seed` decides when a stored copy is replaced (the previous one kept
as `.bak`):

<!-- LINT.IfChange(config_seed_policies) -->
| `--config_seed` | A stored copy is replaced |
|---|---|
| `when_bundle_changes` | when the bundled file differs from its `.seed`: a deploy that changed the file reaches the robot, a restart without one keeps its saves |
| `every_start` | at every start (the simulation's default) |
| `never` | never: only a save changes it |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotConfigDirectory.h:config_seed_policies) -->

A store that cannot be created or written is an ERROR in the log; the robot then reads the bundled files in place and
answers every save FAILED. A stored configuration never keeps the robot from starting: both binaries set up and start
through `runRobot()` (`RobotStartup.h`, `bringUpRobot()`). A start that fails on stored copies is tried on the bundled
files in place, touching no stored file: when the bundle starts, the stored copies are renamed `.rejected` and the
robot starts on the bundle's files; when it fails too (a bus port in use, a backend that does not come up), the
failure is not the stored copies', which are kept, the boot marker is withdrawn and the process ends with the error. A
`.booting` left by a start that died before it was confirmed makes the next start reject the stored copies the same
way, unless the machine booted in between: a power cut or a reboot ended that start, not its files. `.rejected` holds
one generation. Reset the store by removing its directory (tools/deploy/README.md, "The stored configuration").

<!-- LINT.IfChange(boot_confirmation_time) -->
A start is confirmed once its realtime loop has run 10 s (`kBootConfirmationTime`).
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotStartup.h:boot_confirmation_time) -->

A save (`ConfigFileStore`) is checked on the IO thread for what is cheap - the payload's schema, the file kind, the
configuration (`config_path` against the bundled file's identity, compared and never used as a path), the robot
(`robot_name`), the file message's schema fingerprint, at most 1 MiB, a repeat of a save already answered (answered
again, not written again) - and queued (4 at most; one more is refused as busy). The store's writer thread, never the
realtime thread, then parses the text strictly and runs `checkRobotConfiguration()` - the checks start-up runs: the
task file's `robot_name`, the robot process's settings, the formulation's controller models, the PD gains on the
controller's joints, the contact estimator against the backend's registry and its probe, the telemetry sinks and the
backend's options (`RobotBackendRegistry::checkOptions()`) - on the stored files with the saved one in place. It answers
`UNCHANGED` when the copy holds the text already, and otherwise keeps the copy as `.bak` and writes the text atomically
(`AtomicFileWrite.h`: a temporary of the process's pid, fsync, rename, directory fsync), answering `SAVED` - saying so when the file
was replaced but its directory could not be synced, so that a power loss may undo it - or `FAILED` with the copy
untouched. The watchers then apply a saved task file's controller-side fields and PD gains as they apply
an edit. Without `--config_store_dir` every save is answered `NOT_STORED`, naming the bundled file the robot reads in
place: the laptop's file itself for a robot process of the same checkout (`bazel run` in the dev container), which the
GUI then reports in sync; a file the save did not change otherwise, which it reports as a problem.

### The task file

<!-- LINT.IfChange(robot_task_keys) -->
| Field | Absent | What |
|---|---|---|
| `contact_estimator` | `"cheater_sim"` | the controller's measured contact state (`ContactEstimatorRegistry`; the backend adds `cheater_sim`); hot-reloaded |
| `contact_wrench_gate` | the instantaneous gate | `debounce_time`, `ramp_time` [s] of the touch-down shaping; hot-reloaded as `contact_estimator` |
| `state_estimator` | `"ground_truth"` | the base state the controller is handed (`StateEstimatorRegistry`): `"ground_truth"` the backend's, `"kalman"` estimated from the IMU and the leg kinematics; read at start-up |
| `kalman_state_estimator` | its defaults | the noise model of `"kalman"` (`kalman_state_estimator_config.proto`): the IMU's noise, the random walks of the base and of the stance and swing feet, the kinematics' noise, the stance feet's height |
| `sim_min_force_mjcontact_ewrt_mjcontact_x`, `sim_contact_timeline_window`, `sim_visualizations`, `gantry_hold`, `sim_projectile` | the simulator's; no viewer marker | the MuJoCo backend |
| `sim_ragdoll_joint_damping`, `sim_auto_reset_min_pos_base_rt_world_ewrt_world_z` | 20 N m s/rad, 0.2 m | the MuJoCo backend: the robot joints' damping while the torques are off (on, they have the MJCF's own), and the base height below which the robot is reset |
| `sim_max_base_tilt_angle`, `sim_gantry_catch_lift` | absent (off) | the fall recovery ([`docs/mpc_reset`](../../docs/mpc_reset/README.md)) |
| `mpc_entry_blend_time` | the controller's | [s] the ramp into WB_MPC (centroidal) |
| `safety_decay_time_constant` | the controller's | [s] the SAFETY decay |
| `wb_mpc_feedforward` | `"inverse_dynamics"` | the feedforward of WB_MPC: `"inverse_dynamics"` or `"gravity_compensation"` (centroidal only; a debugging aid) |
| `telemetry_sinks` | none | where the telemetry goes, by name (`TelemetrySinkRegistry`), one entry per sink (`telemetry_sinks: "bus"`); none samples nothing |
| `telemetry_frequency` | min(100 Hz, the control rate) | [Hz] the telemetry rate |
| `mpc_link { policy_timeout }` | 0.5 | [s, robot clock] the remote link's link-loss timeout (`RemoteMpcLink::Config`): longer than one TCP retransmission (Linux: 200 ms at least), so that a lost packet is not a lost link |

Retired, and refused at start-up with their replacement: `enable_telemetry` (now `telemetry_sinks`),
`use_gravity_comp_feedforward` (now `wb_mpc_feedforward`). A value the robot process cannot run with is refused by its
field.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:robot_task_keys, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:robot_controller_keys, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:retired_robot_keys) -->

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
| `OperatorCommandMailbox.h`, `FsmCommand.h`, `config/robot/DodgeballThrowFromMessage.h` | the operator's commands |
| `FsmBridge.h`, `FallRecovery.h`, `RobotSupervisorRegistry.h`, `FsmBridgeState.h`, `FsmStateMailbox.h` | the supervisors by name, and what every FSM bridge keeps |
| `SimFsmBridge.h`, `SimFallRecovery.h`, `InitialSimState.h` | the simulator's FSM and fall recovery (`sim`) |
| `HardwareFsmBridge.h`, `HardwareFallRecovery.h` | the FSM and the fall cut-off of a backend that is no simulator (`hardware`) |
| `test/FakeRobotBackend.h`, `test/RobotTestSupport.h` | `:robot_test_support` (testonly): a backend that is no simulator, a scripted controller, the Atlas files, loopback buses |
| `TelemetrySampler.h`, `TelemetrySink.h`, `BusTelemetrySink.h`, `TelemetrySinkRegistry.h` | the telemetry |
| `MujocoViewerAnnotator.h` | the MPC's viewer annotations in the MuJoCo viewer |
| `RobotProcessSettings.h`, `config/robot/RobotProcessSettingsFromConfig.h`, `TaskFileWatcher.h` | the task file |
| `JointNamesByIndex.h` | the joint names in joint-index order (as `RobotDescription::getJointNames()`) |
| `RobotAppOptions.h`, `RobotAppFlags.h` | the command line (`:robot_app_flags` defines the flags: binaries only) |
| `RobotConfigDirectory.h`, `AtomicFileWrite.h` | the configuration store's files: seeding, the seed policies, the boot marker, crash-safe writes |
| `ConfigFileStore.h`, `RobotConfigurationCheck.h` | a save of `operator/config_save` checked as start-up checks the files, and written |
| `RobotStack.h`, `RobotStartup.h` | what a robot binary sets up, and the start both binaries share: the fallback to the bundled files, the boot confirmation |
| `RobotSetUpSteps.h`, `StoredConfigFileKinds.h` | the steps of the set-up that do not depend on the formulation (the configuration read once, the check context with the controller's PD gains defaults, the process's hooks and configuration), and the kinds of file the store keeps |
| `test/ChildProcess.h`, `test/LoopbackNetwork.h`, `test/ScriptedOperator.h` | `:end_to_end_test_support` (testonly): a binary started as a process and stopped with SIGTERM, a loopback network file, and the operator the end-to-end tests script |

## Tests

| Target | What |
|---|---|
| `:test_realtime_loop_runner` | pacing on absolute deadlines (no drift with half the period computing), overruns counted, a snapshot per window, a clean stop, a set-up step the process may not take reported and skipped, a cycle that throws ending the loop and running the fault function once |
| `:test_robot_realtime_allocations` | the realtime side of the mailboxes, the event log, the FSM state, the telemetry sampler (a full ring included) and the loop runner allocate nothing; the realtime side does not wait while the IO thread holds the mailbox's lock; the whole cycle of the process on the fake backend with the `kalman` state estimator, FSM commands included, allocates nothing |
| `:test_hardware_supervisor` | the `hardware` supervisor: the FSM commands switch the backend's torques, the gantry commands are ignored, a tilt past the limit under torque switches them off once; the `sim` supervisor refused for a backend that is no simulator; the whole process running its cycles over the bus on the fake backend with either state estimator |
| `:test_operator_command_mailbox` | FSM commands in order and once, aliases, unknown names; joint targets merged, the newest wins; dodgeballs; the gantry clamp; controller settings resolved, estimators kept alive, unknown names, a whole task file without a gate being the instantaneous gate; PD gains to the controller; the handlers, typed tuning payloads included, on a loopback bus |
| `:test_sim_fsm_bridge` | the FSM against the headless simulator: torques, gantry, one command per cycle, the slider, controller resets and health in the state, joint targets, the dodgeball |
| `:test_sim_fall_recovery`, `:test_base_tilt_angle` | the fall recovery against MuJoCo (ported), and its reports |
| `:test_initial_sim_state` | the simulator's initial state from the MPC's in every formulation: the base orientation through the robot model's `getQuatBaseRtWorldXyzw()` (the stored unit quaternion bit for bit, the rotation of the Euler angles it was set from to round-off, an unwrapped yaw included), the MPC joints by name, every other joint zero |
| `:config_robot_test` | the typed task file's robot settings and the GUI's typed `DodgeballThrow` (`test/data/dodgeball_throw.textproto`, shared with remote_control's test) converted, every refusal |
| `:test_telemetry_sampler` | decimation, the sample round-tripped through `robot/state`'s message, the joint names, drops |
| `:test_robot_backend_registry` | `mujoco` by name (Atlas, `--viewer=none`), an unknown name listing the available ones, the configuration from the options, the viewer settings copied and checked (an unknown viewer, `glfw,none`, a web option out of range refused) |
| `:test_robot_process_settings` | the task file read strictly, errors naming the file; the defaults, every field, wrong types and refused values, the retired booleans; every shipped task file; a YAML task file refused naming the file |
| `:test_task_file_watcher` | a write reported once, an unchanged or vanished file never |
| `:test_config_file_store` | a save written atomically on the writer thread with the replaced copy kept; unchanged; every refusal (malformed, an unknown field, another robot, configuration or fingerprint, oversize, a contact estimator, telemetry sink or simulator option the robot refuses) with the stored copy and its time untouched; a store that cannot be written; no store; a full queue; a repeated save answered again and not rewritten; `stop()`; `NOT_STORED` naming the file read in place |
| `:test_robot_configuration_check` | each check of `checkRobotConfiguration()` refusing on its own, naming the field; a saved file checked in place of its stored copy |
| `:test_robot_config_directory` | an empty store seeded, a restart keeping a save, the seed policies, the contact planner's mirror, a store that cannot be written, the boot marker's fallback (not after the machine booted) and `rejectStoredCopies()`; `bringUpRobot()` rejecting the stored copies only when the bundle starts where they did not, and keeping them, with the marker withdrawn, for a failure the bundle shares |
| `:test_atomic_file_write` | a write replacing the file and leaving no temporary; a write that fails before the rename leaving the file as it was, which is no DataLoss |
| `:test_robot_app_options` | the core lists, the viewer by name (`web,none` parsed, `vnc` refused listing `glfw, web, none`, nothing after `glfw` or `none`), its precedence (`--viewer`, else the `mujoco_viewer` of `--plot_config`, else `web`; the file not read with `--viewer`), a bad plot configuration refused (an unknown name, a list the registry refuses, a misspelled field at its line and column, another message's file, a missing file), the shipped plot configurations' `web`, the web viewer's port, address and allowed hosts, `--viewer_cores=default` as the allocation's viewer cores, `MUJOCO_EGL_DEVICE_ID`, `LP_NUM_THREADS` for the web viewer's cores and its log line, the `MpcLink` over the remote link handed to the controller once |
| `:test_realtime_event_log` | the reports' order, wording and drops, the controller's reports through it |
| `:test_robot_process` | the process without an MPC: a scripted controller, the headless simulator, a loopback bus and an operator - modes, `robot/state`, `robot/fsm_state`, `robot/loop_timing` (skipped periods and lateness included), the period, settings from the GUI and the task file, PD gains, the gantry's resets, health, `stop()`; the cycle that switches the torques on handing the backend its own action; a cycle that throws stopping the loop with the backend in its safe state; a save on the bus stored and applied by the task file's watcher; an MPC parameter update of another configuration installing nothing; the boot report of `runUntilShutdown()` |
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:test_config_save_end_to_end` | the robot binary as a process with a store: saves of the task file and the PD gains stored, refusals, a restart on the saved task file (its telemetry rate), and stored files the robot refuses (or a start that did not confirm its boot) falling back to the bundle, leaving `.rejected` |
| `//humanoid_nmpc/remote_control:test_two_copy_save_end_to_end` | the GUI's tuning tabs against the robot binary with a store: each Save leaves the laptop's bytes in the store, refusals leave it untouched, a restart runs the saved files |
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:test_centroidal_mpc_robot_end_to_end` | the robot binary as a process against an `MpcServer` and an operator: WB_MPC, a policy applied, the period held, the JOINT_PD hold on link loss, SIGTERM |
| `//humanoid_nmpc/humanoid_centroidal_mpc_app:test_centroidal_mpc_node` (its last case) | the robot binary and the MPC node binary as two processes, as every simulation launch runs them: WB_MPC, the gantry released, standing, walking forward and stopping without a fall, the period held, fresh policies, `viz/scene` on the bus, SIGTERM |
