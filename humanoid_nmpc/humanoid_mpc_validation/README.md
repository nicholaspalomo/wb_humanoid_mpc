# humanoid_mpc_validation

The measurement infrastructure of the switch from Euler angles to a quaternion base orientation
(`humanoid_nmpc/docs/quaternion_base_orientation/README.md`, Step 1): golden files, the lockstep MuJoCo closed loop with
its metrics, and the solve benchmark. It runs the production MPC, as the MPC node builds it, and the production MRT joint
controller, as the robot binary builds it, in one thread and without the bus.

```
 scenario (section 4.5)      GUI message            the MPC node's conversion       ProceduralMpcMotionManager
 commandAt(t) ----------> toGuiVelocityCommand --> walkingVelocityCommandFromProto --> setAndScaleVelocityCommand
                                                                                                  |
 MujocoSimInterface (headless,                   RobotController (MrtRobotController)            v next solve
 MujocoRobotBackend::makeConfig())               over the MRT joint controller         InProcessMpcLink
   step x (1 / mrt_desired_frequency) / dt <-- prepareCycle(), computeJointControlAction()  (Execution::kCaller)
   getRobotState() ------------------------->        |  observations, policies  <------>  runSolverIteration()
         |                                                                               every 1 / mpc_desired_frequency
         +--------------------> ClosedLoopMetrics <--- solve times, the SQP's delta_x0
                                      |
                                <robot>_<scenario>.json (ClosedLoopMetricsSchema)
```

## What is here

| Piece | Files | What it does |
|---|---|---|
| Golden files | `io/GoldenIo.h`, `io/Sha256.h`, `io/RunProvenance.h` | Labeled matrices at `%.17g` (NaN, infinities and the sign of zero kept), with a header carrying the commit, the worktree state, the machine and the SHA-256 of every configuration file. |
| JSON | `io/JsonValue.h` | The documents the metrics and the benchmark write; insertion-ordered, shortest round-trip numbers. |
| Lockstep closed loop | `closed_loop/LockstepClosedLoop.h`, `closed_loop/ClosedLoopDriver.h`, `closed_loop/CentroidalClosedLoopDriver.h`, `closed_loop/WholeBodyClosedLoopDriver.h` | The headless simulator, the formulation's MPC as its MPC node builds it, its MRT joint controller as its robot binary builds it, and between the two an `InProcessMpcLink` whose solver iterations the runner calls (`InProcessMpcLink::Execution::kCaller`, `runSolverIteration()`). One thread, no bus. |
| Scenarios | `closed_loop/ClosedLoopScenario.h` | The commands of section 4.5, driven through the GUI's message. |
| Metrics | `closed_loop/ClosedLoopMetrics.h`, `closed_loop/ClosedLoopMetricsSchema.h`, `closed_loop/MetricBands.h` | The metrics of section 4.5, their JSON schema, and the bands against a baseline. |
| Solve benchmark | `benchmark/MpcSolveBenchmark.h`, `exe/benchmarkMpcSolveMain.cpp` | Section 4.6: recorded walking states replayed through the controller, solve times per phase and the tape operation count of every CppAD library. |
| Recorded data | `data/closed_loop/<label>/`, `data/benchmark/<label>/`, `data/benchmark/states/` | The baselines (M0, B0, ...) with a README each saying how they were made. |

## The lockstep closed loop

A run is reproducible and independent of the machine's speed:

1. Every control cycle (`1 / mpc.mrt_desired_frequency` of the task file, a whole number of 0.5 ms simulation steps) runs in
   the order of `RobotProcess::cycle()` (`humanoid_common_mpc_app/robot`): the runner reads the simulator's state,
   hands the controller the mode and the JOINT_PD posture (`RobotController::prepareCycle()`, in the order of the
   formulation's binary), computes the action, applies the controller-side settings of the task file, serves the operator's
   commands of this cycle, which take effect in the controller from the next cycle on, and applies the action last.
2. A solve is due every `1 / mpc.mpc_desired_frequency` of simulation time. It runs in the control cycle it falls in, after the
   action, on that cycle's observation, and its policy is in use from the next cycle on. A solve takes no simulation
   time: the run measures the controller and the formulation, not the computer or the bus. On the robot the MPC node
   solves meanwhile on another thread or machine and adds the solve's and the link's latency.
3. The operator's sequence: JOINT_PD with the torques on, on the locked gantry, for 1 s; WB_MPC (the controller resets
   the MPC and holds, then ramps over `mpc_entry_blend_time`); 0.5 s after the entry is over the gantry is released as the
   robot process releases it (`unlockGantry()`, and the reset of the MPC the fall recovery requests when it sees the
   gantry unlocked); the robot stands for the scenario's `standingTime` (2 s); then the commands, over which the metrics
   are evaluated.
4. A tilt beyond `sim_max_base_tilt_angle` (1 rad), a base below half its initial height, or a reset by the simulator is a
   fall and ends the run; the document says when and why.

Runs are reproducible, with the configured solver threads too: two runs of one configuration give the same documents
bit for bit, the whole-body ones included. `test_lockstep_determinism` holds runs of the smoke scenario with one solver
thread to 1e-9: two in one process, and a third in a process of its own, since a comparison across a code change always
compares separate processes (a per-process source of nondeterminism, such as Abseil's hash seed ordering
`RobotDescription`'s joint list, shows only there).

The whole-body runs used to differ in their last bits from one run to the next. The cause was CppAD's code generation,
not the solver threads. CppAD's recorder files every constant of a tape under a hash code, and shares a parameter only
with the constant last filed under the same code (`lib/ocs2/thirdparty/include/cppad/local/recorder.hpp`,
`put_con_par()`). CppADCodeGen defined no hash code for its `CG` type, so CppAD's default hashed the bytes of the object
(`cppad/local/hash_code.hpp`), and a `CG` holds the heap address of its value (`cppad/cg/cg.hpp`). Whether equal
constants shared a parameter therefore followed the heap's layout, and so did the subexpressions `optimize()` merges by
parameter index and the constant terms of its sums. Two generations of the whole-body G1's libraries gave different C
sources for six of its 28 models: one subexpression merged in one generation only, sums in another order, `0 - v`
against `-0 - v`. A sandboxed test generates the libraries afresh in every run, and runs that loaded the same libraries
agreed bit for bit with any number of threads. The vendored CppADCodeGen now hashes a constant by its value, and the
generator stamp regenerates every library made before (`lib/ocs2/README.md`, "Local changes to the vendored CppAD /
CppADCodeGen"). The SQP's per-thread performance sums do follow the thread schedule, but they only decide the line
search's comparisons, and no recorded run has changed from them. Comparisons across a code change use the bands below,
since a code change moves the results.

What the robot process does and the runner does not: the bus and the remote MPC link (the runner's link solves in the
loop, without latency or transport), the start in ZERO_TORQUE (the runner starts in JOINT_PD with the torques on, and
solves the first policy before the simulation runs), the FSM bridge (the runner commands the modes itself), the fall
recovery's catch-and-settle (a fall ends the run), the viewer, telemetry, and the operator's parameter, PD-gain and
Dodgeball messages. The MPC side is the MPC node's MPC, served by `InProcessMpcLink` (`MPC_MRT_Interface`) rather than
the node's `MpcServer`: both reset the MPC to the same reset target from the observation the reset is served at, and
solve the same problem. Files on disk are still watched: the parameter updater reloads the task file into the MPC, as on
the MPC node of either formulation, and the controller-side settings (`contact_estimator`, `contact_wrench_gate`) are re-read,
typed, when the task file's textproto changes, checked once a second of simulation time as the robot process checks them
once a second of wall time. The file is the whole file, as for the robot process: a task file without a
`contact_wrench_gate` block sets the instantaneous gate, one whose block is refused keeps the gate in use.
The task files are read through the runfiles, which link into the checkout, so an edit of a task file during a run
reaches the run: do not edit the configuration while a baseline is recorded.

The JOINT_PD posture is the posture the robot is spawned in (`createInitialSimState()`, before any physics step), as the
robot process's FSM bridge takes it.

### On the main line, against the baselines recorded before it

M0 and B0 were recorded on the quaternion worktree before the ROS removal, where the runner mirrored the ROS sim nodes.
On the main line it mirrors the robot process and the MPC node instead, and these differences move the recorded metrics
(numerically: the runs are not those of M0 bit for bit; compare them with the bands, never exactly):

- **The operator's FSM commands are served after the cycle's action and solve**, as `RobotProcess::cycle()` serves the
  FSM commands and the fall recovery after the action and before applying it; before, the runner served them at the
  start of the cycle, before the action. The switch to WB_MPC and the gantry release therefore take effect one control
  period later, the release's MPC reset is requested after the release cycle's solve and served by the next one, and
  the first unlocked physics steps run the action computed before that request. The end of the entry moves by one MPC
  period instead where the solve that serves the controller's reset at WB_MPC used to run in the switching cycle: it is
  now the next one. The scenario's commands are no FSM command (the GUI's message reaches the MPC node, never the robot
  process), so they still start `standingTime` after the release cycle, at the top of the cycle, as in M0.
  `entry_end_time_s`, `gantry_release_time_s` and `command_start_time_s` keep their definitions (the cycle the event is
  seen in, the first cycle of the commands); on the centroidal robots the release and the commands come 10 ms later
  than in M0, on the whole-body Unitree G1 (MPC at 60 Hz) 16 ms. Relative to the solve that serves the release's reset,
  the commands of the centroidal robots start 10 ms earlier than in M0 (1.99 s instead of 2.00 s), those of the
  whole-body G1 as in M0.
- **A failed solve's back-off ends early on a reset requested after it**, as `MpcResetSupervisor::waitBeforeRetry()`
  ends it on the solver thread (`LockstepSolveSchedule`, `resetRequestedSinceLastFailure()`), and the period counts
  from the retry. No recorded run backs off, so this changes none of them.
- **The whole-body controller gets its posture before its mode** (`CycleInputOrder::kPostureThenMode`, as the
  whole-body binary and the ROS sim did); the runner used to hand both formulations the mode first. The posture is the
  same in every cycle, so this changes nothing measured.
- **The controller-side settings come from the robot process's path** (`TaskFileWatcher`,
  `controllerSideSettingsFromConfig()`, applied as `RobotProcess::applyControllerSettings()`) for both formulations;
  the parameter updater applies the MPC's fields only. Without an edit during a run this changes nothing.
- **The task file's robot fields are read with `loadRobotProcessSettings()`**, which refuses the retired
  `use_gravity_comp_feedforward` (now `wb_mpc_feedforward`), and the simulator is configured by
  `MujocoRobotBackend::makeConfig()` (its viewer-only fields, `sim_contact_timeline_window` and `sim_visualizations`,
  change nothing headless).
- **The solve benchmark's later passes** start from a full reset requested at the start of the pass and served by its
  first solve, because the link starts once; before, each pass restarted the controller's MPC directly. The first
  `warmupSolves` of every pass are not timed either way.

`data/closed_loop/M0_main/` re-runs eleven of M0's runs on the main line: every configuration's `standing` and walking
scenario, and the whole-body G1's `turn_in_place_720`. Seven stay within the bands of M0. Four leave them, all from
the first difference above alone, as walks that change their pattern under a 10 to 16 ms shift of the operator's
timeline: EngineAI SA01's `walk_0p3` walks 13 mm higher but slips three times as much, the Unitree G1's `walk_0p5` and
the Unitree R1's `walk_0p3` tilt up to 25 % further, and the whole-body G1's maximum tilt in the 720-degree turn rises
from 0.103 to 0.147 rad at the Euler yaw's first wrap. With the runner's operator sequence moved back to the start of
the cycle, the main line reproduces M0 to a relative 2e-5 or better, and one re-recorded run bit for bit. The
whole-body walk reproduced it only to 0.5 %: that was the spread its CppAD libraries then had from one generation to the
next (above). Its README has the numbers. Its three whole-body documents and the turn's time series were recorded again
on the deterministic libraries (2026-10-06), so a whole-body run of the same code repeats them bit for bit; the walk now
slips 8.9 mm at most, against 5.2 mm before, and the turn's tilt maximum stays 0.147 rad. Its eight centroidal documents
come from libraries generated before the code generation was made deterministic, and the same runs on the
deterministic libraries differ from them at the rounding level: the walks by a relative 1.2e-9 or less, the standing
runs by 4e-8 or less (1.2e-5 in a base-height deviation of 1e-10 m). Compare a main-line Euler run with M0_main where
it has the run, within the bands. Re-record the other baselines on the main line (under new labels, before Step 3)
when a comparison has to be exact up to the bands' noise rather than up to these differences.

## Scenarios

<!-- LINT.IfChange(scenario_names) -->
| Name | Commands (then 2 s at rest, but for standing and smoke) |
|---|---|
| `standing` | at rest for 10 s |
| `walk_0p5` | 0.5 m/s forward for 15 s |
| `walk_0p3` | 0.3 m/s forward for 15 s, the slower variant for a robot limited at 0.5 m/s |
| `lateral_0p2` | 0.2 m/s to the left for 10 s |
| `turn_in_place_720` | +0.5 rad/s through 720 degrees, then -0.5 rad/s back |
| `turn_1radps` | 1 rad/s for 12.6 s |
| `arc` | 0.3 m/s forward while turning at 0.3 rad/s for 15 s |
| `smoke` | 0.3 m/s and 0.2 rad/s for 1.5 s after 0.5 s of standing: the determinism test |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h:scenario_names) -->

A command is sent as the base-controller GUI sends it: the sticks at `command / limit` (`max_displacement_velocity_x`,
`max_displacement_velocity_y`, `max_rotation_velocity` of the reference file, `guiCommandScalingFromConfig()`), clamped
to [-1, 1], and the pelvis height at `default_base_height`. A command beyond a robot's limit is cut there and the document says `command_saturated: true`.

## Metrics

<!-- LINT.IfChange(metrics_schema) -->
Every document has `schema: humanoid_mpc_validation.closed_loop_metrics.v1`, the label, robot, formulation and scenario,
an open `provenance` and `settings` object, and these fields (null where nothing was measured):

| Field | Meaning |
|---|---|
| `evaluation.{start_time_s, end_time_s, control_cycles, solves}` | the window of the commands |
| `survival.{survived, fall_time_s, fall_reason}` | the fall flag |
| `base_height.{mean_m, std_m, rms_error_m}` | the base height, and its error against the commanded height |
| `tilt.{rms_rad, max_rad}` | the tilt `|tau|` of the heading-tilt split (the angle of the base from upright) |
| `velocity.rms_error_mps` | the base velocity in the heading frame against the reference velocity (after the command filter and ramps) |
| `yaw_rate.rms_error_radps` | the world yaw rate against the reference yaw rate |
| `heading.{cumulative_final_rad, cumulative_max_rad, cumulative_min_rad}` | the twist heading, unwrapped from the first command |
| `stance_foot_slip.{max_m, rms_m, stance_phases}` | the horizontal drift of a contact frame from its touch-down while in measured contact |
| `joint_torque.rms_nm` | the commanded torque of the MPC joints |
| `initial_state_gap.{max_rotation_rad, max_rotation_time_s, max_norm}` | `delta_x0` of each solve's first SQP iteration, as the solver records it: the observation against the node 0 it warm-started from (the previous solution after `trajectorySpread`), zero after a reset the solve served first |
| `quaternion_norm.max_deviation` | `max | |xi_k| - 1 |` over the nodes of every solution; null for the Euler formulation |
| `non_finite_values` | NaN or infinite values in the observations, policy inputs and joint torques |
| `solve_time_ms.{total, lq_approximation, solve_qp, linesearch, compute_controller}.{mean, p50, p99, max}` | the solver iteration's wall time and the SQP's phases |
| `failures.{failed_solves, resets_served, full_resets_served, simulator_resets, unhealthy_cycles}` | during the commands; simulator resets over the whole run |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopMetricsSchema.cpp:metrics_schema) -->

## Bands against a baseline

<!-- LINT.IfChange(metric_bands) -->
`compareClosedLoopMetrics()` holds a run to the bands of section 4.5: survival equal or better; height within 5 mm; tilt
within max(20 %, 0.005 rad); velocity within max(15 %, 0.02 m/s); yaw rate within max(15 %, 0.02 rad/s); slip within
max(20 %, 5 mm); p99 solve time at most +10 % (only on the machine of the baseline, `--compare_solve_time`); quaternion
norm below 1e-9.

`compareClosedLoopRuns()` adds the 720-degree turn exception of section 4.5 when the baseline's heading crosses an odd
multiple of pi, the cut of the Euler formulation's wrapped yaw: the velocity and yaw-rate errors are recomputed from both
time series outside +-0.5 s windows around the baseline's crossings; the run must turn at least as far as the baseline
(and to the commanded 4 pi where the baseline got there); and its initial-state gap must stay below pi / 2, which a 2 pi
wrap exceeds. A run on Euler coordinates (no quaternion norm) wraps where its own heading crosses the cut, as its Euler
baseline does, and its gap then decays over a few solves: against a baseline whose gap reached pi / 2 too, it is checked
solve by solve from its time series' `solve_initial_state_rotation_gap`, and a solve at or above pi / 2 must lie within
+-0.5 s of one of the run's own crossings and below 2 pi + pi / 2, no more than a wrap of a gap below pi / 2. A spike
elsewhere, a larger one, or one the series cannot locate is a violation. The crossings come from the baseline's time series, so `make closed-loop-metrics`
archives the time series of the turning scenarios (`turn_*`, `arc`) beside their documents, and
`test_closed_loop_metrics` requires one for every recorded run whose heading reaches +-pi.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/MetricBands.h:metric_bands) -->

## How to run

Inside the dev container, one robot at a time (each run compiles the robot's CppAD libraries and simulates for minutes):

```bash
make closed-loop-metrics ROBOT=drc_atlas LABEL=M0                       # every scenario
make closed-loop-metrics ROBOT=unitree_g1 SCENARIO=walk_0p5 LABEL=M1 BASELINE=M0
make closed-loop-metrics ROBOT=unitree_g1 LABEL=M0 RECORD_STATES=1     # also keep the walking states
make benchmark-mpc-solve ROBOT=unitree_g1 LABEL=B0
make benchmark-mpc-solve ROBOT=unitree_g1 LABEL=B3 BASELINE=B0       # and hold it to the real-time gate of section 4.6
```

The benchmark replays `data/benchmark/states/<robot>_<walking scenario>_states.txt`: `walk_0p5`, or `walk_0p3` for
EngineAI SA01 and Unitree R1, which fall at 0.5 m/s (`RobotConfiguration::walkingScenario`). `compareSolveBenchmarks()`
(`benchmark/SolveBenchmarkGate.h`) is the gate: mean +5 %, p99 +10 %, LQ approximation +8 %, p99 below 80 % of the MPC
period, every library's tape operation count and their total +5 %, a library matched to itself across the layout-tagged
CppAD folder of Step 8 (`normalizedLibraryKey()`).

The provenance's `worktree_state` (`tools/worktree_state.sh`) hashes the diff against HEAD, the status with every
untracked file listed, and the contents of every untracked file, so that an edit of a file not yet committed changes it.
The documents recorded before this (M0, B0) hashed only the diff and the collapsed status, which leaves the untracked
files' contents out.

The configuration files a run depends on (`RobotConfiguration::configurationFiles()`) are the robot's typed textprotos
of [`humanoid_mpc_config`](../humanoid_mpc_config/README.md) - `config/mpc/task.textproto`,
`config/command/reference.textproto`, `config/controller/joint_pd_gains.textproto`, the contact planner's
`config/mpc/contact_planning.textproto` where the robot has one, and the gait file `gait.textproto` - plus its URDF and
MuJoCo scene, and the provenance records the SHA-256 of each. The documents recorded before the textproto migration
(M0, M0_main's centroidal documents and the other labels under `data/`) name and hash the YAML files they ran on, which
held the same values; M0_main's whole-body documents, recorded again since, hash the textprotos.

`ROBOT` is `drc_atlas`, `engineai_sa01`, `unitree_g1`, `unitree_r1` or `unitree_g1_wb`. The targets run the manual tests
`closed_loop_<robot>` and `benchmark_mpc_solve_<robot>` under `bazel test`, so that the machine lock of `tools/bazel`
keeps them from running next to another build, and copy the documents into `data/`.

<!-- LINT.IfChange(output_names) -->
Each closed-loop run writes `<robot>_<scenario>.json` (the metrics), `<robot>_<scenario>_timeseries.txt` (the base and
its reference at 50 Hz, and the rotation part of each solve's initial-state gap, a golden file whose matrices
`TimeSeriesLabels.h` names) and, walking, `<robot>_<scenario>_states.txt` (the robot at 300 solves from 2 s into the
commands, the input of the benchmark) into the test's undeclared outputs; the benchmark writes
`<robot>_solve_benchmark.json`.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h:output_names) -->

<!-- LINT.IfChange(recorded_state_labels) -->
A recorded-states file holds, one row per solve, `time`, `base_position`, `base_quaternion_xyzw`,
`base_linear_velocity_local`, `base_angular_velocity_local`, `joint_positions` and `joint_velocities` (every joint of the
URDF, named in the `joint_names` note), `contact_flags` and `gui_command`. It is the robot the MRT reads, not the MPC
state, so states recorded from the Euler formulation drive the quaternion one through the same observation path.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/RecordedRobotStates.cpp:recorded_state_labels) -->

## Tests

| Target | What it checks |
|---|---|
| `test_validation_io` | SHA-256 against the FIPS test vectors; JSON round trips (numbers bit for bit); golden files bit for bit with their provenance; malformed input rejected with its position |
| `test_closed_loop_metrics` | each metric on synthetic samples with a known answer; the schema test, including every document under `data/closed_loop/` |
| `test_closed_loop_scenarios` | the scenario set, the GUI conversion and its clamp against the MPC node's, the configurations and their files against the robots' launch files, the bands and the 720-degree turn exception |
| `test_lockstep_solve_schedule` | the runner's solve schedule: the pacing, a failed solve's pause and its end by a reset requested after the failure |
| `test_recorded_robot_states` | a recording reads back as the recorded states; another robot is refused |
| `test_lockstep_determinism` | smoke runs on the Unitree G1 with one solver thread agree to 1e-9, two in one process and one in a process of its own (exclusive: it compiles CppAD libraries) |
| `test_driver_settings_from_config` | the GUI's command scaling from its own fields of the reference file, a limit that is absent or not positive refused by name, every configuration's reference file; the contact wrench gate of a whole task file: no block is the instantaneous gate, a refused block keeps the gate in use, every configuration's task file sets its own |
| `test_robot_scene_gantry_hold` | the gantry hold each configuration's task file names (pinned: a change is a change of its runs), its MuJoCo scene supporting it, the robot process's mujoco backend running on it without falling back to `kinematic_teleport`; each scene that declares the weld failing the check with it taken out |
| `test_solve_benchmark_gate` | the real-time gate: every recorded baseline passes against itself, each criterion fails alone, libraries match across the layout-tagged folder; every robot's default recorded states exist |
| `test_worktree_state` | the provenance's worktree state in throwaway git repositories: untracked contents change it, ignored files do not |
| `closed_loop_<robot>`, `benchmark_mpc_solve_<robot>` | manual, exclusive: the full scenarios and the benchmark |

The link the runner solves through, `InProcessMpcLink` with `Execution::kCaller`, is tested in
`humanoid_common_mpc:testInProcessMpcLink` and, under the MRT joint controllers, in
`humanoid_centroidal_mpc:testMrtJointControllerReset` and `humanoid_wb_mpc:testWBMpcMrtJointController`; the CppAD
library observer the benchmark counts tape operations with in `//lib/ocs2:test_cppad_library_observer`.

## Couplings with the main line

The runner mirrors code it cannot share, each pair tied with `LINT.IfChange` / `LINT.ThenChange` both ways:

| Here | There |
|---|---|
| `LockstepClosedLoop.cpp:robot_process_cycle` | `RobotProcess.cpp:robot_process_cycle` (the order of the control cycle) |
| `ClosedLoopDriver.cpp:controller_side_updates` | `RobotProcess.cpp:controller_side_updates` (`applyControllerSettings()`) |
| `ClosedLoopDriver.cpp:robot_backend_options` | `CentroidalMpcRobotMain.cpp`, `WBMpcRobotMain.cpp`: `robot_backend_options` |
| `CentroidalClosedLoopDriver.cpp:centroidal_robot_controller`, `WholeBodyClosedLoopDriver.cpp:whole_body_robot_controller` | the controller set-up of `CentroidalMpcRobotMain.cpp`, `WBMpcRobotMain.cpp` |
| `CentroidalClosedLoopDriver.cpp:centroidal_mpc_node_wiring`, `WholeBodyClosedLoopDriver.cpp:whole_body_mpc_node_wiring`, `ClosedLoopDriver.cpp:command_path` | `CentroidalMpcNode.cpp:mpc_wiring`, `WBMpcNode.cpp:mpc_wiring` |

What it shares instead of copying: `loadRobotProcessSettings()`, `createInitialSimState()`,
`MujocoRobotBackend::makeConfig()`, `MrtRobotController`, `TaskFileWatcher`, `loadTaskFile()` and
`controllerSideSettingsFromConfig()` (the controller-side settings of a changed task file, applied as a whole file as the
robot process applies them, `wholeFileContactWrenchGate()`), `loadReferenceFile()` and `referenceSettingsFromConfig()`
(the GUI's command scaling, `guiCommandScalingFromConfig()`), the MPC
node's `node::applyWalkingVelocityCommand()` (the GUI's message into the motion manager, as the node's subscription
applies it) and `node::clampWalkingVelocityCommand()` (the clamp the saturation flag and the commanded height are
computed with), and `MpcResetSupervisor::resetRequestedSinceLastFailure()` (what ends a back-off). The robot
configurations' file paths (`RobotConfiguration.cpp`) are checked by `test_closed_loop_scenarios` against the variables
of the robots' `launch/robot.textproto` and `launch/mpc.textproto`, the files the robot process and the MPC node run on.
