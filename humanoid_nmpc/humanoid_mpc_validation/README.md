# humanoid_mpc_validation

The closed-loop measurements of the MPC formulations: the lockstep MuJoCo closed loop with its metrics, the comparison
of two recorded labels within the bands of the metrics, and the solve benchmark with its real-time gate; beside them the
Euler chart the equivalence suites map their Euler oracle through. The closed loop runs the production MPC, as the MPC
node builds it, and the production MRT joint controller, as the robot binary builds it, in one thread and without the
bus. The design these measurements verify is `humanoid_nmpc/docs/quaternion_base_orientation/README.md` (sections 4.5
and 4.6).

```
 scenario (section 4.5)      GUI message            the MPC node's conversion       ProceduralMpcMotionManager
 commandAt(t) ----------> toGuiVelocityCommand --> walkingVelocityCommandFromProto --> setAndScaleVelocityCommand
                                                                                                  |
 MujocoSimInterface (headless,                   RobotController (MrtRobotController)            v next solve
 MujocoRobotBackend::makeConfig())               over the MRT joint controller         InProcessMpcLink
   step x (1 / mrt_desired_frequency) / dt <-- prepareCycle(), computeJointControlAction()  (Execution::kCaller)
   getRobotState() ------------------------->        |  observations, policies  <------>  runSolverIteration()
         |                                                                               every 1 / mpc_desired_frequency
         +--------------------> ClosedLoopMetrics <--- solve times, the SQP's delta_x0 and step
                                      |
                                <robot>_<scenario>.json (ClosedLoopMetricsSchema)
```

## What is here

| Piece | Files | What it does |
|---|---|---|
| Golden files | `io/GoldenIo.h`, `io/Sha256.h`, `io/RunProvenance.h` | Labeled matrices at `%.17g` (NaN, infinities and the sign of zero kept), with a header carrying the commit, the worktree state, the machine and the SHA-256 of every configuration file: the format of a run's time series and of the recorded robot states. |
| JSON | `io/JsonValue.h` | The documents the metrics and the benchmark write; insertion-ordered, shortest round-trip numbers. |
| Lockstep closed loop | `closed_loop/LockstepClosedLoop.h`, `closed_loop/ClosedLoopDriver.h`, `closed_loop/CentroidalClosedLoopDriver.h`, `closed_loop/WholeBodyClosedLoopDriver.h` | The headless simulator, the formulation's MPC as its MPC node builds it, its MRT joint controller as its robot binary builds it, and between the two an `InProcessMpcLink` whose solver iterations the runner calls (`InProcessMpcLink::Execution::kCaller`, `runSolverIteration()`). One thread, no bus. |
| Scenarios | `closed_loop/ClosedLoopScenario.h` | The commands of section 4.5, driven through the GUI's message. |
| Metrics | `closed_loop/ClosedLoopMetrics.h`, `closed_loop/ClosedLoopMetricsSchema.h`, `closed_loop/MetricBands.h`, `closed_loop/SqpStepDiagnostics.h`, `exe/compareClosedLoopMain.cpp` | The metrics of section 4.5 and each solve's SQP step, their JSON schema, the band rules against a baseline label, and the offline comparison of two recorded labels. |
| Solve benchmark | `benchmark/MpcSolveBenchmark.h`, `benchmark/SolveBenchmarkGate.h`, `exe/benchmarkMpcSolveMain.cpp` | Section 4.6: recorded walking states replayed through the controller, solve times per phase and the tape operation count of every CppAD library, and the real-time gate of a benchmark against a baseline label. |
| Recorded data | `data/closed_loop/<label>/`, `data/benchmark/<label>/`, `data/benchmark/states/` | The labels the Make targets record (["Recording and comparing labels"](#recording-and-comparing-labels)), and the walking states the benchmark replays. |
| Euler chart | `chart/EulerChart.h` | Phi, Phi^-1, dPhi / dx_e and P_e between the ZYX Euler coordinates of a state and the physical state of the quaternion formulation (design section 2.12), through `orientation/EulerBoundary.h`: what the equivalence suites of `humanoid_common_mpc/test/equivalence` map their Euler oracle through. |

## The lockstep closed loop

A run is reproducible and independent of the machine's speed:

1. Every control cycle (`1 / mpc.mrt_desired_frequency` of the task file, a whole number of 0.5 ms simulation steps) runs in
   the order of `RobotProcess::cycle()` (`humanoid_common_mpc_app/robot`): the runner reads the simulator's state,
   passes it through the task file's `state_estimator` as the robot process does (`createRobotProcessStateEstimator()`:
   `ground_truth` hands the controller the simulator's state itself, `kalman` its estimate of the base; the fall check,
   the metrics and the joints' feedback torques read the true state, the recorded robot states the controller's), hands
   the controller the mode and the JOINT_PD posture (`RobotController::prepareCycle()`, in the order of the
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
search's comparisons. Comparisons across a code change use the bands below, since a code change moves the results.

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
reaches the run: do not edit the configuration while a label is recorded.

The JOINT_PD posture is the posture the robot is spawned in (`createInitialSimState()`, before any physics step), as the
robot process's FSM bridge takes it.

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
Every document has `schema: humanoid_mpc_validation.closed_loop_metrics.v2`, the label, robot, formulation and scenario,
an open `provenance` and `settings` object, and these fields (null where nothing was measured). v2 added `sqp_step`; a v1
document has none of it, and a comparison skips a metric either document lacks:

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
| `quaternion_norm.{max_deviation, non_finite_solves}` | `max | |xi_k| - 1 |` over the nodes of every solution; null without a solve, and the count of solves with non-finite quaternion deviations |
| `non_finite_values` | NaN or infinite values in the observations, policy inputs and joint torques |
| `solve_time_ms.{total, lq_approximation, solve_qp, linesearch, compute_controller}.{mean, p50, p99, max}` | the solver iteration's wall time and the SQP's phases |
| `sqp_step.{steps, full_steps, reduced_steps, zero_steps, full_steps_refused_above_g_max}` | the SQP's step in each successful solve (its last iteration, `SqpSolver::getIterationRecords()`): how many solves took the full step, a reduced one or none, and in how many the full step was refused for a constraint violation above `g_max` |
| `sqp_step.accepted_by.{constraint, dual, cost}` | the filter line search's rule that accepted the step: the step's violation above `g_max`; every other step not above `g_max` - between `g_min` and `g_max`, or below `g_min` after an iterate at or above it or without a descent direction (a decrease of the merit or of the violation); the violation below `g_min` before and after the step on a descent direction (Armijo) |
| `sqp_step.longest_violation_growth` | the longest run of consecutive solves whose constraint violation grew over the solve before (a failed solve ends a run) |
| `sqp_step.constraint_violation.{mean, p50, p99, max}` | the line search's constraint violation after each step: the shooting gaps, the initial-state gap and the equality constraints |
| `sqp_step.swing_normal_velocity_residual.{mean, p50, p99, max}` | the largest residual of the hard swing-foot normal-velocity rows (`<contact>_normalVelocity`) over each new solution's nodes, over the solves with a swing in the horizon. The rows differ by formulation, and so do their units: m/s on the centroidal MPCs (`v_z - zdot_ref + k_p (z - z_ref)`), m/s^2 on the whole-body MPC (`k_v (v_z - zdot_ref) + k_a (a_z - zddot_ref) + k_p (z - z_ref)`, the gains of `model_settings.foot_constraint`; 10, 1 and 100 on the Unitree G1), so the numbers of a centroidal and of a whole-body configuration are not alike |
| `failures.{failed_solves, resets_served, full_resets_served, simulator_resets, unhealthy_cycles}` | during the commands; simulator resets over the whole run |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopMetricsSchema.cpp:metrics_schema) -->

The `sqp_step` group is what shows a Gauss-Newton SQP cycling from one solve to the next (design README section 6): runs
of full steps whose violation grows (`longest_violation_growth`), each ended by a full step refused above `g_max`
(`full_steps_refused_above_g_max`) and a reduced step. A run's time series has the step of every solve
(`solve_sqp_step`: the step size, the codes of the rules that accepted the step and judged the full step, the full
step's and the step's constraint violation) and the swing residual of every solve with a swing
(`solve_swing_normal_velocity_residual`), so a plot shows where in the gait the cycle runs (`TimeSeriesLabels.h`).

## Bands against a baseline

<!-- LINT.IfChange(metric_bands) -->
`compareClosedLoopMetrics()` holds a run to the bands of section 4.5 around the same run of a baseline label, each in the
direction its metric may move (`MetricBand::direction`), so that no gate fails on the improvement a change is meant to
bring:

| Metric | Band | Direction |
|---|---|---|
| `base_height.mean_m` | 5 mm | either way (`kTwoSided`): a level, not an error |
| `base_height.rms_error_m` | 5 mm | no worse (`kNoWorse`) |
| `tilt.rms_rad`, `tilt.max_rad` | max(20 %, 0.005 rad) | no worse |
| `velocity.rms_error_mps` | max(15 %, 0.02 m/s) | no worse |
| `yaw_rate.rms_error_radps` | max(15 %, 0.02 rad/s) | no worse |
| `stance_foot_slip.max_m` | max(20 %, 5 mm) | no worse |

A `kNoWorse` metric fails only when the candidate is worse than its baseline by more than the band. Better by more than
the band, it passes and is reported as an improvement. A comparison returns both (`ClosedLoopComparison`): the
`violations`, of which a passing run has none, and the `improvements`, which the candidate label's README lists, so that
a widened gate is documented rather than discovered. Survival must be equal or better (surviving where the baseline fell
is an improvement; a run that fell in the baseline is compared on survival alone), the p99 solve time at most +10 %
(only on the machine of the baseline, `--compare_solve_time`), and the quaternion norm below 1e-9. A metric that is null
in the baseline is not compared; one that the baseline measured and the candidate did not (null, or NaN in a run's
document still in memory) fails, since no band can hold a value that is not there.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/MetricBands.h:metric_bands) -->

## Recording and comparing labels

A label is a directory of documents, one per run: `make closed-loop-metrics ROBOT=<robot> LABEL=<label>` writes
`data/closed_loop/<label>/<robot>_<scenario>.json`, and `make benchmark-mpc-solve ROBOT=<robot> LABEL=<label>` writes
`data/benchmark/<label>/<robot>_solve_benchmark.json`. No label is checked in: record a baseline on the code to compare
against and a candidate on the change, on the same machine and container image, and give each label a README saying
what it ran (the commit and the worktree state are in every document's provenance) and, for a candidate, the
improvements its comparison found. Only the metrics documents are copied into the label; the time series and the
`<robot>_<scenario>_comparison.txt` of each run stay in the test's outputs
(`.bazel/testlogs/humanoid_nmpc/humanoid_mpc_validation/closed_loop_<robot>/test.outputs/`).

With `BASELINE=<label>`, `make closed-loop-metrics` compares each run in the loop with the same run of that label
(`compareClosedLoopMetrics()`), prints the violations and improvements at its end and fails on a violation; a run the
baseline has no document of is only recorded. `make compare-closed-loop CANDIDATE=<label> BASELINE=<label>
[ROBOT=<robot>]` makes the same comparison offline (`compareArchivedClosedLoopRuns()`): every run of
`data/closed_loop/<CANDIDATE>/` that `data/closed_loop/<BASELINE>/` has too, from the archived documents, which read
back bit for bit, so the verdict is the run's. No simulation runs. The target runs the manual test
`compare_closed_loop_labels` under `bazel test`, inside the machine lock, prints each run's violations and improvements
with a summary that names the runs without a baseline, and fails when a run has a violation or no run has a baseline.
Neither target compares the p99 solve time; the binaries' `--compare_solve_time` does, between labels of one machine.

## How to run

Inside the dev container, one robot at a time (each run compiles the robot's CppAD libraries and simulates for minutes):

```bash
make closed-loop-metrics ROBOT=drc_atlas LABEL=<baseline>                                   # every scenario
make closed-loop-metrics ROBOT=unitree_g1 SCENARIO=walk_0p5 LABEL=<candidate> BASELINE=<baseline>
make closed-loop-metrics ROBOT=unitree_g1 LABEL=<label> RECORD_STATES=1                    # also keep the walking states
make benchmark-mpc-solve ROBOT=unitree_g1 LABEL=<baseline>
make benchmark-mpc-solve ROBOT=unitree_g1 LABEL=<candidate> BASELINE=<baseline>             # held to the real-time gate
make compare-closed-loop CANDIDATE=<candidate> BASELINE=<baseline>                          # two recorded labels, offline
```

`RECORD_STATES=1` overwrites the states the benchmark replays: never set it on a comparison run.

<!-- LINT.IfChange(real_time_gate) -->
The benchmark replays `data/benchmark/states/<robot>_<walking scenario>_states.txt`: `walk_0p5`, or `walk_0p3` for
EngineAI SA01 and Unitree R1, which fall at 0.5 m/s (`RobotConfiguration::walkingScenario`). `compareSolveBenchmarks()`
(`benchmark/SolveBenchmarkGate.h`) is the gate: mean +5 %, p99 +10 %, LQ approximation +8 %, p99 below 80 % of the MPC
period, and every library's tape operation count and their total +5 %. Libraries are matched by their name in
`tape_operation_counts`; one that only one document counts is compared through the total alone. It compares solve times
only between two documents of one machine and one toolchain (below) and refuses them otherwise, while it still checks
the candidate's p99 against its MPC period and the tape operation counts, which do not depend on the machine. A
container rebuild changes the toolchain: record the baseline again on the new image.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/benchmark/SolveBenchmarkGate.h:real_time_gate) -->

<!-- LINT.IfChange(provenance_toolchain) -->
The provenance names the toolchain too: `provenance.toolchain` of a document, the `compiler`, `hpipm_target` and
`container_image` notes of a golden file's header. `compiler` is `__VERSION__` of the compiler that built the run;
`hpipm_target` is HPIPM's `-DTARGET` (`GENERIC`), which the Make targets read from `bazel/system_libs.bzl`;
`container_image` is the id of the image, which the container cannot see itself: pass it from the host,
`VALIDATION_CONTAINER_IMAGE=$(docker inspect --format '{{.Image}}' wbmpc-noros-dev)`, else it is `unknown`, and the
label's README records `docker image inspect` by hand.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/io/RunProvenance.cpp:toolchain_fields) -->

The provenance's `worktree_state` (`tools/worktree_state.sh`) hashes the diff against HEAD, the status with every
untracked file listed, and the contents of every untracked file, so that an edit of a file not yet committed changes it.

The configuration files a run depends on (`RobotConfiguration::configurationFiles()`) are the robot's typed textprotos
of [`humanoid_mpc_config`](../humanoid_mpc_config/README.md) - `config/mpc/task.textproto`,
`config/command/reference.textproto`, `config/controller/joint_pd_gains.textproto`, the contact planner's
`config/mpc/contact_planning.textproto` where the robot has one, and the gait file `gait.textproto` - plus its URDF and
MuJoCo scene, and the provenance records the SHA-256 of each.

`ROBOT` is `drc_atlas`, `engineai_sa01`, `unitree_g1`, `unitree_r1` or `unitree_g1_wb`. The targets run the manual tests
`closed_loop_<robot>`, `benchmark_mpc_solve_<robot>` and `compare_closed_loop_labels` under `bazel test`, so that the machine lock of `tools/bazel`
keeps them from running next to another build, and copy the documents into `data/`.

<!-- LINT.IfChange(output_names) -->
Each closed-loop run writes `<robot>_<scenario>.json` (the metrics), `<robot>_<scenario>_timeseries.txt` (the base and
its reference at 50 Hz, and the rotation part of each solve's initial-state gap, a golden file whose matrices
`TimeSeriesLabels.h` names) and, walking, `<robot>_<scenario>_states.txt` (the robot at 300 solves from 2 s into the
commands, the input of the benchmark) into the test's undeclared outputs; the benchmark writes
`<robot>_solve_benchmark.json`. `make closed-loop-metrics` copies the metrics documents into the label (and, with
`RECORD_STATES=1`, the states into `data/benchmark/states/`), `make benchmark-mpc-solve` the benchmark's.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h:output_names) -->

<!-- LINT.IfChange(recorded_state_labels) -->
A recorded-states file holds, one row per solve, `time`, `base_position`, `base_quaternion_xyzw`,
`base_linear_velocity_local`, `base_angular_velocity_local`, `joint_positions` and `joint_velocities` (every joint of the
URDF, named in the `joint_names` note), `contact_flags` and `gui_command`. It is the robot the MRT reads, not the MPC
state, so states recorded with one formulation drive another through the same observation path.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/RecordedRobotStates.cpp:recorded_state_labels) -->

## Tests

| Target | What it checks |
|---|---|
| `test_validation_io` | SHA-256 against the FIPS test vectors; JSON round trips (numbers bit for bit); golden files bit for bit with their provenance, every recorded file under `data/` (the recorded states, and the documents of any label) formatting back to its bytes; malformed input rejected with its position; the toolchain through a document and a golden header, read from the environment the Make targets pass, a missing or malformed one refused |
| `test_euler_chart` | the chart's blocks where `initial_state` puts them; Phi is the Euler boundary's conversion; dPhi / dx_e and P_e against central differences, P_e = E+ dPhi on the angles; the inverse chart (the angles wrapped); the rate coupling; the singularity; malformed layouts and a zero quaternion refused |
| `test_closed_loop_metrics` | each metric on synthetic samples with a known answer, the SQP steps of a cycle among them (growth runs, refusals above `g_max`, the rules that accepted each step); the schema test; the band directions, an improvement passed and reported, a regression failed, the height mean two-sided; the offline comparison of two labels archived from synthetic runs: the verdict of the runs as held in memory, run by run and in the order of the configurations and scenarios, the runs without a baseline named, one robot alone, the solve times only on request, and a missing directory, an unknown robot and an unreadable document refused |
| `test_sqp_step_diagnostics` | the SQP step a solve records: the hard normal-velocity terms found by their contacts' names, their largest residual over the nodes where they are active (none without a swing), and every line-search verdict's kind |
| `test_closed_loop_scenarios` | the scenario set, the GUI conversion and its clamp against the MPC node's, the configurations and their files against the robots' launch files, the bands (each the larger of its parts, survival, the quaternion norm) |
| `test_lockstep_solve_schedule` | the runner's solve schedule: the pacing, a failed solve's pause and its end by a reset requested after the failure |
| `test_recorded_robot_states` | a recording reads back as the recorded states; another robot is refused |
| `test_stride_motion_of_recorded_walks` | the stride means of the base's planar motion on the recorded straight walks of `data/benchmark/states/` (design section 2.9): neither a turn nor a sideways walk over the cycle of the gait they walk, where the pelvis's own yaw rate and sideways velocity sway past the gait thresholds; an added turn reads exactly as that turn; a stall of the solves of up to 1 s is bridged; a history restarted from the instantaneous twist reads the sway as a turn. Builds no model |
| `test_joint_feedback_torques` | the joint torques a control cycle records: the PD torque plus the feedforward of each MPC joint, in the order of the MPC's joints; a joint the action leaves out is an internal error naming it |
| `test_lockstep_determinism` | smoke runs on the Unitree G1 with one solver thread agree to 1e-9, two in one process and one in a process of its own, and a run's per-solve gaps, SQP steps and swing residuals hold its document's largest (exclusive: it compiles CppAD libraries) |
| `test_initial_state_gap` | the initial-state gap of a solve (`initialStateGapOf`): the state manifold's rotation angle wherever its layout puts the rotation block, zero without a rotation block; and the quaternion norm deviation of a solution (`maxQuaternionNormDeviationOf`), the largest distance of a stored quaternion from the unit sphere, none without a node |
| `test_driver_settings_from_config` | the GUI's command scaling from its own fields of the reference file, a limit that is absent or not positive refused by name, every configuration's reference file; the contact wrench gate of a whole task file: no block is the instantaneous gate, a refused block keeps the gate in use, every configuration's task file sets its own |
| `test_robot_scene_gantry_hold` | the gantry hold each configuration's task file names (pinned: a change is a change of its runs), its MuJoCo scene supporting it, the robot process's mujoco backend running on it without falling back to `kinematic_teleport`; each configuration on `weld_constraint` (all five) failing the check with its scene's weld taken out |
| `test_hot_field_coverage` | the reload classes of the task file's schema against each formulation's parameter updater (T8, `humanoid_mpc_config/README.md`): the centroidal and the whole-body MPC each apply exactly the `RELOAD_HOT` fields they read, neither applies a field only the other formulation, the robot process or the GUI reads, and every applied name is a field of the schema. From the updaters' static field lists: builds no MPC |
| `test_solve_benchmark_gate` | the real-time gate on synthetic documents: a document passes against itself (one without a toolchain compares its tape counts only), each criterion fails alone, only the libraries both documents count compared one by one, a solve time of another machine, compiler, HPIPM target or image refused while the tape counts still compare; every robot's default recorded states exist |
| `test_worktree_state` | the provenance's worktree state in throwaway git repositories: untracked contents change it, ignored files do not |
| `closed_loop_<robot>`, `benchmark_mpc_solve_<robot>` | manual, exclusive: the full scenarios and the benchmark; `closed_loop_scenarios` and `benchmark_mpc_solve`, their `bazel run` binaries, are not manual, so that `bazel build //...` compiles their sources |
| `compare_closed_loop_labels` | manual: two recorded labels compared offline (`make compare-closed-loop`) |

The link the runner solves through, `InProcessMpcLink` with `Execution::kCaller`, is tested in
`humanoid_common_mpc:testInProcessMpcLink` and, under the MRT joint controllers, in
`humanoid_centroidal_mpc:testMrtJointControllerReset` and `humanoid_wb_mpc:testWBMpcMrtJointController`; the CppAD
library observer the benchmark counts tape operations with in `//lib/ocs2:test_cppad_library_observer`.

## Couplings with the main line

The runner mirrors code it cannot share, each pair tied with `LINT.IfChange` / `LINT.ThenChange` both ways:

| Here | There |
|---|---|
| `LockstepClosedLoop.cpp:robot_process_cycle` | `RobotProcess.cpp:robot_process_cycle` (the order of the control cycle) |
| `LockstepStateEstimator.cpp:robot_process_state_estimator` | `RobotProcess.cpp:robot_process_state_estimator` (the state estimator by name, `StateEstimatorContext`) |
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
