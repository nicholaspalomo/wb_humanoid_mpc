# M0: closed-loop metrics of the Euler formulation

The first baseline of the quaternion switch's verification ladder (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`,
section 4.3). It records the closed-loop metrics of section 4.5 for the five configurations and every scenario, with the
base orientation still in ZYX Euler angles. Later labels (M1 at Step 3, M2 at Step 7, M3 at Step 10) are compared with
it within the bands of section 4.5 (`compareClosedLoopMetrics()`, or `make closed-loop-metrics ... BASELINE=M0`).

## On the main line

M0 was recorded before the ROS removal, with the runner mirroring the ROS sim nodes and the MRT joint controllers running
their own solver iteration. On the main line the runner mirrors the robot process and the MPC node and solves through
`InProcessMpcLink::runSolverIteration()`; the switch to WB_MPC and the gantry release move by one control period (the
end of the entry on the whole-body G1 by one MPC period), the gantry release's MPC reset is served one solve later, and
the commands keep their `standingTime` after the release (`../../../README.md`, "On the main line, against the
baselines recorded before it"). A main-line run is compared with M0 within the bands only. `../M0_main/` re-records
eleven of these runs on the main line and shows that the ordering is the only difference: with the old order the main
line reproduces these documents, the whole-body walk within its rounding-level spread. Four of its eleven runs leave
these bands: their gait takes another pattern under that shift of the timeline.

## How it was made

- **Command:** `make closed-loop-metrics ROBOT=<robot> LABEL=M0 RECORD_STATES=1`, one robot at a time. This runs the
  lockstep closed loop (`humanoid_nmpc/humanoid_mpc_validation/README.md`) under `bazel test` and copies the 35 metrics
  documents here; the walking states the solve benchmark replays went to `../../benchmark/states/`.
- **Code:** commit `c331ddd76dcf3917bdd9d0578eee6f518a915ba8` with the uncommitted work of the worktree
  `wb_humanoid_mpc_quat`: the Boost-free state, Steps 0, 2, 4 and 5, and this Step 1. This is not the "HEAD as-is" of the
  design's ladder: the harness's metrics need Step 5's heading-tilt split, and no snapshot of the worktree before Steps 2
  and 4 exists. So M0 cannot catch a behavior change of Step 2 or Step 4 (an M0 ≈ M1 comparison would contain both on
  both sides); their neutrality rests on their own evidence, which is bitwise but not closed-loop:
  - Step 2's fixes were verified bit for bit by their tests (the G1 whole-body foot weights on frozen copies of the block
    before and after the loader fix);
  - Step 4's manifold support is not used (`nullptr`), and its flat SQP path reproduces a toy-problem golden recorded
    before the change bit for bit (`test_sqp_flat_parity`, data hash pinned);
  - Step 5's library is not linked into the MPC (only into these metrics);
  - Step 1 lets the MRT joint controllers run their solver iteration without their thread, the same function the
    thread runs.

  `provenance.worktree_state` names the state each document was recorded on: `d39e5c37...` for DRC Atlas, `13855b2a...`
  for the four others. That hash covers the diff of the tracked files and the collapsed `git status`, not the contents of
  untracked files - and most of Steps 1, 4 and 5 were untracked - so it cannot show that the two differ by comments,
  lint fixes and the Makefile only, as the recorder states. Documents recorded from now on hash the untracked contents
  too (`tools/worktree_state.sh`).
- **Configuration:** the shipped task, reference, gait, PD-gain, URDF and MuJoCo files of each robot, unmodified; their
  SHA-256 are in `provenance.configuration_sha256`.
- **Solver threads:** the configured ones (DRC Atlas 8, the others 4). Runs with more than one thread are not bit
  reproducible, so compare them within the bands, not exactly. A rerun of the Unitree G1 reproduced its first run to
  three decimals in every metric that does not time the machine.
- **Machine:** Intel Core Ultra 7 265K, 20 logical cores, 30.2 GiB, in the dev container `devcontainer-app-1`. Other
  agents' builds took turns with these runs through the machine lock and never ran beside them; the solve times are
  comparable on this machine only.
- **Environment:** `BASH_ENV` unset and the worktree's own `setup_env.sh` sourced, which sources ROS 2 Jazzy. The
  container's `BASH_ENV` sources the main checkout's `setup_env.sh`, which no longer sets up ROS; this worktree still
  links the ROS libraries, and its test binaries do not start without them.
- **Date:** 2026-10-01.

## Headline numbers

Tilt is the angle of the base from upright; the velocity error is the base velocity in the heading frame against the
reference velocity; the yaw-rate error is the world yaw rate against the reference yaw rate; the heading peak is the
largest unwrapped heading change; the rotation gap is the largest rotation part of the initial-state gap; the solve time
is the p99 of the solver iteration. A fall's time is counted from the first command. Every other metric is in the
documents.

| Robot | Scenario | Survived | Tilt RMS [rad] | Velocity RMS error [m/s] | Yaw-rate RMS error [rad/s] | Heading peak [rad] | Largest rotation gap [rad] | Solve p99 [ms] |
|---|---|---|---|---|---|---|---|---|
| drc_atlas | standing | yes | 0.006 | 0.000 | 0.000 | 0.00 | 0.000 | 7.20 |
| drc_atlas | walk_0p5 | yes | 0.043 | 0.215 | 0.052 | 0.02 | 0.015 | 6.94 |
| drc_atlas | walk_0p3 | yes | 0.043 | 0.149 | 0.051 | 0.02 | 0.009 | 6.87 |
| drc_atlas | lateral_0p2 † | yes | 0.066 | 0.204 | 0.302 | 0.01 | 0.361 | 7.07 |
| drc_atlas | turn_in_place_720 | yes | 0.049 | 0.072 | 0.480 | 0.22 | 0.006 | 7.17 |
| drc_atlas | turn_1radps | yes | 0.047 | 0.069 | 0.903 | 0.28 | 0.009 | 7.44 |
| drc_atlas | arc | yes | 0.044 | 0.144 | 0.280 | 0.12 | 0.012 | 7.29 |
| engineai_sa01 | standing | yes | 0.003 | 0.000 | 0.000 | 0.00 | 0.000 | 3.25 |
| engineai_sa01 | walk_0p5 | fell at 4.0 s (tilt) | 0.239 | 0.364 | 0.322 | 0.05 | 0.103 | 3.23 |
| engineai_sa01 | walk_0p3 | yes | 0.109 | 0.211 | 0.089 | 0.08 | 0.028 | 3.28 |
| engineai_sa01 | lateral_0p2 | fell at 3.4 s (tilt) | 0.280 | 0.454 | 0.423 | 0.15 | 0.405 | 3.41 |
| engineai_sa01 | turn_in_place_720 | yes | 0.065 | 0.128 | 0.489 | 0.01 | 0.017 | 3.30 |
| engineai_sa01 | turn_1radps | yes | 0.061 | 0.123 | 0.923 | 0.01 | 0.017 | 3.35 |
| engineai_sa01 | arc † | fell at 5.4 s (tilt) | 0.226 | 0.362 | 0.575 | 0.11 | 0.350 | 3.57 |
| unitree_g1 | standing | yes | 0.002 | 0.001 | 0.000 | 0.00 | 0.000 | 8.02 |
| unitree_g1 | walk_0p5 | yes | 0.073 | 0.296 | 0.303 | 0.17 | 0.022 | 7.91 |
| unitree_g1 | walk_0p3 | yes | 0.047 | 0.189 | 0.169 | 0.08 | 0.010 | 7.52 |
| unitree_g1 | lateral_0p2 | yes | 0.039 | 0.201 | 0.166 | 0.44 | 0.015 | 8.55 |
| unitree_g1 | turn_in_place_720 | yes | 0.046 | 0.129 | 0.497 | 0.11 | 0.018 | 7.96 |
| unitree_g1 | turn_1radps | yes | 0.063 | 0.135 | 0.923 | 0.25 | 0.031 | 8.18 |
| unitree_g1 | arc | yes | 0.051 | 0.192 | 0.317 | 0.27 | 0.014 | 7.45 |
| unitree_r1 | standing | yes | 0.001 | 0.002 | 0.000 | 0.00 | 0.003 | 7.25 |
| unitree_r1 | walk_0p5 | fell at 5.5 s (tilt) | 0.170 | 0.402 | 0.633 | 0.48 | 0.467 | 8.07 |
| unitree_r1 | walk_0p3 | yes | 0.048 | 0.241 | 0.277 | 0.09 | 0.017 | 7.41 |
| unitree_r1 | lateral_0p2 | yes | 0.040 | 0.204 | 0.108 | 0.18 | 0.015 | 7.94 |
| unitree_r1 | turn_in_place_720 | fell at 13.4 s (tilt) | 0.144 | 0.264 | 0.542 | 0.33 | 0.230 | 8.02 |
| unitree_r1 | turn_1radps † | fell at 5.5 s (height) | 0.171 | 0.332 | 1.065 | 0.47 | 0.540 | 8.93 |
| unitree_r1 | arc † | fell at 6.5 s (tilt) | 0.152 | 0.350 | 0.575 | 0.14 | 0.396 | 8.21 |
| unitree_g1_wb | standing | yes | 0.004 | 0.001 | 0.000 | 0.00 | 0.000 | 8.63 |
| unitree_g1_wb | walk_0p5 | yes | 0.045 | 0.301 | 0.223 | 0.12 | 0.012 | 10.06 |
| unitree_g1_wb | walk_0p3 | yes | 0.047 | 0.180 | 0.254 | 0.08 | 0.012 | 10.22 |
| unitree_g1_wb | lateral_0p2 | yes | 0.048 | 0.177 | 0.058 | 0.13 | 0.007 | 9.53 |
| unitree_g1_wb | turn_in_place_720 | yes | 0.044 | 0.158 | 0.330 | 5.27 | 6.285 | 10.40 |
| unitree_g1_wb | turn_1radps | yes | 0.036 | 0.139 | 0.603 | 4.97 | 6.286 | 10.34 |
| unitree_g1_wb | arc | yes | 0.045 | 0.159 | 0.278 | 1.62 | 0.012 | 10.60 |

No run had a failed solve or a non-finite value, and no command was saturated.

† Re-recorded on 2026-10-02, after the harness was changed to read the initial-state gap from the SQP itself (below):
the four runs that served a reset during the commands, the only ones whose gap the change could alter beyond the
trajectory spread. Same command, configuration and machine; the worktree states are in their provenance. Every metric
that does not time the machine reproduced the first recording to three decimals, the rotation gap included, except the
EngineAI SA01 arc's gap, 0.350 rad against 0.287 (its maximum lies 0.2 s before the fall). The other 31 documents are
those of the first recording.

## What the baseline shows, and what to keep in mind when comparing with it

- **The centroidal robots do not follow a yaw-rate command.** With the shipped task files none of DRC Atlas, EngineAI
  SA01, Unitree G1 and Unitree R1 turns: at 1 rad/s for 12.6 s the heading changes by 0.01 to 0.47 rad, not 12.6 rad.
  Every centroidal task file schedules the contacts with the gait schedule, so the foot cost has no yaw reference
  (it has one only from the contact planner); every locomotion heuristic, `in_place_turning` included, is off; and
  defect D2 (`ProceduralMpcMotionManager` reads `L_x / m` as the yaw rate on the centroidal model) holds a turn in the
  slow gaits. The centroidal scenarios therefore never cross the ±π heading of section 4.5's 720-degree exception: their
  rotation gap stays below 0.6 rad.
- **The whole-body G1 turns, and shows the Euler wrap.** Its heading reaches 5.3 rad in `turn_in_place_720` and comes back
  to 0.02 rad, and both turning runs record a rotation gap of 6.285 rad, i.e. 2π: the measured yaw is `atan2`-wrapped
  against an unwrapped warm start (`DynamicsHelperFunctions.h`, `SqpSolver.cpp` `delta_x0`). This is the ±π transient the
  quaternion switch has to remove; M3's rotation gap on these two runs should stay far below π.
- **No robot reaches 0.5 m/s.** Over the last 10 s of the 15 s walk, DRC Atlas averages 0.36 m/s at 0.5 m/s commanded
  and 0.22 m/s at 0.3; Unitree G1 0.29 and 0.17 (centroidal), 0.26 and 0.18 (whole-body); EngineAI SA01 and Unitree R1
  0.14 and 0.13 at 0.3 m/s, and both fall within 4 to 6 s of the 0.5 m/s command. The slower variant `walk_0p3` that
  section 4.5 allows is the walking scenario those two robots survive, and its states are the ones the solve benchmark
  replays for them.
- **DRC Atlas walking sideways at 0.2 m/s survives with three divergence resets** (`failures.resets_served`) and a 0.36
  rad rotation gap. When M0 was first recorded the harness measured the gap against the previous solution before the
  solve, although a solve that serves a reset discards that solution first (the SQP then starts from the observation,
  a zero gap), and without the trajectory spread. It now reads the SQP's own first-iteration `delta_x0`
  (`SqpSolver::getInitialStateGap()`), and the four runs with a reset (†) were re-recorded with it. Their maxima stood:
  they are gaps the SQP itself saw, in solves before a reset was served, not artifacts of the reset-serving solves.
- **Unitree R1 falls in every turning scenario and in the arc; EngineAI SA01 falls sideways and in the arc.** Neither has
  been validated in simulation on these task files.
- **Time series.** The two whole-body G1 turning runs, the only M0 runs whose heading crosses ±π, keep their time series
  here (`unitree_g1_wb_turn_in_place_720_timeseries.txt`, `unitree_g1_wb_turn_1radps_timeseries.txt`, copied from the
  test outputs of the first M0 recording, whose documents they match byte for byte): the 720-degree turn exception of
  section 4.5 locates the crossings in them. The re-recorded turning runs (†: `engineai_sa01_arc`,
  `unitree_r1_turn_1radps`, `unitree_r1_arc`) keep theirs as `make closed-loop-metrics` now archives every turning
  scenario's; the other runs' time series were not archived.
- The runs measure the controller, not the real-time behavior: a solve takes no simulation time and its policy is in use
  from the next control cycle. The threaded MuJoCo sims add the solve's latency on top.
