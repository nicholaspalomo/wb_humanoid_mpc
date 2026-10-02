# B0: solve benchmark of the Euler formulation

The real-time baseline of the quaternion switch (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`, section
4.6). B3 is recorded at Step 10 on the same machine and the same recorded states, and must stay within the gate of
section 4.6:
- mean solve time at most +5 %;
- p99 at most +10 %;
- LQ approximation at most +8 %;
- p99 below 80 % of each robot's MPC period;
- every CppAD library's tape operation count at most +5 %.

## On the main line

B0 was recorded before the ROS removal. On the main line the benchmark replays the same states through the robot
process's controller path and solves through `InProcessMpcLink::runSolverIteration()`; its later passes start from a
full reset served by their first, untimed, solve (`../../../README.md`, "On the main line, against the baselines
recorded before it"). Solve times also depend on the machine and its load: re-record a baseline on the main line, on the
machine B3 will be measured on, before the gate of section 4.6 is applied.

## How it was made

- **Command:** `make benchmark-mpc-solve ROBOT=<robot> LABEL=B0 [STATES=...]`, one robot at a time, under `bazel test`
  (the manual tests `benchmark_mpc_solve_<robot>`). The default states are now each robot's own walking scenario, so
  `STATES` is no longer needed for EngineAI SA01 and Unitree R1. B3 is checked with
  `make benchmark-mpc-solve ROBOT=<robot> LABEL=B3 BASELINE=B0` (`compareSolveBenchmarks()`, which matches each library
  across the layout-tagged CppAD folder of Step 8).
- **What a run does:** builds the formulation's MPC stack as the lockstep closed loop does (`ClosedLoopDriver`, the task
  file's solver threads). It then replays the recorded robot states through the MRT joint controller in WB_MPC, with
  one solver iteration per state.
- **Timing:** each run makes three passes over the 300 states. Each pass starts afresh, and its first 25 solves are not
  timed, so 825 solves are timed per robot.
- **Input states** (`../states/`): recorded by the M0 closed-loop runs (`../../closed_loop/M0/README.md`). Each is the
  robot at 300 consecutive solves, from 2 s into the walk:
  - `walk_0p5` (0.5 m/s) for DRC Atlas, Unitree G1 and Unitree G1 whole-body;
  - `walk_0p3` (0.3 m/s) for EngineAI SA01 and Unitree R1, which fall at 0.5 m/s.

  The states are what the MRT reads, not the MPC state. They drive the quaternion formulation through the same
  observation path, so B3 replays these same files. Their hashes are in each document's `provenance`.
- **Tape operation counts:** `CppAdInterface::getTapeOperationCount()`, collected through
  `CppAdInterface::setLibraryObserver()` for every library the problem builds. They do not depend on the machine.
- **Code, machine and environment:** as for M0: commit `c331ddd`, the worktree with Steps 0, 2, 4, 5 and 1 uncommitted,
  Intel Core Ultra 7 265K with 20 logical cores and 30.2 GiB, and the worktree's own `setup_env.sh`. Other agents' builds
  took turns with these runs through the machine lock. The machine was not otherwise idle.
- **Not "HEAD as-is":** the design's ladder asked for B0 at Step 1 on HEAD as it was. B0 includes Steps 2 and 4, so B3
  against B0 measures the switch on top of Step 4's flat-path overhead (a null-manifold check per node and per step),
  not that overhead itself; Step 4's flat path is held bitwise by `test_sqp_flat_parity` instead (design section 4.3).
- **`provenance.worktree_state`** hashes the diff against HEAD and the collapsed `git status`, not the contents of
  untracked files, so it does not identify the code exactly: most of Steps 1, 4 and 5 were untracked files. Documents
  recorded from now on hash those contents too (`tools/worktree_state.sh`).
- **Date:** 2026-10-01.

## Results

The solve time is the wall time of the MRT joint controller's solver iteration. That covers the synchronized modules
and the SQP, which runs one iteration per solve on every robot.

| Robot | States | Threads | Timed solves | Mean [ms] | p50 [ms] | p99 [ms] | Max [ms] | LQ approximation p99 [ms] | Solve QP p99 [ms] | MPC period [ms] | p99 / period | CppAD libraries | Tape operations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drc_atlas | walk_0p5 | 8 | 825 | 6.39 | 6.34 | 8.43 | 8.75 | 2.95 | 2.77 | 20.0 | 0.42 | 32 | 56337 |
| engineai_sa01 | walk_0p3 | 4 | 825 | 3.13 | 3.11 | 3.67 | 3.91 | 2.15 | 0.97 | 12.5 | 0.29 | 24 | 25368 |
| unitree_g1 | walk_0p5 | 4 | 825 | 7.62 | 7.65 | 8.43 | 8.79 | 5.21 | 2.57 | 12.5 | 0.67 | 31 | 61552 |
| unitree_r1 | walk_0p3 | 4 | 825 | 7.07 | 7.04 | 8.05 | 10.44 | 4.86 | 2.65 | 12.5 | 0.64 | 31 | 59150 |
| unitree_g1_wb | walk_0p5 | 4 | 825 | 9.23 | 9.29 | 9.97 | 12.64 | 5.54 | 4.15 | 16.7 | 0.60 | 28 | 86674 |

- No solve failed.
- Every robot keeps its p99 below 80 % of its MPC period. The tightest are Unitree G1 at 67 % and Unitree R1 at 64 %,
  13 and 16 percentage points from that limit; the +10 % p99 gate binds first on every robot.
- The per-library counts are in each document under `tape_operation_counts`.
