# M0_main: the M0 scenarios on the main line, after the ROS removal

The parity check of the ROS removal. M0 (`../M0/`) was recorded on the quaternion worktree before the switch-over, with
the MRT joint controllers as they were then. These are the same runs on the merged main line: the robot process and
the MPC node instead of the ROS sim nodes, `InProcessMpcLink` in place of the controllers' own solver loop, and phase A
of the quaternion switch merged in. The runs are still on the Euler base orientation. Each is held to M0 within the
bands of section 4.5 (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`).

The subset is every configuration's `standing` and its walking scenario (`RobotConfiguration::walkingScenario`:
`walk_0p5`, or `walk_0p3` for EngineAI SA01 and Unitree R1), plus `turn_in_place_720` on the whole-body Unitree G1.
That G1 run is the only M0 run whose heading crosses ±pi.

The three whole-body documents and the turn's time series were recorded again on 2026-10-06, once CppAD's code
generation had been made deterministic ([below](#the-whole-body-documents-recorded-again-on-deterministic-code-generation)).
The eight centroidal documents are the first recording's.

## How it was made

- **Command:** `make closed-loop-metrics ROBOT=<robot> LABEL=M0_main BASELINE=M0 SCENARIO=<scenarios>
  VALIDATION_GIT_COMMIT=<commit> VALIDATION_WORKTREE_STATE=<state>`, one robot at a time, in the ROS-free dev container
  `wbmpc-noros-dev`. `SCENARIO` was a gtest filter: for example `SCENARIO='standing:*/walk_0p5'` expands to
  `--test_filter='*/standing:*/walk_0p5'`.
- **Code:** commit `c331ddd76dcf3917bdd9d0578eee6f518a915ba8` plus the uncommitted main-line state: the ROS removal,
  phase A merged into it, and the fixes of the review of that merge (the runner's command start, its back-off, the
  localized 720-degree turn exception, the per-solve gaps in the time series).
  - Every document's `provenance.worktree_state` is `2010 changed paths, diff sha256 428fb8a4730b76da`: the state
    `tools/worktree_state.sh` computes, with this directory left out, so that the documents of one robot do not change
    the state the next robot records. No file outside this directory changed between the first recording and the
    last. Afterwards only documentation and a test of these documents changed: this README, the package README, M0's
    README and `test/testClosedLoopMetrics.cpp` (which holds the turn's per-solve gaps to its document and to the
    720-degree turn exception).
  - A first recording of this label on the same day, before those fixes, is superseded by this one. Where it differs it
    is named below: its commands started one control period later.
  - The whole-body documents and the turn's time series are of commit `f0f55ce92819198127541ffeef5d77d76694f201` plus
    the uncommitted state `1663 changed paths, diff sha256 cccb6a1acad348f0` (this directory left out again): the
    typed textproto configuration, live tuning on both formulations, the robot's configuration store and the
    deterministic code generation. They hash the textproto files they ran on.
- **Configuration:** the shipped task, reference, gait, PD-gain, URDF and MuJoCo files. The task files' SHA-256 differ
  from M0's, but no closed-loop value does:
  - the ROS removal renamed two robot keys: `enableTelemetry` became `telemetrySinks: [bus]`, and
    `useGravityCompFeedforward: false` became `wbMpcFeedforward: inverse_dynamics`;
  - the visualization publisher's keys were added;
  - comments and LINT labels were added;
  - the Unitree R1's `telemetryFrames` were renamed.

  None of these reaches the closed loop except `wbMpcFeedforward`, which has the old value under its new name.
- **Solver threads:** the configured ones (DRC Atlas 8, the others 4).
- **Machine:** the same Intel Core Ultra 7 265K as M0, but in another container: `wbmpc-noros-dev`, not
  `devcontainer-app-1`. HPIPM is built from source with `TARGET=GENERIC` (`bazel/system_libs.bzl`), where M0 linked the
  old container's colcon build (`TARGET=AVX`). The solve times are comparable with M0's only roughly.
- **Date:** 2026-10-02; the whole-body documents 2026-10-06.

## Headline numbers

M0_main first, then M0 (`../M0/`). The metrics are those of the M0 README. "Outside" names the bands a run leaves. The
run that crosses ±pi is compared with `compareClosedLoopRuns()`, with the 720-degree turn exception.

| Robot | Scenario | Survived | Tilt RMS [rad] | Velocity RMS error [m/s] | Yaw-rate RMS error [rad/s] | Solve p99 [ms] | Against M0 |
|---|---|---|---|---|---|---|---|
| drc_atlas | standing | yes / yes | 0.0056 / 0.0056 | 0.000 / 0.000 | 0.000 / 0.000 | 7.19 / 7.20 | within |
| drc_atlas | walk_0p5 | yes / yes | 0.0425 / 0.0426 | 0.214 / 0.215 | 0.051 / 0.052 | 6.90 / 6.94 | within |
| engineai_sa01 | standing | yes / yes | 0.0032 / 0.0032 | 0.000 / 0.000 | 0.000 / 0.000 | 3.41 / 3.25 | within |
| engineai_sa01 | walk_0p3 | yes / yes | 0.0958 / 0.1086 | 0.208 / 0.211 | 0.079 / 0.089 | 3.39 / 3.28 | outside: base height mean 0.770 / 0.757 m, base height RMS error 0.047 / 0.060 m |
| unitree_g1 | standing | yes / yes | 0.0016 / 0.0016 | 0.001 / 0.001 | 0.000 / 0.000 | 7.91 / 8.02 | within |
| unitree_g1 | walk_0p5 | yes / yes | 0.0766 / 0.0728 | 0.320 / 0.296 | 0.429 / 0.303 | 8.51 / 7.91 | outside: tilt max 0.132 / 0.106 rad, yaw-rate RMS error |
| unitree_r1 | standing | yes / yes | 0.0011 / 0.0012 | 0.002 / 0.002 | 0.000 / 0.000 | 7.52 / 7.25 | within |
| unitree_r1 | walk_0p3 | yes / yes | 0.0556 / 0.0483 | 0.256 / 0.241 | 0.285 / 0.277 | 7.17 / 7.41 | outside: tilt max 0.107 / 0.088 rad |
| unitree_g1_wb | standing | yes / yes | 0.0037 / 0.0037 | 0.001 / 0.001 | 0.000 / 0.000 | 9.08 / 8.63 | within |
| unitree_g1_wb | walk_0p5 | yes / yes | 0.0454 / 0.0450 | 0.300 / 0.301 | 0.221 / 0.223 | 10.31 / 10.06 | within |
| unitree_g1_wb | turn_in_place_720 | yes / yes | 0.0455 / 0.0441 | 0.160 / 0.158 | 0.330 / 0.330 | 9.42 / 10.40 | outside: tilt max 0.147 / 0.103 rad |

Every run had no failed solve, no reset during the commands, no non-finite value and no saturated command. The p99
solve times are within -10 % to +8 % of M0's (-3 % to +8 % in the first recording: a solve time varies from run to
run). The whole-body G1's turn peaks at a heading of 5.39 rad, against M0's 5.27 rad. Like M0's, its initial-state gap
reaches 2 pi (6.286 rad), from the Euler wrap; its time series is the first to carry the per-solve gaps
(`solve_initial_state_rotation_gap`), which show the wrap at each of its own two crossings (19.13 s and 39.72 s) and
its decay below pi / 2 within 0.17 s. Nothing else in the run spikes.

## Why four runs leave the bands: the runner's cycle order, not the controllers

On the main line the runner serves the operator's FSM commands after the cycle's action and solve, as
`RobotProcess::cycle()` does. When M0 was recorded it served them at the start of the cycle. See the package README,
"On the main line, against the baselines recorded before it". The operator's events therefore come later than in M0,
and the gantry release's MPC reset, requested after the release cycle's solve, is served by the next solve:

| Event (time since the start) | Centroidal robots (control 10 ms, MPC 12.5 ms or 20 ms) | Whole-body G1 (control 2 ms, MPC 16.7 ms) |
|---|---|---|
| end of the entry into WB_MPC | +10 ms | +16 ms (the reset the controller requests on WB_MPC is served by the next solve) |
| gantry release | +10 ms | +16 ms |
| start of the commands | +10 ms: `standingTime` after the release cycle, as in M0 | +16 ms |

The commands keep their distance to the cycle the gantry is released in. Relative to the solve that serves the
release's reset, which restarts the gait schedule, they start 10 ms earlier than in M0 on the centroidal robots
(1.99 s instead of 2.00 s: in M0 the release cycle's own solve served it), and as in M0 on the whole-body G1 (1.986 s,
the next solve in both).

To isolate the ordering, a copy of the main-line state was built before the fixes of this recording. In it, the
runner's operator sequence is moved back to the start of the cycle, and the whole-body controller is again handed its
mode before its posture. Nothing else changed, and the copy ran the same tests. It reproduces M0:

| Run | Largest relative difference from M0 over the banded metrics (with the M0 cycle order) |
|---|---|
| drc_atlas standing, walk_0p5 | 2e-5 (a yaw-rate error of 2e-6 rad/s), 9e-8 |
| engineai_sa01 standing, walk_0p3 | 3e-6, 4e-9 |
| engineai_sa01 arc (a re-recorded † run of M0) | 0: all 50 compared fields bit for bit, the initial-state gap included |
| unitree_g1 standing, walk_0p5 | 3e-9, 1e-8 |
| unitree_r1 standing, walk_0p3 | 2e-10, 6e-13 |
| unitree_g1_wb standing, turn_in_place_720 | 1e-8, 4e-9 (tilt max 0.10260 rad, M0's) |
| unitree_g1_wb walk_0p5 | 0.5 % but for the stance-foot slip maximum (7.1 against 5.5 mm), see below |

So every violation is the ordering change. None is a change of the controllers, the MPC or the simulator. Three of the
walks are this sensitive to timing: a shift of 10 ms in when the gantry is released or when the commands start, against
the gait schedule, puts them in another walking pattern. The runs are deterministic: on EngineAI SA01's walk, three
reruns and runs with 1, 2, 3 and 8 solver threads reproduced the first recording to every printed digit.

- **EngineAI SA01, `walk_0p3`:** this robot is marginal at this speed. M0 tilts it by 0.11 rad RMS, and it falls at
  0.5 m/s. With the entry and the release 10 ms later its walk takes another pattern. The first recording, with the
  commands a further 10 ms later, walks the same pattern (every metric below within 0.1 %), so the release decides it,
  not the commands. The new
  pattern is not simply better:

  | | M0_main | M0 |
  |---|---|---|
  | base height mean (banded) | 0.770 m | 0.757 m |
  | tilt RMS (banded) | 0.096 rad | 0.109 rad |
  | yaw-rate RMS error (banded) | 0.079 rad/s | 0.089 rad/s |
  | stance-foot slip maximum (banded) | 3.9 mm | 1.6 mm |
  | stance-foot slip RMS (not banded) | 1.31 mm | 0.41 mm |
  | stance phases (not banded) | 27 | 35 |
  | joint torque RMS (not banded) | 13.5 N m | 14.4 N m |

  It stands 13 mm higher and tilts less, but its feet slip 3.2 times as much RMS (2.4 times at the maximum, which stays
  within the band only through its 5 mm floor) in fewer, longer stance phases. The main line linked against M0's colcon
  HPIPM reproduces it, so the HPIPM build plays no part.
- **Unitree G1, `walk_0p5`, and Unitree R1, `walk_0p3`:** the first recording of this label, whose commands started
  2.00 s after the release's reset as in M0, stayed within the bands (G1: yaw-rate RMS error 0.284, tilt maximum
  0.107 rad; R1: tilt maximum 0.082 rad). Starting them 1.99 s after it walks the G1 with a yaw-rate RMS error of
  0.429 rad/s and a tilt maximum of 0.132 rad, and the R1 with a tilt maximum of 0.107 rad in 62 stance phases
  (M0: 45), turning further off its heading (final -1.04 against -0.76 rad).
- **Whole-body Unitree G1, `turn_in_place_720`:** the maximum tilt lies 0.7 s after the heading first crosses pi. That
  is the transient of the Euler yaw's wrap, which the quaternion switch removes.
  - In M0 the same transient peaks at 0.100 rad, and its largest tilt lies at the second crossing.
  - With the entry and the release 16 ms later, the first crossing falls at another point of the gait, and the
    transient peaks at 0.147 rad. The first recording, with the commands 2 ms later still, has the same tilt RMS and
    maximum to four digits.
  - In that first recording, outside 16 to 20 s into the commands, the tilt maxima over every window of 3 to 5 s agree
    with M0's within 0.005 rad, at the second crossing included (0.100 against 0.102 rad).
- **Whole-body Unitree G1, `walk_0p5`:** within the bands. Before the code generation was deterministic, this walk
  spread at the rounding level from one generation of its CppAD libraries to the next, even with the M0 cycle order: the
  two copies with that order (one with this HPIPM, one with M0's colcon HPIPM) agreed with M0 to 0.5 % in every banded
  metric but the slip maximum. Over those runs, M0, the first recording and the recording of 2026-10-02:
  - stance-foot slip maximum: 5.2 to 7.1 mm;
  - final heading: 0.05 to 0.11 rad;
  - stance phases: 111 to 127.

  The libraries the deterministic generator makes walk it with a slip maximum of 8.9 mm, a final heading of 0.104 rad
  and 127 stance phases, every time (below); against M0 that is base-height RMS error +3.8 %, slip maximum +61 % (both
  within their bands) and every other banded metric within 2.2 %. Compare it within the bands only.

## What to keep in mind when comparing with it

- **The bands do not see every change of gait.** The stance-foot slip RMS and the number of stance phases are not
  banded, and the slip maximum's 5 mm floor is wide against these robots' 1 to 4 mm. EngineAI SA01's `walk_0p3` here
  is a different gait from M0's with three times its slip; compare a later Euler run of it with this document, and read
  its `stance_foot_slip` fields, not only the verdict.
- **The initial-state gap fields of M0 are mostly of the old definition.** The documents of M0 that the M0 README does
  not mark † measured the gap against the previous solution, before the solve. M0_main reads the SQP's own
  first-iteration `delta_x0`. So their `initial_state_gap` fields differ by definition, for example `max_norm` 11.1
  against 4.8 on the whole-body walk. With the M0 cycle order, the † run above matches M0 there bit for bit. Compare a
  gap with these documents, or with the † documents of M0, never with the other M0 documents.
- **The 720-degree turn exception and an Euler candidate.** As first written, the exception's "no spike" criterion
  (gap below pi / 2) also failed this Euler run, by construction: its baseline wraps by 2 pi too. `compareClosedLoopRuns()`
  now checks a run on Euler coordinates (no quaternion norm) against a baseline whose own gap spiked solve by solve, from
  its time series' per-solve gaps: a solve at or above pi / 2 must lie within 0.5 s of one of the run's own crossings
  and below 2 pi + pi / 2. M1 and M2 on Euler coordinates are checked the same way. A quaternion run is still held to
  the criterion everywhere.
- **What this baseline is for:** a main-line Euler baseline for these eleven runs, recorded with the runner a later
  Euler step (M1, M2) runs on, so that a comparison with it holds no ordering differences. For every other scenario, M0
  and its README still apply, within the timing sensitivity above.

## The whole-body documents recorded again on deterministic code generation

On 2026-10-06 the three whole-body documents and the turn's time series were recorded again; the centroidal ones were
not.

**Why.** The documents of 2026-10-02 came from one of several variants of the whole-body G1's CppAD libraries. CppAD's
recorder shared a parameter between equal constants only when their hash codes matched, and CppADCodeGen's `CG` type was
hashed by its bytes, which hold a heap address, so two generations of the same tapes gave different C sources for six
of the 28 whole-body models, and the runs moved in their last bits with them; the walk moved further
(`humanoid_mpc_validation/README.md`, "Runs are reproducible"). The vendored CppADCodeGen now hashes a constant by its
value (`lib/ocs2/README.md`, "Local changes to the vendored CppAD / CppADCodeGen"), so a generation is the same on any
heap, and the documents here are the only ones the main line makes. Two runs, each on libraries generated afresh, one of
them with glibc's thread cache off (`GLIBC_TUNABLES=glibc.malloc.tcache_count=0`, another heap layout), gave all six
files bit for bit alike.

**What else changed, and what did not.** Live tuning came with the same tree: both MPC nodes, and the closed-loop
runner, now register a parameter updater. Fed nothing, it changes nothing. The weights the whole-body tapes use already
entered the generated code as parameters (the foot cost's and the joint torque cost's), so live tuning added no
parameter, changed no taped function and renamed no library; the runs equal the post-determinism ones of the day
before bit for bit. These documents differ from the first recording's only by the code generation.

**Before and after** (2026-10-02 / 2026-10-06; everything else of the documents, the survival, the failures and the
initial-state gaps included, agrees to a relative 5e-7 or better, but for the walk's):

| Metric | standing | walk_0p5 | turn_in_place_720 |
|---|---|---|---|
| Tilt RMS [rad] | 0.0037 / 0.0037 | 0.0453 / 0.0454 | 0.0455 / 0.0455 |
| Tilt max [rad] | 0.0124 / 0.0124 | 0.0755 / 0.0755 | 0.1467 / 0.1467 |
| Velocity RMS error [m/s] | 0.001 / 0.001 | 0.299 / 0.300 | 0.160 / 0.160 |
| Yaw-rate RMS error [rad/s] | 0.000 / 0.000 | 0.218 / 0.221 | 0.330 / 0.330 |
| Base height mean [m] | 0.7909 / 0.7909 | 0.7787 / 0.7789 | 0.7801 / 0.7801 |
| Base height RMS error [m] | 0.0003 / 0.0003 | 0.0140 / 0.0138 | 0.0121 / 0.0121 |
| Stance-foot slip max [mm] | 0.0 / 0.0 | 5.2 / 8.9 | 9.2 / 9.2 |
| Stance-foot slip RMS [mm] (not banded) | 0.00 / 0.00 | 1.33 / 1.66 | 0.94 / 0.94 |
| Stance phases (not banded) | 2 / 2 | 127 / 127 | 334 / 334 |
| Final heading [rad] (not banded) | 0.001 / 0.001 | 0.093 / 0.104 | 0.141 / 0.141 |
| Solve p99 [ms] | 8.73 / 9.08 | 9.89 / 10.31 | 10.20 / 9.42 |

Each run of 2026-10-06 is within the bands of the 2026-10-02 documents, and against M0 the verdicts are those of the
headline table: standing and the walk within, the turn outside by its tilt maximum alone (0.147 against 0.103 rad, the
Euler wrap's transient above). The turn's heading still peaks at 5.39 rad and its initial-state gap at 2 pi, at
39.72 s.

**The centroidal documents** were left as recorded. The same runs on the deterministic libraries pass their bands and
repeat bit for bit, but differ from them in 14 or 15 of 52 fields: the walks by a relative 1.2e-9 or less, the standing
runs by 4e-8 or less, but for the DRC Atlas's base-height deviation, whose 1e-10 m differ by 1.2e-5. Hashing by value
merges equal constants the old hash kept apart, and no deterministic hash reproduces merges that followed the heap.
Whether to record them again is open; compare a centroidal run with them within the bands.
