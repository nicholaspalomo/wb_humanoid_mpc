# M0_main: the M0 scenarios on the main line, after the ROS removal

The parity check of the ROS removal. M0 (`../M0/`) was recorded on the quaternion worktree before the switch-over, with
the MRT joint controllers as they were then. These are the same runs on the merged main line: the robot process and
the MPC node instead of the ROS sim nodes, `InProcessMpcLink` in place of the controllers' own solver loop, and phase A
of the quaternion switch merged in. The runs are still on the Euler base orientation. Each is held to M0 within the
bands of section 4.5 (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`).

The subset is every configuration's `standing` and its walking scenario (`RobotConfiguration::walkingScenario`:
`walk_0p5`, or `walk_0p3` for EngineAI SA01 and Unitree R1), plus `turn_in_place_720` on the whole-body Unitree G1.
That G1 run is the only M0 run whose heading crosses ±pi.

All eleven documents and the turn's time series were recorded again on 2026-10-06, on CppAD code generation made
deterministic and with the Unitree G1's MuJoCo scene holding the robot on a gantry weld
([below](#all-eleven-documents-recorded-again-deterministic-code-generation-and-the-g1-gantry-weld)). They replace
the documents of 2026-10-02 and the whole-body documents recorded again earlier on 2026-10-06
([below](#the-whole-body-documents-recorded-again-on-deterministic-code-generation)). Where the analysis below quotes
those, it says so. Later that day the Unitree R1's two documents were recorded once more, on the gantry weld its scene
now declares too ([below](#the-unitree-r1s-documents-recorded-again-its-gantry-weld)).

## How it was made

- **Command:** `make closed-loop-metrics ROBOT=<robot> LABEL=M0_main BASELINE=M0 SCENARIO=<scenarios>
  VALIDATION_GIT_COMMIT=<commit> VALIDATION_WORKTREE_STATE=<state>`, one robot at a time, in the ROS-free dev container
  `wbmpc-noros-dev`. `SCENARIO` was a gtest filter: for example `SCENARIO='standing:*/walk_0p5'` expands to
  `--test_filter='*/standing:*/walk_0p5'`.
- **Code:** commit `b41650f060e24cee9b29dec4561a6dc957bc57f1`. On top of the ROS removal it has phase A, the fixes of
  the review of that merge (the runner's command start, its back-off, the localized 720-degree turn exception, the
  per-solve gaps in the time series), the typed textproto configuration, live tuning on both formulations, the
  deterministic code generation, the deletion of OCS2's DDP and the G1's gantry weld. The Unitree R1's two documents
  are of that commit with the R1's gantry weld on top, uncommitted.
  - Every document's `provenance.worktree_state` but the Unitree R1's is `2 changed paths, diff sha256
    ff8d903eadc63349`. That is the state `tools/worktree_state.sh` computed when the recording started, with this
    directory left out, so that the documents of one robot do not change the state the next robot records. The R1's
    two documents have `36 changed paths, diff sha256 61aac8ea80ef8f10`, computed the same way when their recording
    started: the R1's weld and its task file, the review's fixes of tests and documentation, and the edits in progress
    in the checkout (the Rerun viewer, the robots' `launch/*.textproto` files, the Makefile, `.devcontainer/`). No
    closed-loop run reads any of those edits.
  - The two paths were edits that were in progress in the checkout, in the Makefile's `kill-sims` target and in
    `.devcontainer/start_vnc.sh`. While the last three robots ran, the Makefile's `RERUN_SINK` default and the
    robots' `launch/*.textproto` files were edited as well. No run reads any of these files, and no source or Bazel
    file changed. Every document's `configuration_sha256` is the hash of that file at the commit, but the R1's
    `R1.xml` and `task.textproto`, which carry its weld.
  - The documents of 2026-10-02 were of commit `c331ddd76dcf3917bdd9d0578eee6f518a915ba8` plus the uncommitted state
    `2010 changed paths, diff sha256 428fb8a4730b76da`. A first recording of this label on that day, before the fixes
    above, was superseded by them; the analysis below names it "the first recording". Its commands started one control
    period later.
  - The whole-body documents recorded earlier on 2026-10-06 were of commit `f0f55ce92819198127541ffeef5d77d76694f201`
    plus `1663 changed paths, diff sha256 cccb6a1acad348f0`.
- **Configuration:** the shipped typed textprotos (task, reference, gait, PD gains and, where the robot has one, the
  contact planner), URDF and MuJoCo scene. Their SHA-256 are in `provenance.configuration_sha256`. They differ from
  M0's, which hash the YAML files they replaced, but the only value that reaches the closed loop differently from M0 is
  the gantry hold:
  - the ROS removal renamed two robot keys: `enableTelemetry` became `telemetry_sinks`, and
    `useGravityCompFeedforward: false` became `wb_mpc_feedforward: "inverse_dynamics"` (the old value under its new
    name);
  - the visualization publisher's keys, comments and LINT labels were added, and the Unitree R1's telemetry frames were
    renamed; none of these reaches the closed loop;
  - the gantry hold: both G1 configurations and the Unitree R1 name `gantry_hold: "weld_constraint"`, and
    `g1_29dof.xml` and `R1.xml` now have the `gantry` weld it needs. M0 held both robots by `kinematic_teleport`,
    since their scenes had no weld.
- **Solver threads:** the configured ones (DRC Atlas 8, the others 4).
- **Machine:** the same Intel Core Ultra 7 265K as M0, but in another container: `wbmpc-noros-dev`, not
  `devcontainer-app-1`. HPIPM is built from source with `TARGET=GENERIC` (`bazel/system_libs.bzl`), where M0 linked the
  old container's colcon build (`TARGET=AVX`). The solve times are comparable with M0's only roughly.
- **Date:** 2026-10-06 (the first documents 2026-10-02).

## Headline numbers

M0_main first, then M0 (`../M0/`). The metrics are those of the M0 README. "Outside" names the bands a run leaves. The
run that crosses ±pi is compared with `compareClosedLoopRuns()`, with the 720-degree turn exception.

| Robot | Scenario | Survived | Tilt RMS [rad] | Velocity RMS error [m/s] | Yaw-rate RMS error [rad/s] | Solve p99 [ms] | Against M0 |
|---|---|---|---|---|---|---|---|
| drc_atlas | standing | yes / yes | 0.0056 / 0.0056 | 0.000 / 0.000 | 0.000 / 0.000 | 7.38 / 7.20 | within |
| drc_atlas | walk_0p5 | yes / yes | 0.0425 / 0.0426 | 0.214 / 0.215 | 0.051 / 0.052 | 7.25 / 6.94 | within |
| engineai_sa01 | standing | yes / yes | 0.0032 / 0.0032 | 0.000 / 0.000 | 0.000 / 0.000 | 3.42 / 3.25 | within |
| engineai_sa01 | walk_0p3 | yes / yes | 0.0958 / 0.1086 | 0.208 / 0.211 | 0.079 / 0.089 | 3.34 / 3.28 | outside: base height mean 0.770 / 0.757 m, base height RMS error 0.047 / 0.060 m |
| unitree_g1 | standing | yes / yes | 0.0006 / 0.0016 | 0.001 / 0.001 | 0.000 / 0.000 | 8.40 / 8.02 | within |
| unitree_g1 | walk_0p5 | yes / yes | 0.0721 / 0.0728 | 0.289 / 0.296 | 0.276 / 0.303 | 8.22 / 7.91 | within |
| unitree_r1 | standing | yes / yes | 0.0015 / 0.0012 | 0.002 / 0.002 | 0.000 / 0.000 | 7.50 / 7.25 | within |
| unitree_r1 | walk_0p3 | yes / yes | 0.0480 / 0.0483 | 0.237 / 0.241 | 0.273 / 0.277 | 7.91 / 7.41 | within |
| unitree_g1_wb | standing | yes / yes | 0.0028 / 0.0037 | 0.001 / 0.001 | 0.000 / 0.000 | 8.98 / 8.63 | within |
| unitree_g1_wb | walk_0p5 | yes / yes | 0.0453 / 0.0450 | 0.301 / 0.301 | 0.226 / 0.223 | 9.89 / 10.06 | within |
| unitree_g1_wb | turn_in_place_720 | yes / yes | 0.0454 / 0.0441 | 0.176 / 0.158 | 0.331 / 0.330 | 9.88 / 10.40 | outside: tilt max 0.139 / 0.103 rad, heading peak 5.12 / 5.27 rad |

Every run had no failed solve, no reset during the commands, no non-finite value and no saturated command. The p99
solve times are within -5 % to +9 % of M0's (-10 % to +8 % in the documents of 2026-10-02: a solve time varies from
run to run). The whole-body G1's turn peaks at a heading of 5.12 rad, against M0's 5.27 rad. Like M0's, its
initial-state gap reaches 2 pi (6.285 rad), from the Euler wrap. Its time series carries the per-solve gaps
(`solve_initial_state_rotation_gap`), which show the wrap at each of its own two crossings (18.92 s and 38.57 s) and
its decay below pi / 2 within 0.17 s. Nothing else in the run spikes.

## Why two runs leave the bands: the runner's cycle order and the G1's gantry hold, not the controllers

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

To isolate the ordering, a copy of the main-line state was built before the fixes of the documents of 2026-10-02, and
before the G1's gantry weld: its G1 ran on `kinematic_teleport`, as M0's did. In it, the runner's operator sequence is
moved back to the start of the cycle, and the whole-body controller is again handed its mode before its posture.
Nothing else changed, and the copy ran the same tests. It reproduces M0:

| Run | Largest relative difference from M0 over the banded metrics (with the M0 cycle order) |
|---|---|
| drc_atlas standing, walk_0p5 | 2e-5 (a yaw-rate error of 2e-6 rad/s), 9e-8 |
| engineai_sa01 standing, walk_0p3 | 3e-6, 4e-9 |
| engineai_sa01 arc (a re-recorded † run of M0) | 0: all 50 compared fields bit for bit, the initial-state gap included |
| unitree_g1 standing, walk_0p5 | 3e-9, 1e-8 |
| unitree_r1 standing, walk_0p3 | 2e-10, 6e-13 |
| unitree_g1_wb standing, turn_in_place_720 | 1e-8, 4e-9 (tilt max 0.10260 rad, M0's) |
| unitree_g1_wb walk_0p5 | 0.5 % but for the stance-foot slip maximum (7.1 against 5.5 mm), see below |

So every violation of the documents of 2026-10-02 was the ordering change, not a change of the controllers, the MPC or
the simulator. Since then the G1's two configurations and the Unitree R1 also differ from M0 by their gantry hold, which
changes the state they are released in ([below](#what-the-weld-changes-the-state-at-the-release)). Of the two runs that
leave M0's bands, EngineAI SA01's walk leaves them by the cycle order, and the whole-body G1's turn by both: its tilt
maximum by the cycle order, its heading peak by the weld (on `kinematic_teleport` it turned further than M0). Three of
the walks are this sensitive to timing: a shift of 10 ms in when the gantry is released or when the commands start,
against the gait schedule, puts them in another walking pattern. The runs are deterministic: on EngineAI SA01's walk,
three reruns and runs with 1, 2, 3 and 8 solver threads reproduced the first recording to every printed digit, and every
run here repeats bit for bit
([below](#all-eleven-documents-recorded-again-deterministic-code-generation-and-the-g1-gantry-weld)).

- **EngineAI SA01, `walk_0p3`:** this robot is marginal at this speed. M0 tilts it by 0.11 rad RMS, and it falls at
  0.5 m/s. With the entry and the release 10 ms later its walk takes another pattern. The first recording, with the
  commands a further 10 ms later, walks the same pattern (every metric below within 0.1 %), so the release decides it,
  not the commands. The new pattern is not simply better:

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
- **Unitree R1, `walk_0p3`:** within the bands again since its weld. On `kinematic_teleport` the first recording,
  whose commands started 2.00 s after the release's reset as in M0, stayed within the bands (tilt maximum 0.082 rad),
  and starting them 1.99 s after it walked the R1 with a tilt maximum of 0.107 rad in 62 stance phases (M0: 45),
  turning further off its heading (final -1.04 against -0.76 rad). Released from the weld it walks with a tilt maximum
  of 0.097 rad in 43 stance phases and ends at a heading of -0.78 rad
  ([below](#the-unitree-r1s-documents-recorded-again-its-gantry-weld)).
- **Unitree G1, `walk_0p5`:** within the bands again since the weld. The documents of 2026-10-02, on
  `kinematic_teleport`, left them like the R1: the first recording stayed within (yaw-rate RMS error 0.284, tilt
  maximum 0.107 rad), and starting the commands 1.99 s after the release's reset walked it with a yaw-rate RMS error of
  0.429 rad/s and a tilt maximum of 0.132 rad. Released from the weld it walks with 0.276 rad/s and 0.106 rad (M0:
  0.303 rad/s and 0.106 rad), its base 0.1 mm from M0's mean height.
- **Whole-body Unitree G1, `turn_in_place_720`:** the maximum tilt, 0.139 rad, lies 0.84 s after the heading first
  crosses pi. That is the transient of the Euler yaw's wrap, which the quaternion switch removes.
  - In M0 the same transient peaks at 0.100 rad, and its largest tilt lies at the second crossing.
  - With the entry and the release 16 ms later, the first crossing falls at another point of the gait. On
    `kinematic_teleport` (the documents of 2026-10-02 and of earlier on 2026-10-06) the transient peaked at 0.147 rad,
    0.72 s after the first crossing, and the turn reached 5.39 rad. The first recording, with the commands 2 ms later
    still, had the same tilt RMS and maximum to four digits, and outside 16 to 20 s into the commands its tilt maxima
    over every window of 3 to 5 s agreed with M0's within 0.005 rad, at the second crossing included (0.100 against
    0.102 rad).
  - Released from the weld, the robot crosses pi at 18.92 s (19.14 s on `kinematic_teleport`, 18.78 s in M0) but
    turns less far: its heading peaks at 5.12 rad, 3 % short of M0's 5.27 rad. The 720-degree exception requires a run
    to turn at least as far as its baseline, so the turn now also leaves M0's bands by its heading.
- **Whole-body Unitree G1, `walk_0p5`:** within the bands. Before the code generation was deterministic, this walk
  spread at the rounding level from one generation of its CppAD libraries to the next, even with the M0 cycle order: the
  two copies with that order (one with this HPIPM, one with M0's colcon HPIPM) agreed with M0 to 0.5 % in every banded
  metric but the slip maximum. Over those runs, M0, the first recording and the recording of 2026-10-02:
  - stance-foot slip maximum: 5.2 to 7.1 mm;
  - final heading: 0.05 to 0.11 rad;
  - stance phases: 111 to 127.

  On the deterministic libraries and the weld it walks with a slip maximum of 6.6 mm, a final heading of 0.133 rad and
  118 stance phases, every time. Against M0 that is a slip maximum of +20 % (within its band), a tilt maximum of -7.8 %,
  a base-height RMS error of +2.4 % and every other banded metric within 1.8 %. Compare it within the bands only.

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
- **The gantry hold of the G1 and the R1.** The two G1 configurations and the Unitree R1 are released from the weld
  here, and from `kinematic_teleport` in M0. A run of theirs compared with M0 holds that difference besides the cycle
  order; compared with these documents, none.
- **What this baseline is for:** a main-line Euler baseline for these eleven runs, recorded with the runner a later
  Euler step (M1, M2) runs on, so that a comparison with it holds no ordering differences. For every other scenario, M0
  and its README still apply, within the timing sensitivity above.

## The whole-body documents recorded again on deterministic code generation

Earlier on 2026-10-06 the three whole-body documents and the turn's time series were recorded again; the centroidal
ones were not. All eleven were recorded again later that day
([below](#all-eleven-documents-recorded-again-deterministic-code-generation-and-the-g1-gantry-weld)), so the documents
of this section are superseded too; its numbers are the "before" of the whole-body columns there.

**Why.** The documents of 2026-10-02 came from one of several variants of the whole-body G1's CppAD libraries. CppAD's
recorder shared a parameter between equal constants only when their hash codes matched, and CppADCodeGen's `CG` type was
hashed by its bytes, which hold a heap address, so two generations of the same tapes gave different C sources for six
of the 28 whole-body models, and the runs moved in their last bits with them; the walk moved further
(`humanoid_mpc_validation/README.md`, "Runs are reproducible"). The vendored CppADCodeGen now hashes a constant by its
value (`lib/ocs2/README.md`, "Local changes to the vendored CppAD / CppADCodeGen"), so a generation is the same on any
heap. Two runs, each on libraries generated afresh, one of them with glibc's thread cache off
(`GLIBC_TUNABLES=glibc.malloc.tcache_count=0`, another heap layout), gave all six files bit for bit alike.

**What else changed, and what did not.** Live tuning came with the same tree: both MPC nodes, and the closed-loop
runner, now register a parameter updater. Fed nothing, it changes nothing. The weights the whole-body tapes use already
entered the generated code as parameters (the foot cost's and the joint torque cost's), so live tuning added no
parameter, changed no taped function and renamed no library; the runs equal the post-determinism ones of the day
before bit for bit. These documents differed from those of 2026-10-02 only by the code generation.

**Before and after** (2026-10-02 / earlier on 2026-10-06; everything else of the documents, the survival, the failures
and the initial-state gaps included, agreed to a relative 5e-7 or better, but for the walk's):

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

Each of those runs was within the bands of the 2026-10-02 documents. The centroidal documents were left as recorded
then: the same runs on the deterministic libraries passed their bands and repeated bit for bit, but differed from them
in 14 or 15 of 52 fields. The next section records them again.

## All eleven documents recorded again: deterministic code generation and the G1 gantry weld

Later on 2026-10-06 every document and the turn's time series were recorded again. The Unitree R1's two documents of
this recording were replaced afterwards, on the R1's own weld
([below](#the-unitree-r1s-documents-recorded-again-its-gantry-weld)); this section describes them as they were.

**Why.** Two changes since the documents above:

1. **The deterministic code generation.** The eight centroidal documents came from libraries generated before the code
   generation was deterministic. Hashing a constant by its value merges equal constants that the old hash kept apart,
   and no deterministic hash reproduces merges that followed the heap. So the same runs on the deterministic libraries
   differ from those documents at the rounding level, and the deterministic libraries are the only ones the main line
   makes. Recording the documents from them settles what the section above left open.
2. **The G1's gantry weld.** `g1_29dof.xml` now has the inactive `gantry` weld from the world to `pelvis` that the DRC
   Atlas and EngineAI SA01 scenes have. Both G1 configurations hold the robot on `weld_constraint` while it is locked,
   as their task files name it, instead of logging an error and falling back to `kinematic_teleport`
   (`test_robot_scene_gantry_hold` holds every configuration's scene to its task file's hold). That changes every G1
   run.

**Before the recording, against the documents it replaces.** Each run was made by the recording's command with
`BASELINE=M0_main` and the old documents in place:

- **DRC Atlas, EngineAI SA01, Unitree R1:** all six runs stayed within the bands of the documents they replace. They
  differ from them in 14 or 15 of 52 fields, by a relative 1.2e-5 at most (DRC Atlas standing's base-height deviation
  of 1e-10 m), the walks by 2e-11 or less. They are bit for bit, every field and time series, the runs on the
  deterministic libraries from before the weld and the deletion of OCS2's DDP. That includes the Unitree R1, whose task
  file then named `kinematic_teleport`, the hold it fell back to before.
- **Unitree G1, both formulations:** the weld is the whole difference. For a check, both G1 task files were set back to
  `kinematic_teleport`, with the scene keeping its weld, inactive. The whole-body runs then gave the documents they
  replace bit for bit, every field and the turn's time series. The centroidal runs gave the deterministic generation's
  pattern above (15 of 52 fields, 1.2e-9 at most). So the inactive weld in the scene changes nothing, and the hold
  changes the G1's runs. On the weld, the standing runs and the whole-body walk stay within the old documents' bands.
  The centroidal walk leaves them by its yaw-rate RMS error (0.276 against 0.429 rad/s, into M0's band). The turn
  leaves them by its heading peak (5.12 against 5.39 rad).

### What the weld changes: the state at the release

`kinematic_teleport` writes the base's pose before each step and leaves the base unsupported during it. The gravity
torques the controllers command on the gantry then lift the limbs instead of holding them. In the 1.5 s on the gantry
(JOINT_PD, then the entry into WB_MPC), the knees bend past the JOINT_PD posture and the arms move off it. The weld
carries the base while the dynamics are integrated, and the posture holds. A run of each G1 configuration on each hold,
with no standing time and 2 s at rest, measured this. It sends the zero command the scenarios send before theirs, so
its window is the first 2 s after the release of every run above. The diagnostic is not kept in the tree.

| From the release to the commands | G1 centroidal: `kinematic_teleport` / weld | G1 whole-body: `kinematic_teleport` / weld |
|---|---|---|
| joint posture against JOINT_PD's at the release, RMS (largest) | 1.7° (knee 3.7°) / 0.9° (knee 2.8°) | 2.2° (knee 5.7°) / 0.6° (knee 1.6°) |
| base tilt at the release | 0 / 5.4 mrad | 0 / 1.1 mrad |
| base drop over the next 0.3 s | 5.7 / 1.4 mm | 5.6 / 2.6 mm |
| base tilt when the commands start, 2 s later | 5.7 / 1.7 mrad | 12.5 / 8.5 mrad |

Released from the drifted posture, the robot sinks further, and 2 s later it is more tilted. The standing runs carry the
tail of that transient: each one's tilt maximum is its tilt at the start of the commands (0.0057 to 0.0017 rad on the
centroidal G1, 0.0124 to 0.0085 rad on the whole-body G1). The walks and the turn start from another state, and the
gait-sensitive ones change their pattern.

**Before and after** for the G1 (the replaced documents / these). The centroidal "before" is of 2026-10-02, the
whole-body "before" of earlier on 2026-10-06:

| Metric | unitree_g1 standing | unitree_g1 walk_0p5 | unitree_g1_wb standing | unitree_g1_wb walk_0p5 | unitree_g1_wb turn_in_place_720 |
|---|---|---|---|---|---|
| Tilt RMS [rad] | 0.0016 / 0.0006 | 0.0766 / 0.0721 | 0.0037 / 0.0028 | 0.0454 / 0.0453 | 0.0455 / 0.0454 |
| Tilt max [rad] | 0.0057 / 0.0017 | 0.1318 / 0.1059 | 0.0124 / 0.0085 | 0.0755 / 0.0711 | 0.1467 / 0.1392 |
| Velocity RMS error [m/s] | 0.001 / 0.001 | 0.320 / 0.289 | 0.001 / 0.001 | 0.300 / 0.301 | 0.160 / 0.176 |
| Yaw-rate RMS error [rad/s] | 0.000 / 0.000 | 0.429 / 0.276 | 0.000 / 0.000 | 0.221 / 0.226 | 0.330 / 0.331 |
| Base height mean [m] | 0.7904 / 0.7907 | 0.7554 / 0.7585 | 0.7909 / 0.7909 | 0.7789 / 0.7790 | 0.7801 / 0.7801 |
| Base height RMS error [m] | 0.0009 / 0.0005 | 0.0377 / 0.0342 | 0.0003 / 0.0003 | 0.0138 / 0.0136 | 0.0121 / 0.0123 |
| Stance-foot slip max [mm] | 0.0 / 0.0 | 1.8 / 1.8 | 0.0 / 0.0 | 8.9 / 6.6 | 9.2 / 4.8 |
| Stance-foot slip RMS [mm] (not banded) | 0.0 / 0.0 | 0.8 / 0.8 | 0.0 / 0.0 | 1.7 / 1.7 | 0.9 / 0.8 |
| Stance phases (not banded) | 2 / 2 | 37 / 38 | 2 / 2 | 127 / 118 | 334 / 389 |
| Heading peak [rad] (banded in the turn) | 0.000 / 0.000 | 0.242 / 0.163 | 0.001 / 0.000 | 0.154 / 0.174 | 5.386 / 5.120 |
| Final heading [rad] (not banded) | 0.000 / -0.000 | 0.069 / 0.058 | 0.001 / 0.000 | 0.104 / 0.133 | 0.141 / -0.116 |
| Joint torque RMS [N m] (not banded) | 2.35 / 2.35 | 10.58 / 10.35 | 2.37 / 2.37 | 9.48 / 9.47 | 9.56 / 9.77 |
| Initial-state gap, max norm (not banded) | 0.003 / 0.001 | 0.188 / 0.159 | 0.151 / 0.111 | 11.128 / 11.118 | 9.618 / 12.711 |
| Solve p99 [ms] | 7.91 / 8.40 | 8.51 / 8.22 | 9.08 / 8.98 | 10.31 / 9.89 | 9.42 / 9.88 |

Against M0 the G1's centroidal walk is within the bands again, and the turn leaves them by its tilt maximum and its
heading peak (the headline table). The other six documents differ from the ones they replace only at the rounding level
above.

**Determinism.** After the recording every run was made once more by the same command, each on CppAD libraries
generated afresh, and held to these documents. All eleven documents and every time series repeat bit for bit: every
field but `provenance.worktree_state` (the checkout's edits had moved on) and the solve times, which time the machine.
The run before the recording, above, gave them too.

## The Unitree R1's documents recorded again: its gantry weld

Later on 2026-10-06 the Unitree R1's two documents were recorded once more; the other nine are those of the section
above.

**Why.** `R1.xml` declared no `gantry` weld, so the R1 hung from `kinematic_teleport`: it fell back to it from the
default with an ERROR at every start, and after the G1's weld its task file named it. On that hold GRAVITY_COMP cannot
keep the R1's limbs still on the gantry, which is where the mode is operated: the base is in free fall inside each step,
so the gravity torque lifts the limbs into their stops. `R1.xml` now has the weld of the other scenes, from the world
to `pelvis_link`, and the R1's task file names `weld_constraint`. `test_mujoco_sim_interface_gantry` measures both
holds on every scene, commanding each joint's gravity torque on the locked gantry with no feedback (MuJoCo's own, so
the model is exact). The largest excursion of any joint:

| | DRC Atlas | EngineAI SA01 | Unitree G1 | Unitree R1 |
|---|---|---|---|---|
| weld, from the catch, over 11 s (the swing as the soft weld takes the weight, and the settling) | 2.4 mrad | 0.7 mrad | 4.1 mrad | 7.6 mrad |
| weld, over 3 s after 8 s of settling | 3e-5 rad | 5e-18 rad | 4e-18 rad | 2e-20 rad |
| `kinematic_teleport`, from the catch, over 3 s | 2.23 rad | 1.22 rad | 2.54 rad | 2.11 rad |

**Before the recording, against the documents it replaces.** The recording's command with `BASELINE=M0_main` and the
replaced documents in place: both runs stayed within their bands, differing in 15 (standing) and 16 (walk) of 52
fields. The two documents' `configuration_sha256` differ from the replaced ones in `R1.xml` and the task file only, and
the code is the same, so the hold is the whole difference.

**What the weld changes.** As on the G1, the robot is released from another posture. The runs' time series start with
the commands, 2 s after the release: there the R1's base tilts by 2.1 mrad on the weld against 4.2 mrad on
`kinematic_teleport`, at the same height (0.7339 against 0.7338 m), and that tilt is the standing run's tilt maximum.
The diagnostic that measured the G1's release transient needs a package in the checkout, and was not repeated for the
R1.

**Before and after** (the replaced documents / these):

| Metric | unitree_r1 standing | unitree_r1 walk_0p3 |
|---|---|---|
| Tilt RMS [rad] | 0.0011 / 0.0015 | 0.0556 / 0.0480 |
| Tilt max [rad] | 0.0042 / 0.0021 | 0.1073 / 0.0967 |
| Velocity RMS error [m/s] | 0.002 / 0.002 | 0.256 / 0.237 |
| Yaw-rate RMS error [rad/s] | 0.000 / 0.000 | 0.285 / 0.273 |
| Base height mean [m] | 0.7362 / 0.7363 | 0.7054 / 0.7085 |
| Base height RMS error [m] | 0.0044 / 0.0043 | 0.0374 / 0.0337 |
| Stance-foot slip max [mm] | 11.6 / 11.6 | 9.9 / 7.2 |
| Stance-foot slip RMS [mm] (not banded) | 11.6 / 11.6 | 3.3 / 3.1 |
| Stance phases (not banded) | 2 / 2 | 62 / 43 |
| Heading peak [rad] | 0.000 / 0.000 | 0.111 / 0.112 |
| Final heading [rad] (not banded) | 0.000 / 0.000 | -1.040 / -0.776 |
| Joint torque RMS [N m] (not banded) | 1.82 / 1.80 | 8.17 / 7.66 |
| Initial-state gap, max norm (not banded) | 0.016 / 0.017 | 0.163 / 0.150 |
| Solve p99 [ms] | 7.57 / 7.50 | 8.05 / 7.91 |

Against M0 both runs are now within the bands. The walk is close to M0's pattern again: 43 stance phases against 45, a
final heading of -0.78 against -0.76 rad and a tilt maximum of 0.097 against 0.088 rad, where the replaced document
left M0's bands by its tilt maximum of 0.107 rad.

**Determinism.** After the recording both runs were made once more by the recording's Bazel command, on CppAD libraries
generated afresh, and held to these documents: both documents and both time series repeat bit for bit, every field but
`provenance.worktree_state` and the solve times. The run before the recording, above, gave them too.
