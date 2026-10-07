# tools/config_dump

Every value the start-up of a robot configuration's stack builds from its configuration files, as text: what the MPC,
the robot process and the lockstep driver actually run with. A debugging aid: dump a configuration before and after an
edit of its files, or of the code that reads them, and compare the two.

```bash
bazel run //tools/config_dump -- --robot=drc_atlas > before.txt
# edit the configuration, or the code that reads it
bazel run //tools/config_dump -- --robot=drc_atlas | diff before.txt -
```

The robots are the five configurations of the closed-loop validation (`RobotConfiguration`): `drc_atlas`,
`engineai_sa01`, `unitree_g1`, `unitree_r1` (centroidal MPC) and `unitree_g1_wb` (whole-body MPC). Run one robot at a
time: building an MPC interface loads, or the first time generates and compiles, the robot's CppAD libraries. The tool
runs in the checkout (`$BUILD_WORKSPACE_DIRECTORY`), so it reads the checkout's files and reuses its `cppad_code_gen/`.
The dump goes to the standard output; logs and errors go to the standard error.

## What is dumped

The stack is built through the entry points that take the files by path - the MPC interfaces' `Create`, the target
trajectories calculators' `Create`, `ProceduralMpcMotionManager::Create`, the MRT joint controllers' `Create`,
`loadRobotProcessSettings`, `loadVisualizationConfig` and `loadKeyboardCommandLimits` - the calls the binaries make at
start-up. In order:

| Section | From |
|---|---|
| `[model_settings]`, `[solver]` | the MPC interface: `ModelSettings`, the MPC, SQP and rollout settings |
| `[interface]` | the initial state, the formulation's names, the basis vectors, the names of the terms |
| `[reference_manager]` | the swing trajectory settings, the initial mode schedule, the terrain height |
| `[target_trajectories]` | the targets of a velocity command beyond every limit (which the limits clamp) and of a pose command |
| `[motion_manager]` | the gaits, the acceleration ramps and the command filter |
| `[pd_gains]` | the MRT joint controller's PD gains, resolved over its defaults |
| `[robot_process]`, `[visualization]`, `[keyboard_teleop]` | the robot process's, the visualization publisher's and the keyboard teleoperation's settings |
| `[problem]` | the optimal control problem on a walking schedule (`ProblemDump.h`): every term's name, and at three points of the horizon and at its end every cost and soft-constraint term's value and quadratic approximation and the linear-quadratic approximation of the whole problem |

The weights are shown through the problem rather than through accessors: a weight is the Hessian of its term, whose
diagonal is dumped element by element. A double is written `%a (%.17g)`, so that equal lines are equal bits; a matrix
too large to read is written as its fingerprint, its size, nonzeros, sum and a hash of its elements' bits
(`ValueDump.h`), which a change of any bit changes.

The contact planner's configuration is not dumped: no shipped robot plans its contacts, so its file is not part of
what any stack builds (its conversion's own tests, humanoid_common_mpc/test/config/contact_planning/, check it).

Nothing compares a dump with a recorded one: a recorded dump pins tuned values, which the tests of the configuration
must not (humanoid_nmpc/humanoid_mpc_config/README.md, "Tests").
