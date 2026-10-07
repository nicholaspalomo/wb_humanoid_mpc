# humanoid_common_mpc

The parts of the humanoid MPCs that the centroidal MPC (`humanoid_centroidal_mpc`) and the whole-body MPC
(`humanoid_wb_mpc`) share: the robot model settings and the Pinocchio model, the costs and constraints and the factory
that builds them, the gaits, the reference managers and the target trajectories, the contact planner, the locomotion
heuristics and the MRT joint controllers' common parts.

## The configuration it reads

The library is configured by textprotos, each a message of `humanoid_nmpc/humanoid_mpc_config` parsed strictly: a field
the schema does not know, a value of the wrong type or a field given twice is refused with its file, line and column,
and a retired key is refused with what replaced it ([`humanoid_mpc_config/README.md`](../humanoid_mpc_config/README.md)).

| File | Message | Read through |
|---|---|---|
| `robot_models/<robot>/<package>/config/mpc/task.textproto` | `TaskFile` | `loadTaskFile()` |
| `.../config/command/reference.textproto` | `ReferenceFile` | `loadReferenceFile()` |
| `.../config/mpc/contact_planning.textproto` (Atlas, SA01) | `ContactPlanningFile` | `loadContactPlanningFileBeside()`: the file beside the task file, or none |
| `.../config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | `loadJointPdGainsFile()`, `loadJointPdGains()` |
| `config/command/gait.textproto` (this package, shared by every robot) | `GaitFile` | `loadGaitFile()` |

The loaders are in `config/ConfigFiles.h`. Each gives the generated struct of its message
(`ocs2::humanoid::mpc_config::<Name>`), and the conversions in `config/<area>/*FromConfig.h` turn its blocks into
the library's own types:

| Area | Conversions |
|---|---|
| `solver` | `solverSettingsFromConfig()`: the SQP, rollout and MPC settings |
| `model` | `ModelSettings::Create(TaskFile, ...)`, `mpcFormulationTasksFromConfig()`, the contact schedule source, the contact input parameterization, the centroidal model type |
| `weights` | the state and input weights and the initial state, addressed by coordinate and joint name on a `StateInputLayout` |
| `costs` | contacts, wrench cones and bases, task-space costs, joint limits, collision spheres, the knee mimic joints |
| `swing` | the swing trajectories and the locomotion heuristics |
| `reference` | `referenceSettingsFromConfig()`, the default joint state, the gaits and mode schedules |
| `contact_planning` | `contactPlanningConfigFromConfig()`, or the library defaults for a robot without a file |
| `robot` | the joint PD gains and the controller-side settings of the robot process |

A conversion refuses a value the library cannot run with, naming the field of the file
(`state_weights.joint_positions[joint=back_bkz].value`, `contacts.contact_wrench_cone_soft_constraint.friction_coefficient`).

The roots of the configuration take the typed files, and keep a form that takes the file's path for the binaries and
the tests that start from one: `ModelSettings::Create()`, `loadCustomPinocchioInterface()`, `GaitSchedule::Create()`,
`ProceduralMpcMotionManager::Create()` and the target calculators of the two MPCs. `HumanoidCostConstraintFactory`
makes the terms of a typed task file. Hot reloads hand typed settings to the running objects
(`TargetTrajectoriesCalculatorBase::applyCommandLimits()`, `ProceduralMpcMotionManager::applyCommandLimits()`).

## Tuning

Edit the textproto by field name, or tune it live with the tuning GUI, which renders every hyperparameter from the
schema and saves the file without losing its comments
([`humanoid_nmpc/remote_control/README.md`](../remote_control/README.md), "How the GUI saves"). The running stack
applies a field marked `RELOAD_HOT` at once and a `RELOAD_START_UP` one at the next start.

A new hyperparameter of this library is a field of its schema in `humanoid_mpc_config` whose default is the library's
default, a line of the conversion that reads it, and the tests of the conversion under `test/config/<area>/` (every
field of a block reaches the struct, the schema defaults are the struct's defaults); the GUI needs no code for it.

## Tests

| Directory | What |
|---|---|
| `test/` | the library's units; a test that needs a configuration writes a textproto, or reads a shipped one through the loaders |
| `test/config/<area>/` | the conversions of the typed files (`config_<area>_test`) and the roots on every shipped robot (`config_test`) |
