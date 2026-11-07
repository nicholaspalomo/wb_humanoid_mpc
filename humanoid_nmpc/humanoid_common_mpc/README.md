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

## Contacts and modes

The contact set is run-time data, `ModelSettings::contactNames`: the 3-DoF point contacts of
`model_settings.contacts_3dof` first, then the 6-DoF contacts of `model_settings.contact_names_6dof`, which is OCS2's
input layout. `ModelSettings::numContacts()` is its size. Any mix is valid, a set of point contacts alone included (a
point-footed quadruped's four feet), and `ModelSettings::Create()` refuses more than `kMaxNumContacts` (32) contacts.

Generic code, anything meaningful for any contact count, holds per-contact data in two allocation-free types:
- `ContactArray<T>` (`common/ContactArray.h`) holds one value per contact: an `std::array` of capacity
  `kMaxNumContacts` with a size set once, at construction or with `assign()`. Indexing is unchecked, and the container
  never allocates, so the realtime path holds its wrenches, frames and forces in one. `kMaxNumContacts` is paired with
  MuJoCo's contact mask capacity (`MujocoUtils.h`).
- `ContactMask` (`common/ContactMask.h`) holds the active contacts of a set, stored as the mode number itself. It is
  trivially copyable and is passed by value.

A mode number is the bitmask of the active contacts: contact `i` is bit `numContacts - 1 - i`, so contact 0 is the most
significant bit, and a set of `n` contacts has `2^n` modes (`numModeNumbersOf()`). A number with a bit at or past `n`
is no mode, and no contact is active in it. `contactIsActiveInMode()`, `contactModeBit()` and `numModeNumbersOf()`
(`gait/MotionPhaseDefinition.h`) are the reference semantics, and `ContactMask::fromModeNumber()` follows them. With
the feet listed last, the biped's modes keep their numbers in any larger set: `STANCE` is 3, the two feet.

Code written for a left and a right foot uses the biped vocabulary:
- `common/BipedFeet.h`: `kContactLeftIndex` (0), `kContactRightIndex` (1), `kBipedFootCount`, `BipedFeetArray<T>`
  and `makeBipedFeetArray()`;
- `gait/BipedModes.h`: `kBipedModeFly`, `kBipedModeRf`, `kBipedModeLf` and `kBipedModeStance` (0 to 3),
  `kNumBipedModes`, and `bipedFootContactsOfMode()` / `bipedModeOfFootContacts()` between a mode and the two feet.
  The gait files' mode names `RF` and `LF` are the biped's.

A gait or reference file names a mode as `parseModeNumber()` reads it against the contact set's `GaitModeLayout`
(`gaitModeLayoutOf()`, `gait/GaitSchedule.h`): `FLY` and `STANCE` on any set, `LF` and `RF` on a set with two 6-DoF
feet, or one digit 0 or 1 per contact in contact order (`"1001"`: contacts 0 and 3 of four). `STANCE` is the two feet
(3) on a set ending in two 6-DoF feet, the shipped robots and the DRC Atlas multi-contact set included, and every contact
on any other set. `GaitSchedule` starts the schedule, separates gaits and ends the schedule with that stance.

The biped-only components refuse a contact set they cannot run on when they are built, never per tick:
- the contact planner: `LipContactPlanner` and `HlipContactPlanner` in `Create()` and `setConfig()`, and the terms
  `foot_separation`, `zmp_support_region`, `yaw_torque_budget` and `alternating_feet` and the foothold integrator in
  `ContactPlanningTerm::validateContactSet()`, which `ContactPlanningProblem::finalize()` calls. Each returns the
  InvalidArgument of `checkTwoContactCount()` (`common/TwoContactCount.h`), which names the component and the
  contact count: "FootSeparationConstraint is written for two contacts; the contact set has 4 contacts". The check
  reads the count alone, so a set of two point contacts passes it as a left and a right contact;
- the knee mimic blocks (`kneeMimicJointsFromConfig()`), with the same message;
- the reference manager's nominal footholds and swing references, which are `std::nullopt` on a set other than the
  biped's.

The planner rules `no_flight` and `phase_durations` and the `no_flight` row are written over the number of feet, and
`minimum_double_support` is a minimum support count (every foot by default). The costs, constraints and heuristics
written for two feet are refused by the MPC that selects them: the centroidal MPC's `checkCentroidalContactSet()`
names each one ([`multi_contact_planning`](../docs/multi_contact_planning/README.md), section 7.2), and the whole-body
MPC runs on the biped's two 6-DoF feet alone and refuses every other set.

The tests of the contact types are `testContactArray`, `testContactMask`, `testContactModeBits` and
`testContactSetAllocations` (zero allocations on the per-tick operations); `testContactSetReference` runs the reference
manager and the swing planner on four and six contacts.

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
