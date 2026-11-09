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
| `config/command/gait.textproto` (this package, shared by the bipeds) | `GaitFile` | `loadGaitFile()` |
| `config/command/quadruped_gait.textproto` (this package, shared by the quadrupeds) | `GaitFile` | `loadGaitFile()` |

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

The legs group the contacts (`ModelSettings::legs`, from `model_settings.legs`). A leg has a name, the contacts at its
end (one point foot, one wheel or one 6-DoF sole, or several), and a hip. The hip is the frame `hip_frame_name` names,
or else the first joint below the floating base on the way up from the leg's first contact (`hipJointOf()`,
`pinocchio_model/pinocchioUtils.h`, the walk the locomotion heuristics use too). `ModelSettings::Create()` reads the
hip's position relative to the base at the URDF's neutral configuration and derives the leg's side from its y
(`lateralSign`, +1 left, -1 right; a hip within `kHipSideTolerance` of the base's x-z plane is refused) and its front
or hind position from its x against the mean hip x (`longitudinal`, `kCenter` within `kHipFrontHindTolerance`). Sides and positions are never
typed. A biped contact set whose task file lists no legs gets two: `left` on contact 0 and `right` on contact 1,
refused when a derived side contradicts that order, so no shipped file names its legs. A foot without a parent joint
of the URDF keeps the side of its index and has no hip. Any other set without `model_settings.legs` has no legs: the
multi-contact Atlas has none. `legOfContact()` finds a contact's leg, and `isLeggedContactSet()` is true when every
contact is in a leg.

The pendulum length above `feetmean` (`computeProxyPosComRtFeetmeanEwrtWorldZ()`) is the CoM height above the mean of
the legs' contacts: the two feet of a biped, by the arithmetic it always had, every contact of a leg on another legged
set, and the lowest contact on a set without legs.

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
`plannedContactSwitchTimes()` (`gait/ModeScheduleSwitchTimes.h`) reads a mode schedule per contact: the time since its
flag last changed and until it next changes, +infinity where the schedule shows no change, without allocating (the
robot process passes them to the state estimator's contact-trust ramp).

Code written for a left and a right foot uses the biped vocabulary:
- `common/BipedFeet.h`: `kContactLeftIndex` (0), `kContactRightIndex` (1), `kBipedFootCount`, `BipedFeetArray<T>`
  and `makeBipedFeetArray()`;
- `gait/BipedModes.h`: `kBipedModeFly`, `kBipedModeRf`, `kBipedModeLf` and `kBipedModeStance` (0 to 3),
  `kNumBipedModes`, and `bipedFootContactsOfMode()` / `bipedModeOfFootContacts()` between a mode and the two feet.
  The gait files' mode names `RF` and `LF` are the biped's.

A gait or reference file names a mode as `parseModeNumber()` reads it against the contact set's `GaitModeLayout`
(`gaitModeLayoutOf()`, `gait/GaitSchedule.h`), trying the forms in this order:
- `FLY` (no contact) and `STANCE` on any set;
- `LF` and `RF` on a set whose `STANCE` is the two feet (below), which therefore refuses a leg of either name;
- distinct leg names joined by `_`, the contacts of those legs: `LF_RH` is 9 on a quadruped whose contacts and legs
  are LF, RF, LH and RH;
- distinct contact names joined by `+`: `sole_l_toe_contact+sole_r_heel_contact`;
- one digit 0 or 1 per contact in contact order (`"1001"`: contacts 0 and 3 of four).

A name listed twice, a leg or contact the set does not have and any other string are refused with the forms and the
legs of the set. `STANCE` is the two feet (3) on a set ending in two 6-DoF feet, the shipped robots and the DRC Atlas
multi-contact set included, and every contact on any other set. `GaitSchedule` starts the schedule, separates gaits and
ends the schedule with that stance. `modeNumber2String()` prints a mode back in the first form that names it: `FLY`,
`STANCE`, `LF` or `RF`, the legs joined by `_` when the mode is a union of whole legs, the digits otherwise, so the
bipeds print FLY, RF, LF and STANCE as before.

### Gait files

The bipeds share `config/command/gait.textproto`, whose modes are FLY, RF, LF and STANCE. The quadrupeds share
`config/command/quadruped_gait.textproto` (filegroup `:quadruped_gait_config`): stance, trot, standing_trot,
flying_trot, pace, static_walk and dynamic_walk, named by their stance legs (`LF_RH`, `RF_LH`, ...) for contacts and
legs ordered LF, RF, LH, RH, with the timings of the OCS2 legged_robot ANYmal example. A robot that needs other gaits
ships its own gait file in its package and points its launch files' `gait_file` variable (`launch/mpc.textproto`,
`launch/dummy_sim.textproto`) and the validation's `RobotConfiguration::gaitFile` at it. `testModeSequenceTemplate`
checks every gait of the biped file against its literal modes and times, and `testQuadrupedGaitFile` reads the quadruped
file on a four-leg layout.

The procedural motion manager (`ProceduralMpcMotionManager`) switches gaits along the gait file's `gait_switching`
table, one row up or down at a time, from the first row. A row names a gait and the command and stride-mean motion
thresholds at which the manager switches to the row after or before it (`gait_switching_table_config.proto` says what each one means). A
file without the table gets `defaultGaitModeStates()` (`reference_manager/GaitSwitchingTable.h`): stance, slow_walk,
walk, slower_trot, slow_trot, trot and run, the biped table the manager always had. `Create()` refuses a malformed
table naming the row and the field, and a table naming a gait the file lacks (`checkEveryGaitIsLoaded()`). The
quadruped file has its own table: stance, static_walk, trot and flying_trot.

A new gait starts at the first event after 70 % of the horizon (`updateGaitSchedule()`, `gait/GaitScheduleUpdater.h`).
When the phase that ends at that event has the mode the new gait starts its cycle with (`cycleStartMode()`: its first
mode in which the first leg stands and another leg swings, LF for every biped gait and LF_RH for a quadruped trot), the
gait starts where that phase started instead. A gait without such a mode, or a set without legs, aligns to
`defaultCycleStartMode()`: LF on a biped, nothing on any other set. A biped gait therefore always aligns to LF, as
before: `very_slow_walk` (RF, STANCE, LF, STANCE), `right_leg` and `jump` included.

"STANCE" is every contact of a set that has legs and is not the biped, a quadruped on four 6-DoF soles included; there
the legs name LF and RF, and the biped's foot aliases do not apply. The biped and a mixed set without legs (the
multi-contact Atlas) keep STANCE as their last two contacts, the feet.

The biped-only components refuse a contact set they cannot run on when they are built, never per tick:
- the contact planner: `LipContactPlanner` and `HlipContactPlanner` in `Create()` and `setConfig()`, and the terms
  `foot_separation`, `zmp_support_region`, `yaw_torque_budget` and `alternating_feet` and the foothold integrator in
  `ContactPlanningTerm::validateContactSet()`, which `ContactPlanningProblem::finalize()` calls. Each returns the
  InvalidArgument of `checkTwoContactCount()` (`common/TwoContactCount.h`), which names the component and the
  contact count: "FootSeparationConstraint is written for two contacts; the contact set has 4 contacts". The check
  reads the count alone. Before them `makeContactPlanner()` and `contactPlannerSummary()` take the contact set
  (`ContactSetSummary`, which `ContactPlannerModule::Create()` fills from the reference manager's model) and refuse
  every set but the biped's two 6-DoF feet, two point contacts included, naming `planner.type` and the count. That
  covers the planners' step widths (`hlip.nom_pos_lfoot_rt_rfoot_ewrt_heading_y`,
  `step_width.nom_pos_lfoot_rt_rfoot_ewrt_heading_y`), which nothing else reads;
- the knee mimic blocks (`kneeMimicJointsFromConfig()`), with the same message;
- the procedural arm swing, which swings the arms against the biped's gait phase: `configureArmSwing()` at start-up
  and `setArmSwingSettings()` on a hot reload refuse `arm_swing.generator: "procedural"` while the swing is active
  (the robot names `model_settings.arm_joint_names` and the centroidal MPC does not track the ACoM), naming the
  generator and the contact count. A reload that is refused keeps the running settings;
- the reference manager's nominal footholds and swing references on the `step_width` anchor
  (`nominal_foothold.anchor`), which are `std::nullopt` on a set other than the biped's. The `nominal_stance` anchor
  places every contact of a legged set at its nominal stance point relative to the predicted base, and is refused on
  a set without legs ([`locomotion_heuristics`](../docs/locomotion_heuristics/README.md), section 4).

Per leg, on any legged set: `getSwingPhaseVariable(legIndex, time)` is the phase of the leg's swing, 0 in stance and 0
to 1 from lift-off to touch-down (`gait/LegSwingPhase.h`); the foothold heuristics read the leg's side, hip and
nominal stance point (`FootholdHeuristicContext`). `getPhaseVariable()` stays the biped's gait phase, which the arm
swing and `periodic_orientation` read.

The planner rules `no_flight` and `phase_durations` and the `no_flight` row are written over the number of feet, and
`minimum_double_support` is a minimum support count (every foot by default). The costs, constraints and heuristics
written for two feet are refused by the MPC that selects them: the centroidal MPC's `checkCentroidalContactSet()`
names each one ([`multi_contact_planning`](../docs/multi_contact_planning/README.md), section 7.2), and the whole-body
MPC runs on the biped's two 6-DoF feet alone and refuses every other set.

The swing trajectory planner plans every contact of the set, each from its own bit of the mode schedule, and the input
reference (`getDesiredInput()`) shares the weight among the active contacts of any mode. The normal-velocity servo
reads the planner's references (`HumanoidPreComputation`) on the biped and on a legged set.

The tests of the contact types are `testContactArray`, `testContactMask`, `testContactModeBits` and
`testContactSetAllocations` (zero allocations on the per-tick operations); `testContactSetReference` runs the reference
manager and the swing planner on four and six contacts, and `testSwingTrajectoryPlannerContacts` the swing planner on a
four-contact trot and walk.

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
