# legged_centroidal_mpc

The centroidal MPC: the centroidal-momentum model of the robot (`CentroidalMpcRobotModel`), its dynamics, costs and
constraints, the interface that assembles them from a typed task file (`CentroidalMpcInterface::Create()`), the
initializer, the MRT joint controller and the multi-contact plan tracking (`multi_contact/`). The processes that run it
are in [`legged_centroidal_mpc_app`](../legged_centroidal_mpc_app/README.md); the shared parts are in
[`legged_common_mpc`](../legged_common_mpc/README.md), whose "Contacts and modes" section defines the contact set,
`ContactArray`, `ContactMask` and the mode numbers.

## The contact set

The centroidal MPC runs on any contact set of `model_settings`: 3-DoF point contacts, 6-DoF surface contacts or a mix,
up to `kMaxNumContacts` (32). The shipped bipeds (DRC Atlas, EngineAI SA01, Unitree G1 and R1) plan on two 6-DoF
feet, and their assembled problems are recorded bit for bit in `testCentroidalContactSetParity`. ANYbotics ANYmal B
plans on four 3-DoF feet, `LF_FOOT`, `RF_FOOT`, `LH_FOOT` and `RH_FOOT`, one per leg, on the shared quadruped gait file
([`robot_models/anybotics_anymal_b`](../../robot_models/anybotics_anymal_b/README.md)); it is not in the parity record,
and `testAnymalBLockstepLoop` runs its shipped configuration instead. LimX Tron1 plans on two rolling contacts, one per
wheel (below); it is not in the parity record either, and `testTron1LockstepLoop` runs its shipped configuration in the
MuJoCo lockstep loop. The wheeled ANYmal C plans on four rolling contacts, one per leg, and is not in the parity
record; `testAnymalCWheelsContactSet` builds its shipped configuration and `testAnymalCWheelsLockstepLoop` runs it in
the MuJoCo lockstep loop, on the wheeled quadrupeds' gait file, which turns it in a trot. The tests over every shipped configuration
(`ShippedCentroidalRobot`: `testSrbdQuaternion`, `testCentroidalCommandYawEquivariance`, and `testNominalPendulum` over
its own list) include all three.

- The input is the force of each 3-DoF contact, the wrench of each 6-DoF contact, then the joint velocities:
  `3 n3 + 6 n6 + nj` entries, with the offsets of OCS2's `CentroidalModelInfo`.
- The dynamics are the centroidal momentum rate of every contact's force and moment. They take every contact's input
  in every mode; the mode reaches the problem through the per-contact constraints: `zero_wrench` holds an inactive
  contact's input at zero and `zero_velocity` holds an active contact in place (three translation rows for a point).
- `contact_wrench_cone` builds a wrench cone for a 6-DoF contact and a point friction cone
  (`PointFrictionConeConstraint`) for a 3-DoF one; `friction_force_cone` runs on both.
- The weight-compensating initializer (`CentroidalWeightCompInitializer`) and the input reference share the robot's
  weight equally among the active contacts.
- `contact_input_parameterization: "basis_vectors"` takes any number of 6-DoF contacts, one wrench cone basis each,
  and no 3-DoF contact: its generator sets span a surface contact's wrench cone, and a point contact has none
  ([`contact_basis_vectors`](../docs/contact_basis_vectors/README.md)).
- The terms written for the biped's two feet, the online contact planner and the locomotion heuristics are refused by
  name on any other set, before a model is built, by `checkCentroidalContactSet()` (`CentroidalContactSet.h`; the list
  is in [`multi_contact_planning`](../docs/multi_contact_planning/README.md), section 7.2). The swing terms
  `task_space_foot_cost` and the soft and hard `normal_velocity`, which track the swing trajectory planner, also run
  on a set that `model_settings.legs` describes. A 3-DoF contact's foot cost has no orientation to track: its rotvec
  and angular-velocity weights are zero, at start-up and after every hot reload of `task_space_foot_cost`
  (`CentroidalMpcEndEffectorFootCost::setWeights()`). `DcmTerminalCost::Create()` refuses a non-biped set itself, for a
  cost built by hand.

A wheel is a rolling contact (`model_settings.contacts_rolling`; the LimX Tron1 plans on two,
[`robot_models/limx_tron1`](../../robot_models/limx_tron1/README.md), and the wheeled ANYmal C on four,
[`robot_models/anybotics_anymal_c_wheels`](../../robot_models/anybotics_anymal_c_wheels/README.md)): a 3-DoF contact whose frame is the wheel center
and whose force acts a wheel radius below it, at the wheel's lowest point. The `zero_velocity` entry builds
`RollingContactConstraintCppAd` on it, the no-slip constraint of the wheel material at that point, registered as
`<contact>_rollingNoSlip`; `normal_velocity` and `task_space_foot_cost` read the contact point, and the dynamics take
the moment of its force there. In stance the reference manager moves its reference with the predicted base, and the
input cost's reference of its wheel joint velocity is the rolling speed of the commanded twist. The derivation, the
references, the costs of the wheel joints and the terms a set with a wheel refuses are in
[`rolling_contact`](../docs/rolling_contact/README.md).

The MRT joint controller's realtime inverse dynamics apply each contact's planned wrench at its own frame:
`getContactFrameIndexArray()` (`legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h`) returns every contact's
frame in contact order. Without a policy, the controller shares the weight among the measured contacts at their
frames. A rolling contact's frame is its wheel center, so the controller gives its force the moment (-r n) x f about
that frame, and the wheel joint's feedforward torque includes r times the traction force
(`legged_centroidal_mpc_test:test_rolling_contact_mrt_feedforward`).

| Test | What |
|---|---|
| `testCentroidalContactSetParity` | the shipped robots' assembled problems, bit for bit against a record |
| `testMixedContactSet` | Atlas's feet with a 3-DoF contact on each hand: the input layout, the hand-derived momentum rate, the cones in the patch frame, the weight sharing, the contact frames of the inverse dynamics, the refusals |
| `testPointContactQuadrupedLikeSet` | four point contacts and no 6-DoF contact (the corners of Atlas's soles, in a quadruped's leg order): the interface builds, its LQ approximation is finite and gates each contact by its mode on a trot, the weight is shared among the active contacts, the biped's `STANCE` is the feet of a six-contact set, every biped-only term, the procedural arm swing and a hand-built `DcmTerminalCost` are refused by name, and the swing terms run once the set's legs are described |
| `testPointContactSwingTerms` | the same four contacts as legs LF, RF, LH and RH with `task_space_foot_cost` and the hard `normal_velocity`: no orientation weight on a point contact's foot cost, also after a hot reload, and each contact's swing height and servo following its own swing on a trot |
| `testPointContactQuadrupedTrot` | the same legged four contacts trotting on the `trot` of `quadruped_gait.textproto`, read by leg name, with the `nominal_stance` foothold anchor: several SQP iterations from the standing state converge to a finite plan whose stance contacts hold still and whose swing contacts rise to the apex and land within 1 cm of their footholds. Each foot pivots about its stance corner, since its toe and heel share the ankle link |
| `testRollingContactConstraint` | the rolling no-slip constraint on Tron1: its linear approximation against central finite differences at random states and inputs, zero for pure rolling, non-zero for longitudinal and lateral slip, the z row against the gain times the height error, active in stance only, and its value against Pinocchio's own velocity of the wheel material at the contact point |
| `testRollingContactSet` | Tron1's configuration: the input layout (two forces and eight joint velocities), the rolling no-slip, normal-velocity and friction terms, no joint-limit row for a wheel, every term refused on a rolling set, and the moment of the wheel forces about the center of mass at the contact points in the flow map |
| `testRollingContactSwing` | a Tron1 wheel in swing: its normal-velocity servo is active, references the wheel center a radius above the swing reference of the contact point, and its foot cost tracks the contact point |
| `testRollingContactReferences` | the references of Tron1's wheels: the wheel joint velocity reference of a forward command (v / r with the sign of the axle), of a turn in place (opposite signs, in the ratio of the half tracks), of no command and of a wheel in swing (0); the stance reference that advances with the predicted base, the lift-off from it and the foot cost that tracks it in stance; and the fixed point contacts of the legged four-point set, which keep no stance reference, lift off where they were measured and get no joint velocity reference |
| `testAnymalCWheelsContactSet` | the wheeled ANYmal C's shipped configuration: the input layout (four forces and sixteen joint velocities), a rolling no-slip term and a normal-velocity term per wheel, no joint-limit row for a wheel and twelve for the legs, and the wheel joint velocity reference of a forward command on every wheel |
| `testTron1LockstepLoop` | Tron1's shipped task, reference and gait files in the headless MuJoCo lockstep loop of `legged_mpc_validation` (`LockstepClosedLoop`, ground-truth contacts): 2 s of standing, a 0.5 m/s drive for 4 s, a 0.5 rad/s turn in place for 4 s and 1 s at rest. It survives, every solve succeeds, the quaternion stays unit to 1e-12, both wheels stay in measured contact at every control cycle (one stance phase each), and the robot covers at least half the commanded distance (2 m) and half the commanded angle (2 rad) (large, exclusive: it compiles the robot's CppAD libraries) |
| `testAnymalCWheelsLockstepLoop` | the wheeled ANYmal C's shipped task, reference and gait files in the headless MuJoCo lockstep loop of `legged_mpc_validation` (`LockstepClosedLoop`, ground-truth contacts), in two runs after 2 s of standing: a 0.5 m/s drive for 4 s, in which every wheel stays in measured contact at every control cycle (one stance phase each) and the robot covers at least half the commanded 2 m, and a 0.5 rad/s turn in place for 4 s, in which the wheels step (`turning_trot`), the tilt stays below 0.1 rad and the robot turns through at least half the commanded 2 rad. Both survive, every solve succeeds and the quaternion stays unit to 1e-12 (large, exclusive: it compiles the robot's CppAD libraries) |
| `testAnymalBLockstepLoop` | ANYmal B's shipped task, reference and gait files in the lockstep loop of the MPC's own rollout (`CentroidalMpcLoop`, the full interface and SQP): standing for 2 s, a 0.3 m/s walk for 5 s and a 0.4 rad/s turn for 4 s. Every solve succeeds, the quaternion stays unit to 1e-12, the walk reaches the `trot` and the stand does not, and the robot covers at least half the commanded distance (1.5 m) and half the commanded angle (1.6 rad) (large, exclusive: it compiles the robot's CppAD libraries) |
| `testBasisInputsModelDecorator` | the basis-vector inputs on three and four 6-DoF contacts |
