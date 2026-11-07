# humanoid_centroidal_mpc

The centroidal MPC: the centroidal-momentum model of the robot (`CentroidalMpcRobotModel`), its dynamics, costs and
constraints, the interface that assembles them from a typed task file (`CentroidalMpcInterface::Create()`), the
initializer, the MRT joint controller and the multi-contact plan tracking (`multi_contact/`). The processes that run it
are in [`humanoid_centroidal_mpc_app`](../humanoid_centroidal_mpc_app/README.md); the shared parts are in
[`humanoid_common_mpc`](../humanoid_common_mpc/README.md), whose "Contacts and modes" section defines the contact set,
`ContactArray`, `ContactMask` and the mode numbers.

## The contact set

The centroidal MPC runs on any contact set of `model_settings`: 3-DoF point contacts, 6-DoF surface contacts or a mix,
up to `kMaxNumContacts` (32). The shipped robots are bipeds, two 6-DoF feet, and their assembled problems are recorded
bit for bit in `testCentroidalContactSetParity`.

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
  is in [`multi_contact_planning`](../docs/multi_contact_planning/README.md), section 7.2).

The MRT joint controller's realtime inverse dynamics apply each contact's planned wrench at its own frame:
`getContactFrameIndexArray()` (`humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h`) returns every contact's
frame in contact order. Without a policy, the controller shares the weight among the measured contacts at their
frames.

| Test | What |
|---|---|
| `testCentroidalContactSetParity` | the shipped robots' assembled problems, bit for bit against a record |
| `testMixedContactSet` | Atlas's feet with a 3-DoF contact on each hand: the input layout, the hand-derived momentum rate, the cones in the patch frame, the weight sharing, the contact frames of the inverse dynamics, the refusals |
| `testPointContactQuadrupedLikeSet` | four point contacts and no 6-DoF contact (the corners of Atlas's soles, in a quadruped's leg order): the interface builds, its LQ approximation is finite and gates each contact by its mode on a trot, the weight is shared among the active contacts, the biped's `STANCE` is the feet of a six-contact set, and every biped-only term is refused by name |
| `testBasisInputsModelDecorator` | the basis-vector inputs on three and four 6-DoF contacts |
