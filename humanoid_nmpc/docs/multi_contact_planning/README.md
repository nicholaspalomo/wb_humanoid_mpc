# Multi-Contact Planning: Stairs, Climbing onto a Box, Car Egress and Getting Up

> I. Mordatch, E. Todorov, Z. Popović,
> *Discovery of Complex Behaviors through Contact-Invariant Optimization*,
> ACM Transactions on Graphics (SIGGRAPH) 31(4), 2012,

adapted to this repository's centroidal NMPC, with a sampling-based stance planner beside it and one stability
certifier that both planners and the tests share. The scenarios are four dummy sims for the DRC Atlas:
- climbing stairs;
- climbing onto a box;
- getting out of a utility vehicle (the egress task of the DARPA Robotics Challenge Finals);
- getting up from lying on the ground.

Feet, knees, hands and forearms are all valid contacts. For egress the pelvis and the backs of the thighs are too, and
for getting up the back and the chest.

**In progress; see section 10 for the milestones.** Every piece is opt-in: the scenarios are new launch targets, the
multi-contact formulation is a new task file, and no shipped robot configuration changes its behavior. Section 11
lists the decisions.

---

## 1. Why the current stack cannot do this

Three assumptions run through the code, each of them correct for walking on a floor and each of them wrong here.

**Two contacts, both feet, fixed at compile time.**
- `kNumContacts = 2` (`humanoid_common_mpc/common/BipedFeet.h`).
- `feet_array_t<T> = std::array<T, kNumContacts>` and `contact_flag_t = feet_array_t<bool>` (`common/Types.h`).
- About 130 first-party files use them.
- The input layout of both models is `6 * kNumContacts` wrench entries followed by the joint velocities
  (`CentroidalMpcRobotModel.h`, `WBAccelMpcRobotModel.h`).
- The mode numbers are a 2-bit mask: `FLY`, `RF`, `LF` and `STANCE` (`gait/MotionPhaseDefinition.h`).
- Eight contact-planning rules and constraints `static_assert(kNumContacts == kBipedFootCount)`.
- The basis-vector contact parameterization, which Atlas and SA01 ship, requires exactly two 6-DoF contacts
  (`CentroidalDynamicsBasisInputsAD.cpp`).

The good news is underneath. OCS2's `CentroidalModelInfo` and its centroidal dynamics are generic over any number of
mixed 3-DoF (point) and 6-DoF (surface) contacts (`numThreeDofContacts`, `numSixDofContacts`,
`AccessHelperFunctionsImpl.h`), and OCS2's `ModeSchedule` takes any `size_t` mode.

**Flat ground everywhere.**
- The terrain is one scalar, `terrain_height`, owned by `SwitchedModelReferenceManager`.
- The swing planner broadcasts it to every phase.
- The friction cones and the slip terms are written about the world `z` axis (`ForceWeightedSlipConstraint.h` says so
  and lists what a slope would need).
- `GroundPenetrationConstraint` and `ContactComplementarityConstraint` subtract that one height.
- The MuJoCo scenes are one MJCF per robot, with a floor plane and nothing else.

**Contacts are planned online, in 2-D, a step at a time.**
- `hlip` and `lip_miqp` plan footholds in the plane under a linear inverted pendulum, over a horizon of about one
  second.
- There is no offline planner.
- There is no way to give the MPC a trajectory or a mode schedule from outside: the MPC node subscribes to the
  velocity command and the parameter updates only (`MpcNodeRuntime.cpp`).

Climbing onto a box takes five to fifteen seconds and a sequence of contact changes the robot cannot find by looking
one second ahead. A knee goes onto the box before the foot can, and a hand pushes so that the other knee can follow.
So the design has two layers:
- an **offline planner** that finds the whole contact sequence and a dynamically consistent motion through it;
- the **online centroidal MPC**, which tracks that plan and rejects disturbances.

## 2. Stability: what "statically and dynamically stable" means here

Both planners and every test use the same two conditions, implemented once (section 6).

**Contacts.**
- A contact `i` touches a surface patch with unit normal `n_i` and friction coefficient `mu_i`.
- A **point contact** (knee, hand, forearm, thigh) carries a force `f_i`.
- A **surface contact** (foot, pelvis on the seat) carries a wrench `w_i = (f_i, tau_i)` about its frame, bounded by the
  contact wrench cone of its rectangle: friction, center of pressure inside the rectangle and torsional friction, as
  `ContactWrenchConeConstraint` already writes it, but in the patch frame instead of the world frame.

All cones are linearized into polyhedral cones, with four or eight edges per friction cone, so the conditions below are
linear in the forces.

**Static equilibrium of a stance.** A stance `S` is a set of active contacts with their positions `p_i`. The robot,
standing still with its center of mass `c`, is in static equilibrium on `S` when there are contact forces in their
cones that balance gravity:

```
sum_i f_i + m g = 0
sum_i (p_i x f_i + tau_i) + c x m g = 0
f_i in K_i (friction cone, or wrench cone for a surface contact)
```

Because gravity is the only external wrench, the set of `c` that satisfies this is a vertical prism over a convex
polygon in the horizontal plane: the CoM height does not matter (Bretl and Lall). On flat ground with feet only, that
polygon is the familiar support polygon; with a hand on a box it is wider, and it can extend past the feet.

**The static stability margin** `rho_s(S, c)` is the distance from `c`'s horizontal projection to the boundary of that
polygon.
- It is computed from 16 LPs that do not depend on `c`: for each horizontal direction `d_k`, one LP finds the CoM
  position farthest along `d_k` that is still in equilibrium (the support point of the polygon in that direction).
- The convex hull of the 16 support points lies inside the polygon and touches its boundary at each of them, so the
  signed distance from `c` to the hull's boundary is a lower bound on the true margin that tightens with more
  directions. (The distance along a ray `c + t d_k` would not be: it is at least the true distance in every
  direction.)
- It is positive inside, negative outside, and its sign flips where `c` crosses the hull's boundary. One more LP first
  tests whether any CoM position is in equilibrium; when none is (a hand on a wall alone), the margin is absent.
- The hull is also the polygon the playback draws, and planner A's solve 3 constrains the CoM to it (section 4.3).
- A scenario's `support_polygon: "refined"` (`"sampled"` by default) has the certifier and planner B grow that hull: one
  more LP along the outward normal of each edge, until no support point lies more than 0.1 mm beyond its edge or 32 more
  points were added (`SupportPolygon::kRefined`, Bretl and Lall's projection). The refined hull contains the sampled one
  and is still inside the polygon. It matters for a thin polygon whose corner lies between two of the 16 directions: on
  the get-up from the back's roll (two pads along the torso box's edge, parallel to x, and a sole far ahead), the corner
  at the inner pad is a support point only along -y, where the LP's tie between the two pads decides whether the hull
  keeps it. The plan's knots kept it; the tracked trajectory, its edge a few milliradians off x, lost it, and the margin
  of a CoM the MPC held within 1 cm of the plan's fell from 0.05 m to -0.045 m. Planner A keeps the sampled hull, which
  lies inside the refined one.

**Torque feasibility.** For a configuration `q`, the joint torques that hold the stance are
`tau_joints = g(q) - sum_i J_i(q)^T f_i`. Under a scenario's `joint_limits: "position_and_effort"` (the default) they
must lie within the effort limits of the URDF, scaled by a margin. Without this check, a statically "stable" stance on a
box can need more shoulder torque than Atlas has. The constraint is linear in the forces for a given `q`, so it is added
to the same LP. Under `joint_limits: "position_only"`, which every shipped scenario sets (section 11, decision 6), the
torques are reported and never judged.

**Dynamic stability of a motion.** While the CoM accelerates, the gravito-inertial wrench must lie in the contact
wrench cone of the stance, CWC(S) (Caron, Pham and Nakamura). Here `c_ddot` is the CoM acceleration and `L_dot` the
rate of change of the angular momentum about the CoM:

```
( m (c_ddot - g),  m c x (c_ddot - g) + L_dot )  in  CWC(S)
```

This is exactly what the centroidal dynamics plus per-contact cones already enforce inside the MPC. The certifier
checks it on a planned trajectory by solving, at every knot, the feasibility LP for contact forces that produce the
planned centroidal momentum rate.

**The policy for these behaviors**, as the request asks:
1. **Every instant is statically stable.** At every knot of the plan, the robot could stop where it is, with zero
   velocity, on its active contacts: `rho_s(S(t), c(t)) >= delta_s` (default `delta_s = 0.03` m) and, under
   `"position_and_effort"`, torque-feasible.
   The motion is quasi-static, and a stalled solve or a dropped message never leaves the robot mid-fall.
2. **The motion is dynamically feasible.** The planned momentum rates lie in CWC(S(t)) with the same cones the MPC
   uses, so the MPC is asked for nothing it cannot do.
3. **Contact changes keep the rest of the stance stable.** A contact is broken only when the stance without it already
   satisfies condition 1. A contact is made only after it has been reached in a stance that satisfies condition 1
   without it. This is the stance-sequence rule of Hauser, Bretl and Latombe: consecutive stances differ by one
   contact, and the transition happens in the smaller of the two.

Condition 1 is stricter than climbing needs: a running jump onto a box fails it. It is the right first target because
it makes every plan robust to timing errors. The margin `delta_s` is a scenario parameter, and setting it to `-inf`
leaves only condition 2.

## 3. Contacts and terrain

### 3.1 The contact set

The contact set is runtime data, not `kNumContacts`. Each candidate has:
- a name;
- a kind: `point` (3-DoF) or `surface` (6-DoF, with a rectangle);
- a parent joint;
- a local offset, plus the local normal of its contact pad;
- the radius of the collision geometry it sits on.

For the DRC Atlas, from the link names in `atlas.urdf` and the collision geoms of `atlas.xml`:

| Contact | Kind | Parent joint | Pad | Used in |
| --- | --- | --- | --- | --- |
| `foot_l_contact`, `foot_r_contact` | surface, 0.24 x 0.11 m | `l_leg_akx`, `r_leg_akx` | the existing foot frames | all |
| `knee_l_contact`, `knee_r_contact` | point | `l_leg_kny`, `r_leg_kny` | front of the shin capsule (r 0.055 m), just below the knee | box, egress |
| `hand_l_contact`, `hand_r_contact` | point | `l_arm_wrx`, `r_arm_wrx` | the hand sphere (r 0.04 m) | stairs (optional), box, egress |
| `forearm_l_contact`, `forearm_r_contact` | point | `l_arm_elx`, `r_arm_elx` | middle of the `l_larm` / `r_larm` capsule (r 0.04 m) | box, egress |
| `seat_contact` | surface, 0.24 x 0.20 m | the pelvis (base, `root_joint`) | under the pelvis, at the underside of the thighs when seated | egress |
| `thigh_l_contact`, `thigh_r_contact` | point | `l_leg_hpy`, `r_leg_hpy` (on `l_uleg`) | back of the thigh capsule (r 0.06 m) | egress |
| `torso_back_contact` | surface, about 0.36 x 0.30 m | `back_bkx` (on `utorso`) | the back face of the `utorso` collision box (x = -0.22 m in its frame) | getting up |
| `torso_front_contact` | surface, about 0.36 x 0.30 m | `back_bkx` (on `utorso`) | the chest face of the `utorso` collision box (x = +0.14 m) | getting up |
| `torso_edge_r_low_25_contact`, `torso_edge_r_high_25_contact`, `torso_edge_r_low_65_contact`, `torso_edge_r_high_65_contact` | point (geom r 0.001 m) | `back_bkx` (on `utorso`) | the back-right edge of the `utorso` collision box (x = y = -0.22 m), near its two ends, with normals 25 and 65 degrees from the back face toward the right side | getting up from the back |

Notes:
- Atlas's old DRC naming puts `l_larm` after the elbow and `l_farm` after the wrist's y joint. The forearm contact is
  on `l_larm`, the long segment, on its front (+x) side: the side that faces the floor with the arms at the sides of a
  robot lying face down.
- The thigh capsule hangs from the hip pitch joint, so the thigh contact's parent joint is `l_leg_hpy` (the joint whose
  frame is `l_uleg`'s), not the knee.
- The thigh capsules (radius 0.06 m from z = -0.066 m) reach below the pelvis box (bottom at z = -0.07 m), so a seated
  Atlas rests on them, not on the box: the seat pad sits at z = -0.143 m, the height of the underside of the thighs
  with the hips flexed, rather than on the box's bottom face.
- The set is `robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/contact_set.textproto`
  (`contact_set.proto`); a test checks every parent joint against `atlas.urdf`.
- The MPC model fixes the wrist joints (`fixed_joint_names` lists `l/r_arm_wry` and `l/r_arm_wrx`). Hand contacts work
  with locked wrists, but the hand cannot turn to face a surface. Unlocking the wrists is a separate decision
  (section 11).
- A point on a round geom is an approximation: the true contact point rolls around the capsule as the limb rotates. The
  pad's offset is placed at the radius along the pad normal, and the slip and orientation terms allow a cone of
  approach angles (`max_pad_tilt`, default 30 degrees) instead of requiring the pad normal to equal the surface
  normal.
- A forearm resting along a box edge is closer to a line contact. Version 1 uses one point. Two points per forearm,
  which give it a moment, are a later option that needs no new kind.
- The torso edge pads let the robot roll from its back onto its right side: a box edge is a line contact, so each
  roll angle has two points, one near each end of the edge. The two angles' points sit at different heights along the
  edge, because the planner drops a pad that coincides with another active pad as overlapping. The pads lie on the
  edge itself, so their geom radius is a token 1 mm.
- A scenario lists which candidates it allows. Stairs allow the feet, optionally the hands. The box allows feet, knees,
  hands and forearms. Egress allows all but the torso contacts. Getting up from the front allows all but the four torso
  edge pads, and getting up from the back allows all seventeen.

With seventeen candidates there are 2^17 = 131072 contact modes. A mode number is the bitmask of the active contacts in
contact-set order, which OCS2's `ModeSchedule` already carries. The biped's `FLY` / `RF` / `LF` / `STANCE` names remain
the biped's.

### 3.2 The terrain model

`TerrainScene` (new proto, `terrain_scene.proto`) lists convex primitives in the world frame. Each primitive is a box
(center, orientation, half extents) or a capsule (two end points, radius), plus a friction coefficient and a name. The
floor is a large box whose top face is at `z = 0`. Stairs are stacked boxes; the vehicle is boxes for the floor pan,
seat cushion, seat back and dashboard, and capsules for the roll-cage tubes.

`TerrainModel` answers three queries, templated on the scalar so that CppAD can record them:
- `signedDistance(p)`, and the nearest surface point and outward normal: what the contact-invariant cost, the
  non-penetration constraints and the swing planner read. A box is exact: the distance to the nearest face, edge or
  corner, negative inside; a capsule is exact too. Across primitives it is the smooth minimum (log-sum-exp with a
  scale of 5 mm), so the gradient is defined where two primitives meet, such as the inside corner of a stair. It lies
  below the true minimum by at most 5 mm × log(n): 3.5 mm where two primitives are equally near, nothing where one is
  nearer by a few centimeters. The normal is the gradient's direction (the softmin-weighted normals, normalized), and
  the nearest point `p - d(p) n(p)`, exact where one primitive dominates.
- `patchFrame(contact)`: for a contact assigned to a face, the face's frame, with its normal and its polygon. That is
  what the friction cones and slip constraints are written in. `patchFrameAt(p)` assigns a point to the face of the
  nearest primitive whose plane is nearest, or to a capsule's tangent plane.
- `sphereDistance(center, radius)`: the robot's collision spheres (section 5.4) against the scene.

Every branch of the templated queries is a CppAD conditional expression, and a norm is guarded before its square
root, so a tape recorded at one point is valid in every region (face, edge, corner, inside) and runs on CppADCodeGen
variables. The scene's numbers enter a tape as constants: a tape belongs to one scene.

The same `TerrainScene` textproto generates the MuJoCo scene (section 8) and the Rerun meshes, so the planner, the
MPC, the dummy sim and MuJoCo can never disagree about where the box is.

**Flat ground is the special case.** A scene with only the floor box at height `h` reproduces today's
`terrain_height = h`. That is the parity test of section 9.

## 4. Planner A: contact-invariant optimization

### 4.1 The paper's formulation

Mordatch, Todorov and Popović optimize one vector `s` that holds the motion: the body's trajectory as splines over a
sequence of `K` phases of fixed duration, plus, for each candidate contact `i` and phase `phi`, a scalar
`c_{i,phi} >= 0` that says whether `i` should be in contact during `phi`. The objective is a sum of four terms:

```
L(s) = L_CI(s) + L_physics(s) + L_task(s) + L_hint(s)

L_CI(s) = sum_t sum_i c_{i,phi(t)} ( ||e_i(s, t)||^2 + ||e_dot_i(s, t)||^2 )
```

- `e_i` is the contact's violation: its distance to the nearest surface and its misalignment with it.
- `e_dot_i` is the contact's velocity, which is slip once it touches.
- A large `c` therefore costs the contact for being away from a surface or sliding on it.
- In `L_physics`, the contact forces solve a regularized least-squares problem that best explains the motion. The
  force of contact `i` is penalized by `k0 / (c_i^2 + k1)`, so a contact with a small `c` cannot push.
- The optimization is not told when contacts happen. It discovers the sequence by trading `L_CI` against
  `L_physics`.
- It runs in stages of continuation on the weights.

### 4.2 Adaptation to this repository

The paper's soft physics is replaced by the centroidal dynamics with full kinematics that the centroidal MPC already
models (Dai, Valenzuela and Tedrake). The planner is an OCS2 SQP problem, the same solver and the same CppAD
machinery as the MPC, solved offline over the whole motion.

**State.** The centroidal MPC's state (centroidal momentum, base pose as a quaternion, joint positions), plus one
contact variable `c_i` per candidate.

**Input.** The centroidal MPC's input (one force or wrench per candidate, then the joint velocities), plus one `u_c_i`
per candidate, with `c_dot_i = u_c_i`.

**Dynamics.** The centroidal momentum dynamics over all candidates (exact, as in the MPC), the joint kinematics, and
`c_dot = u_c`.

**Hard constraints.**
- Every candidate's force lies in its cone, in the frame of the patch nearest to it. Here `nearest` is a smooth
  blend weighted by `exp(-d / sigma)` for the planner, frozen to the assigned patch for the MPC.
- No candidate and no collision sphere penetrates the scene: `signedDistance >= 0`, the multi-primitive generalization
  of `GroundPenetrationConstraint`.
- `0 <= c_i <= 1`.
- Joint position and velocity limits.

**Costs.**
- `c_i (d_i(q)^2 + w_slip ||v_i^t||^2 + w_tilt (1 - n_pad . n_patch)^2)`: the contact-invariant term. `d_i` is the
  distance to the scene, `v_i^t` the tangential velocity in the patch frame, `n_pad` the pad normal and `n_patch` the
  patch normal.
- `k0 ||f_i||^2 / (c_i^2 + k1)`: the paper's force gate. It is smooth and bounded, and stiff when `k1` is small.
- `||u_c||^2`: `c` changes slowly, which takes the place of the paper's phase-constant `c` (see below).
- The task cost: the base pose and CoM reaching the goal region by the final time, from a goal written as a target
  stance or a base pose.
- The quasi-static terms of section 2: `||c_ddot||^2` and `||L_dot||^2`, and a hinge on `rho_s` (section 4.3).
- Regularization: posture (the scenario's nominal pose), joint velocities and force smoothness.

**Why `c` is a state and not a per-phase parameter.** An OCS2 problem has no decision variables that are constant over
a phase, and fixed phase boundaries are what the paper's results depend on least. A slow state with a velocity cost
gives the same behavior: `c` is free to switch, but switching costs something, so it switches rarely. Afterwards `c` is
thresholded at 0.5 with a minimum dwell of 0.3 s into a stance sequence.

**Continuation, as in the paper.** Three solves, each warm-started from the last:
1. `k0` small and the contact-invariant weight small. Forces are almost free everywhere, and the task cost moves the
   body toward the goal through something like a ghost motion.
2. Raise `k0` and the contact-invariant weight, so that the forces concentrate on candidates that are touching
   something.
3. Raise them again, add the static-margin hinge, threshold `c`, and re-solve with the force of every non-contact
   candidate fixed to zero and its `c` term removed: the hard-contact problem the MPC will track.

The third solve is an ordinary multi-contact centroidal problem with a known mode schedule. Its result passes the
certifier of section 6, or the planner reports which knot fails, which condition, and by how much.

**Horizon.** Ten to twenty seconds at 0.05 s per node is 200 to 400 nodes, unremarkable for an offline multiple-shooting
SQP. Egress has three surface contacts (the feet and the seat) and eight point contacts, so with `n_j` MPC joints it
has 6 + 7 + n_j + 11 states and 3·6 + 8·3 + n_j + 11 = 53 + n_j inputs; the box is smaller. The solve is offline,
so its time is reported, not bounded; it runs on four threads (section 4.4).

### 4.3 Making static stability differentiable

`rho_s` is an LP's optimal value, which is not smooth in `c(t)` and the contact positions. The planner instead uses a
smooth surrogate, and the certifier checks the exact quantity:
- **During solves 1 and 2:** no static term. The quasi-static costs keep accelerations small.
- **In solve 3:** the stance is known, so the static equilibrium polygon of each phase is fixed up to the small motion
  of the contacts within it. The planner computes the polygon at the start of solve 3 (its H-representation, by
  projecting with the 16 directions of section 2). It adds `A c_xy <= b - delta_s` as linear inequality constraints per
  knot, which are exact for that polygon. If the certifier then fails, the polygon is recomputed at the solution and
  solve 3 is repeated, at most three times.

### 4.4 Planner A as built (milestone 5)

`ContactInvariantPlanner` (`humanoid_multi_contact/planner/`, registered as `contact_invariant`) implements sections
4.2 and 4.3 on OCS2's SQP and the centroidal MPC's own model, with these choices:

- **The problem is the MPC's.** The physical problem of solves 1 and 2 is the scenario's multi-contact centroidal MPC
  (`mpc_task_file` and `mpc_reference_file` of the scenario, `CentroidalMpcInterface` built on an in-memory plan of the
  guess with every allowed candidate in contact): its dynamics (`CentroidalDynamicsAD`), posture, input and terminal
  costs, joint limits, the surface candidates' contact wrench cones and the zero wrench of the candidates the scenario
  does not allow. `augmentProblem()` (`contact_invariant/ContactInvariantProblem.h`) wraps every one of those terms on
  the augmented state and input (`AugmentedStateInputCost`, `AugmentedStateCost`, `AugmentedStateInputConstraint`,
  `AugmentedDynamics`, `AugmentedPreComputation`, `AugmentedReferenceManager`), with the state manifold of the
  quaternion base and a Euclidean block for `c`, and adds the contact-invariant terms. The task cost is the MPC's
  terminal cost toward the goal pose; its running state cost, at 1 % of the task file's weights, is the posture
  regularization toward the guess. Where the task file is tuned for tracking a plan, planner A keeps its own values,
  so that the plans do not move with that tuning: the base position weights (25, 25, 15; `plannerMpcConfigOf()`), the
  wrench cones' barrier of solves 1 and 2 `(0.2, 5 N)`, and solve 3's settings below.
- **The contact features** of every allowed candidate are one CppAD tape (`ContactFeatureFunction`,
  `RobotContactFeatures.h`, named by the scene's and the contact set's hashes): the pad's signed distance, its velocity
  tangent to the terrain, the tilt of its normal (`1 - n_pad . n` for a surface pad, the excess over the 30 degree
  cone for a point pad), the friction cone of a point pad's force and the clearance of its geometry, all about the
  blended normal `n(p)`, which is the terrain model's softmin-weighted normal (section 3.2; its scale, 5 mm, is the
  `sigma` of section 4.2), and the exact normalized momentum rate of the centroidal dynamics. The contact-invariant
  cost (`ContactInvariantCost`), the force gate (`ContactForceGateCost`), `|u_c|^2` (`ActivationRateCost`), the
  quasi-static terms (`MomentumRateCost`) and the cone and clearance constraints (`ContactFeatureConstraint`) read it
  from the pre-computation. The non-convex Hessians of the first two in `(c, e)` and `(c, f)` are projected onto the
  positive semidefinite cone before the chain rule (`projectToPositiveSemidefinite()`); the gate clamps `c` into
  `[0, 1]`, the contact-invariant cost only from below, so that a `c` above 1 buys no force and still pays.
- **Inequalities are penalties.** OCS2's SQP hands HPIPM its equality constraints only (it linearizes the inequality
  collections for its metrics, not for the QP), so the "hard" constraints of section 4.2 are relaxed log barriers, as
  the MPC's cones are: the cones and the clearances of the pads and of the collision spheres (`TerrainClearanceConstraint`,
  every sphere of section 5.4 at margin 0, none exempt). `0 <= c <= 1` is a squared hinge (no push inside the box,
  where `c` must be free to fall to 0), with one more row, `sum c >= 0.9`: the robot always claims nearly one full
  contact, so no solve buys its motion with a flight phase. The certifier judges the exact conditions.
- **Continuation.** Solve 1 with `k0 = 1e-3` and the contact-invariant weight 1; solve 2 in three steps, each raising
  both by 10 (to `k0 = 1` and 1000), warm-started from the last (`solveContinuation()`); `k1 = 0.01`, the slip weight
  1 (s^2/m^2 against m^2), forces in units of the weight. The first node's `c` is the initial stance, the others start
  at 0.5. Nodes every 0.05 s.
- **Thresholding** (`thresholdActivations()`): `c >= 0.5` where the pad is within 2 cm of the terrain (the contact-
  invariant cost has brought it there; a pad in the air is not a contact whatever its `c`), the initial and the goal
  stance at the ends, phases and gaps shorter than 0.3 s merged, no node without a contact (the contact that lifted
  off last is kept until another has touched down), changes 0.25 s apart, one contact at a time, and every change
  delayed to the next multiple of 0.1 s, so that the tracking MPC's period (0.02 s on the Atlas), the plan's knots and
  the nodes of solves 1 and 2 all divide the times the stance changes at.
- **Solve 3** (`HardContactRefinement`) is the MPC of the task file tracking the thresholded plan over the whole
  duration in one offline SQP (`CentroidalMpcInterface::Create()` with the plan in memory): the stances are the mode
  schedule, the solve-2 states the targets, the swings rebuilt by the generalized swing planner, every inactive
  force zero (`zero_wrench`) and every stance pad held (`zero_velocity`). On top of the MPC's terms it adds the
  quasi-static joint torques `g(q) - sum J' w` within the scenario's torque margin less 0.02
  (`QuasiStaticTorqueConstraint`, under `joint_limits: "position_and_effort"` only, the stairs too), the collision spheres' clearance, and with a static
  margin the CoM in the support polygon of the smallest stance around each knot at the margin plus 5 mm
  (`SupportPolygonConstraint`, the polygons recomputed at the solution at most three times when the certifier fails a
  static condition). Four adjustments were needed for the certifier to pass a dynamic plan: the contact that lifts
  off is unloaded in the 0.1 s before (`LiftOffUnloadingCost`: the certifier's momentum rate at the first knot after
  a lift-off is the central difference over the interval before it); a sole lands flat in the 0.2 s before its
  touch-down (`LandingAlignmentCost`, so that its corner spheres do not reach the step first); the swings keep 12 cm
  of clearance (a sole's pad is its center; the solve's own, whatever the task file's) and are servoed at weight 1000,
  without the task file's landing alignment (section 7.2, item 3), with which the solve converged to plans that
  failed the sphere clearance by tenths of a millimeter; the stance pads' normal gain is 20/s, the
  wrench cones' barrier `(1, 1 N)` and their sole rectangle 1 cm smaller on every side. The solver's nodes are
  0.0125 s apart (the stance pads are held at the nodes only), the plan's knots 0.025 s. These are planner A's own
  settings: the task file the dummy sim tracks the plan with keeps its online tuning.
- **Output.** The plan of the solve-3 trajectory at the knots (`planOfTrajectory()`: one phase per run of active
  knots, until the knot that breaks it, on the patch under the run's mean pad), stamped and certified;
  FailedPrecondition naming the first failing knots, NotFound when the goal stance, its primitives or the base region
  are not reached (`checkGoalReached()`).
- **The combined planner** (`stance_graph_then_contact_invariant`, `StanceGraphThenContactInvariantPlanner`, section
  5.2): planner B's certified sequence, retimed to the scenario's duration (`retimedPlan()`: every run of one stance
  keeps its share of the time, at least 0.3 s, and the changes are rounded to the thresholding's 0.1 s), is the seed
  of solve 3; solves 1 and 2 are skipped. Its failures are planner B's when B finds no sequence. It is the stairs
  scenario's planner: its plan is quasi-static (its smallest static margin -3 cm), and the tracking MPC follows it
  within the certifier's conditions (below).

Two changes to milestone 4's code came with it: `CentroidalMpcInterface::Create()` takes a plan in memory, and the
planned-motion reference manager starts a contact's later swings within the horizon at their own phase's pad instead
of where the contact is now (an offline solve's horizon holds every swing of the plan; an online one rarely holds two
of one contact).

**Status (milestone 5, every scenario at `joint_limits: "position_only"`, section 11, decision 6).** Per scenario and
planner, from `plan_multi_contact --v=1`:

<!-- LINT.IfChange(contact_invariant_status) -->
| Scenario | Planner | Plans | Where it stops |
| --- | --- | --- | --- |
| `stairs` | `contact_invariant` | yes: 13 stances, feet only, 8.0 s, certified at all 321 knots, torque utilization <= 0.42 (`back_bky`), smallest static margin -0.31 m (a dynamic gait: the right foot skips step 1, and the left lifts off the floor with the CoM 0.3 m behind the stance foot), smallest sphere clearance -4.7 mm; 30 s | tracked in the dummy sim's loop (not the shipped plan), 14 of 452 samples fail the certifier: the momentum rate outside the cone (residual <= 0.03) in the 0.1 s after the two lift-offs where the CoM is far behind the stance foot, the MPC's CoP at the heel's edge, and three touch-downs 5-12 mm above the step; the base 5.1 cm off the plan's |
| `stairs` | `stance_graph_then_contact_invariant` (the scenario's) | yes: 11 stances, feet only, 8.0 s, certified at all 321 knots, torque utilization <= 0.70 (`back_bkz`, reported only), smallest static margin -0.035 m, smallest sphere clearance -3.0 mm; 52 s | the shipped plan, tracked to the goal, the certifier passing at every sample, the base within 4.7 cm of the plan's (`climbsTheStairsOfTheShippedPlan`) |
| `stairs_steep` | `contact_invariant` | no | solve 2 thresholds to one 8.3 s stance on the left foot (1.3-9.6 s); solve 3: 316 of 401 knots fail, first a sphere of `l_lleg` 14 mm into a step and the landing pads 24-55 mm off it |
| `box_climb` | `contact_invariant` | no | solves 1-2 find hand, knee, knees, foot (left hand on the box at 0.5 s, left knee on the top at 2.6 s, both knees at 5.0 s, the right foot at 8.7 s), but the thresholded sequence holds the robot on one point for seconds (the hand alone 0.8-2.6 s, the left knee alone 2.9-5.0 s and 7.5-8.7 s), which the inactive candidates' leaked force (0.04-0.37 of the weight each) held up in solve 2; solve 3: 446 of 481 knots fail, first condition 1 at the first lift-off (static margin 21 mm against 30 mm) |
| `box_climb_high` | `contact_invariant` | no | the same single-point stances (the right hand alone 0.8-5.9 s); the certifier's equilibrium LP on such a stance fails (HPIPM MAX_ITER, 52 variables), which ends the plan with Internal |
| `car_egress` | `contact_invariant` | no | the thresholded sequence sits on the seat pad alone (1.2-4.1 s) and the right hand alone (4.4-15 s); solve 3: 502 of 601 knots fail, first a sphere of `l_hand` 41 mm into the vehicle |
| `get_up_supine` | `contact_invariant` | no | the thresholded sequence stands on the left foot alone from 0.5 s to 12 s; solve 3: 301 of 481 knots fail, first the soles 6-10 mm off the floor and a sphere of `r_hand` 17 mm into it |
| `get_up_prone` | `contact_invariant` | no | the thresholded sequence kneels on the right knee alone (1.1-8.9 s); solve 3: 378 of 481 knots fail condition 1, first the margin of the knees and a hand -4.9 mm against 30 mm |
| `box_climb` | `stance_graph_then_contact_invariant` | no (the scenario now names `stance_graph`) | planner B's waypoint sequence (section 5.5, its earlier 31 stances) retimed to 12 s: solve 3 fails 184 of 481 knots, first a sphere of `r_hand` 5.7 mm into the box; to 60 s: 868 of 2401 knots, the hand and sole spheres and the static margin (0.024 m) at the single-foot stances |
| `box_climb_high`, `car_egress` | `stance_graph_then_contact_invariant` | no | planner B finds no sequence (section 5.5) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/app/BUILD.bazel:planner_scenarios) -->

What stops planner A on the static scenarios is solve 3 inheriting a solve-2 motion that is not quasi-static: with
eight or more candidates the soft gate (`k0 = 1`, `k1 = 0.01`) lets the candidates out of contact carry 0.1 to 0.2 of
the weight each, so the thresholded sequence asks the hard-contact problem for a motion it cannot hold with the
remaining contacts inside the 3 cm polygon, and the polygon constraint, a barrier, gives way. Lifting the effort
limits (section 11, decision 6) removed solve 3's torque constraint but not this: the sequences are the same.
Two attempts did not help and were not kept:
- two more continuation steps raising `k0` to 10 and 100 at the contact-invariant weight 1000: the box's thresholded
  sequence keeps its single-point stances, and solve 3 fails 466 of 481 knots;
- thresholding make-before-break (a contact broken less than 1 s, or 3 s, before another is made stays until that
  touchdown): it removes the first single-point stance of the box at 1 s, and at 3 s all of them, but the contacts it
  keeps have left the terrain in the solve-2 motion, so solve 3 starts from a guess whose stance pads are in the air,
  and the certifier's equilibrium LP of the five-contact stance fails (HPIPM MAX_ITER).

The box climb now plans with planner B through the scenario's waypoints (section 5.5), and the scenario names
`stance_graph`. That sequence as the seed of solve 3 (the combined planner, table above) still fails: retimed to the
scenario's 12 s, 184 of 481 knots, and to 60 s, 868 of 2401, the hands' and soles' spheres and the static margin at
the single-foot stances; the 60 s solve also met a wrench cone LP HPIPM stopped on, which the certifier now reports as
a failing knot (section 6) instead of an error. The next steps are solve 2 with the static margin as a hinge of the CoM against a smooth support region, a solve 2
repeated on the repaired schedule (make before break, with the kept contacts' pads held), and seeding solves 1-2 from
planner B's partial sequences.

### 4.5 Why not mixed-integer contact planning

Mixed-integer formulations make the contact-to-surface assignment explicit with binaries (Deits and Tedrake for
footsteps; Aceituno-Cabezas et al. for simultaneous contact, gait and motion). They are attractive because the
assignment is exact and the convex relaxations give certificates. Here they scale badly:
- There are 11 candidates × about 10 surfaces × about 40 phases.
- The angular momentum term `p_i x f_i` is bilinear, so it needs McCormick envelopes or a piecewise-linear
  decomposition, which loosen the dynamics.
- The repository's `lip_miqp` is a branch-and-bound over HPIPM for a 2-D pendulum, so the solver would have to be new.

The mixed-integer approach stays a later option for the discrete layer only: choosing which patch each contact uses,
over a stance graph like the one in section 5.

## 5. Planner B: sampling over stances

### 5.1 The stance graph

The second planner follows Hauser, Bretl and Latombe's non-gaited planning:
- **Nodes** are stances: an assignment of some candidates to points on scene faces.
- **Edges** add or remove one contact.
- **A node is feasible** when there is a configuration `q` that:
  - reaches every assigned point within 5 mm, with its pad normal in the cone of section 3.1;
  - is in static equilibrium with margin `delta_s` and is torque-feasible (section 2);
  - respects the joint limits;
  - is collision-free (section 5.4).

  That is one small NLP per node: Pinocchio inverse kinematics with the CoM constrained to the polygon from the LP. The
  first draft solved it with OCS2's SQP as a single-node problem; milestone 3 solves it with a dedicated trust-region
  SQP on HPIPM's dense QP (section 5.5), which is smaller and is tested on its own.
- **An edge from `S` to `S + {i}`** is feasible when a motion inside stance `S` brings contact `i` from where it was to
  its target without penetrating the scene. Each such motion is a short planner-A solve 3 with the stance fixed.

**Search.** Weighted A* over the graph:
- The cost is the number of contact changes plus a weighted path length of the base.
- The heuristic is the base's distance to the goal divided by the largest base displacement one contact change has
  achieved in that scenario.
- Contact targets are sampled on each face, on a 5 cm grid inside the face's polygon shrunk by the pad radius plus a
  margin, and the samples are pruned by the reach of the limb from the current base.

The output is a stance sequence with a feasible configuration at every node and a motion along every edge. By
construction it satisfies all three conditions of section 2.

### 5.2 Planner B as the initializer of planner A

The stance sequence fixes when each contact is active. Planner A's `c` is initialized to it, its state to the
configurations, and solve 3 alone smooths the motion and redistributes the forces. Mordatch's method needs no
initialization, so the two planners are useful separately:
- **Planner A alone** can find a contact sequence the grid would never sample, such as a forearm along an edge.
- **Planner B alone** is complete up to its sampling resolution, and its failures are explicit: the node or edge that
  failed and why.

Both run in the tests of section 9, and each is the other's baseline.

### 5.3 Selection by name

Following the repository's rule, a scenario selects its planner, and the joint limits its plan respects (section 11,
decision 6), by name:

<!-- LINT.IfChange(planner_names) -->
- `planner: "contact_invariant"` (section 4);
- `planner: "stance_graph"` (section 5);
- `planner: "stance_graph_then_contact_invariant"` (section 5.2).
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/scenarios/include/humanoid_multi_contact/scenarios/Scenario.h:planner_names) -->

<!-- LINT.IfChange(joint_limits_names) -->
- `joint_limits: "position_and_effort"` (the default): the joint position limits, and the quasi-static torques within
  `torque_margin` of the effort limits;
- `joint_limits: "position_only"`: the joint position limits alone;
- `joint_limits: "relaxed_position"`: as `"position_only"`, with the position limits the stance graph planner keeps to
  widened by the scenario's `joint_limit_relaxation` (in (0, 0.3] rad) on both sides (decision 8).
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/stability/include/humanoid_multi_contact/stability/JointLimits.h:joint_limits_names) -->

The registry is `MultiContactPlannerRegistry`, built the way the cost and constraint registries are.

### 5.4 Collision geometry

The robot's environment collision model is a set of spheres on its links, as `FootCollisionSpheres` does for the legs,
extended to the shins, thighs, forearms, hands, pelvis and torso. Each sphere has a radius taken from the MJCF capsule
it approximates. A link's spheres are placed along the capsule's axis at a spacing of one radius.

The constraint is `sphereDistance >= margin` for every sphere except those of the contacts that are active, plus a
5 cm neighborhood around each active pad. Without those exceptions, every contact would collide with the surface it
stands on.

Self-collision keeps the existing `FootCollisionCbfConstraint` and adds pairs between the forearms and the thighs,
which kneeling brings together.

**As built (milestone 3).** `CollisionModel` (`humanoid_multi_contact/collision/`) reads
`collision_model.textproto` beside the contact set (`collision_model_file` of a scenario): per link, a capsule, a box
or given spheres, in the frame of the joint of the MJCF body it is copied from (`test_collision_model_mjcf` compares
every link with `atlas.xml`). The Atlas model covers the pelvis and torso boxes, the head, the upper arms, both forearm
segments and the hands, the thighs, the shins and the four corner spheres of each sole: 83 spheres. A capsule gets
spheres of its radius at most one radius apart; a box is filled with spheres of its smallest half extent on a grid of
its other two axes, inside the box. A sphere is exempt when its center is closer than its radius plus 5 cm to an active
pad's region (the pad's point, or a surface pad's rectangle), so a torso lying on its back face is exempt over the
whole face. A point pad exempts only the spheres of its own link and the links below it (`exemptJointsOf()`): a knee
exempts the shin and the foot, not the other leg or the arms. What it does to the spheres of the link above it, within
the neighborhood - the end of the thigh at the knee joint, whose capsule has a larger radius than the shin's, so that a
kneel on a flat top with the shin level keeps it about 5 mm in the surface - the scenario names
(`point_pad_link_above`, `LinkAbove`): `"exempt"` (the default, the box climb's kneels on the box's edge, which the
thigh's end crosses) exempts them too; `"rests"` (the get-ups) lets them lie up to 6 mm in the terrain
(`kRestingLinkDepth`) and no deeper, by adding 6 mm to their clearance (`restingJointsOf()`,
`CollisionModel::restingAllowance()`, in the certifier, the planner's checks and the stance IK alike). Exempting
every link near the knee once let the planned kneels press the thigh 1-2 cm into the box; on the floor, the exempt
thigh's end sank 1.2 cm under a kneel and lifting the knee then swept it 9 mm through the floor between two knots,
which under `"rests"` cannot happen; counting it whole left no kneel on a flat top within the planner's tolerance.
A pad whose contact the plan does not count but which still rests on the terrain (its center within the 5 mm contact
tolerance) treats the spheres as an active pad does, a sole exempting its foot only: at the knot that breaks a knee's
contact the knee has not moved, and its tilted shin and the end of its thigh rest beside it as they did a knot
before. The self-collision pairs are the legs (thigh and shin against the other leg's, the soles) and each arm's
forearm segments and hand against the same side's thigh and shin; the stance IK keeps them 1 cm apart. The certifier
reports the smallest sphere clearance at every knot and fails a knot whose spheres penetrate the terrain by more than
its 5 mm contact tolerance (section 6).

### 5.5 Planner B as built (milestone 3)

`StanceGraphPlanner` (`humanoid_multi_contact/planner/`, registered as `stance_graph`) implements section 5.1 with
these choices:

- **Nodes and edges.** A node is a stance (each candidate on a point of a face, ordered by candidate) and the
  configuration that entered it. Removing contact `i` from `S` needs a *transition configuration* that holds all of
  `S` with the CoM in the polygon of `S - {i}`; adding `j` needs one that holds `S + {j}` with the CoM in the polygon of
  `S`. Either way the configuration holds the larger stance and is statically stable on the smaller, which is
  condition 3. The edge is feasible when the *motion* inside the stance being left, from the node's configuration to
  the transition, is.
- **Motions.** Each knot of a motion is an IK of the stance whose posture reference is the previous knot moved toward
  the transition by the step's share of the rest of the way (consecutive knots stay on one branch of the IK), with
  the CoM interpolated, the pad that broke contact lifting off along its surface normal and hovering there, and the
  pad that makes contact swinging to its target: lift-off, a traverse 6 cm above the surfaces (an arc raised just
  enough to keep 3 cm plus the pad's extent off the terrain when the straight line does not, as a foot stepping onto a
  box rises while it advances), and touch-down along the target's normal. The motion's last knot is the transition
  solved again from the knot before it, so that the motion ends on its own branch; when the transition cannot be
  solved directly (a foot that must go over a box's edge, a CoM that must travel far), the motion comes first, with the
  start as its guide and the CoM moving toward the middle of the smaller stance's polygon, and its end must then be a
  transition. Each knot must be statically feasible on the stance; a knot that moves more than twice the knot limits
  (8 cm of CoM or swinging pad, 0.25 rad of a joint) is retried at half the step, down to 1/48 of the motion, and then
  bridged by knots solved between the two (four halvings, each knot feasible), a remaining fast step being slowed by the
  timing law. The goal is reached when the stance is the goal's on the goal's primitives and a last motion inside it
  reaches the goal pose (shifted with the reached feet) with the base in the goal region.
- **Feasibility.** Every planned configuration: the held pads within 5 mm of their targets, a point pad's normal within
  30 degrees and a surface pad's within 0.02 rad of the surface normal, the joint limits (the IK keeps 0.1 rad inside
  them where a pose is not already closer, since lifting a foot from a straight knee is a singularity), no collision
  sphere more than 5 mm into the terrain (the certifier's tolerance) or into its self-collision pair, the CoM
  `delta_s + 1 cm` inside the polygon, static equilibrium, under `"position_and_effort"` torque feasibility within the
  scenario's torque margin, and, with a static margin, the exact margin of the stance at the reached pads. The torque
  check applies with or without a static margin: a slow motion cannot exceed the effort limits either. Under
  `"position_only"` no torque LP runs. Every knot of a motion also passes the certifier's own sphere test: no
  collision sphere more than 5 mm into the terrain outside 5 cm of the stance's pads. The IK's clearance exempts the
  swinging pad's start and target regions as well, and without this test a sole swinging past a riser could be planned
  6 cm into it; the torque check had hidden this on the stairs by refusing the long single-leg swings that cause it. Without a static margin the CoM
  keeps 2 cm inside the polygon, room for the momentum rates of condition 2. An LP HPIPM cannot solve marks its stance
  infeasible instead of aborting the search.
- **The IK is not OCS2's SQP** but a dedicated trust-region SQP (`StanceInverseKinematics`) on Pinocchio and HPIPM's
  dense QP, the solver of the certifier's LPs: one QP of the 36 velocity coordinates and one slack per inequality per
  iteration. The held pads are equalities with an exact L1 penalty (1e5 per meter or radian), the CoM polygon, the
  clearances (spheres within 8 cm of the terrain, self-collision pairs within 5 cm) and the point pads' normal cones
  inequalities with a smaller one (1e3), the posture, the swinging pads and the CoM target least squares. A trial step
  is corrected back onto the held pads by at most 1 cm (a second-order correction against the Maratos effect of the
  exact penalty) before the merit judges it; the trust region shrinks by four on a rejected step and doubles on an
  accepted one. The exemption of the spheres near a pad ramps off over a centimeter instead of switching, which keeps
  the merit continuous. A transition that fails is solved again in two stages, the pads first and then the CoM's
  polygon. It is small, needs no CppAD model, and has unit tests of its own. A transition's posture reference is the
  node's configuration and, first, a quarter of the way to the goal pose.
- **Search.** A lazy weighted A* (`lazyWeightedAStar`, epsilon 2, at most 6000 validations or 40 minutes): a node's
  candidates enter the open list with an estimated cost of one change and are validated only when popped. The cost is the number of changes plus the base's
  displacement in meters; the heuristic the larger of the contacts that differ from the goal stance (one per goal
  contact absent, two per goal contact on another primitive, a half per other contact, less than the change that
  removes it, so that the stances a climb needs are not all worse than every stance on the floor) and the base's
  distance to the goal over the largest displacement one change has achieved (0.3 m until one is larger). Stances are
  keyed by candidate, primitive, face and a 5 cm cell (the sampling grid) of each pad.
- **Waypoints** (`waypoints` of a scenario, hints in the sense of Mordatch's `L_hint`). A scenario may list stances
  on the way to its goal, each a goal stance (contacts on primitives, an optional base region and an optional pose).
  The search then runs in legs: from the initial stance to the first waypoint, from each to the next and from the
  last to the goal, one lazy weighted A* per leg within the one budget, each starting where the last ended (with the
  change that entered it, so that a contact broken at the end of one leg lifts off in the next). A leg to a waypoint
  makes only the waypoint's contacts. A waypoint with a pose fixes where: each contact on the face its pose's pad
  touches and within the 10 cm sample spacing of that pad, the pose's pad itself tried first; its changes are solved
  toward the pose as well as a quarter of the way to it, and, when they fail from the node's configuration, from the
  pose itself and from the pose of the waypoint the leg starts at (the change then moves there). A sole that swings
  to a touch-down in a leg to a waypoint with a pose is held level with its target the whole way (a climb's free leg
  folds far, and a free sole flipped its ankle by 0.7 rad on the way). The goal's leg is the search without
  waypoints, so the stairs, which has none, plans as before. The motions changed for the contacts the feet-only
  stairs never breaks: a pad that only lifts off (its contact is not the one the segment makes) is held level if it
  is a sole, and a point pad (a knee folding under the body) goes on from where its lift ended to where the
  segment's end has it; a transition keeps the free pads of the allowed contacts out of the terrain; and a segment's
  end solved again from a last knot that already holds it replaces that knot instead of repeating it a millisecond
  later (the momentum rate of the pair was outside the cone by 0.02).
- **Sampling.** Every face whose normal the candidate's kind accepts (point pads: no ceilings; surface pads: within 45
  degrees of level), on a 5 cm grid of the face's frame inside the face shrunk by the pad (a point pad's geom radius, a
  surface pad's rectangle at its current yaw) plus 1.5 cm, within the limb's length plus 0.4 m of trunk travel from the
  limb's root (the trunk contacts within 0.4 m of their pad); a capsule on its top and on its side facing the limb. A
  sample whose pad would be under another primitive, farther from another pad of the stance than the kinematic chain
  between them allows, or on one surface with another pad, is dropped. Of the rest, six, at most three per face
  and 10 cm apart, are tried, taken in turn from the samples nearest two anchors, the pad moved half way and all the way
  toward its goal (the goal pose's pad for a goal contact, the base's displacement to the goal otherwise), scored by
  the distance to the anchor plus twice the distance beyond the limb's own length plus 0.2 m per unit of `1 - n_z`.
- **Plans the MPC tracks as they are.** Planner B alone (`stance_graph`) makes plans the dummy sim tracks without
  solve 3, and its defaults are made for that; as planner A's seed (`stance_graph_then_contact_invariant`) it keeps
  the certifier's own tolerances, which solve 3 re-solves, so the stairs plans bit for bit as before. Alone:
  - it breaks a contact at the knot that ends the dwell before the motion that lifts its pad (`ContactBreak`, by
    name: `kAtDwellEnd`; the seed's `kAtSecondKnot`), and every phase lasts to the knot that breaks it, where the
    MPC's mode switches: with knots seconds apart, an MPC that held a sole down while the plan already lifted it, or
    whose sole had no phase of its own (its patch then the next phase's, on the box) for a knot, diverged at the box
    climb's first single-foot stance;
  - it keeps every collision sphere of its knots within 2.5 mm of the terrain, half the certifier's 5 mm
    (`StanceIkOptions::penetrationTolerance` and `StanceMotionOptions::sphereTolerance`; the seed's are the
    certifier's), and the CoM 2 cm beyond the static margin (`staticMarginBuffer`, 1 cm for the seed): the MPC follows
    the plan a few millimeters off;
  - it checks the motion between every two consecutive knots as the MPC is asked to track it
    (`StanceMotionOptions::sweep`, `StanceMotion::sweptClearance()`): the configurations of the MPC's reference
    interpolation (`knotInterpolation()`: the base position and the joints linearly, the orientation slerped, not
    pinocchio's screw motion), at least ten per step and one per 0.02 rad of the largest joint's turn and per cm of the
    base's travel, keep their spheres within the same 2.5 mm and their CoM within 1 cm of the knots' margin, and the
    straight path from knot to knot of every free pad - which the MPC's reference has a swing longer than
    `arc_swing_duration_limit` follow (section 7.2) - keeps a point pad's sphere and a sole's corners within 2.5 mm.
    A step whose sweep fails is treated like a jump: halved down to 1/48 of the segment, then bridged by knots solved
    in between, and the segment fails when a sweep still fails at the deepest halving. The knots alone missed what the
    MPC then tracked: the box climb's first plan put its knots a second apart around a knee's swing onto the box, and
    the thigh and the shin went up to 4.5 cm through the box's edge between them. A jump that remains at the deepest
    halving, between knots that are all feasible, is a fast motion the timing law slows down, and is kept only when
    its CoM keeps the full CoM margin, without the sweep's slack (`StanceMotion::jumpComDefect()`): the box climb's
    replan for seventeen candidates swung the free leg 1.2 rad in one such step, its CoM within the slack of the
    margin, and the MPC tracked it to 0.029 m, under the certified 0.03 m;
  - a knee that lifts off keeps facing the floor while it lifts, so that its shin and the end of its thigh rise with
    it, and a sole held level for its touch-down that starts tilted (a foot whose toes rest on the floor behind a
    kneeling robot) turns level over the first half of its swing instead of at the first knot, where its ankle
    flipped 1.1 rad;
  - a segment that sets a knee down brings the posture to its end's by half of the segment
    (`pointTouchdownPostureLead`, 1 for the seed), so the leg folds before the knee reaches the box's edge, and the foot
    below that knee, whose contact the segment broke, follows the leg to the end's position instead of staying by the
    floor where it lifted off: held there, the knee lagged its path by 0.2-0.4 m and jumped to the box at the end;
  - a segment's end that the last knot cannot reach without a defect is solved again toward the last knot alone, and
    taken when the step to it has none; and a segment that sets a pad down ends at the first knot, bridged ones
    included, where the pad has reached its target, with the end solved from there (`endOnReach()`): a bridge to the
    end held a sole a millimeter above the box for nine seconds while the body moved 1 rad, and the MPC, which keeps a
    swinging sole's center off the terrain but not its heel, pressed the heel 1.3 cm into the box;
  - a plan whose knots fail condition 2 alone is retimed in the failing knots' segments only
    (`RetimingScope::kFailingSegments`; the seed's `kWholePlan`): one fast fold no longer slows the whole climb 1.5 or
    2.25 times.
- **Timing law** (`PlanTiming.h`). Every motion is quasi-static between rests: its knots are timed by a minimum-jerk
  profile `10 tau^3 - 15 tau^4 + 6 tau^5` of a path parameter that advances with each interval's "time at the speed
  limits" (CoM 0.15 m/s, base 0.2 m/s, any pad 0.3 m/s, any joint or the base turn 0.5 rad/s), over the longest of
  1.875 times that time (the profile's peak speed), `sqrt(5.774 L / 0.15 m/s^2)` for the CoM's path length `L` (its
  peak acceleration) and 0.8 s. Every contact change sits in a 0.4 s dwell at rest; a contact is made at the last knot
  of the motion that brings it, and broken after the dwell at the end of the motion before the one that lifts it, so
  condition 3 holds knot by knot. The velocities are central differences of the configurations, the momenta
  `A_G(q) v`, and the contact wrenches of a knot the torque LP's static forces (for playback). A plan shorter than the
  scenario's `duration` is stretched to it; one that fails condition 2 is retimed 1.5 times slower, at most four times
  (the whole plan for the seed, the failing segments for planner B alone).
- **Output.** The plan is stamped with the scenario's hashes and certified by the scenario's certifier before it is
  returned; a search that fails returns NotFound naming the stance nearest the goal and why its last changes failed
  (the transition or the motion, the knot and the condition), and a plan the certifier refuses FailedPrecondition
  naming its first failing knots.

**Status (milestone 3, every scenario at `joint_limits: "position_only"` but the box climb's and the get-ups'
`"relaxed_position"`, section 11, decisions 6 and 8).** Planner B plans the stairs, and the box climb and both get-ups
through their waypoints; it does not yet plan the other three scenarios within its budget, and lifting the effort
limits did not change that: the searches now reach stances the arms could not hold (a foot on the box with a hand on its top), and stop on the IK. What the searches
reach and where they stop, from the planner's own failure reports (`--v=1`):

<!-- LINT.IfChange(planner_status) -->
| Scenario | Plans | Where the search stops |
| --- | --- | --- |
| `stairs` | yes: 9 stances, feet only, certified (static margin >= 2 cm, torque <= 1.01 at `back_bkz`, reported only), 133 knots, 135 s of motion, 14 s (44 validations); the shipped plan is the combined planner's (section 4.4) | |
| `stairs_steep` | no (40 min, 1851 validations) | nearest the goal: the right foot on step 1; adding a hand on the landing makes the IK change branch (the other hand's pad jumps 0.43-0.45 m between two knots, and no bridge continues it) |
| `box_climb` | yes, through its seven waypoints (the plan of `make plan-drc-atlas-box-climb`, `plans/box_climb.binpb`): 51 stances (the hands on the box, the right foot a step forward, the left knee and then the right on the box's top 5 cm behind its edge, the hands forward, the right foot and then the left on the box, the hands off, a step forward), feet, knees and hands, 594 knots, 455 s of motion, certified at every knot and clear of the terrain between them (static margin >= 0.049 m, sphere clearance >= -0.2 mm, torque 2.8 at `l_arm_elx`, reported only), the position limits relaxed by 0.1 rad (`r_leg_hpy`, `l_arm_shz` and `back_bky` 0.100 past their ranges), 1198 validations, 13 min | tracked in the dummy sim's loop to the top, every sample certified (section 9) |
| `box_climb_high` | no (40 min, 1764 validations) | stances with a knee or both hands on the box are reached; nearest the goal stays the floor stance |
| `car_egress` | no (40 min, 1525 validations) | nearest the goal: the right foot on the pan and the left thigh on the cushion; lifting the thigh leaves the CoM 1-2 cm inside a polygon that needs 4 cm, and a foot onto the ground beside the vehicle stalls the IK 0.23 m from its target with the foot 13 cm into the terrain |
| `get_up_supine` | yes, through its twelve waypoints (the plan of `make plan-drc-atlas-get-up-supine`, `plans/get_up_supine.binpb`): 36 stances (the right arm along the head, the right foot off and the back-right edge of the torso box down beside the back, the back off onto that edge, the roll on to its 65-degree pads, the right foot set down behind, the right hand down past the head, the right foot in front of the hips, the edge off and the left hand down, on the feet and the hands, the feet and the hands walked toward the goal, the hands off, up), the feet, the hands, the torso's back and its four edge pads, 426 knots, 302 s of motion, certified at every knot (static margin >= 0.050 m on the refined polygon, sphere clearance >= +5.1 mm, torque 3.5 at `r_arm_shx`, reported only), the position limits relaxed by 0.05 rad (both hips' pitch, `r_leg_aky`, `l_arm_shz` and `back_bky` 0.050 past their ranges), the spheres 2 cm off the floor, 327 validations, 8 min. Its waypoints hold each planted sole's yaw, as the stance IK does, and each holds its CoM in the polygon of the stance it shares with a neighbor one change away; without them the search's seeds missed by 11 cm (a sole turned 25 degrees on the floor) or found no CoM position | tracked in the dummy sim's loop to standing, every sample certified (section 9) |
| `get_up_prone` | yes, through its four waypoints (the plan of `make plan-drc-atlas-get-up-prone`, `plans/get_up_prone.binpb`): 9 stances (the left knee off and the left foot forward between the knees and the hands, the left hand off, the right knee off and the right foot beside it, a squat on the hands, the left hand set down again and off, the right hand off, up), 104 knots, 79 s of motion, certified at every knot and clear of the floor between them (static margin >= 0.050 m, sphere clearance >= +0.7 mm, torque 1.38 at the right elbow, reported only), the position limits relaxed by 0.1 rad (`back_bky` 0.099, `l_leg_aky` 0.054) and the thigh's end beside a knee resting (`point_pad_link_above: "rests"`), 19 validations, 19 s | tracked in the dummy sim's loop to standing, every sample certified (section 9) |

Without its waypoints, and with a 3 h budget (an experiment, not the shipped 40 min), the box climb's search got
closer and still found no plan (8919 validations): it reached the left foot and the right hand on the box, where a
forearm onto the box stalled the IK 0.3-0.5 m from its target. With the same budget `stairs_steep` (6923) reaches the left foot on the landing and the right on step 2,
where lifting the right foot needs the CoM 22 cm over, which the IK does not find; `box_climb_high` (6939) and
`get_up_prone` without its waypoints (9652) stop where they did in 40 min.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/app/BUILD.bazel:planner_scenarios) -->

These are limits of the local IK and of the search, not proofs that the scenarios cannot be done: the transitions they
need are large motions of the whole body (a step onto a box at 68 % of the leg's length, sitting up from the back),
which a projection from the previous configuration does not find. The arms of the DRC Atlas are weak for carrying the
body (`l_arm_ely` 63 N m, `l_arm_shx` 99 N m), which the shipped scenarios now disregard (section 11, decision 6).
Planner A (milestone 5, section 4.4), which optimizes the whole motion at once, plans the stairs in 8 s; posture seeds
for the IK per kind of change are the next step for planner B; the box climb's waypoints are such seeds, written in
the scenario. The three other planning tests are tagged `manual` until their scenarios plan.

## 6. The stability certifier

`StabilityCertifier` takes a plan (section 7) and the contact set and scene. At every knot it reports:
- the active stance;
- the static margin `rho_s`;
- the smallest torque margin, and which joint has it;
- whether the planned centroidal momentum rate lies in the CWC;
- the smallest distance from a collision sphere to the scene (section 5.4), outside the neighborhood of the active
  pads, and its link; the clearance of every inactive candidate's pad geometry (a sphere of its geom radius); and the
  largest distance of an active pad to the terrain, which must be within 5 mm (condition 2).

Each knot passes or fails conditions 1 to 3 of section 2, and, when the scenario names a collision model, fails when a
sphere penetrates the terrain by more than the 5 mm contact tolerance (`collision_free`). A knot whose support polygon,
equilibrium or wrench cone LP HPIPM cannot solve fails, naming the LP, instead of ending the certification with an
error (solve 3 on the box climb's sequence hit one at MAX_ITER). Before that, an LP HPIPM stops on is solved again:
once with 2000 iterations at 1e-8 (`solveLinearProgram()`), and then with every generator's scaling bounded by ten
body weights (`StaticEquilibrium`'s `solveBoundedOnFailure()`). The LPs are degenerate: internal forces between two
contacts add scalings that change no wrench, so their optimal sets are unbounded, and HPIPM's interior point, on the
1e-9 regularization alone, ran out of iterations on the static wrench of the box climb's two feet and one hand
(`equilibrium_test` keeps that LP). The bound makes the set compact and excludes no force a stance can use; it changed
ten LP failures of the box climb's earlier plan (of 4341 configurations certified between its knots) into passes. A
static equilibrium LP that fails is named too (it failed condition 1 silently, at a margin of 5 cm).

`certify_plan --between_knots=N` certifies N configurations between every two knots as well, interpolated as the MPC's
reference interpolates the plan (`planBetweenKnots()`: the base position, the joints and the momenta linearly, the
orientation slerped): what the MPC is asked to track, rather than the knots alone. Its active pads, which the plan
holds by IK at the knots only, drift off their patches in between, and those failures say nothing about the motion.

**The LPs** run on HPIPM's dense QP solver (already a dependency through OCS2 and `lip_miqp`), with the Hessian
1e-9 I: strictly convex, so each has one solution, the LP optimum of least norm.
- The forces are combinations `lambda >= 0` of the edges of linearized friction pyramids, eight per cone, inscribed
  (`mu cos(pi / 8)`), in units of the robot's weight, with moments about the mean of the contact positions.
- A surface contact contributes a pyramid at each corner of its rectangle, which spans its contact wrench cone exactly
  (friction, center of pressure and torsion). With 17 candidates, 5 of them surfaces, that is at most
  (12 + 5 × 4) × 8 = 256 variables.
- A feasibility test (is this wrench in the cone?) minimizes the L1 norm of slack on the six wrench rows, so it always
  has a solution, and its value says by how much a knot fails (feasible below 1e-6).
- The support-point LPs of the margin and the torque LP use the equalities exactly, with their linearly dependent rows
  removed (a rank-revealing QR), after a feasibility test.
- **The torque LP** minimizes `u` subject to static equilibrium and `|tau_j| <= u effort_j` for every joint with a
  limit: `u` is the smallest fraction of the limits that holds the stance, and the stance is torque-feasible when `u`
  is at most the scenario's `torque_margin`. Under `joint_limits: "position_only"` the report carries `u` per knot for
  information (absent where HPIPM cannot solve the LP, which then fails nothing), names the joint limits and no torque
  margin, and condition 3 solves no torque LP.

The momentum rate of a knot is the central difference of the plan's momenta. The certifier is used:
- by both planners, as their acceptance test;
- by the dummy sims, on the robot's actual trajectory, logged to Rerun as a margin trace;
- by the tests, which require every shipped scenario's plan to pass.

## 7. The plan, and how the MPC tracks it

### 7.1 The plan file

`MultiContactPlan` (new proto, `multi_contact_plan.proto`) holds:
- the scenario name and the hashes of the contact set and the scene it was planned for (64-bit FNV-1a of the
  deterministic serialization of each message as its struct rewrites it, so a default spelled out or left out hashes
  alike);
- the knot times;
- per knot, the state, the input and the active-contact bitmask, by name and in physical units rather than as the
  MPC's vectors: the base pose (position and quaternion), the joint positions and velocities of the plan's joint
  names, the linear and angular momentum, and the contact wrenches. A plan therefore does not depend on an MPC's state
  layout or fixed joints;
- per contact phase, the assigned face and the target pad pose;
- the certifier's report.

A plan whose hash does not match the loaded contact set or scene is refused at load time (`PlanFile.h`,
FailedPrecondition). It is written as a binary proto (`<name>.binpb`) beside a textproto summary without the knots
(`<name>.summary.textproto`), by `make plan-drc-atlas-<scenario>`, and read by the MPC node at startup from a path
in the launch file. A bus topic for plans is not needed for the dummy sims; it is a later addition for replanning.

### 7.2 Changes to the centroidal MPC

The online formulation is the planner's solve 3, over a receding horizon and with the robot's estimated state instead
of the plan's.

1. **A runtime contact set** in the centroidal MPC (milestone 2, done). The input layout, the dynamics and the
   per-contact constraints come from `CentroidalModelInfo`'s mixed 3-DoF and 6-DoF offsets instead of `6 * index`:
   - The task file's `model_settings.contacts_3dof` lists the 3-DoF contacts: a name, a parent (a joint of the MPC
     model, or a joint it fixes, such as Atlas's locked wrists) and a translation. The model gets a frame for each.
   - The contact set is `ModelSettings::contactNames`: the 3-DoF contacts first, then the 6-DoF ones, which is OCS2's
     input layout. The input is the force of each 3-DoF contact, the wrench of each 6-DoF contact, then the joint
     velocities, 3 n3 + 6 n6 + nj entries. `CentroidalMpcRobotModel` and `StateInputLayout` take their offsets from it.
     A 3-DoF contact's input weight is its force block alone.
   - A mode number is the bitmask of the active contacts in contact-set order, with the first contact as the most
     significant bit (`contactIsActiveInMode()`). With the feet listed last, the biped's `FLY` / `RF` / `LF` / `STANCE`
     keep their numbers and their meaning in any larger set. Every per-contact term asks
     `SwitchedModelReferenceManager::isInContact()`.
   - Per contact, `contact_wrench_cone` builds the contact wrench cone for a 6-DoF contact and the new
     `PointFrictionConeConstraint` for a 3-DoF one. `friction_force_cone`, `zero_wrench` and `zero_velocity` run on
     both (three translation rows for a point). `contact_moment_xy` is built for the 6-DoF contacts only. The
     initializer and the input reference share the weight among the active contacts.
   - The terms that are biped by nature keep `kNumContacts` and their `static_assert`s. On any other contact set,
     `checkCentroidalContactSet()` refuses them by name before a model is built.
     <!-- LINT.IfChange(biped_only_terms) -->
     They are:
     - the costs `task_space_foot_cost`, `external_torque_cost` and `dcm_terminal_cost`;
     - the soft constraints `foot_collision`, `foot_collision_cbf`, `normal_velocity` and the three contact-implicit
       terms;
     - the hard constraints `normal_velocity` and `knee_joint_mimic`;
     - `contact_schedule_source: "contact_planner"`, which brings the H-LIP and MIQP planners, the eight
       contact-planning rules, `FootSeparationConstraint`, `ZmpSupportRegionConstraint` and the no-flight and
       yaw-torque-budget constraints;
     - `contact_input_parameterization: "basis_vectors"`;
     - every locomotion heuristic.
     <!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/CentroidalContactSet.cpp:biped_only_terms) -->

     The biped's swing planner, and every term that reads it, stays the biped's: milestone 4 swings every other
     contact on a generalized planner of its own and new terms that read it (`contact_swing`, item 3), so no term of
     this list is ungated.
   - Version 1 is the centroidal MPC only. The whole-body MPC refuses any contact set but the biped's when it builds its
     models, as it refuses the contact-implicit terms today.
   - Parity: `testCentroidalContactSetParity` compares every term's name, value and derivatives of each shipped
     robot's assembled problem with a record made before the change, bit for bit. Three variants on Atlas and G1 cover
     the wrench cone, the contact-implicit terms and a soft yaw-holding stance. `testMixedContactSet` checks Atlas
     with two 3-DoF hand contacts against the hand-derived centroidal momentum rate.
2. **Cones and contact constraints in the patch frame.** `FrictionForceConeConstraint`, the new
   `PointFrictionConeConstraint` and the zero-velocity constraint take a rotation from the world to the patch frame,
   and the zero-velocity constraint also takes the patch's offset along its normal (`ContactPatchFrame`). That
   rotation is a parameter of the contact phase, read on every evaluation from
   `SwitchedModelReferenceManager::getContactPatchFrame()`, which milestone 4's reference manager overrides. It is
   never recorded on a tape. The default is the world-aligned patch at the applied terrain height, and a world-aligned
   patch evaluates exactly today's arithmetic.

   `ContactWrenchConeConstraint` takes no rotation. Contrary to an earlier draft of this section, it is not written in
   the world frame: its rows bound `l_R_w(q) W`, the wrench in the sole's own frame, because the rectangle is the
   sole's. On a sole resting flat on a patch, that frame is the patch's frame whatever the patch's orientation, so the
   cone is already patch-correct. Rotating it again would be wrong, and replacing `l_R_w(q)` with a constant patch
   rotation would change today's term.
3. **A planned-motion reference manager** (milestone 4, done). `PlannedMotionReferenceManager`
   (`humanoid_centroidal_mpc/multi_contact/`) is selected by name with the existing selection field of where the mode
   schedule comes from, `contact_schedule_source: "multi_contact_plan"`, beside `gait_schedule` and
   `contact_planner`. The task file's `multi_contact_plan` block names the scenario and the plan (the MPC node's and
   the dummy simulator's `--multi_contact_scenario` and `--multi_contact_plan` replace them) and tunes the tracking. A
   plan whose hashes do not match the scenario's contact set and terrain is refused at start-up (FailedPrecondition).
   It does four things before every solve:
   - It provides the mode schedule from the plan's bitmasks, shifted by the plan clock's offset. A contact is mapped
     **by name**, never by its bit: the contact set numbers the candidates feet first, the MPC lists its 3-DoF
     contacts first and its feet last, and `PlanContactOrder` maps one to the other once, at load. An MPC contact that
     is not a candidate is never active; a candidate the MPC lacks is refused when the plan makes it active.
   - It provides target trajectories from the plan's states (`PlannedMotion`, converted once at load: the base
     quaternion, the MPC's joints by name, the momenta normalized by the mass), shifted the same way.
   - It rebuilds every swing on the generalized swing planner (`ContactSwingPlanner`, `ContactSwingTrajectory`),
     which moves any contact, not only a foot: from where the contact last was in contact, a cubic Bezier arc that
     leaves along the normal of the patch it lifts off from and arrives along the normal of its target patch, its
     inner control points raised until every sample away from its ends keeps the terrain's signed distance above
     `multi_contact_plan.swing.clearance`, timed by a smoothstep so that it starts and ends at rest. A contact the plan
     starts in the air reaches its first pad along that pad's normal. `HumanoidPreComputation` reads the planner's
     reference of every contact, and the soft constraint `contact_swing` (`ContactSwingConstraint`) holds each
     swinging contact to it: `v - v_ref + k (p - p_ref) = 0` under a quadratic penalty. That servo moves only the
     pad's center; with `contact_swing.landing_alignment_weight` positive (off by default; the Atlas's multi-contact
     task file sets it, milestone 5) every 6-DoF contact also lands flat: over the last `landing_window` seconds of a
     swing, `z_c x n = 0` for its frame's z axis and the normal of the patch it lands on, under a quadratic penalty
     (`SwingLandingAlignmentConstraint`). Without it the Atlas's soles reached the stairs' steps pitched by up to 30
     degrees, a heel or toe sphere first; its task file holds them flat over the whole swing (weight 3000, a 3 s
     window: the stairs' swings are shorter, and the get-up from prone's sole spends 2.9 s on its last step before it
     lands, where a 2 s window let its heel dip 6.5 mm into the floor), with 15 cm of swing clearance, the wrench cones' barrier `(1, 0.5 N)` and base position weights of 75
     along x and y, the settings with which the stairs plan's tracked trajectory certifies (section 9). With
     `multi_contact_plan.arc_swing_duration_limit` positive (0, every swing on an arc, by default; the Atlas's
     multi-contact task file sets 2 s, longer than every swing of the stairs) a contact out of contact for longer
     follows the plan's own pad instead: straight segments between the pad's places in the plan at its knots, each
     timed by the same smoothstep (segments 0.25 s apart, following the kinematics of the interpolated configurations
     more closely, made the box climb's tracked trajectory fail the certifier at more samples, not fewer). The stance graph's quasi-static plans hold a hand or a knee in the air for
     tens of seconds on a path they checked against the terrain knot by knot; an arc over that whole window pulled the
     box climb's arms toward the box a minute early, against the plan's posture, and an arc over its last 2 s alone
     raised a foot stepping toward the box 15 cm and drove its thigh 3.6 cm into the box's front.
   - It answers each contact's patch frame (`getContactPatchFrame()`) from the plan's phase it rests in, or the one
     it swings to: the frame of the phase's target pad normal, world-aligned exactly when the normal is the world z
     axis.
   The MPC's initial state, and the dummy simulator's, is the plan's first knot.
4. **Environment non-penetration** (milestone 4, done for the pads), as a soft constraint with the relaxed-barrier
   penalty the existing soft constraints use: `terrain_non_penetration` (`TerrainNonPenetrationConstraint`) keeps the
   collision sphere of every contact that is not in contact (its pad's geometry, the contact set's `geom_radius`) out
   of the terrain, the scene recorded on the tape as constants and the library named by the scene's hash. The
   collision spheres of the links (section 5.4) are a later addition.

The MPC does not re-plan contacts. If the robot falls behind the plan, the reference manager holds the plan's clock
until the current phase's stance is reached within a tolerance. It never skips a contact change, because skipping one
breaks condition 3. Precisely (`PlanClock`): the plan starts at the first solve and runs with the solves until it is
`multi_contact_plan.clock_hold.lead` before its next contact change; there the stance the robot is on is judged, every
contact of it on its patch within `contact_tolerance` along the patch normal and the center of mass within
`com_tolerance` horizontally of the plan's at that time. Reached, the change is released; otherwise the clock holds
there, solve after solve, the change `lead` ahead in the horizon. At most one change is released per solve, so a
stalled solve cannot skip one.

### 7.3 Playing a plan forward and in reverse

The task file's `multi_contact_plan.playback` names how the tracked plan plays (`PlanPlayback.h`; the MPC node's and
the dummy simulator's `--multi_contact_playback` replaces it):
- `once`, the default: from the first knot to the last, then holding the last state, as section 7.2 describes;
- `forward_and_reverse`: forward, `playback_dwell` seconds (default 1.5 s) standing at the end, the plan in reverse
  time back to its start, `playback_dwell` standing there, and again, forever. The stairs dummy sim plays its plan this
  way (the Makefile's `MULTI_CONTACT_PLAYBACK_stairs`): Atlas walks up the stairs, stands on the top, walks backward
  down them still facing up, stands at the bottom and starts again. Every other scenario plays once, a get-up or a box
  climb being nothing to undo.

**Why the reversed plan is as stable as the plan.** Reversing time, tau = 2 tN + d - t, keeps every configuration
q(tau) = q(t) and negates every velocity, q'(tau) = -q_dot(t), while the accelerations are even, q''(tau) = q_ddot(t).
The centroidal momentum h = A(q) q_dot is odd, so its rate is even:
- the linear rate m c_ddot is even with c_ddot;
- the angular rate d/dt (A(q) q_dot) = A q_ddot + A_dot q_dot is even, A q_ddot because q_ddot is, and A_dot q_dot
  because A_dot = dA/dq q_dot and q_dot are both odd.

At every mirrored instant the momentum rate, the configuration and the stance are the plan's, so the same contact
wrenches realize it: the contact wrench cone (condition 2), the static margin, the joint position limits, the pads on
the terrain and the collision clearances hold exactly where they held forward. The certifier's central differences of
the momenta agree too (to 1e-9 relative on the stairs), since the reversed knots are the plan's in mirrored order with
their momenta negated.
Liftoffs become touchdowns and touchdowns liftoffs.

**The cycle.** `forwardAndReverseCycle()` builds one period of it as a plan, once, at load: the plan's knots; a knot
at rest in the middle of the top dwell (the plan's last configuration, zero momenta, its last wrenches); the knots in
reverse order at tau, their joint velocities and momenta negated; a knot at rest in the middle of the bottom dwell;
and the first knot again at t0 + T, the period T = 2 (tN - t0) + 2 d. A knot's stance and wrenches hold from it to the
next knot, so in reverse time they hold from it back to the knot before: the reversed knot of knot i carries the
active contacts and the wrenches of knot i - 1. The reversed mode schedule is then the exact mirror of the plan's, a
change at t reversed at tau(t), and every contact phase [s, e] mirrors to [tau(e), tau(s)]; a phase that lasts to tN
merges with its mirror through the top dwell, and the mirror of a phase from t0 lasts through the bottom dwell to
t0 + T. The certifier passes the cycle at every knot, the reversed knots at a change of stance in the stance before it
included.

The first version gave each reversed knot its own stance. Its schedule then lagged the mirrored phases by one knot
interval (25 ms): at every reversed lift-off a contact stood on no phase, `getContactPatchFrame()` fell back to the
next phase's patch, the pad a step below, and the hard zero-velocity constraint pulled the stance foot 18 cm into the
step. The closed-loop descent's cost jumped at each lift-off, its base error grew from 1 cm to 9 cm by the third step
down, and the robot fell at 13.6 s (the stairs dummy sim at 15.4 s). `test_plan_playback` now requires every contact
the schedule puts down to rest on a phase, its planned pad within 1 cm of that phase's patch, at every millisecond of
two periods.

The shipped stairs plan starts at rest but ends with momenta of a few N s and N m s, which the resting knot of the top
dwell brings to zero and then to their negation: the reference stays continuous at both seams, and its momentum rate
over the dwell (about 7 N and 13 N m) is well inside the cone of the two feet.

**The periodic lookup.** `PlannedMotion` converts that cycle with `PlanRepetition::kPeriodic` and answers every plan
time through the period: plan time t is the cycle's at t0 + (t - t0) mod T, and every result is shifted back by the
whole periods. The mode schedule of a solve unrolls the periods over its horizon (and the swings' margin), the targets
the knots of the periods it spans, the swings the windows of those periods, and the patch frames, the stance checks of
the clock hold and the planned wrenches look their phase or knot up the same way. The reversed half's swings are the
cycle's own: the swing planner builds each from the pad the reversed sequence lifts off (the old touchdown) to the one
it lands on (the old liftoff), so on the stairs every reversed swing steps down, and `contact_swing`,
`terrain_non_penetration` and the landing alignment read them as any other. `PlanClock`'s changes repeat with the
period, change k being change k mod n of period k / n: the clock holds before a change after the wrap as before any
other, and never ends. A seam needs no special case, since the cycle ends in the stance and the state it starts in
(`PlannedMotion::Create()` refuses a periodic plan that does not).

## 8. The four dummy sims

### 8.1 The scenarios

A scenario is a textproto (`scenario.proto`, in `humanoid_multi_contact/scenarios/config/`) with these fields:
- `robot`, and its `urdf_file` and `contact_set_file`;
- `terrain_scene`;
- `initial_pose` (a named pose: the base pose, the joint positions and the active contacts);
- `goal` (the goal stance, each contact on a named primitive; an optional base-pose region; an optional final pose);
- `allowed_contacts`;
- `planner`;
- `static_margin`, absent for the stairs;
- `joint_limits` (`"position_only"` in every shipped scenario, section 11, decision 6) and `torque_margin`, read only
  under `"position_and_effort"`;
- `duration`;
- `simulation_start`: when MuJoCo's tilt catch and gantry catch arm, by name, `at_start` or `when_standing` (with a
  standing base height), for the scenarios that start seated or lying (section 8.3).

Every number below is an initial value, set in the scenario file and expected to be tuned.

**1. Stairs** (`stairs.textproto`).
- **Scene:** four steps, rise 0.18 m, run 0.28 m, width 1.2 m, starting 0.5 m in front of the robot, the fourth a 1 m
  landing. A second variant, `stairs_steep`, has a 0.25 m rise and a handrail capsule 0.9 m above the line of the
  nosings, on the left.
- **Allowed contacts:** feet, plus hands for the steep variant.
- **Goal:** both feet on the top step.

The feet-only variant is also the first test of the generalized swing planner and the patch-frame cones, and is the
bridge to today's locomotion stack. Its dummy sim plays the plan `forward_and_reverse` (section 7.3): up the stairs and
backward down, over and over.

**2. Climbing onto a box** (`box_climb.textproto`).
- **Scene:** a 1.0 × 1.0 m box, 0.6 m high, 0.4 m in front of the robot. A second variant, `box_climb_high`, is at the
  robot's hip height (measured from the model at the nominal pose: the hip pitch joints are 0.857 m above the floor;
  the box is 0.86 m).
- **Allowed contacts:** feet, knees, hands and forearms.
- **Goal:** standing on the box, feet on its top face, pelvis above it.

The expected sequence, which planner A is not told: hands on the box's top edge, one knee onto the top face, the
second knee, a foot, then stand up. The high variant should discover forearms on the top face before a knee can
reach it.

**3. Car egress** (`car_egress.textproto`). This is the DRC Finals egress task, with a parametric vehicle modeled on
the Polaris Ranger XP 900 used there.
- **Scene:**
  - floor pan 0.30 m above the ground;
  - seat cushion 0.45 m above the floor pan (Atlas's shins, 0.50 m from the knee to the sole, need more than the
    first estimate of 0.35 m), 0.50 m deep, a utility vehicle's bench. Atlas's hip pitch limit (-1.61 rad) keeps its
    thigh capsules sloping about 6 degrees down to the knees when it sits, so they rest on the cushion's front edge;
    milestone 1's 0.59 m cushion put the thighs 1 cm into it ahead of the thigh pads, which the collision spheres of
    section 5.4 caught;
  - seat back;
  - dashboard;
  - an A-pillar and a roof-rail capsule (radius 0.025 m);
  - an open driver's side with no door.

  All dimensions are scene parameters, to be checked against the vehicle model before the scenario is used as
  evidence.
- **Allowed contacts:** all but the two torso contacts.
- **Initial pose:** sitting, with `seat_contact`, both `thigh_*_contact` and both feet on the floor pan.
- **Goal:** both feet on the ground beside the vehicle, standing, with no contact on the vehicle.

The hard part is the transfer of weight from the seat to one foot on the ground 0.75 m below, with a hand pushing on the
pillar or the seat. The hand contacts are **push-only** in version 1, a unilateral cone like every other contact. A
grasp on the roll cage can pull, which is a bilateral wrench bounded by a box rather than a cone. It is a later
addition and changes the static equilibrium test (section 11).

The robot starts sitting, so this scenario also needs the MPC's initial state to be the planned initial state, and the
MuJoCo gantry and tilt catch (`sim_max_base_tilt_angle`) to be configured for a seated start (section 8.3).

**4. Getting up from the ground** (`get_up_supine.textproto`, `get_up_prone.textproto`).
- **Scene:** the floor only.
- **Allowed contacts:** all seventeen on the back, all but the four torso edge pads face down.
- **Initial pose:** lying on the back or face down, with the arms at the sides:
  - on the back: `torso_back_contact` and both soles flat on the floor, the knees drawn up. Atlas's collision boxes put
    the back face of the torso 9 cm below the pelvis's and 20 cm below the backs of the thighs, so neither the seat
    pad nor the thighs touch the floor when the back does, and the backs of the heels are no candidate;
  - face down: on the hands and knees, the arms straight below the shoulders (milestone 3). Atlas cannot lie flat on
    the chest face of its torso box, as milestone 1's pose did: its head sphere (r 0.10 m about a point of the neck
    joint, which sits 11.5 cm in front of the chest face) would be 16 cm into the floor, and no neck motion moves the
    joint itself; the collision spheres of section 5.4 caught it. The pose was found with the stance IK at the
    scenario's margins.
- **Goal:** standing on both feet at the nominal pose, with no other contact.

The sequences the planners should find are the ones people use. From the back: sit up onto the seat and the hands,
draw the feet in, shift the weight forward onto the feet and hands, and stand. From the front (from the hands and
knees): bring one foot forward, and stand.

Three things are new in this scenario:
- **The base starts pitched by 90 degrees.** That is the singularity of the Z-Y-X Euler angles. The state is a unit
  quaternion, so the formulation is unaffected
  ([quaternion_base_orientation](../quaternion_base_orientation/README.md)). But the Euler-written tuning and the
  Euler display in Rerun pass through the singular point. The base-orientation reference comes from the plan as a
  quaternion, and the scenario's base-orientation weights are written isotropic, so that nothing depends on the Euler
  map near the singularity.
- **The early phases have the CoM about 0.15 m above the floor.** The torque check of section 2 matters most here:
  sitting up from the back takes most of the hip flexors' effort, and the planner must find the hands' push that
  relieves them.
- **MuJoCo's tilt catch fires at once.** It catches a robot whose base tilts past `sim_max_base_tilt_angle`, which a
  lying robot does from the first step. The scenario disarms the tilt catch and the gantry until the robot stands
  (section 8.3).

### 8.2 What the dummy sim shows, and what it does not

The dummy sim's plant is the MPC's own model, rolled out under the latest policy (`DummySimLoop`). Contact in the
dummy sim is whatever the MPC's constraints enforce. It has no contact physics, so a foot on a stair is on the stair
because the plan and the constraint say so.

**What it shows:**
- that the plan is consistent with the model;
- that the MPC can track the plan with the generalized swing planner, the patch-frame cones and the planned-motion
  reference manager, solve after solve, in real time;
- that the certifier passes on the tracked trajectory, not just the planned one.

**What it does not show:** that the contacts would hold in a physics engine. That is MuJoCo's job (section 8.3).

Each dummy sim also has a **plan playback** mode that plays the plan kinematically in Rerun, with no MPC. It shows:
- the scene;
- the contact forces as arrows at every active contact;
- the static equilibrium polygon and the CoM's projection;
- the certifier's margins.

That is the first thing to look at when a planner's output is wrong.

Launch targets, generated by a new `MULTI_CONTACT_SCENARIOS` table in the Makefile, with an IFTTT pair to
`ROBOT_CONFIGURATIONS`, `robot_models/tests/test_launch_files.py` and the launch files:

```
make plan-drc-atlas-stairs                 # run the offline planner, write the plan, print the certifier report
make playback-drc-atlas-stairs             # kinematic playback in Rerun
make launch-drc-atlas-stairs-dummy-sim     # the centroidal MPC tracking the plan in the dummy sim
```

There are equivalent targets for `box-climb`, `car-egress`, `get-up-supine` and `get-up-prone`, and `-vnc` variants as
the existing targets have. The table's rows are `<target name>:<scenario>` (`box-climb:box_climb`); the dummy sim of
a scenario runs `robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/multi_contact_dummy_sim.textproto` with
`scenario` set, tracking the plan `$(MULTI_CONTACT_PLAN_DIR)/<scenario>.binpb`
(`humanoid_nmpc/humanoid_multi_contact/plans/`), where the planner's `make plan-drc-atlas-<scenario>` writes it.

### 8.3 MuJoCo, after the dummy sims

A Bazel genrule generates `atlas_<scenario>.xml` from `atlas.xml` and the scenario's `TerrainScene`. It copies the
robot and its `gantry` weld, and replaces the floor with the scene's primitives. The robot's `mjcf_file` in a scenario
launch file names the generated scene. `test_robot_scene_gantry_hold` covers the generated scenes as it covers the
shipped ones.

`CheaterSimContactEstimator` counts any contact between a contact point's body and a body outside the robot, so knees
on a box register without change. The new contact candidates need MJCF sites or a body mapping. Egress needs the
robot to start seated: an initial `qpos` from the scenario, and a gantry catch that is not armed until the robot
stands.

## 9. Tests

New code ships with tests, and a new formulation ships with parity tests.

**Unit tests:**
- **`TerrainModel`:** the signed distance, nearest point and normal of a box and a capsule against closed-form cases,
  covering faces, edges, corners and inside points. CppAD derivatives against finite differences. The smooth minimum
  at an inside stair corner.
- **Static equilibrium:**
  - On flat ground with two feet, the CoM region equals the convex hull of the foot rectangles (to the linearization of
    the cones).
  - A hand on a wall widens it in the right direction.
  - A single point contact admits equilibrium only with the CoM directly above it.
  - The margin's sign flips at the boundary.
  - Torque feasibility fails when the limits are scaled down.
- **CWC check:** a planned motion that accelerates the CoM out of the cone fails, and the same motion slowed down
  passes.
- **The patch-frame cones:** with the identity rotation, the values and derivatives equal today's terms exactly. With
  a rotated frame, they equal today's terms written in that frame.

**Parity:** the multi-contact centroidal MPC with only the two feet, a flat `TerrainScene` at height `h`, and a plan of
standing or of the shipped gait equals today's centroidal MPC with `terrain_height = h`. Every cost, constraint, value
and derivative is compared at the probe points of `test/equivalence`, the same pattern as the existing equivalence
tests.

**Planner tests:**
- Each scenario plans successfully with each planner.
- The certifier passes at every knot.
- The plan reaches the goal.
- Planner A on the box discovers a sequence that uses a knee or a forearm. This is the check that it is doing more
  than walking.

These solves take minutes, so they run in their own Bazel test targets with a timeout, in the CI batch for their
package: `//humanoid_nmpc/humanoid_multi_contact/app:<planner>_<scenario>_test`, planner A
(`contact_invariant_<scenario>_test`) and planner B on every shipped scenario and the combined planner on the scenarios
that name it, each a certified plan to the goal (planner A's over the scenario's duration). Planner A's units have
tests of their own (`humanoid_multi_contact/planner/`): every term against finite differences of its value on a toy
ball and on Atlas's multi-contact model, and the robot terms against the certifier's stance model
(`contact_invariant_terms_test`, `robot_contact_invariant_terms_test`), the thresholding (`stance_thresholding_test`),
and the continuation and the thresholding on a ball that must discover the floor to reach its goal
(`contact_invariant_toy_test`). For planner B they are `//humanoid_nmpc/humanoid_multi_contact/app:stance_graph_<scenario>_test`, one per
shipped scenario (`size = "large"`, `timeout = "eternal"`): the plan is found, every knot passes the certifier, it is
stamped with the scenario's hashes, its last knot is the goal stance on the goal's primitives with the base in the goal
region, the box plans use more than the feet and the stairs plan only the feet. Planner B's units have tests of their
own (`humanoid_multi_contact/planner/`): the lazy weighted A* on a grid with a wall (the shortest path, the weighted
bound, the budgets, an unreachable goal), the sampler on a stair and a rail, the stance IK on Atlas (holding a stance,
the CoM onto one foot and onto a foot on a step, a hand onto a box, an unreachable target, a squat that keeps the shins
off the floor, the joint limits), a step forward through `StanceMotion`, the overlap of pads, the timing law and the
registry; the collision model's spheres, validation, exemption and clearances (`collision_model_test`) and its match with
`atlas.xml` (`test_collision_model_mjcf`).

**Dummy sim:** an end-to-end test per scenario runs the MPC tracking the plan to the goal in the dummy sim's loop,
faster than real time. It requires the certifier to pass on the tracked trajectory. Its knots are the loop's samples;
the momentum rate of a sample is the forward difference of the momenta over the step the simulator took from it, and
its stance the mode schedule's in the middle of that step: the mean rate of a rollout in one mode, since the plans'
stances change at multiples of the loop's period (a central difference would average two modes at every change).
`test_multi_contact_plan_tracking` tracks the sway, the hand-written stair step and the stairs scenario's shipped plan
this way, and the stairs plan played `forward_and_reverse` for three periods (58 s), its base within 5 cm of the plan's
throughout, on the top step and back at the bottom in each period, and every tracked sample certified. It tracks the
box climb's shipped plan the same way (`climbsOntoTheBoxOfTheShippedPlan`): every sample of its 455 s certifies,
the base within 2.7 cm of the plan's, every active pad within 1.2 mm of its patch, the momentum rate in the cone
(residual 6e-10) and the static margin at least 0.034 m, with no hold of the plan's clock; and the get-up from prone's
(`getsUpFromProneOfTheShippedPlan`): every sample of its 79 s certifies, the base within 1.9 cm of the plan's, every
active pad within 2.3 mm of its patch, the momentum rate in the cone (residual 5e-10) and the static margin at least
0.042 m, with no hold. And the get-up from the back's (`getsUpFromSupineOfTheShippedPlan`): every sample of its 302 s
certifies, the base within 3.6 cm of the plan's, every active pad within 1.1 mm of its patch, the momentum rate in the
cone (residual 6e-10) and the static margin, on the refined polygon, at least 0.040 m, with no hold. Its earlier
plans failed in three ways, each fixed in the scenario (decision 11): the sampled polygon lost the corner of the
roll's stance (298 samples, the margin down to -0.045 m), the right forearm went up to 1.8 cm into the floor beside
the hand (1093 samples), and with the hips 0.1 rad past their ranges the MPC diverged. The plan before it, whose
knots alone planner B checked, failed 593 of 25284 samples: as a knee swung onto the box the reference's
interpolation between two knots a second apart swept a thigh or a shin up to 4.5 cm into the box's edge, the static
margin fell to 0.019 m on the stance of one foot and one hand, and HPIPM stopped on a few wrench cone LPs; with the
checks between knots (section 5.5) a sole that the plan's bridge to a segment's end held a millimeter above the box
for nine seconds was the last failure (29 samples, a heel 1.3 cm into the box), which ending the segment where the
sole lands removed.
`test_plan_playback` certifies the stairs cycle at every knot, with each reversed knot's momentum rate its forward
knot's, and checks the periodic lookups: both seams continuous, the reversed momenta negated, the stances a palindrome,
the reversed swings stepping down, every lookup alike periods away, and the clock hold and the schedules across the
wrap.

## 10. File layout and order of work

```
humanoid_nmpc/humanoid_multi_contact/          new package
  terrain/          terrain_scene.proto, TerrainModel (box, capsule, smooth minimum)
  contact_set/      contact_set.proto, ContactSet (the candidates of section 3.1)
  stability/        StaticEquilibrium, TorqueFeasibility, ContactWrenchConeCheck, StabilityCertifier
  planner/          MultiContactPlanner, the registry, ContactInvariantPlanner, StanceGraphPlanner
  plan/             multi_contact_plan.proto, PlanFile
  scenarios/        scenario.proto, stairs, stairs_steep, box_climb, box_climb_high, car_egress, get_up_supine,
                    get_up_prone
  app/              certify_plan (certifies a plan, writes it with its report), plan_playback (Rerun),
                    plan_multi_contact (offline planner CLI, with the planners)
humanoid_nmpc/humanoid_centroidal_mpc/         runtime contact set, PlannedMotionReferenceManager
humanoid_nmpc/humanoid_common_mpc/             patch-frame cones, generalized swing planner, collision spheres
robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/   contact set and task file for Atlas
```

**Milestones**, each landing with its tests:
1. Terrain model, contact set, scenarios, the stability certifier, and plan playback of a hand-written plan. No MPC
   changes. **Landed:** `humanoid_nmpc/humanoid_multi_contact/` with `terrain/`, `contact_set/`, `stability/`,
   `plan/`, `scenarios/` and `app/` (`certify_plan`, `plan_playback`, the hand-written plan
   `app/testdata/box_climb_sway.textproto`).
2. The runtime contact set and the patch-frame cones in the centroidal MPC, with the parity test against today's MPC.
3. Planner B (stance graph). Every scenario is to have a certified plan. **Landed:** `humanoid_multi_contact/collision/`
   (section 5.4, with the certifier's sphere clearance), `humanoid_multi_contact/planner/` (section 5.5),
   `app:plan_multi_contact` and `make plan-drc-atlas-<scenario>` / `make playback-drc-atlas-<scenario>` for every row of
   `MULTI_CONTACT_SCENARIOS`. Only the stairs plan so far (section 5.5, "Status"); the other scenarios' planning tests
   are tagged `manual`.
4. The planned-motion reference manager and the generalized swing planner. The stairs dummy sim tracks its plan.
   **Landed:** `humanoid_centroidal_mpc/multi_contact/` (`PlanContactOrder`, `PlannedMotion`, `PlanClock`,
   `PlannedMotionReferenceManager`), `humanoid_common_mpc/contact_swing/` and the `contact_swing` and
   `terrain_non_penetration` terms; the multi-contact task file of the Atlas
   (`robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/task.textproto`: the
   seventeen candidates, 6-DoF pads of their own frames through `contacts.surface_pads`, the wrists unlocked), checked against
   the contact set; the launch file `multi_contact_dummy_sim.textproto` and `make launch-drc-atlas-<scenario>-dummy-sim`
   for every scenario, whose Rerun bridge draws the terrain (`--multi_contact_scenario`) and the contact force of every
   contact. `test_multi_contact_plan_tracking` runs the dummy simulator's loop faster than real time on milestone 1's
   hand-written sway and on a hand-written feet-only step onto a 0.10 m stair (`test/support/StairStepPlan.h`), to the
   goal, with the certifier passing on the tracked trajectory. The stairs scenario's own plan is planner B's (milestone
   3): `make launch-drc-atlas-stairs-dummy-sim` tracks `humanoid_nmpc/humanoid_multi_contact/plans/stairs.binpb`.
5. Planner A (contact-invariant optimization), alone and initialized from planner B. The box, egress and get-up
   dummy sims. **Landed:** `humanoid_multi_contact/planner/contact_invariant/` and the planners `contact_invariant`
   and `stance_graph_then_contact_invariant` (section 4.4); `make plan-drc-atlas-<scenario>` runs the scenario's own
   planner (`MULTI_CONTACT_PLANNER=` overrides it); the stairs scenario names the combined planner, whose plan,
   `plans/stairs.binpb`, `test_multi_contact_plan_tracking` tracks to the goal in the dummy sim's loop with the
   certifier passing on the tracked trajectory, after the tracking task file's landing alignment
   (`SwingLandingAlignmentConstraint`, section 7.2) and settings. Planner A does not plan the box, egress and get-up scenarios yet
   (section 4.4, "Status"); their tests are tagged `manual`.
6. Generated MuJoCo scenes, and the closed loop in MuJoCo.

Planner B comes before planner A in this order because it is simpler to validate, and it gives milestone 4 a plan to
track. The design does not depend on the order.

## 11. Decisions

Settled by the owner (2026-10-25):

1. **The contact set is made runtime in place** (section 7.2): one centroidal formulation, the biped-only terms gated
   by name, and the parity test in the same milestone.
2. **Atlas's wrists are unlocked** in the multi-contact task file, so that the hands can face a surface.
3. **Hand contacts are push-only** in version 1. A bilateral grasp on the roll cage is a later addition.
4. **The solver is OCS2's SQP**, for the planner as for the MPC. No new NLP dependency.
5. **Static stability is relaxable per scenario.** Condition 2 of section 2 (dynamic feasibility) is always required.
   Condition 1 is enforced when a scenario sets `static_margin`, and skipped when it leaves the field unset. The
   shipped stairs scenario leaves it unset, because walking up stairs is a dynamic gait. The box, egress and get-up
   scenarios set it to 0.03 m: a slip on the box or the vehicle is a fall from a height, and getting up is slow by
   nature. Condition 3 applies only with condition 1.

Settled by the owner (2026-10-26):

6. **The dummy sims' plans disregard the joints' effort and velocity limits.** Every shipped scenario sets
   `joint_limits: "position_only"`: its plan respects the joint position limits, the friction and contact wrench cones
   (condition 2, always), collision avoidance and its static margin, but not the URDF's effort limits. The DRC Atlas's
   arms (elbow 112 N m, shoulder 87-99 N m, `ely` 63 N m) cannot carry its body, which blocked every arm-assisted
   climb, and the dummy sims show contact sequences, not hardware. The certifier still reports the torque utilization
   at every knot. The enforcing policy, `"position_and_effort"`, stays the scenario's default for the hardware work: it
   restores the stance graph's torque check, solve 3's `QuasiStaticTorqueConstraint` and the certifier's torque
   condition with `torque_margin`. No planner reads the URDF's joint velocity limits (6.3-100 rad/s on the Atlas) under
   either policy: the timing law moves every joint at most 0.5 rad/s, and planner A's solves carry no velocity
   limit.
7. **The stairs dummy sim walks up and backward down, over and over** (section 7.3). The descent is the ascent's plan
   in reverse time, not a second plan: reversing time keeps every configuration and momentum rate, so the descent
   certifies wherever the ascent does, and the robot keeps facing up the stairs. The loop is a playback of the tracked
   plan, `multi_contact_plan.playback`, selected by name and off by default (`once`), not a longer plan file: it runs
   forever, with a dwell of `playback_dwell` at the top and at the bottom. Only the stairs launch selects
   `forward_and_reverse`.

Settled by the owner (2026-10-27):

8. **A dummy sim's plan may bend a joint a little past its range.** `joint_limits: "relaxed_position"` widens the
   URDF's position limits by the scenario's `joint_limit_relaxation` on both sides for the stance graph planner's
   configurations, a name beside the two of decision 6, the enforcing ones staying the default. The box climb uses
   0.1 rad: setting the second foot onto the box from a kneel, with the CoM inside the polygon of the first foot and
   the hands, takes the hips, the knees and the back's pitch up to 0.1 rad past their range; within the range the
   search did not find that change in 20 minutes. The plan that also keeps clear of the terrain between its knots
   uses the margin more widely (r_leg_hpy, l_arm_shz and back_bky 0.100, r_arm_shz 0.059). The get-up from
   prone uses 0.1 rad too (back_bky 0.099, l_leg_aky 0.054): its squat on the hands folds the back,
   the ankles and the hips to the ends of their ranges. The get-up from the back uses 0.05 rad (both hips' pitch,
   r_leg_aky, l_arm_shz and back_bky 0.050): at 0.1 rad its plans held both hips at the widened limit with a foot in
   the air, where the MPC diverged (decision 11).
   The certifier does not judge the position limits, and the tracking MPC keeps its own joint-limit barrier at the
   URDF's range, which the tracked plan touches only there. Planner A's solves read the MPC's limits under every name.
9. **What planner B plans alone is checked between its knots as the MPC tracks it** (2026-10-27). The knots of a
   quasi-static plan are a second or more apart, and what the MPC follows between them is its reference's
   interpolation (and, for a long swing, the straight path of the pad), not the IK's: the box climb's knots all
   certified while the tracked robot swept a thigh 4.5 cm through the box's edge. The sweeps are checked at the
   knots' resolution of the MPC (section 5.5), within 2.5 mm, half the certifier's tolerance, at the knots too:
   kneeling on a flat top keeps the end of the thigh a few millimeters in the box beside the knee, and 0 mm left no
   kneel. Only planner B alone checks them; as planner A's seed it keeps the knots-only checks, which solve 3
   re-solves, so the stairs plan bit for bit as before.
10. **The end of the thigh at a kneeling knee is named per scenario** (2026-10-28). `point_pad_link_above: "exempt"`
    (the default) keeps the box climb's kneels on the box's edge as they were planned and tracked; `"rests"` lets the
    sphere lie 6 mm in the terrain and no deeper, which the get-up from prone's kneels on the flat floor need (exempt,
    a lifting knee swept the thigh 9 mm through the floor between two knots). The get-up from the back rests too.
11. **The get-up from the back rolls over on the torso's edge** (2026-10-28). The contact set had no contact on the
    body's side, and from the crab on the feet and the hands no quasi-static way led onto the feet within 0.3 rad past
    the joint ranges. Of the two ways on, a side contact or a start that is not flat on the back, the user chose the
    side contact: four point pads on the back-right edge of the torso box (section 3.1), allowed only in
    `get_up_supine`, whose plan rolls onto them, puts the right foot and hand down and comes up over the feet and the
    hands (section 5.5). The MuJoCo foot model stays four corner spheres, and the planner keeps its usual 2 cm beyond
    the certified 0.03 m margin: a buffer of 5 mm found a plan, which the MPC then tracked below the margin. Three
    settings of that scenario alone made its plan track: `support_polygon: "refined"` (section 2), `sphere_clearance:
    0.02` (the MPC, with no collision spheres on its links, tracked the forearm beside a hand on the floor up to 2 cm
    below the plan's 1 cm) and `joint_limit_relaxation: 0.05` (the MPC, whose joint limits are a soft barrier at the
    URDF's ranges, diverged where the plan held both hips 0.1 rad past them with a foot in the air).
    The four pads are candidates of the one Atlas contact set and its one multi-contact task file, which every
    scenario shares. That changed the other plans, not only their hashes: planner B caps the speed of every
    candidate's pad and planner A solves on the MPC model of all seventeen, so the stairs, the box climb and the
    get-up from the front were planned again and their tracking checked again (the user's choice, 2026-10-29, over a
    contact set and a task file of the get-up from the back's own).

## References

- I. Mordatch, E. Todorov, Z. Popović, *Discovery of Complex Behaviors through Contact-Invariant Optimization*, ACM
  Transactions on Graphics (SIGGRAPH) 31(4), 2012.
- K. Hauser, T. Bretl, J.-C. Latombe, *Non-Gaited Humanoid Locomotion Planning*, IEEE-RAS Humanoids, 2005.
- T. Bretl, S. Lall, *Testing Static Equilibrium for Legged Robots*, IEEE Transactions on Robotics 24(4), 2008.
- S. Caron, Q.-C. Pham, Y. Nakamura, *Leveraging Cone Double Description for Multi-Contact Stability of Humanoids with
  Applications to Statics and Dynamics*, Robotics: Science and Systems, 2015.
- H. Dai, A. Valenzuela, R. Tedrake, *Whole-Body Motion Planning with Centroidal Dynamics and Full Kinematics*, IEEE-RAS
  Humanoids, 2014.
- R. Deits, R. Tedrake, *Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization*, IEEE-RAS
  Humanoids, 2014.
- B. Aceituno-Cabezas et al., *Simultaneous Contact, Gait, and Motion Planning for Robust Multilegged Locomotion via
  Mixed-Integer Convex Optimization*, IEEE Robotics and Automation Letters 3(3), 2018.
- A. W. Winkler, C. D. Bellicoso, M. Hutter, J. Buchli, *Gait and Trajectory Optimization for Legged Systems Through
  Phase-Based End-Effector Parameterization*, IEEE Robotics and Automation Letters 3(3), 2018.
