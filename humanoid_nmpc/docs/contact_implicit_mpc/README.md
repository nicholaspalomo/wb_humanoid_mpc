# Contact-Implicit Whole-Body MPC

The second half of

> S. A. Esteban, V. Kurtz, A. B. Ghansah, A. D. Ames,
> *Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion*,
> [arXiv:2502.15630](https://arxiv.org/abs/2502.15630),

adapted to this repository's centroidal NMPC. The first half, the reduced-order planner that proposes the nominal
gait, is [hlip_contact_planner](../hlip_contact_planner/README.md).

**Off by default.** It changes the closed loop and has not been validated in simulation on hardware-like conditions.
Section 5 says how to switch it on.

---

## 1. Why the paper's construction does not transfer verbatim

The paper's CI-MPC is *Inverse Dynamics Trajectory Optimization*: the generalized positions `q_0 … q_N` are the only
decision variables, velocities and accelerations come from finite differences, and the contact forces are a smooth
function of the configuration under a compliant contact model,

```
lambda_k(q) = func(q_{k+1}, v_{k+1}(q)),
B u(q) = D(q) v_dot + H(q, v) - J_c(q)^T lambda_k(q).
```

Contact is implicit because `lambda` is not a variable at all: it is whatever the compliant model says the
configuration produces, so a trajectory that puts a foot on the ground *is* a trajectory with contact force, and the
optimizer moves between contact modes by moving `q`.

This repository's MPC is an OCS2 SQP over a centroidal model whose **input vector carries the contact wrenches**
(or the basis scalings of `BasisInputsModelDecorator`). Making `lambda` a function of `q` would mean replacing the
dynamics, the solver and the whole-body cost structure — a different controller, not a change to this one.

## 2. What is implemented instead

The equivalent freedom is obtained from the other direction: stop letting the mode schedule decide which foot carries
load, and ask the optimizer for the **complementarity conditions of rigid contact** directly. For each foot, with
`f_n` its normal contact force, `h` the **gap** -- the height of the lowest point of the footprint above the terrain --
and `v_xy` the tangential velocity of the contact frame:

```
f_n >= 0                     friction cone, or the non-negativity of the basis scalings -- but see section 2a
h_i >= 0 for every corner i  GroundPenetrationConstraint      (one-sided squared hinge, one row per footprint corner)
f_n h   = 0                  ContactComplementarityConstraint (quadratic penalty, h = smooth min of the corner heights)
f_n [v_x, v_y, omega_z] = 0  ForceWeightedSlipConstraint      (quadratic penalty)
```

## 2a. The first condition is not free: the cones have to be un-gated

`f_n >= 0` was listed above as "already enforced", and that was wrong. Every contact cone in this repository switched
itself off while the mode schedule called a foot a swing foot -- `ContactWrenchConeConstraint`,
`FrictionForceConeConstraint`, `ContactMomentXYConstraintCppAd` and `BasisScalingNonNegativityConstraint` all gated
`isActive()` on `getContactFlags(time)`. That gate was sound for exactly one reason: the hard `zero_wrench` constraint
had already pinned the swinging foot's wrench to zero, so there was nothing left for a cone to bound. This formulation
removes `zero_wrench`, and the gate then leaves a foot the schedule calls a swing foot with an **unbounded** wrench:
adhesion, unlimited friction, a centre of pressure anywhere. On a robot running `useContactBasisVectorInputs: true`
it is worse still, because the explicit wrench cone is skipped in favour of the structural guarantee that `lambda >= 0`
gives, so the non-negativity barrier is the only bound there is -- and it was gated too. A negative scaling is an
adhesive, outside-the-cone wrench, and the complementarity product is sign-blind and does not object to it.

So the gate now follows `zero_wrench` rather than the schedule: `contactConstraintsAreScheduleGated(formulationTasks)`
is true exactly when `zero_wrench` is listed, and the four terms are built with it. That is the narrower and the
stronger condition -- it is also correct for a task file that drops `zero_wrench` without listing the three terms
below, which the loader permits.

### Un-gating a cone that is not there enforces nothing

`f_n >= 0` is the **first** of the three conditions, and the other two do not imply it: `ground_penetration` is a
statement about height alone, and `f_n h = 0` is satisfied at `h = 0` by **any** `f_n`, a negative one included. So a
foot resting on the floor could pull on it, without limit, and nothing in the formulation would object.

The bound is the cone's, and the cones stop bounding anything the moment `zero_wrench` goes. `loadMpcFormulationTasks`
therefore refuses a task file that drops `zero_wrench` and lists neither `contact_wrench_cone` nor
`friction_force_cone`. Either will do at that level -- the first carries the friction, centre-of-pressure and torsional
rows together, the second carries `mu Fz - |F_xy| >= 0`, which bounds the normal force below on its own. The rule is
keyed off `zero_wrench` rather than off the three terms, because a task file that drops the pin without listing them
has the same hole.

On `useContactBasisVectorInputs: true` that is not enough, and `CentroidalMpcInterface` narrows it to
`contact_wrench_cone` specifically: that is the branch which builds `BasisScalingNonNegativityConstraint`, and
`lambda >= 0` is the whole of the cone there. A friction cone bounds the *assembled* wrench and says nothing about the
individual scalings, so the centre-of-pressure and torsional limits would go unenforced and a negative scaling would
still be free.

One caveat that is a tuning decision rather than a rule, and is therefore reported rather than patched: un-gated,
`contacts.basisNonNegativityBarrier.mu` is the **only** thing holding every contact wrench inside its cone, at every
node. It ships at `0.01`, which was tuned while the term was a redundant regulariser -- the swing foot's scalings were
pinned by `zero_wrench` and the stance foot's pulled positive by `R`. The same job is done by
`contactWrenchConeSoftConstraint.mu = 0.2` in the wrench parameterization. Note in particular that a scaling pair
`(+a, -a)` costs only `mu a^2` and leaves the load indicator `f_n = sum(lambda)` at zero, so **both** contact-implicit
products stay blind to it. `CentroidalMpcInterface` logs a warning when the un-gated barrier is the softer of the two;
tune it against a foot in flight before trusting the formulation on hardware.

### The friction cone has to linearise through the parameterization in use

`FrictionForceConeConstraint` wrote its input Jacobian as a fixed `3x3` block at `getContactForceStartIndices()`, which
assumes the input stores the three force components there. That is true of the wrench-space model and false under
`BasisInputsModelDecorator`, where the same index begins an eleven-wide block of scalings and the force is
`B_local lambda`: the block landed on the first three scalings and was wrong in every entry, while `getValue()` stayed
correct throughout -- which is exactly why nothing caught it. The term now probes the model once at construction
(`contactForceInputJacobian()` in `contact/ContactInputJacobian.h`, which also backs `normalContactForceRow()`) and
linearises through the result, so `friction_force_cone` is safe to list in either parameterization.

Un-gating is not quite enough on its own, because an always-active cone is evaluated on a foot at **zero wrench**, and
two of the three were not satisfied there:

* `ContactWrenchConeConstraint` carries `minNormalForce` and `mu * gripperForce` in its constant column. Both are
  statements about a foot the schedule has declared loaded -- the first demands a normal force a foot in flight cannot
  produce. They are dropped when the term is un-gated, leaving the homogeneous cone, which the zero wrench satisfies
  exactly and which is the same set `ContactWrenchConeBasisMatrix` verifies its generators against.
* `FrictionForceConeConstraint` reads `-sqrt(regularization)` at zero force -- `-5` with the default -- because the
  regularization that keeps its gradient finite also buys a parabolic safety margin. The constant is added back when
  the term is un-gated, which moves the cone's zero onto the zero force without touching the gradient or the Hessian.
* the four centre-of-pressure rows are already homogeneous and need no change.

All three do need a different **penalty**, though. A relaxed log barrier never reaches zero: at zero slack its
derivative is `-2 mu / delta`, which for the shipped centre-of-pressure settings is `-40` per row. On a cone that is
tight at the origin that pays the solver to leave the origin -- i.e. to invent a normal force on a foot in the air,
which is exactly the force floor that dropping the affine offsets removes. Un-gated, the three are wrapped in a
squared hinge with `delta = 0` instead, which is zero in value **and** in gradient at zero slack and quadratic below
it.

### What `f_n` actually is

`normalContactForceRow()` reads the normal component of the model's own contact-force parameterization, and which
frame that is depends on the model: `MpcRobotModelBase::getContactForce()` returns the **world-frame** wrench for the
wrench-space models and the **local contact-frame** wrench under `BasisInputsModelDecorator`. Both are correct here,
because these terms use `f_n` as a **load indicator** -- a number that vanishes exactly when the foot carries no
wrench -- rather than as a physical force. Under the basis decorator the indicator is exact: every generator of
`ContactWrenchConeBasisMatrix` is built with a local normal force of exactly 1, so `f_n` is the sum of the foot's
scalings and vanishes if and only if every one of them does. The constructor CHECKs that the row is non-negative and
non-zero, which is the property a new input parameterization could otherwise break silently.

The slip term constrains **three** of the six components of the foot's twist: the two tangential linear velocities and
the spin about the contact normal. The pivot row matters as much as the two linear ones — the `zero_velocity`
constraint this replaces was a full six-row twist constraint with `constrainOrientation: true`, and a loaded foot left
free to yaw walks the robot sideways out from under itself. The normal velocity and the two rocking rates are left
free on purpose: the first because forbidding it would forbid lift-off under load and reintroduce the scheduling this
formulation removes, and the second because rolling the foot about its heel and toe edges under load is exactly how a
heel-to-toe strike happens (the paper's Fig. 3).

Penalised rather than imposed — hence *relaxed* complementarity, the standard smoothing of contact for a solver that
linearises. The consequences are the ones the paper's architecture needs:

* a foot may carry load **wherever it touches the ground**, whatever phase the nominal gait says it is in, so an
  early touch-down, a late lift-off, or simply keeping a foot down are all available to the solver;
* a foot that carries load is **held still** by the third condition, which is what the stance constraint used to do,
  except that now the condition is keyed off the load the solver chose and not off the schedule;
* the reduced-order planner's contact sequence survives only as the **reference of the swing-foot cost** — a
  suggestion, exactly as in the paper.

All three terms are bilinear in the force and the foot kinematics, so their linear approximations are assembled in
closed form from the end-effector kinematics and the constant row that maps the input to the normal force
(`normalContactForceRow`). No automatic differentiation and no code generation is added by this formulation.

The normal direction is the sole's, the third component of the model's contact force. On the flat terrain the
reduced-order planner assumes that is the world vertical, and on a tilted foot it is the physically correct normal.

## 3. What it replaces

| Removed | Was |
| --- | --- |
| `zero_wrench` (hard) | forced the swinging foot's wrench to zero, gated by the mode schedule |
| `zero_velocity` (hard or soft) | held the stance foot still (all six twist rows), gated by the mode schedule |
| `normal_velocity` (hard) | forced the swing foot's vertical velocity onto the swing trajectory's reference |

`loadMpcFormulationTasks` refuses the half-way combinations: `contact_complementarity` with `zero_wrench`,
`force_weighted_slip` with either `zero_velocity`, `contact_complementarity` without `force_weighted_slip` or without
`ground_penetration` (the complementarity product is also satisfied by a *negative* height, so the unilateral condition
has to be there), any of the three with the hard `normal_velocity`, and -- see section 2a -- a missing `zero_wrench`
with neither cone listed.

`normal_velocity` has to go **from `hard_constraints`**, and an earlier version of this document was wrong to leave it
listed there as a mere shaping term. It is a **hard equality** on the contact frame's vertical velocity for every foot the schedule
calls a swing foot, so -- the foot's height at lift-off being given -- it fixes the entire height profile of the swing.
The solver can then neither land early, nor land late, nor keep a foot down, which are the three freedoms this
formulation exists to provide; and where it does want load early, the only variable it has left is the normal force, so
the complementarity product gets driven to zero by removing the force rather than by closing the gap. That is the
formulation backwards.

It must be **listed in `soft_constraints` in its place**, though, and that half is not optional either. Deleting the
row outright was tried, and the robot shuffled its feet along the ground without lifting them. The reason is worth
stating precisely, because it is not where one would look:

* the only vertical term left is `task_space_foot_cost_weights.pos_z`, at 150;
* against it stand the leg-joint entries of `Q`, whose reference posture is the standing crouch of
  `reference.yaml`'s `defaultJointState` - and `defaultBaseHeight` equals the pelvis-to-sole distance at that posture
  to five decimals, so the posture prior is literally *the foot on the floor*. Every millimetre of lift is charged
  against `Q` at scaling 85, and the `orientation_x/y` weights of 800 add the ankle's share by insisting the sole stay
  flat. The effective vertical stiffness at the sole is of order 1e4, so the fight is about 100:1 and the static
  equilibrium swing height is a couple of millimetres;
* and the same leg at that crouch is roughly **eighty times more compliant horizontally than vertically**, so the
  cheapest way to serve a forward command is to slide the foot rather than lift it. The shuffle is not a tuning
  accident; it is what that stiffness ratio asks for.

None of this bit while `normal_velocity` was a hard equality, because an equality has infinite weight and `Q` never got
a vote on swing height. As a soft constraint the same row - `v_z - zdot_ref - positionErrorGain_z (z_ref - z)`, weight
`model_settings.foot_constraint.normalVelocitySoftConstraintWeight` - shapes the swing and is overruled whenever
anything else pays more, which is exactly what a schedule-derived reference should be.

One weight rather than two, deliberately: the hard row combines a velocity feedforward and a position feedback in a
fixed ratio, and the foot cost cannot reproduce that at ANY pair of `pos_z` / `lin_velocity_z`, because only its
velocity rows are faded by the impact-proximity factor while its position rows are not. Splitting the servo across two
differently-faded channels gives a different controller at every point of the swing.

### A soft schedule-derived reference is not a schedule gate

There is a real distinction here, and getting it wrong stops the robot walking. Three different things read the mode
schedule, and only two of them may be removed:

* a schedule-derived **hard constraint** (`zero_wrench`, `zero_velocity`, `normal_velocity`) makes departing from the
  plan impossible, and must go;
* a schedule-derived **gate on a bound** (the cones above) switches off a statement that is true everywhere, and must
  go;
* a schedule-derived **soft reference** is how the reduced-order plan guides the whole-body MPC in the first place, and
  must STAY. The solver overrules it whenever anything else pays more, which is exactly the freedom the formulation is
  for.

`StateInputQuadraticCost` is the third kind. It regularises the input towards `weightCompensatingInput(contactFlags)`,
which gives the whole body weight to the schedule's stance feet and none to the swinging one. That looks like a soft
`zero_wrench`, and removing it along with the hard one was tried. It does not work, and the failure is worth recording:
spreading the weight over all feet makes the nominal force on the foot that should be in the air half the body weight,
so R pulls it there, and the complementarity penalty's curvature on that foot's height is
`complementarityWeight (f_n/f_ref)^2 / heightReference^2` - about 1950 at half body weight, against the 150 of
`task_space_foot_cost_weights.pos_z` trying to lift it. **The foot rises about six millimetres and stops.** With the
schedule-derived nominal restored, R pulls the swing foot's force towards zero, the complementarity term goes quiet as
it does, and the swing height reference is free to lift the foot. That nudge is the only asymmetry in an otherwise
symmetric contact problem: without it the solver has no reason to prefer lifting one foot over the other, or over
neither.

One more place did keep deciding contact from the schedule, and is fixed:

* `GroundPenetrationConstraint` was evaluated at the **centre of the sole** only. This formulation deliberately leaves
  the foot's rocking rates free, so a foot pitched about a centre held at ground level buries its toe for nothing -- on
  the DRC Atlas the footprint corners are 0.12 m fore and aft, and the shipped 0.08 rad of swing pitch alone is about
  10 mm. It is now evaluated at every corner of the contact polygon, whose frames `createPinocchioModel()` already
  adds, so this costs no new mathematics.

## 3a. The two terms have to measure the same foot

Moving the hinge to the corners and leaving the product on the sole centre created a second, quieter disagreement: the
two terms are the two halves of one condition -- no load above the ground, no foot below it -- and they were reading
different geometry. A foot rocked onto its heel had a positive **centre** height while carrying the whole robot, so the
complementarity product charged full price for a contact that physically existed, and the only way for the solver to pay
less was to take the load off a foot that was genuinely on the floor. Since this formulation deliberately leaves the
rocking rates free, that is the ordinary case and not an exceptional one.

Both terms now share one `FootprintCornerHeights`, so they cannot disagree again. `h` in the product is the height of the
**lowest** corner. The exact minimum is not differentiable precisely at the flat-footed stance where the robot spends
most of its time, so it is blended over `contact_implicit.gapSmoothing` (1 mm):

```
softmin(h) = m - gapSmoothing * log( (1/N) sum_i exp(-(h_i - m) / gapSmoothing) ),   m = min_i h_i
```

The `1/N` is the part that matters, and getting it wrong is worse than not smoothing at all. Without it this is the
standard log-sum-exp softmin, which at a **flat** foot returns `m - log(N) * gapSmoothing` = `m - 1.39 mm`: it
under-reports the gap, and does so worst in the common case. The complementarity penalty is two-sided, so a negative
reported gap is minimised by pushing the foot **up** until the reported gap reaches zero -- a permanent 1.39 mm hover
under full load, with nothing to oppose it, because the penetration hinge is identically zero above the ground and
`task_space_foot_cost_weights.activeInStance` is `false`. That is the same disease as the log barrier section 6 removed.

Normalised by `N` the bound is exact in value and in gradient at a flat foot, never falls below the true minimum, and
errs only in the safe direction. The bias is `gapSmoothing * log(N / k)`, where `k` is how many corners sit at the
minimum -- not a single number, which is easy to get wrong:

| corners down | when | reported gap minus true | equilibrium penetration at full load |
|---|---|---|---|
| 4 | flat foot -- most of stance | 0 (exact) | 0 |
| 2 | an edge: the ordinary heel strike or toe-off | `s log 2` = 0.69 mm | 0.094 mm |
| 1 | a single corner: needs pitch *and* roll at once | `s log 4` = 1.39 mm | 0.187 mm |

The equilibrium column balances the complementarity gradient `complementarityWeight * (f_n/f_ref)^2 / heightReference^2`
= 7812 at full body weight against `penetrationWeight` = 5e4, so the bias is attenuated about 7.4x. Both numbers are
below the compliance of any real sole.

`FootprintCornerHeights` also exists for a build reason. The obvious implementation, a
`PinocchioEndEffectorKinematicsCppAd` over the corner frames, generates **six** code-generated models per frame set --
position, velocity, orientation, orientation error, orientation against a plane, angular velocity -- of which only the
first is ever read. Over a four-corner set per foot that is five large dead models each, enough to push a test past its
900 s timeout on a clean tree. One tape of `N` heights replaces them.

## 4. The three weights

Both products are **normalised before they are penalised**. The residuals the solver sees are

```
complementarity:  g = (f_n / f_ref) (h / h_ref)
slip:             g = (f_n / f_ref) [v_x / v_ref, v_y / v_ref, omega_z / w_ref]
```

with `f_ref` the robot's weight, taken from the model rather than configured so it cannot drift from the URDF.

This is not cosmetic. The penalty around each term is quadratic, so the curvature it puts on the foot height is

```
d2/dh2 [ w g^2 / 2 ]  =  w f_n^2 / (f_ref h_ref)^2,
```

proportional to `f_n^2`. Un-normalised, on a 160 kg robot that factor ranges over nine orders of magnitude between a
foot in flight and a foot carrying the whole body, and **no single weight is right at both ends**. Chosen for the
loaded foot it is negligible in flight; chosen for the flight foot it dwarfs every other term in the problem — in
particular the swing height reference. And since `f_n h = 0` is satisfied just as well by pressing the foot down as by
taking the force off it, an oversized weight buys its reduction by **landing the foot early**. That failure mode is
not a tuning accident, it is what the un-normalised term asks for.

Normalisation also repairs a second defect in the slip term: two of its rows are linear velocities in m/s and the
third is a yaw rate in rad/s, and squaring them under one weight declared one rad/s to be exactly as bad as one m/s —
a statement about SI units, not about the robot. Each row now carries a reference in its own units.

`config/mpc/task.yaml`, block `contact_implicit`:

| Key | Meaning |
| --- | --- |
| `complementarityWeight` | cost of a foot at `heightReference` carrying full body weight |
| `slipWeight` | cost of a foot sliding at `velocityReference` under full body weight |
| `heightReference` | [m] normally `swing_trajectory_config.swingHeight` |
| `velocityReference`, `angularVelocityReference` | [m/s], [rad/s] a slide and a pivot that would already be failures |
| `penetrationWeight` | the one-sided quadratic hinge on `h >= 0`: cost `w h^2 / 2` below the ground, exactly zero on or above it |
| `gapSmoothing` | [m] the length the footprint corners' minimum height is blended over; see section 3a |

Where the ground is is NOT in this block: it is the top-level `terrainHeight` of the task file, shared with the swing
trajectories and the landing targets. There used to be two answers -- this block had its own key, while
`SwitchedModelReferenceManager::adaptToCurrentGroundHeight()` computed an estimate from the stance feet and then
discarded it on the next line, silently returning 0. They agreed only because both were pinned to the same constant.

The two residuals are dimensionless and O(1) at their worst case, so `complementarityWeight` and `slipWeight` are
comparable **with each other**. They are NOT comparable with `task_space_foot_cost_weights`, and an earlier version of
this document said they were. That claim was wrong, and wrong in the direction that hides the problem.

`pos_z` multiplies a residual in **metres**, not a normalised one. Its worst case over a swing is the apex,
`heightReference` = 0.08 m, so the cost it can ever charge is

```
0.5 * pos_z * heightReference^2  =  0.5 * 150 * 0.0064  =  0.48
```

against the complementarity term's `0.5 * complementarityWeight * 1 = 25` for a foot at that height carrying full body
weight. Comparing the raw numbers 150 and 50 suggests the swing reference wins three to one; in cost it loses fifty to
one. The factor between them is `heightReference^2 = 0.0064`, i.e. **156x**, and it is pure unit mismatch.

Two consequences follow, and they are the ones to tune against:

* **the break-even load.** Holding the foot at the apex costs `0.5 * w_c * (f_n/f_ref)^2`; planting it costs
  `0.5 * pos_z * heightReference^2`. They cross at `f_n/f_ref = heightReference * sqrt(pos_z / w_c)`, which for the
  shipped values is **14% of body weight**. Leak more than that onto the swing foot and landing early is simply the
  cheaper option.
* **the curvature on foot height.** The complementarity term contributes
  `complementarityWeight * (f_n/f_ref)^2 / heightReference^2`, which is 7800 at full body weight but only 150 - level
  with `pos_z` - at that same 14%. Quoting the 7800 figure without the load fraction overstates the effect at any
  realistic operating point.

Neither of these is the largest vertical term in the problem, though. See the paragraphs on `normal_velocity` above:
the leg-joint entries of `Q` regularise the swing leg towards a posture whose reference is the foot on the floor, at an
effective vertical stiffness of order 1e4. The honest ordering of what holds a swing foot down is `Q` first,
complementarity second, and `pos_z` is not in the fight at 150.

### Why the penetration term is a hinge and not a barrier

A relaxed log barrier is the wrong object for a unilateral condition whose solution lies exactly **on** the boundary,
which `h >= 0` is for every foot that is carrying the robot. It never reaches zero: with the values that shipped here
(`mu = 0.1`, `delta = 0.01`) its derivative at `h = 0` is `-2 mu / delta = -20`, constant and upward. The only term
pulling a foot back down was the complementarity penalty, whose gradient in `h` is
`complementarityWeight (f_n/f_ref)^2 h / heightReference^2`. Balancing the two gives

```
h = 2.6 mm   at full body weight,     h = 1.0 cm   at half body weight
```

so throughout double support the MPC's own solution stood a centimetre off the floor, carrying a permanent
complementarity residual that no weight could tune away -- and the four signals the tuning order below asks you to log
would have shown it as an unexplained floor on `f_n h`. The hinge is zero in value and in gradient at `h = 0`, so it
has no such equilibrium: it does nothing at all until the foot is actually below ground.

### Tuning order

1. **Check `terrainHeight` first.** Log `h` for a foot in stance. If it is not within a few millimetres of zero, the
   contact frame sits off the sole and every stance foot carries a permanent residual that no weight can fix — set
   `terrainHeight` to that offset instead.
2. **Penetration.** `penetrationWeight = 5e4`: a 1 cm penetration then has a restoring gradient of 500, decisively
   above anything that could push a foot down there, while its curvature is the same order as the complementarity
   term's own (`complementarityWeight / heightReference^2 = 7.8e3`), so it does not wreck the QP's conditioning.
3. **Complementarity.** Watch the normal force on a foot in flight. Raise the weight until it is negligible; if the
   foot starts landing early instead, you have passed `pos_z` and should raise `pos_z` rather than the weight.
4. **Slip, last.** Watch the tangential velocity of a loaded foot.

The four signals worth logging are `h`, `f_n`, `f_n h` and `f_n v_xy` per foot. They separate the two failure modes —
force in flight versus early touchdown — without guesswork.

## 5. Switching it on

In the robot's `config/mpc/task.yaml`:

```yaml
hard_constraints: []                  # zero_wrench, zero_velocity AND normal_velocity all removed from HERE

soft_constraints:
  - normal_velocity                   # ...and normal_velocity re-listed HERE, as a cost
  - joint_limits
  - foot_collision
  - contact_wrench_cone               # REQUIRED once zero_wrench is gone; see section 2a
  - contact_complementarity
  - force_weighted_slip
  - ground_penetration
```

`contact_wrench_cone` is not optional here. With `zero_wrench` gone it is what supplies `f_n >= 0`, and on a robot
running `useContactBasisVectorInputs: true` it is the only key that will do -- the loader accepts `friction_force_cone`
in its place, but `CentroidalMpcInterface` does not.

Then validate in MuJoCo before hardware, in this order: stand still (no foot should leave the ground and no force
should appear in flight), walk on flat ground at a low command, then a push. The symptom to watch for is a foot that
hovers with force, which means `complementarityWeight` is too low for the scale of the other costs.

## 6. Tests

Five test targets cover this formulation, and the split matters: the shipped task file has it **off**, so a test that
only reads the shipped configuration proves nothing about it. That is exactly why every defect described above went
unnoticed -- the one test that reached for these terms, `MpcParameterUpdaterModuleTest.ContactImplicitTuningReachesEveryTerm`,
called `GTEST_SKIP()` because they were not enabled.

`humanoid_centroidal_mpc:testContactImplicitFormulation` writes a task file with the formulation switched **on** and
builds the real `CentroidalMpcInterface` from it. It holds that the loader refuses every half-way combination
(including the hard `normal_velocity`), that the shipped file still has the formulation off and its cones still gated,
that all three terms are built for every foot, that the schedule-gated hard constraints are gone, that the basis-scaling
barrier is no longer gated, that ground penetration is checked at all four footprint corners with a penalty that is zero
in value and gradient on the ground, that the gap the product measures is bracketed by the smallest penetration row and
that row plus `log(N) * gapSmoothing` -- i.e. that the two terms really were built on the same geometry -- and that one
terrain height reaches both terms that need it.

`humanoid_centroidal_mpc:testContactConstraintScheduleGating` holds the gate itself: each of the four terms is inactive
during a scheduled swing when gated and active when not; the un-gated wrench cone reads exactly zero at the zero wrench
while the gated one is violated there; the un-gated friction cone reads exactly zero at the zero force with an unchanged
gradient; a squared hinge has zero value and gradient at zero slack where a relaxed barrier has `-40`; and the gate, the
active flag and the dropped offsets all survive the clone the SQP solver makes of the problem per worker thread -- two of
those copy constructors used to drop `isActive_`.

`humanoid_centroidal_mpc:testContactImplicitTermsOnBasisInputs` exercises the terms against the **basis-vector**
parameterization the shipped robot actually runs, which nothing did before: the normal-force row is all ones over the
foot's block, the indicator vanishes exactly when the foot carries no wrench and is blind to the other foot's load, both
products are the normalised products they claim to be, the penetration term reads all four corner heights, and every
analytic derivative matches a central finite difference **on that model** -- the check that would catch a frame or a
scale applied to the value but not to the Jacobian.

`humanoid_centroidal_mpc:testRelaxedContactConstraints` builds the three terms on the wrench-space DRC Atlas model and
holds:
the normal-force row reads exactly this foot's contact block; each term's value is the product it claims to be; the
slip term constrains three twist components and leaves the rocking rates free; a consistent configuration (foot down,
or no load) costs nothing; and every analytic derivative matches a central finite difference of the value.

It also pins the normalisation, which is the part most likely to be undone by accident: the complementarity residual
is exactly one at the two references, each residual is divided by both of its references, the slip term's yaw row uses
the angular reference rather than the linear one, and the derivatives still match finite differences once the scales
are applied — a normalisation applied to the value but not to the Jacobian would pass every other test here and hand
the solver a wrong gradient.

It also pins the **gap**, on a foot pitched 0.25 rad onto one edge: the product reports the lowest corner's clearance
rather than the sole centre's, which on that configuration differ by more than a centimetre, and the penetration hinge
sits exactly at its boundary where the product says the foot is touching. The derivatives are checked on the tilted foot
as well as the flat one, because at a flat foot every softmin weight is `1/N` and the state Jacobian is the plain mean of
the corners' -- which would hide a wrong contraction.

`humanoid_common_mpc:testFootprintCornerHeights` covers the smoothed minimum on its own, away from any robot model, so
the properties the formulation depends on are stated where they can be read: it is **exact** in value and gradient at a
flat foot -- the property the `1/N` buys and the reason the plain log-sum-exp softmin could not be used -- it never
reports less clearance than the lowest corner, its bias is the whole family `gapSmoothing * log(N / k)` for `k` corners
down -- exact flat, 0.69 mm on an edge, 1.39 mm on a single corner, and monotone between them -- its weights are a convex
combination that really is its gradient, it is exact under a rigid lift of the foot, and it stays finite when corners are
far enough apart for the exponentials to underflow.

## 7. If the robot still walks itself sideways

The contact-implicit terms are only half of what the lateral direction needs. The other half is that the whole-body
MPC has to be *asked* for the centre-of-mass motion the reduced-order planner is placing feet for — see section 3c of
[hlip_contact_planner](../hlip_contact_planner/README.md) and the `planned_com_override` execution rule. Without it,
narrowing steps and a sideways drift appear whether or not this formulation is enabled.
