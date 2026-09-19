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
`f_n` its normal contact force, `h` the height of its contact frame above the terrain and `v_xy` the tangential
velocity of that frame:

```
f_n >= 0                     already enforced   (friction cone, or the non-negativity of the basis scalings)
h   >= 0                     GroundPenetrationConstraint      (relaxed barrier)
f_n h   = 0                  ContactComplementarityConstraint (quadratic penalty)
f_n [v_x, v_y, omega_z] = 0  ForceWeightedSlipConstraint      (quadratic penalty)
```

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

`loadMpcFormulationTasks` refuses the half-way combinations: `contact_complementarity` with `zero_wrench`,
`force_weighted_slip` with either `zero_velocity`, and `contact_complementarity` without `ground_penetration` (the
complementarity product is also satisfied by a *negative* height, so the unilateral condition has to be there).

`normal_velocity` remains schedule-gated and is left listed; it shapes the swing's approach to the ground rather than
the contact decision, but it is the next term to reconsider if the solver still refuses to depart from the nominal
schedule.

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
| `penetrationMu`, `penetrationDelta` | the relaxed barrier on `h >= 0`: weight, and the width [m] of its relaxation |
| `terrainHeight` | [m] the ground under the contact frames |

Because the residuals are now dimensionless and O(1) at their worst case, the two weights are **directly comparable
with `task_space_foot_cost_weights`**. That comparison is the tuning rule: `complementarityWeight` above `pos_z` means
the complementarity term outvotes the swing height reference and the foot will land early; well below it means force
can linger on a foot in flight. Both shipped values are 100, against `pos_z: 100`.

### Tuning order

1. **Check `terrainHeight` first.** Log `h` for a foot in stance. If it is not within a few millimetres of zero, the
   contact frame sits off the sole and every stance foot carries a permanent residual that no weight can fix — set
   `terrainHeight` to that offset instead.
2. **Penetration.** `mu = 0.1`, `delta = 0.01` (1 cm of quadratic relaxation, the right order for foot-height noise).
3. **Complementarity.** Watch the normal force on a foot in flight. Raise the weight until it is negligible; if the
   foot starts landing early instead, you have passed `pos_z` and should raise `pos_z` rather than the weight.
4. **Slip, last.** Watch the tangential velocity of a loaded foot.

The four signals worth logging are `h`, `f_n`, `f_n h` and `f_n v_xy` per foot. They separate the two failure modes —
force in flight versus early touchdown — without guesswork.

## 5. Switching it on

In the robot's `config/mpc/task.yaml`:

```yaml
hard_constraints:
  - normal_velocity                   # zero_wrench and zero_velocity removed

soft_constraints:
  - joint_limits
  - foot_collision
  - contact_wrench_cone
  - contact_complementarity
  - force_weighted_slip
  - ground_penetration
```

Then validate in MuJoCo before hardware, in this order: stand still (no foot should leave the ground and no force
should appear in flight), walk on flat ground at a low command, then a push. The symptom to watch for is a foot that
hovers with force, which means `complementarityWeight` is too low for the scale of the other costs.

## 6. Tests

`humanoid_centroidal_mpc:testRelaxedContactConstraints` builds the three terms on the DRC Atlas model and holds:
the normal-force row reads exactly this foot's contact block; each term's value is the product it claims to be; the
slip term constrains three twist components and leaves the rocking rates free; a consistent configuration (foot down,
or no load) costs nothing; and every analytic derivative matches a central finite difference of the value.

It also pins the normalisation, which is the part most likely to be undone by accident: the complementarity residual
is exactly one at the two references, each residual is divided by both of its references, the slip term's yaw row uses
the angular reference rather than the linear one, and the derivatives still match finite differences once the scales
are applied — a normalisation applied to the value but not to the Jacobian would pass every other test here and hand
the solver a wrong gradient.

## 7. If the robot still walks itself sideways

The contact-implicit terms are only half of what the lateral direction needs. The other half is that the whole-body
MPC has to be *asked* for the centre-of-mass motion the reduced-order planner is placing feet for — see section 3c of
[hlip_contact_planner](../hlip_contact_planner/README.md) and the `planned_com_override` execution rule. Without it,
narrowing steps and a sideways drift appear whether or not this formulation is enabled.
