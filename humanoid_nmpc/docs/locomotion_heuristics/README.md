# Locomotion Heuristics: Bledt's Regularized Predictive Control in this MPC

An implementation of the ten regularization heuristics of

> Gerardo Bledt, *Regularized Predictive Control Framework for Robust Dynamic Legged Locomotion*, <!-- NOLINT(claudisms): cited paper title -->
> PhD thesis, Massachusetts Institute of Technology, February 2020.
> Chapter 3 (the RPC formulation), Chapter 4 (heuristic extraction), **Appendix C** (the ten heuristics).
> Related papers: Bledt & Kim, *Implementing Regularized Predictive Control for Simultaneous Real-Time Footstep and
> Ground Reaction Force Optimization*, IROS 2019; Bledt & Kim, *Extracting Legged Locomotion Heuristics with
> Regularized Predictive Control*, ICRA 2020.

Every heuristic is selected **by name** from one of three lists in the robot's `config/mpc/task.textproto`, and every
list ships **empty**, so this subsystem is inert until someone opts a robot in one name at a time.

```textproto
locomotion_heuristics {
  # Each list ships empty, its candidate names commented out beneath it, one line per entry:
  # base_pose: "orientation_compensation"   uncomment ONE line and that name is in the list
  # base_pose: "periodic_orientation"
  # base_pose: "height_compensation"
  # foothold: ...   hip_centered_stepping, capture_point, translational_stepping, in_place_turning, high_speed_turning
  # wrench: ...     impulse_scaling, centripetal_acceleration
  # ... one parameter block per name, named after it: orientation_compensation { ... }
}
```

(A repeated field of a textproto is one line per entry, so uncommenting a line adds exactly that name to its list; a
file without any line for a list leaves it empty.)

## 1. What RPC is, and what it is *not*

The dissertation's thesis is stated in its section 3.1. A nonlinear predictive controller's cost space is full of
local minima, and the optimizer has no idea which of them a legged robot actually wants. But we know a great deal
about legged locomotion already. So rather than letting the optimizer rediscover it, RPC **adds a simple
regularization term** that biases the solution towards a known-good answer, while leaving the optimizer free to
depart from it where the dynamics pay better:

$$J(\mathbf{x}) \;=\; \underbrace{J_D(\mathbf{x})}_{\text{dynamics / task}} \;+\; \underbrace{J_R(\mathbf{x})}_{\text{regularization}}, \qquad J_R(\mathbf{x}) = \sum_{k=0}^{N-1} \tilde{\mathbf{x}}_k^\top W_k \tilde{\mathbf{x}}_k,$$

where the **regularization error** of decision variable $\mathbf{x}$ is its distance from a heuristic value

$$\tilde{\mathbf{x}} \;=\; \mathcal{H}_{\mathbf{x}}(\mathbf{x}, \boldsymbol{\Phi}, \mathbf{x}_d) \;-\; \mathbf{x}.$$

$\mathcal{H}_{\mathbf{x}}$ is the heuristic: a cheap closed-form guess at what the decision variable should be, given
the state, the gait phase $\Phi$ and the operator's command $\mathbf{x}_d$. Figures 3-2 and 3-3 of the dissertation
make the geometric point: a well-chosen $J_R$ smooths the combined cost and removes spurious local minima *without
moving the optimum*, while a badly-chosen one drags the optimum somewhere neither term wanted.

**The key observation for this repository.** The quadratic regularization $J_R$ already exists here - it is
`state_quadratic_cost`, `input_quadratic_cost` and `task_space_foot_cost`, with $W_k$ spelled out as the
`state_weights`, `input_weights` and `task_space_foot_cost` blocks of `task.textproto`. What was missing is $\mathcal{H}$. The references those
costs regularize against are, today:

| Channel | Reference before this subsystem |
| --- | --- |
| base roll, pitch | **a hard zero at every speed** - `TargetTrajectoriesCalculatorBase::integrateRefPoseeulerzyxBaseRtWorld` writes `targetPose[4] = targetPose[5] = 0.0` on every call |
| base height | the operator's commanded height, adapted to the terrain |
| swing foot $x, y$ | **nothing at all** without a contact planner - `nominalFoothold()` returns `std::nullopt` and the foot cost zeroes its own `pos_x` / `pos_y` weights |
| contact wrench | weight compensation: $m g$ split evenly over the scheduled stance feet, purely vertical |

So this is a **reference-shaping layer**, not a new cost family. Nothing here adds a term to the optimal control
problem; each name modifies the reference that a term the task file already lists is regularizing against. Three
consequences follow:

1. **An empty list is the previous behavior**, bit for bit, because the sum of no offsets is zero and
   nothing is recomputed. This is pinned by parity tests on each seam.
2. **Order within a list is documentary**, because the offsets are summed and addition is commutative. (Unlike the
   contact planner's `costs` list, whose order is the floating-point accumulation order.)
3. The **list and the coefficients are separate switches**: on G1 and R1 every fitted coefficient ships at zero, so
   "does listing this change anything at all" is a separate experiment from "what should the number be". On Atlas
   and SA01 the blocks carry **derived, non-zero** coefficients (section 4), so there the list entry *is* the change;
   zero a block first to run the listing-only experiment.

## 2. Where it sits

```
  operator command  (joypad: v_cmd, omega_z,cmd)
            |
            v
  CentroidalMpcTargetTrajectoriesCalculator        <-- integrates the command into a 3-knot TargetTrajectories
            |                                          (roll and pitch hard-zeroed here; NOT the seam)
            v
  +=========================================================================================+
  |                        SwitchedModelReferenceManager                                    |
  |                                                                                         |
  |   captureMeasuredState(t0, x0)   once per solve, on the solver thread                   |
  |      latches: base pose, foot lift-off positions, CoM velocity, CoM height, I_zz        |
  |   getCmdLinvelBaseRtWorldEwrtWorldXy(t) / getCmdRateYawBaseRtWorld(t)   the operator's command (virtual)   |
  |                                    |                                                    |
  |                                    v                                                    |
  |   +---------------------------------------------------------------+                    |
  |   |             LocomotionHeuristicLayer   (immutable)            |                    |
  |   |   base_pose[]   foothold[]   wrench[]   <- the three lists     |                    |
  |   |   LocomotionHeuristicFactory resolves each name                |                    |
  |   |   LocomotionHeuristicModelParameters: m, g, z_com, r_hip       |                    |
  |   +---------------------------------------------------------------+                    |
  |         ^  (A)                  ^  (B)                  ^  (C)                          |
  |         |                       |                       |                               |
  |   shapePoseBaseRtWorld()         nominalFoothold()       getDesiredInput()                     |
  |   per shooting node       every solve             per shooting node                     |
  +=========================================================================================+
         |                       |                          |
         | x_nominal             | landing target           | u_nominal
         v                       v                          v
  StateQuadraticCost      getSwingFootReference()     InputQuadraticCost
  StateInputQuadraticCost         |                   StateInputQuadraticCost
  BasePoseShaped-                 |                          |
   QuadraticStateCost (terminal)  |                          |
  torso task-space cost           |                          |
         |                        v                          |
         |             CentroidalMpcEndEffectorFootCost      |
         |                        |                          |
         +------------------------+--------------------------+
                                  |
                                  v
                    OCS2 SQP whole-body NMPC  --->  MRT joint controller  --->  robot
```

| Seam | Shapes | Evaluated | Reaches |
| --- | --- | --- | --- |
| **(A)** `shapePoseBaseRtWorld()` | base roll, pitch, height of $\mathbf{x}_{nom}$ | per shooting node, per SQP iteration | `state_weights.pos_base_rt_world_ewrt_world.z` (height), `state_weights.eulerzyx_base_rt_world.pitch` and `.roll` (`.yaw` is never shaped), the same entries of `final_state_weights`, and the torso task-space cost's orientation |
| **(B)** `nominalFoothold()` | the swing foot's landing target | at every solve, a landmark that holds still through the swing | `task_space_foot_cost.weights.pos_x` / `.pos_y` |
| **(C)** `getDesiredInput()` | the contact-force reference $\mathbf{u}_{nom}$ | per shooting node, per SQP iteration | `input_weights.contact_wrenches` |

### Why these three seams and not others

**(A) is the one definition of the shaped base pose,** `SwitchedModelReferenceManager::shapePoseBaseRtWorld()`, and every
cost that regularizes the base pose calls it: `StateQuadraticCost` and `StateInputQuadraticCost` through
`getDesiredState()`, the terminal cost (`BasePoseShapedQuadraticStateCost`, a `TuningQuadraticStateCost` whose
`TuningLayoutDeviation` is taken against the shaped pose) and the torso task-space cost (`EndEffectorKinematicsQuadraticCost`, handed the reference
manager). A term left on the unshaped pose would pull against the others: `final_state_weights`, scaled by `terminal_cost_scaling`
on one node, outweighs a whole horizon of running base-pose weight, and the torso's orientation weights of 100 against
`state_weights`' 5 would cancel most of a tilt on R1, whose torso has no pitch joint of its own. The context is built from the
operator's COMMANDED velocity, `getCmdLinvelBaseRtWorldEwrtWorldXy()`, not from the target's momentum channel, which the contact
planner's `planned_com_override` rewrites with the plan's swaying CoM velocity. Shaping the `TargetTrajectories` instead would fail three ways: its knots over the horizon are placed by the command (three on a straight walk, a few more on a turning one), so
a limit cycle at gait frequency (`periodic_orientation`, roughly 2 Hz over a 1 s horizon) is unrepresentable; it is
published one solve ahead of the solver that consumes it; and it is also what the operator's own delta-pose commands
are measured against, so shaping it would make the commanded pose drift away from what was asked for. There is a
precedent in this very function: the procedural arm swing is already a phase-dependent reference heuristic written
there by hand.

**(B) is the base class's opinion about foot placement,** with one caller, `getSwingFootReference()`, which
blends from the actual lift-off position to it with a cubic $-\tau^3 + \tau^2 + \tau$ and hands the result to the
foot task-space cost. That cost switches its $x,y$ position weights **on** precisely when a value is present. It is
re-evaluated at every solve with the latest measurement, so it has to be a **landmark**: the anchor is the base
*predicted at touch-down* (measured at the solve and carried forward by the command), so the target the foot is
steered towards does not move while it swings. (An anchor carried forward only over the time *remaining* until
touch-down shrinks to nothing as the swing ends, and slid the target backwards through every swing.) The command
carries it along the arc of a constant body twist, $\Delta p = \Delta t\,\mathrm{sinc}(\theta/2)\,R_z(\theta/2)\,\bar v$
with $\theta = \bar\omega\Delta t$ and $\bar v = (v_0 + R_z(-\theta)v_1)/2$ the mean of the target's command at the
solve and at touch-down seen from the frame that turns with it (`SwitchedModelReferenceManager::integratePlanarCommand()`):
exact for a turning velocity command, whose knots lie on that arc, and the trapezoid where the command does not turn;
the trapezoid of the two world velocities it was fell short of a turn's arc by
$|v|\Delta t\,(\mathrm{sinc}(\theta/2) - \cos(\theta/2)) \approx |v|\Delta t\,\theta^2/12$, 5.6 cm at 0.41 m/s and
0.8 rad/s over 1.38 s. One
addition inside `nominalFoothold()` therefore reaches the position reference, the velocity reference (the cubic's
analytic rate falls out) and the swing-plane heading query, with no other edit anywhere.

**(C) is new plumbing.** The input channel of `TargetTrajectories` is not a reference at all - it is constructed
all-zero and commented *"they are not used"*, and both input costs ignored it and built their own nominal from
`weightCompensatingInput()`. That nominal is **contact-flag dependent** and the stance set changes several times
inside one horizon, which a three-knot trajectory cannot carry; the reference manager is the only object that knows
the contact flags at an arbitrary node time. `getDesiredInput()` makes exactly the old call when no wrench heuristic
is listed, so routing the two costs through it is a no-op on every robot shipped here. A horizontal force
(`centripetal_acceleration` under basis-vector inputs) is written into the contact frames of the **reference**
configuration at the node, not of the node's state, so that, like (A), the reference is a constant of the
Gauss-Newton step that treats it as one.

### The invariant that makes it thread-safe

**No heuristic performs forward kinematics, and no heuristic holds mutable state.** Everything model-dependent
($m$, $g$, the nominal CoM height, $\mathbf{r}_{hip}$) is derived once at start-up into
`LocomotionHeuristicModelParameters`; everything measurement-dependent is latched once per solve in
`captureMeasuredState()`, on the solver thread, and handed over in a per-call context struct. Every heuristic is
therefore a pure function of (its coefficients, the context), the layer needs no lock on the read path, and
`reconfigure()` - the hot-reload entry point - runs from the pre-solve hook, before any SQP worker exists.

## 3. The ten heuristics

Notation follows Appendix C. $\mathbf{p}$ is the CoM position, $\dot{\mathbf{p}}$ its velocity, $\dot{\mathbf{p}}_d$
the commanded velocity, $\boldsymbol{\Theta}$ the body orientation, $\boldsymbol{\omega}$ the angular rate,
$\dot{\psi}$ the yaw rate, $\Phi$ the gait phase, $\mathrm{PTP}(\cdot)$ the projection onto the ground plane, and
$\mathrm{sgn}_i = +1$ for the left foot, $-1$ for the right.

### Table C.1 - the analytic heuristics

These are derived from physics. They have **no fitted coefficients**, only unit scales, so for these the list entry
alone is the change.

#### 3.1 `hip_centered_stepping` - $\mathcal{H}_{\mathbf{r}}(\boldsymbol{\Theta}) = \mathrm{PTP}\!\left(R(\boldsymbol{\Theta})\,\mathbf{r}_{hip}\right)$

The foot placed under its own hip: this leg's hip position in the base frame, rotated into the world and projected
onto the ground.

$$\Delta\mathbf{r}_i \;=\; R_z(\hat\psi_{td}) \begin{bmatrix} s_x\, r_{hip,i,x} \\ s_y\, r_{hip,i,y} \end{bmatrix}$$

with $\hat\psi_{td}$ the yaw predicted at this foot's touch-down, and the offset added to the base predicted there.

$R(\boldsymbol{\Theta})$ is the full body rotation in the dissertation and $R_z(\psi)$ here. That is a choice of
this heuristic, not a simplification of the control model, which carries the full base rotation as a quaternion: the
roll and pitch at a touch-down are not predicted, only its heading is, and the roll and pitch of a walking base move
a ground-projected hip offset by a second-order amount that $\mathrm{PTP}$ flattens further. $\mathbf{r}_{hip}$ is **derived
from the URDF at start-up** and is never a task-file field; $s_x, s_y$ are the two scales that stretch it.

This is the base case of figure 4-8, the panel every later heuristic is measured against. Alone it "stably stands and
takes a few steps forward" and falls as soon as it has speed, because a foot placed under the hip has no lead on a
moving body. **It is also the one heuristic here that moves the anchor** of the foothold reference: with it listed the
landing target is the predicted base plus the hip offset, so the feet's lateral separation comes from the hips rather
than from half of `nominal_foothold.step_width` to each side of the predicted base. Both anchors are the base
predicted at touch-down plus a lateral offset to the foot's own side, so the velocity leads summed on top mean the
same thing either way. (The other anchor used to be the *stance foot* plus a whole step width across the heading. The
stance foot is where the body was when it landed, so under a lateral command $v_y$ the feet fell behind the body by
$v_y T_{step} - k T_{st} v_y$ at every step, $k T_{st} v_y$ being `translational_stepping`'s lead, without bound:
7.5 cm a step at 0.3 m/s on a 0.5 s step. `testLocomotionHeuristicSeams` walks a sustained sidestep and holds the
midpoint of the feet still relative to the base.) It is registered because
Bledt never uses it alone: it is the term the Table C.2 velocity-dependent heuristics are summed on top of. The layer
warns when it is listed alone, and warns again when the others are listed without it. On a robot whose `step_width` is
0 it is **required** by the rest of the foothold family: start-up refuses a foothold list without it, because nothing
else would keep the feet apart.

#### 3.2 `capture_point` - $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}}, \Phi) = \mathrm{PTP}\!\left(\sqrt{p_z/g}\,(\dot{\mathbf{p}} - \dot{\mathbf{p}}_d)\right)$

$$\Delta\mathbf{r}_i \;=\; k \sqrt{\frac{z_{com}}{g}} \left(\dot{\mathbf{p}}_{meas}(t_0) - \dot{\mathbf{p}}_{cmd}(t_0)\right), \qquad \|\Delta\mathbf{r}_i\| \le \Delta_{max}$$

with both velocities at the same instant $t_0$, the last solve, where the measurement was taken. The command at the
touch-down is not the one to compare with: on a turning command it has rotated by $\dot\psi\,(t_{td} - t_0)$ since,
and a robot tracking the turn perfectly would read as an error of $|\mathbf{v}|$ times that angle - 0.15 m of
spurious outward step on Atlas at 1 m/s, 1 rad/s and 0.45 s ahead, where it used to be compared.

$\sqrt{z/g} = 1/\omega$ is the time constant of a linear inverted pendulum of height $z$ - the same $\omega$ this
repository's DCM terminal cost and its ZMP/DCM viewer markers already use. This is Pratt's capture point applied to
the **velocity error** rather than to the velocity, which is what makes it the one heuristic in the family that is
silent while the robot is tracking its command and grows only when a push, a slip or a model error has opened a gap.

The clamp is on the **magnitude**, not per axis: a per-axis clamp would rotate the offset towards the diagonal
exactly when it is largest, which is when the robot most needs the foot to go where the push came from.

Under this controller this is the heuristic with the clearest mandate. With `zero_velocity` in `hard_constraints` the
stance foot is pinned by the schedule; the foot cost's $x,y$ weights are the only thing in the problem that can move a
foot sideways in response to a push, and nothing is currently writing them a target.

#### 3.3 `impulse_scaling` - $\mathcal{H}_{\mathbf{f}}(\Phi) = \dfrac{m g}{F \beta}$

Over one gait cycle of period $T$ the vertical impulse the feet deliver must equal the weight times the cycle. A foot
that is on the ground for a fraction $\beta$ of the cycle therefore has to average $mg/(F\beta)$ **while it is down**,
not $mg/F$. Bledt credits the idea to the vertical impulse scaling of the MIT Cheetah 2's bounding controller
(section 4.1).

**$F$ is the number of feet the robot HAS (a constant, not the number in contact at this instant).** That
distinction defines the heuristic. The reference this layer shapes is $W / n_{stance}(t)$, which sums
to $W$ at *every* instant; Bledt's is $W/(F\beta)$, the same value on every stance foot whatever
$n_{stance}(t)$ happens to be, so its total varies over the cycle (below $W$ in single support, above it in double
support) and averages, over one cycle,

$$\langle n_{stance}\rangle \cdot \frac{W}{F\beta} \;=\; (F\beta)\cdot\frac{W}{F\beta} \;=\; W,$$

because the mean number of feet on the ground *is* $F$ times the duty factor. That is the impulse budget. Scaling
$W/n_{stance}(t)$ by $1/\beta$ instead would put $W/\beta$ on the ground at every instant and deliver $WT/\beta$ over
the cycle - it would regularize the solver towards accelerating the center of mass upwards for ever.

The offset is therefore written as the difference from the existing reference, with `scale` $s$ blending:

$$\Delta f_{z,i} \;=\; \mathrm{clamp}\!\left(\frac{W}{n_{stance}} + s\left(\frac{W}{F\bar\beta_i} - \frac{W}{n_{stance}}\right),\; 0,\; \rho_{max}\frac{W}{n_{stance}}\right) - \frac{W}{n_{stance}}, \qquad \bar\beta_i = \max(\beta_i, \beta_{min})$$

so an empty list is exactly today's reference. Two sanity checks fall straight out: standing ($\beta = 1$,
$n_{stance} = F$) and a pure alternating single support ($\beta = 1/2$, $n_{stance} = 1$) both give back the existing
reference exactly, so the correction bites only where the two genuinely disagree - a gait with double support.

$\beta_i$ is measured by `stanceDutyFactor()` over **foot $i$'s own stride**, lift-off to lift-off, in the schedule
being solved. The average above needs every foot's $\beta$ to be the cycle's duty factor at every node, and on a
periodic gait the stride measure is exactly that - the same value at every node and for both feet. A window of fixed
length does not have that property unless it happens to be a whole number of strides: a 1.0 s horizon over the 1.4 s
`walk` stride measured anything from 0.40 to 0.80 depending on the phase, gave the two feet different values at the
same node, and put 17% more vertical impulse into the reference than the weight. Standing, and the time before the
first lift-off of any foot, is $\beta = 1$; a foot still in the stance it started in while the other takes its first
step is given the gait's value, not the standing one. The lower clamp at zero and the upper clamp at $\rho_{max}$ times
weight compensation are required rather than defensive: $\beta \to 0$ at the onset of a flight phase and
$1/\beta$ is unbounded there, and a force reference that pulls the robot down through the floor is not one any foot
can track.

What the correction buys on a biped is anticipation: during double support it asks the feet to push a little harder
than static equilibrium requires, because a single-support phase is coming in which one of them will carry everything.

#### 3.4 `centripetal_acceleration` - $\mathcal{H}_{\mathbf{f}}(\boldsymbol{\Theta} \times \dot{\mathbf{p}}) = m\,\boldsymbol{\omega} \times \dot{\mathbf{p}}$

The horizontal force that holds the robot on a circular path. With $\boldsymbol{\omega} = \dot\psi \mathbf{e}_z$ and a
horizontal velocity,

$$\boldsymbol{\omega} \times \dot{\mathbf{p}} = \begin{bmatrix}0\\0\\\dot\psi\end{bmatrix} \times \begin{bmatrix}v_x\\v_y\\0\end{bmatrix} = \begin{bmatrix}-\dot\psi\, v_y\\ \dot\psi\, v_x\\ 0\end{bmatrix}, \qquad \Delta\mathbf{f}_i = \frac{s\,m}{n_{stance}} \left(\boldsymbol{\omega} \times \dot{\mathbf{p}}\right),$$

a force of magnitude $m|\dot\psi||\mathbf{v}|$ pointing towards the center of the turn - the companion of
`high_speed_turning`: one says where to put the foot for a fast turn, the other which way to push once it is there.

**Two caveats before listing it.**

*It is the only heuristic of the ten whose force is horizontal,* so it is the only one whose expression depends on the
foot's orientation. Under `contact_input_parameterization: "basis_vectors"` the input lives in the local contact frame, so
the whole contact-force reference must be written through `setForceContactEwrtWorld()`, which costs one
forward-kinematics pass per shooting node per SQP iteration. The layer switches paths by itself when this name is listed and logs a
warning; the other nine never pay it.

*It may well be the wrong thing to want here.* Bledt's control model is a single rigid body whose force reference is
the only thing telling the QP how hard to push. These dynamics are **full centroidal** and the state cost already
tracks the commanded linear and angular momentum - so $m\,\boldsymbol{\omega} \times \mathbf{v}$ is the derivative of
a momentum this problem is already regulating: an *output* the optimizer must produce, not information the reference
lacks. Supplying it again at lower fidelity can only help where the optimizer would have been slow to find it, and
can fight the momentum cost where the two disagree, as they do in double support and wherever the friction cone binds.
It is implemented because it is one of Bledt's ten and the seam for it is clean; `scale` is the knob for deciding in
simulation whether it earns its place.

The clamp is on the **total** horizontal force, before it is shared over the stance feet, and is a fraction of body
weight by default rather than an absolute force, so it scales across the robots here, whose masses differ by a factor
of five; the feet cannot pull sideways harder than friction allows. A total, because the feet share the normal load as
they share this force: each foot's friction budget is $\mu W / n_{stance}$, and a total clamp $r W$ is the fraction
$r/\mu$ of every foot's budget in single and in double support alike. A clamp on each foot's share, as it was, allowed
twice the fraction in double support that it allowed in single support (80% instead of 40% at the derived
$r = 0.4\mu$). The coefficient is
the listed cone's: 0.5 in `contacts.contact_wrench_cone_soft_constraint` on Atlas and SA01, 0.4 in
`contacts.friction_force_cone_soft_constraint` on G1 and R1. `derive_parameters.py` reads whichever the robot lists.

### Table C.2 - the data-extracted heuristics

These were not designed. Chapter 4 describes the framework that produced them: run the RPC **offline**, without
real-time constraints, over a systematic sweep of commands; collect every state, input and command into a candidate
variable set $V$; fit simple models (polynomials up to degree 9 and sums of up to 8 sines) to every pair; and keep
the relationships with high $R^2$ as heuristic candidates, after validating them on a *different* exploration.

Two cautions from section 4.5 carry over to anyone extending this layer. **Overfitting**: a degree-9 polynomial fit
the pitch–velocity relationship with $R^2 = 0.94$, almost as well as the linear one at $0.95$, and then collapsed on
validation data, which is why the shipped forms are affine and sinusoidal rather than flexible.
**False dependency**: an 8-sine model related forward velocity to *wall-clock time* with $R^2 = 0.99$, purely because
the exploration stepped the command on a fixed schedule. The framework is not a black box; it proposes, the engineer
disposes.

**None of Bledt's fitted numbers is a default here.** They were fitted to a 9 kg quadruped. The coefficients below ship
at zero on G1 and R1; on Atlas and SA01 they carry values **derived** from each robot's geometry, gait and command
limits by `derive_parameters.py` (section 4).

#### 3.5 `orientation_compensation` - $\mathcal{H}_{\boldsymbol{\Theta}}(\dot{\mathbf{p}}) = a_1 \dot{\mathbf{p}} + a_0$

$$\Delta\theta_{roll} = a_1^{\theta} v_y^{base} + a_0^{\theta}, \qquad \Delta\theta_{pitch} = a_1^{\phi} v_x^{base} + a_0^{\phi},$$

each clamped to $\pm\theta_{max}$. Roll follows the **lateral** command and pitch the **forward** one - the pairing the
regressions found, with $R^2 = 0.983$ and $0.953$ (equations 4.11 and 4.12, figure 4-5). The velocities are the
commanded CoM velocity in the *reference* base's yaw frame, so $a_1$ is a tilt per unit of commanded speed and $a_0$ a
standing trim that survives at zero speed.

Bledt reports two distinct benefits (section 4.3, figure 4-12): the robot trots "in a much more controlled manner",
and the **simplified orientation dynamics become more valid**, because a body that is closer to
level at speed is closer to the small-roll-and-pitch approximation the control model is built on. That second argument
does not transfer: the control model here carries the full base rotation as a quaternion
(humanoid_nmpc/docs/quaternion_base_orientation/README.md) and makes no small-angle approximation of roll and pitch.
The first one does, and it is the reason to list this heuristic.

The clamp is not defensive trimming. The command this reads is filtered but not rate limited, and an affine law
applied to it has no bound of its own; a reference tilt the legs cannot reach is worse than no reference at all,
because the state cost pulls towards it at every node and the solver spends iterations on a target it cannot have.

Sign note: the seam edits the base pose as Euler ZYX angles about a z-up world (the robot model's Euler boundary
accessors; the state holds a unit quaternion, humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.9),
in which a **positive** pitch rotates the body's $+x$ axis
towards $-z$ - it puts the nose **down**. Pitching forward into a forward command is therefore a *positive*
`pitch_per_forward_velocity`, which is also the sign of Bledt's own fitted $+0.0725$ rad per m/s.

#### 3.6 `periodic_orientation` - $\mathcal{H}_{\boldsymbol{\Theta}}(\Phi) = b_1 \sin(c_1 \Phi + d_1)$

$$\Delta\theta_{roll} = b_1^{\theta}\sin\!\left(c_1^{\theta}\Phi + d_1^{\theta}\right), \qquad \Delta\theta_{pitch} = b_1^{\phi}\sin\!\left(c_1^{\phi}\Phi + d_1^{\phi}\right).$$

The orientation limit cycle a walking robot falls into on its own. This is the clearest illustration of what the
extraction framework is for: the steady-state pitch during a trot is genuinely complicated, and Bledt makes the point
that it is *"currently too difficult to analyze without data-driven methods"* - yet a single sine at the step
frequency fits it with $R^2 = 0.835$ (equation 4.13, figure 4-6), and summed with the velocity-dependent term of
§3.5 it reproduces the measured pitch closely (figure 4-7):

$$\mathcal{H}_{\phi}(\dot{p}_x, \Phi) \;=\; \underbrace{a_1^{\phi}\dot{p}_x + a_0^{\phi}}_{\text{§3.5}} \;+\; \underbrace{b_1^{\phi}\sin(c_1^{\phi}\Phi + d_1^{\phi})}_{\text{§3.6}}$$

The value of the fit is not its accuracy but that its four numbers *mean* something an engineer can turn: an
amplitude, a frequency and a phase. Bledt's own $c_1 \approx 12.566 \approx 4\pi$ recovers the trot's two
complementary diagonal pairs per gait cycle, which is a check that the fit found the physics rather than the data.

**This is the heuristic that could not be expressed by shaping the target trajectory**, whose knots over a
one-second horizon, placed by the command (three on a straight walk), would sample two or three whole periods and interpolate linearly between them. It only works at the
per-node seam. $\Phi$ comes from `getPhaseVariable(t)` and runs over $[0,1)$ for one left–right cycle, so
$c_1 = 2\pi$ is one period per cycle and $c_1 = 4\pi$ one per step; it **freezes** at 0 or 0.5 through double support
rather than advancing, so the offset is held constant there instead of continuing its cycle. Phase $[0, 0.5)$ is the
`LF` mode and $[0.5, 1)$ the `RF` mode, and those name the foot **in contact**, not the one swinging
(`modeNumber2StanceLeg` maps `LF` to contact flags `{true, false}`) - so the first half of the cycle is left stance
with the right foot in the air.

**The rate goes with the attitude.** A reference that tilts at gait frequency while its rate channel says it holds
still asks the solver for two incompatible things, so the seam writes the derivative of the shaped attitude too:
$\dot\Phi = 1/(2 T_{mode})$ inside a single-support mode of duration $T_{mode}$ and 0 through double support
(`getPhaseVariableRate()`), the offset's rate $b_1 c_1 \cos(c_1\Phi + d_1)\,\dot\Phi$, and from it, on a state that
stores the body angular velocity, $\omega_B = T_B(\Theta_s)\,(T_B(\Theta_n)^{-1}\omega_{B,n} + \dot\delta)$, the body
rate of the shaped attitude $\Theta_s = \Theta_n + \delta$ (`computeShapedAngvelBaseRtWorldEwrtBase()`); on the centroidal state,
which stores the momentum, the angular momentum of that rate with the joints at the reference,
$I_G\,\omega_{tilt}/m$ with $\omega_{tilt} = R(\Theta_s)\,T_B(\Theta_s)\,\dot\delta$ and the composite inertia $I_G$
latched at the solve, added to the target's (`computeTiltAngvelBaseRtWorldEwrtWorld()`). The velocity-dependent heuristics are
constant along a constant command and add no rate. `testLocomotionHeuristicSeams` holds both against a central
difference of the rotation, and `testLocomotionHeuristicIntegration` the momentum on Atlas.

#### 3.7 `height_compensation` - $\mathcal{H}_z(\dot{p}_x) = a_2 v^2 + a_1 v + a_0$

$$\Delta z = \mathrm{clamp}\!\left(a_2 \|\mathbf{v}_{cmd}\|^2 + a_1 \|\mathbf{v}_{cmd}\| + a_0,\; \pm \Delta z_{max}\right)$$

Taken in the **magnitude** of the commanded horizontal velocity, because crouching to walk forwards and rising to walk
backwards is not something any legged system does, and an odd polynomial in a signed speed would do exactly that.

Bledt reports this one as having made the motion "smoother and more consistent" and as having marginally reduced the
solve time, without by itself enlarging the viable operating region (table 4.1) - a fair description of what a
well-chosen regularization does when it is not fixing a failure: it shapes the cost space so the optimizer converges
sooner on something it was going to find anyway. On a humanoid it also buys something a quadruped needs less of:
lowering the base shortens the leg and gives the swing more vertical room within the same joint range, at the cost of
knee torque headroom.

Applied as an **offset** and never as an absolute height: the same channel is where
`adaptToCurrentGroundHeight()` writes the terrain, and an absolute value would silently become a terrain override.

#### 3.8 `translational_stepping` - $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}}) = a_1 \dot{\mathbf{p}} + a_0$

$$\Delta\mathbf{r}_i^{base} = \begin{bmatrix} (a_1^{x} + k^{x} T_{st})\, v_x^{base} + a_0^{x} \\ (a_1^{y} + k^{y} T_{st})\, v_y^{base} + \mathrm{sgn}_i\, a_0^{y} \end{bmatrix}, \qquad \Delta\mathbf{r}_i = R_z(\hat\psi_{td})\,\Delta\mathbf{r}_i^{base}$$

with $T_{st}$ the duration of the stance the foot begins by landing (from the mode schedule) and $k$ the
`forward_stance_fraction` / `lateral_stance_fraction` fields.

Step in the direction of travel. The largest single effect in the dissertation (section 4.3, figure 4-9): a foot placed
under the hip at lift-off has fallen behind the robot by touch-down, so the forces over the stance that follows are
biased into tipping the robot the way it is already going, and the leg reaches its kinematic limit mid-stance. Placing
the step *ahead* of the hip by roughly half of what the robot will travel during the stance "gives it twice as much
stance time capability", and the maximum velocity rose from 1.6 m/s to 3.5 m/s.

**Raibert's rule, and why $a_1$ is split in two.** The anchor this offset is added to is the base predicted at the
touch-down (the hip at the moment of landing, Bledt's own reference point), so placing the foot half a stance of
travel ahead of it is $a_1 = T_{st}/2$. Bledt's $a_1$ is a constant regressed at one gait. This robot's gait
scheduler moves between six gaits by commanded speed (`slow_walk` ... `run`), with stances from 0.3 s to 1.05 s, and one
constant is right in only one of them: sized for `trot`'s 0.5 s stance, it gives 48% of the intended lead in
`slow_walk` (1.05 s) and 167% in `run` (0.3 s). So $a_1$ is Bledt's constant **plus** $k\,T_{st}$: the derived blocks set the constant to 0 and $k = 1/2$, which
is Raibert's rule on every gait; $k = 0$ recovers Bledt's form exactly.

The affine law is applied **in the base's yaw frame and rotated back**, which is not the same as applying it in the
world: the forward and lateral coefficients differ, and the robot's forward is not the world's $x$. The lateral
constant is signed per foot, so one number in the task file widens the stance rather than shifting the whole robot
sideways; the velocity-proportional part is not, because both feet should step the same way when the robot is asked to
move sideways.

#### 3.9 `in_place_turning` - $\mathcal{H}_{\mathbf{r}}(\dot{\psi}) = a_1 \dot{\psi} + a_0$

$$\Delta\mathbf{r}_i^{base} = \begin{bmatrix} -\mathrm{sgn}_i\, (a_1^{x} + \ell\, T_{st})\, \dot\psi_{cmd} + a_0^{x} \\ a_1^{y} \dot\psi_{cmd} + \mathrm{sgn}_i\, a_0^{y} \end{bmatrix}$$

with $\ell$ the `forward_stance_lever` field: Raibert's lead for a hip at lever arm $r$ traveling at $r\dot\psi$ is
$r\dot\psi\,T_{st}/2$, i.e. $\ell = r/2$, for the same gait-tracking reason as §3.8.

The rotational counterpart of §3.8, found the same way and for the same reason: with only the translational term the
feet lagged the hips through a fast spin until they reached the end of their workspace, exactly as they lagged the
body through a fast walk (figure 4-10). Adding it let the feet **lead** the hips into the turn and raised the
achievable yaw rate from 1.8 rad/s to 4.5 rad/s.

Note that the **forward** term is signed per foot here and the lateral one is not - the opposite of §3.8, and that is
what makes this a rotation. Note the sign of the direction: a foot at $\mathbf{r}$ relative to the base moves at
$\boldsymbol{\omega}\times\mathbf{r}$, so the *left* foot at $(0, +d, 0)$ moves at
$(0,0,\dot\psi)\times(0,d,0) = (-\dot\psi d, 0, 0)$ - **backwards** for a counter-clockwise turn, while the right
foot moves forwards. That is how a turn in place is actually done: to turn left you swing the right foot forward and
around it, and the left foot back.

The implementation carries that minus sign, so a **positive** `forward_per_yaw_rate` means "place each foot further along
the direction its own hip is traveling", i.e. lead the hips into the turn. Without it, a positive coefficient would
drive the feet to *trail* the hips by twice the intended amount - precisely the workspace-exhaustion failure the
heuristic exists to remove. The lateral rate term is unsigned because it translates the whole stance sideways, which
is what a turn with a radius does; the lateral *constant* is signed, and that is the one that opens or closes the
stance.

$\dot\psi$ is the **commanded** yaw rate, read from the target knot through the robot model
(`getTargetPlanarCommand()`): on the centroidal model by inverting the angular channel of the target's momentum,
$h_z = I_{zz}\dot\psi / m$, with the composite yaw inertia latched per solve. The target's base *yaw* is not usable:
it blends the measured heading over its first stretch and the planned-heading override rewrites it, so differentiating
it would recover the reference's own smoothing rather than the operator's command.

#### 3.10 `high_speed_turning` - $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}} \times \boldsymbol{\omega}) = a_1 (\dot{\mathbf{p}} \times \boldsymbol{\omega}) + a_0$

The result the dissertation is most pleased with, and the best argument for its framework. With both §3.8 and §3.9 in
place the robot still fell when asked to do both at once: it *"tended to fall outwards along the turning radius as if
it were an object slipping off a spinning plate"*, the outer leading foot running out of workspace with no contact
foot left able to stabilize the body. The framework identified the cross term as statistically significant, and only
**afterwards** did Bledt recognize what it was - equation 4.31, the foot placement that lines up with the resultant of
gravity and the centripetal acceleration, which is visibly what animals do when they corner.

Bledt's own reduction (equation 4.27) drops the vertical velocity and the roll and pitch rates as negligible during
regular locomotion, leaving

$$\dot{\mathbf{p}} \times \boldsymbol{\omega} = \begin{bmatrix}v_x\\v_y\\0\end{bmatrix} \times \begin{bmatrix}0\\0\\\dot\psi\end{bmatrix} = \begin{bmatrix} v_y \dot\psi \\ -v_x \dot\psi \\ 0 \end{bmatrix}, \qquad \Delta\mathbf{r}_i^{base} = \begin{bmatrix} a_1^{x}\, v_y \dot\psi + a_0^{x} \\ a_1^{y}\, (-v_x \dot\psi) + \mathrm{sgn}_i\, a_0^{y} \end{bmatrix}$$

so the two coefficients multiply those two planar scalars directly. This term **vanishes at zero speed however fast
the robot spins**, which is exactly what distinguishes it from §3.9.

It exists only because the designer had thought to put $\dot{\mathbf{p}} \times \boldsymbol{\omega}$ into the candidate
set: the framework fits models between variables it is given and cannot invent a product it was never shown. That
caveat is Bledt's own (section 4.6) and is the thing to remember when extending this layer.

### Summary

<!-- LINT.IfChange(heuristic_summary_table) -->

| Name | Kind | Formula | Table |
| --- | --- | --- | --- |
| `orientation_compensation` | base pose | $a_1 \dot{\mathbf{p}} + a_0$ | C.2 |
| `periodic_orientation` | base pose | $b_1 \sin(c_1\Phi + d_1)$ | C.2 |
| `height_compensation` | base pose | $a_2 v^2 + a_1 v + a_0$ | C.2 |
| `hip_centered_stepping` | foothold | $\mathrm{PTP}(R(\boldsymbol{\Theta})\mathbf{r}_{hip})$ | C.1 |
| `capture_point` | foothold | $\mathrm{PTP}(\sqrt{p_z/g}\,(\dot{\mathbf{p}} - \dot{\mathbf{p}}_d))$ | C.1 |
| `translational_stepping` | foothold | $a_1 \dot{\mathbf{p}} + a_0$ | C.2 |
| `in_place_turning` | foothold | $a_1 \dot\psi + a_0$ | C.2 |
| `high_speed_turning` | foothold | $a_1(\dot{\mathbf{p}} \times \boldsymbol{\omega}) + a_0$ | C.2 |
| `impulse_scaling` | wrench | $mg/(F\beta)$ | C.1 |
| `centripetal_acceleration` | wrench | $m\,\boldsymbol{\omega} \times \dot{\mathbf{p}}$ | C.1 |

<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicFormulation.cpp:known_heuristic_names) -->

## 4. Switching one on

Every list ships empty on every robot, deliberately: each of these changes the closed loop, none has been validated in
simulation under hardware-like conditions on these robots, and Bledt's coefficients were fitted to a quadruped one
fifth the mass. The dissertation's own procedure is the right one to copy - figure 4-13 adds one heuristic per panel
and measures the viable operating region after each.

1. Uncomment **one name** (`base_pose: "..."`, `foothold: "..."` or `wrench: "..."`), which adds
   that name to its list. On G1 and R1 its block is at zero, so this on its own is still a no-op: launch and confirm that nothing
   changed and that the start-up banner lists the heuristic. On Atlas and SA01 the block carries derived values, so
   listing the name IS the change; to run the "nothing changed" check first, zero the block before uncommenting.
2. Sweep **its coefficients** in the tuning GUI. The sliders are generated from the schema, so they appear with no
   code change. *The slider range is derived from the value, so a coefficient at 0 gets a $(0, 1)$ slider and cannot be
   dragged negative* - type the negative value into the entry box, which widens the bounds. Several of these are
   physically negative.
3. Measure the **viable operating region**: the set of $(\dot p_x, \dot p_y,
   \dot\psi)$ commands the robot survives for several gait cycles.
4. Only then **add the next name**.

### Deriving the coefficients: `derive_parameters.py`

Most of these numbers are not free. They follow from the robot's mass and geometry, from the cadence of the gait it
is walking with, and from the command limits the operator can reach - so they are derived by a script rather than
maintained by hand, and the derivation of each is printed beside it:

<!-- LINT.IfChange(derive_parameters_usage) -->
```bash
make derive-heuristic-parameters ROBOT=drc_atlas
```

`ROBOT` takes `drc_atlas`, `engineai_sa01`, `unitree_g1` or `unitree_r1` - the centroidal packages; the whole-body one
has no layer to configure. Calling the script directly gives the rest of its options: `--gait walk` quotes the
stepping leads for another cadence (the stance-proportional coefficients themselves do not depend on it), `--check` compares the shipped block against the derivation instead of printing
it, and `--task-file` / `--urdf` / `--reference-file` take an unregistered robot.

```bash
python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas --check
make test-heuristic-parameters
```

Both run **inside the dev container**, under its system interpreter rather than as Bazel targets: Pinocchio reaches
Python through robotpkg's bindings (`/opt/openrobots`, Python 3.12, on the PYTHONPATH of the dev image and of
`setup_env.sh`) and Bazel's toolchain is a hermetic 3.11 that cannot see them. The same is true of
`tools/hooks/format_code.py`.
<!-- LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots) -->

It reads the URDF, the task file's `initial_state`, `reference.textproto` and `gait.textproto`, and reproduces the C++
`deriveLocomotionHeuristicModelParameters()` - including the walk up the kinematic tree to the last joint before the
floating base, and the contact frames, which are not in the URDF but are added at run time from
`contacts.pos_contact_rt_link_ewrt_link`. Its output splits into three groups:

| Group | Coefficients | Status |
| --- | --- | --- |
| **Closed form** | `capture_point`, `high_speed_turning`, `hip_centered_stepping`, `impulse_scaling`, `centripetal_acceleration`, every clamp | The script is the authority; there is nothing to fit |
| **Sized from the command limits** | `orientation_compensation.pitch_per_forward_velocity`, `height_compensation.height_per_speed` | A design choice (how much tilt, how much crouch), taken from an assumption the script prints |
| **Raibert's rule, per unit stance** | `translational_stepping` stance fractions (1/2), `in_place_turning.forward_stance_lever` (half the hip's lever) | Closed form in the stance duration, so one number fits every gait the scheduler selects; Bledt's constants are left at 0 |
| **Genuinely fitted** | the rest of Table C.2 | Left at **zero** - this is what chapter 4's extraction framework produces from data, and a number invented here would be a guess wearing a derivation's clothes |

**It does not write the task file, and `--check` is not a test.** A coefficient that has been swept in simulation
*should* differ from its starting point; `--check` exists so that a gait or a geometry moving out from under a number
is noticed, and so the differences read as a list of what has been tuned.

`make test-heuristic-parameters` covers the machinery underneath rather than the values, for the same reason: that
each robot's URDF still parses, that `initial_state`'s joint block still lines up with the reduced model, that the
contact frames are still reconstructible, that the gait table still yields a cadence, that the derived block is one
`LocomotionHeuristicConfig::validate()` would accept, and that the coefficients which need *data* are still left at
zero rather than filled in with something plausible.

The script also reports the three configuration facts that decide whether any of this reaches the solver at all
(`com_and_acom_tracking_cost` in `costs`, `contact_schedule_source: "contact_planner"`, and
`task_space_foot_cost.weights.pos_x` / `.pos_y`), because each makes a listed heuristic silently inert, and it reports
a task file still carrying one of the retired fields `use_com_and_acom_tracking`, `use_contact_basis_vector_inputs`,
`use_contact_planning` and `use_dcm_terminal_cost`, which the MPC refuses at start-up.

### What it found on the two robots that ship derived values

The **DRC Atlas** and the **EngineAI SA01** ship derived coefficients; the two Unitree robots still ship zeros.

On Atlas, the derivation exposed two facts that affect tuning:

* The **controller's LIP height** used to disagree with the robot's: `dcm_terminal_cost.com_height` and the contact
  planner's `shared.com_height` were both hand-set to `0.85`, while at `initial_state` the center of mass sits
  **1.0805 m** above the feet (27% higher, which made $\omega$ 13% too large everywhere it was used). Both fields are
  now left out, which each resolves to the model's height with `computeComHeightAboveFeet` (the same quantity this script
  computes), so $\omega = 3.01$ rad/s. The heuristic numbers derived from the pendulum were regenerated with it:
  `capture_point.com_height_override` is `1.0805` and `high_speed_turning`'s lean is `1.0805 / 9.81 = 0.1101`.
  `humanoid_centroidal_mpc:testNominalPendulum` and `test_derive_parameters.py` hold every shipped pendulum number to
  the model's, in C++ and in Python.
* **Atlas's nominal stance** has no outward room: `nominal_foothold.step_width` is `0.45 m`, which is
  `foot_separation.max_step_width`, while the legs naturally splay to `0.223 m`. A capture step pushed outward therefore
  runs past a bound the planner would have enforced, and `hip_centered_stepping` needs a `lateral_scale` of **2.528**
  just to reproduce the shipped stance.

Also note that Atlas lists `com_and_acom_tracking_cost` in `costs`, so **its three `base_pose` heuristics are inert
as shipped** - their values are derived and ready, and they do nothing until base-pose tracking is selected.

SA01 is the easier robot of the two to start on, and the contrast is instructive:

* Its **LIP height was always the model's**: `dcm_terminal_cost.com_height` and the planner's `shared.com_height` wrote
  out the model's `0.6124 m` by hand, and are now left out, which resolves to it, so the script emits no warning. It is
  also the *fastest* pendulum of the four robots ($\omega = 4.00$ rad/s), which makes its capture step the smallest
  and this heuristic the gentlest.
* Switching on **`hip_centered_stepping`** resizes nothing: `nominal_foothold.step_width` is `0`, so the stance to
  reproduce is the one the robot stands in, its contact frames 0.1503 m apart, and the hips at ±0.075 m already give
  0.150 m of it: `lateral_scale: 1.002` - the only robot here where the hips alone are the stance. (On G1, whose hips
  are 0.129 m apart but whose feet stand 0.237 m apart, the derived scale is 1.839; a scale of 1 there would narrow
  the stance by 11 cm the moment the heuristic is listed.)
* Its **coefficients** are larger where its stick is smaller and smaller where its geometry is:
  `pitch_per_forward_velocity` is `0.0611` against Atlas's `0.0407` only because the command limit is 0.8 m/s rather than
  1.2 - the lean at full stick is the same 2.8° on both. `in_place_turning.forward_stance_lever`, by contrast, is a third
  of Atlas's, because the stance it pivots about is a third as wide. A narrow robot has little to gain from that one.
* Its **foothold family needs `hip_centered_stepping`**: with `nominal_foothold.step_width: 0` nothing else keeps the
  feet apart, so start-up refuses a foothold list without it.
* Try **`periodic_orientation`** first: SA01's stance is the narrowest of the four (G1's hips are
  narrower, but G1 stands wider), so its lateral pendulum has the least margin and roll is the primary channel to shape, and the roll phase is the one
  coefficient in the family that is derived rather than fitted.

### Preconditions the layer checks for you

| Situation | What happens |
| --- | --- |
| Unknown name | **Error** at start-up, listing the valid names of that kind |
| A name in the wrong list | **Error**, naming the list it belongs in |
| A name listed twice | **Error** - the offsets are summed, so a repeat would silently double it |
| Foothold heuristic with `contact_schedule_source: "contact_planner"` | **Error** - the planner supplies footholds itself and never consults these, so it would be silently inert; they are the analytic *alternative* to it |
| `hip_centered_stepping` listed alone, or omitted while others are listed | **Warning** - neither half of Bledt's sum is a configuration he validates |
| `centripetal_acceleration` with `contact_input_parameterization: "basis_vectors"` | **Warning** - the reference moves onto the forward-kinematics path; watch the solve time |
| A list entry that is not a string, a list written as a block, or a value that is not a number | **Error** from the strict parser, naming the file, the line and the column - read as an empty list, such a typo would leave the configuration silently inert |
| A foothold heuristic without `hip_centered_stepping` on a robot whose `nominal_foothold.step_width` is 0 | **Error** - nothing would keep the feet apart, and the swing foot would be aimed at the stance foot |
| A base-pose heuristic with `com_and_acom_tracking_cost` in `costs` | **Warning** - the factory zeroes the base-pose block of `state_weights` and `final_state_weights`; it would do nothing |
| A foothold heuristic with `task_space_foot_cost.weights.pos_x` and `pos_y` both 0, or the foot cost unlisted | **Warning** - nothing tracks the landing target; it would not move the feet |
| A clamp set to zero (`capture_point.maximum_offset`, or both `centripetal_acceleration` limits) | **Error** - every clamp here stops a bad measurement reaching the solver, so removing one is not a way to configure it |
| A negative `scale` on either wrench heuristic, a negative `capture_point.gain`, or a negative `hip_centered_stepping` scale | **Error** - it would invert the heuristic rather than reduce it |

### Two things that will make a heuristic do nothing (both warned about at start-up)

* The **base-pose family** is inert while `costs` lists `com_and_acom_tracking_cost`, because the cost factory zeroes
  the base-pose blocks of `state_weights` and `final_state_weights` in that mode and `ComAndAcomTrackingCost` reads the
  target trajectory directly instead.
* The **foothold family** needs `task_space_foot_cost.weights.pos_x` and `pos_y` to be non-zero: they are `0` on every
  robot shipped here, because with `zero_velocity` in `hard_constraints` the stance foot is pinned by the schedule and
  placement follows from it. A landing target multiplied by a zero weight is dead weight. (This trap is already live
  on the DRC Atlas, which ships `nominal_foothold.step_width: 0.45` alongside `pos_x: 0`.)

### One thing a foothold heuristic changes even with those weights at zero

`CentroidalMpcEndEffectorFootCost::getParameters` branches on whether `getSwingFootReference()` returns a value **at
all**. Its `else` branch (the live one today on every robot here) zeroes `pos_x`/`pos_y` *and* installs a swing
velocity reference (the commanded CoM velocity) with a hard-coded sqrt-weight of 1.5, added specifically to stop the
swing foot dragging behind the moving body. Listing any foothold heuristic produces a landing target, which flips the
cost into the `if` branch and drops that override. The anchor the heuristics are summed onto is the base predicted
at touch-down, which moves with the commanded velocity, so the reference they replace it with is of the same kind -
but this is a behavior change independent of the weights, and it is the reason to sweep a foothold heuristic with the
weights raised rather than to leave one listed "harmlessly" at zero.

### Hot reload

The **coefficients** follow edits to `task.textproto`, like the cost weights beside them: the parameter updater
(`MpcParameterUpdaterModule`, `humanoid_common_mpc/parameter_update/`) re-reads the file and its
`LocomotionHeuristicsApplier` calls `LocomotionHeuristicLayer::reconfigure()` from the pre-solve hook. The **lists** are not
reloaded - which heuristics are listed decides how the reference manager and the two input costs are wired at
construction, so changing one under a running solver would be a different controller rather than a retuned one; the
layer says so once per distinct edit and keeps running. A reload applies all of the new coefficients or none: they are
configured into fresh heuristics and swapped in only if every one is accepted, so a rejected edit leaves the running
controller as it was.

### Centroidal only, for now

The layer is built and installed by `CentroidalMpcInterface::setupLocomotionHeuristics()`, and `WBMpcInterface` has no
counterpart - nothing in the **whole-body** path reads a `locomotion_heuristics` block. The whole-body task file
therefore carries a comment saying so rather than a block that would be silently inert. The commanded yaw rate and
velocity the foothold and wrench heuristics read no longer stand in the way: the reference manager reads them through
the robot model's `getTargetPlanarCommand()`, which a whole-body knot answers from its base twist (the defect fix D3 of
humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 4.7). The parameter updater
that hot-reloads the coefficients is shared by both formulations; only the centroidal one registers the
`LocomotionHeuristicsApplier` (the schema marks the block `formulations: "centroidal"`, so the tuning GUI labels it not
applicable on a whole-body file). The base-pose seam itself is in the shared reference manager and would work
unchanged.

## 5. Tests

| Target | Covers |
| --- | --- |
| `//humanoid_nmpc/humanoid_common_mpc:testLocomotionHeuristics` | the registry, the loader, each of the ten formulae, the rejected combinations, and the **parity property** that empty lists are an exact no-op |
| `//humanoid_nmpc/humanoid_common_mpc:testLocomotionHeuristicFormulas` | every formula off the degenerate point (yaw ≠ 0, both feet, every term and clamp, the stance-proportional terms), sums of several heuristics, the anchor and world-frame flags in both list orders, all 42 keys round-tripped, every validation branch, the start-up rejections and warnings, and an all-or-nothing hot reload |
| `//humanoid_nmpc/humanoid_common_mpc:testShippedLocomotionHeuristicBlocks` | every robot's shipped task file loads with all three lists empty, uncommenting any one candidate name gives a valid file with that name listed, and every parameter of the schema is written in the file |
| `//humanoid_nmpc/humanoid_common_mpc:testLocomotionHeuristicSeams` | the foothold anchor keeps the feet with the body through a sustained sidestep at any heading (the stance-foot anchor it replaced as the drifting control), and the shaped attitude's angular velocity and tilt rate are its derivative, against a central difference of the rotation |
| `//humanoid_nmpc/humanoid_common_mpc:testArmSwing` | the procedural arm swing on every robot with arms: the derived joint senses against the forward kinematics, the swing end to end, parity on G1 and R1, independence of the node's state, and the `arm_swing` block |
| `//humanoid_nmpc/humanoid_common_mpc:testStanceDutyFactor` | $\beta$ per stride: the gait's duty factor at every node and for both feet on `walk`, standing, the start and end of a walk, and the default-constructed schedule that carries one `FLY` mode and no events |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testLocomotionHeuristicIntegration` | end to end on the DRC Atlas model through `SwitchedModelReferenceManager`: all three seams with a walking mode schedule, including the landing target holding still through a swing, the impulse reference averaging to the weight over a stride, the base-pose rotation and indices at a non-zero yaw, the gait phase, and the shaped terminal cost; $\mathbf{r}_{hip}$ from the URDF; the task-file errors returned as a `Status` |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testMpcParameterUpdaterModule` | the coefficients hot-reload from the task file, a malformed edit is rejected without changing them, and a list edit is ignored |
| `make test-heuristic-parameters` | `derive_parameters.py`: each robot's model and gait still derive a block the loader accepts, the closed forms hold, the printed block round-trips, and the derived keys are the loader's (not a Bazel target; see above) |

```bash
bazel test //humanoid_nmpc/humanoid_common_mpc:all //humanoid_nmpc/humanoid_centroidal_mpc:testLocomotionHeuristicIntegration
```

## 6. Where the code is

```
humanoid_nmpc/humanoid_common_mpc/{include/humanoid_common_mpc,src}/locomotion_heuristics/
  LocomotionHeuristic.h                 root abstract base: name(), configure(), describe()
  BasePoseHeuristic.h                   kind base (A) + BasePoseOffset + its context
  FootholdHeuristic.h                   kind base (B) + its context
  WrenchHeuristic.h                     kind base (C) + its context
  LocomotionHeuristicFormulation.{h,cpp}  the names, the three kinds, the lists, validate(), warnings()
  LocomotionHeuristicConfig.{h,cpp}       the ten parameter structs and validate()
  LocomotionHeuristicModelParameters.{h,cpp}  m, g, z_com, r_hip, derived once from the URDF
  LocomotionHeuristicFactory.{h,cpp}      name -> instance
  LocomotionHeuristicLayer.{h,cpp}        the assembled layer: start-up checks, sums, hot reload, banner
humanoid_nmpc/humanoid_common_mpc/{include/humanoid_common_mpc,src}/
  reference_manager/SwitchedModelReferenceManager   the three seams: shapePoseBaseRtWorld(), nominalFoothold(), getDesiredInput()
  contact_planning/ContactScheduleAdaptation         stanceDutyFactor(), upcomingStanceDuration(), previousTouchDownTime()
  cost/BasePoseShapedQuadraticStateCost               the terminal cost on the shaped base pose
  base_pose/  OrientationCompensation, PeriodicOrientation, HeightCompensation
  foothold/   HipCenteredStepping, CapturePoint, TranslationalStepping, InPlaceTurning, HighSpeedTurning
  wrench/     ImpulseScaling, CentripetalAcceleration
```

**Adding an eleventh heuristic** is a handful of edits, all tied together by `LINT.IfChange` / `LINT.ThenChange` so
the linter catches a half-finished one: the name in `knownHeuristicNames()`, the line in `LocomotionHeuristicFactory`,
the parameter struct in `LocomotionHeuristicConfig`, its message and field in
`humanoid_nmpc/humanoid_mpc_config/locomotion_heuristics_config.proto` (with a default and, where needed, `tuning`
options; the GUI then renders it with no GUI code) and their conversion in `locomotionHeuristicConfigFromConfig()`,
the block in each robot's `task.textproto`, its derivation in `derive_parameters.py`, the name set in
`test_derive_parameters.py`, and the summary table above. Plus the class itself - one class per header/cpp pair, in the subdirectory of its kind.

### A note on the procedural arm swing

`SwitchedModelReferenceManager::getDesiredState()` contains one Bledt-style heuristic that predates this subsystem:
the procedural arm swing on four arm joints. At gait phase $\Phi$ and commanded forward speed $v$ the left arm swings
forward by $s = \sin(2\pi(\Phi - \phi_0))\,v$ and the right one back by as much; each shoulder turns by $g_s s$ in
the sense that swings its arm forward, each elbow by $g_e s$ in the sense that flexes it. The task file's `arm_swing`
block selects the generator by name (`procedural`, or `none`) and carries $g_s$, $g_e$ and $\phi_0$ (0.15, 0.15, 0.15
by default, the numbers that used to be hard-coded), hot-reloaded by `ReferenceManagerApplier`. The sense of each
joint is not configured: `deriveArmSwingDirections()` takes it from the model at the initial posture, the shoulder's
from the forward velocity of what it carries and the elbow's from the rate at which it closes the elbow. On G1 and R1
that gives the shoulder-pitch convention the swing always used, bit for bit; on DRC Atlas, whose v5 arm has no
shoulder pitch, the task file names the shoulder **yaw** `shz` (the arm hangs lowered by `shx`) and the elbow
**flexion** `elx` (it used to name `shx`, which abducts the arm, and `ely`, which twists the upper arm) with
opposite senses on the two sides. The forward speed is read in the **reference's** heading, not the heading of the
node's state: a reference that depends on the state is not the constant the Gauss-Newton step treats it as.
`testArmSwing` covers the senses against a finite difference of the forward kinematics on every robot with arms, the
motion end to end, the bit-for-bit parity on G1 and R1, and the block's conversion. The swing stays off under
`com_and_acom_tracking_cost`, where the arm motion emerges from the ACoM cost.
