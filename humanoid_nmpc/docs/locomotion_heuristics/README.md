# Locomotion Heuristics: Bledt's Regularized Predictive Control in this MPC

An implementation of the ten regularization heuristics of

> Gerardo Bledt, *Regularized Predictive Control Framework for Robust Dynamic Legged Locomotion*,
> PhD thesis, Massachusetts Institute of Technology, February 2020.
> Chapter 3 (the RPC formulation), Chapter 4 (heuristic extraction), **Appendix C** (the ten heuristics).
> Related papers: Bledt & Kim, *Implementing Regularized Predictive Control for Simultaneous Real-Time Footstep and
> Ground Reaction Force Optimization*, IROS 2019; Bledt & Kim, *Extracting Legged Locomotion Heuristics with
> Regularized Predictive Control*, ICRA 2020.

Every heuristic is selected **by name** from one of three lists in the robot's `config/mpc/task.yaml`, and every
list ships **empty**, so this subsystem is inert until someone opts a robot in one name at a time.

```yaml
locomotion_heuristics:
  base_pose: []   # orientation_compensation, periodic_orientation, height_compensation
  foothold:  []   # hip_centered_stepping, capture_point, translational_stepping,
                  # in_place_turning, high_speed_turning
  wrench:    []   # impulse_scaling, centripetal_acceleration
  # ... one parameter block per name, keyed by the name
```

---

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

**The key observation for this repository.** The quadratic regularization $J_R$ already exists here — it is
`state_quadratic_cost`, `input_quadratic_cost` and `task_space_foot_cost`, with $W_k$ spelled out as the `Q`, `R`
and `task_space_foot_cost_weights` blocks of `task.yaml`. What was missing is $\mathcal{H}$. The references those
costs regularize against are, today:

| Channel | Reference before this subsystem |
| --- | --- |
| base roll, pitch | **a hard zero at every speed** — `TargetTrajectoriesCalculatorBase::integrateTargetBasePose` writes `targetPose[4] = targetPose[5] = 0.0` on every call |
| base height | the operator's commanded height, adapted to the terrain |
| swing foot $x, y$ | **nothing at all** without a contact planner — `nominalFoothold()` returns `std::nullopt` and the foot cost zeroes its own `pos_x` / `pos_y` weights |
| contact wrench | weight compensation: $m g$ split evenly over the scheduled stance feet, purely vertical |

So this is a **reference-shaping layer**, not a new cost family. Nothing here adds a term to the optimal control
problem; each name modifies the reference that a term the task file already lists is regularizing against. Three
consequences follow, and the whole design leans on them:

1. **An empty list is exactly the previous behaviour**, bit for bit, because the sum of no offsets is zero and
   nothing is recomputed. This is pinned by a parity test.
2. **Order within a list is documentary**, because the offsets are summed and addition is commutative. (Unlike the
   contact planner's `costs` list, whose order is the floating-point accumulation order.)
3. **The list and the coefficients are independent switches.** Every fitted coefficient ships at zero, so
   "does listing this change anything at all" is a separate experiment from "what should the number be".

---

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
  |   getDesiredState()       nominalFoothold()       getDesiredInput()                     |
  |   per shooting node       per swing              per shooting node                      |
  +=========================================================================================+
         |                       |                          |
         | x_nominal             | landing target           | u_nominal
         v                       v                          v
  StateQuadraticCost      getSwingFootReference()     InputQuadraticCost
  StateInputQuadraticCost         |                   StateInputQuadraticCost
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
| **(A)** `getDesiredState()` | base roll, pitch, height of $\mathbf{x}_{nom}$ | per shooting node, per SQP iteration | `Q(8,8)`, `Q(9,9)`, `Q(10,10)`, `Q(11,11)` |
| **(B)** `nominalFoothold()` | the swing foot's landing target | once per swing, from the last measurement | `task_space_foot_cost_weights.pos_x / pos_y` |
| **(C)** `getDesiredInput()` | the contact-force reference $\mathbf{u}_{nom}$ | per shooting node, per SQP iteration | `R`'s contact-wrench block |

### Why these three seams and not others

**(A) is the only per-node hook on the state reference,** and it has exactly two callers in the repository —
`StateQuadraticCost` and `StateInputQuadraticCost` — which are precisely the two terms carrying `Q`'s base-pose
block. Shaping the `TargetTrajectories` instead would fail three ways: it has three knots over the whole horizon, so
a limit cycle at gait frequency (`periodic_orientation`, roughly 2 Hz over a 1 s horizon) is unrepresentable; it is
published one solve ahead of the solver that consumes it; and it is also what the operator's own delta-pose commands
are measured against, so shaping it would make the commanded pose drift away from what was asked for. There is a
precedent in this very function: the procedural arm swing is already a phase-dependent reference heuristic written
there by hand.

**(B) is the base class's entire opinion about foot placement,** with one caller, `getSwingFootReference()`, which
blends from the actual lift-off position to it with a cubic $-\tau^3 + \tau^2 + \tau$ and hands the result to the
foot task-space cost. That cost switches its $x,y$ position weights **on** precisely when a value is present. One
addition inside `nominalFoothold()` therefore reaches the position reference, the velocity reference (the cubic's
analytic rate falls out) and the swing-plane heading query, with no other edit anywhere.

**(C) is new plumbing.** The input channel of `TargetTrajectories` is not a reference at all — it is constructed
all-zero and commented *"they are not used"*, and both input costs ignored it and built their own nominal from
`weightCompensatingInput()`. That nominal is **contact-flag dependent** and the stance set changes several times
inside one horizon, which a three-knot trajectory cannot carry; the reference manager is the only object that knows
the contact flags at an arbitrary node time. `getDesiredInput()` makes exactly the old call when no wrench heuristic
is listed, so routing the two costs through it is a no-op on every robot shipped here.

### The invariant that makes it thread-safe

**No heuristic performs forward kinematics, and no heuristic holds mutable state.** Everything model-dependent
($m$, $g$, the nominal CoM height, $\mathbf{r}_{hip}$) is derived once at start-up into
`LocomotionHeuristicModelParameters`; everything measurement-dependent is latched once per solve in
`captureMeasuredState()`, on the solver thread, and handed over in a per-call context struct. Every heuristic is
therefore a pure function of (its coefficients, the context), the layer needs no lock on the read path, and
`reconfigure()` — the hot-reload entry point — runs from the pre-solve hook, before any SQP worker exists.

---

## 3. The ten heuristics

Notation follows Appendix C. $\mathbf{p}$ is the CoM position, $\dot{\mathbf{p}}$ its velocity, $\dot{\mathbf{p}}_d$
the commanded velocity, $\boldsymbol{\Theta}$ the body orientation, $\boldsymbol{\omega}$ the angular rate,
$\dot{\psi}$ the yaw rate, $\Phi$ the gait phase, $\mathrm{PTP}(\cdot)$ the projection onto the ground plane, and
$\mathrm{sgn}_i = +1$ for the left foot, $-1$ for the right.

### Table C.1 — the analytic heuristics

These are derived from physics. They have **no fitted coefficients**, only unit scales, so for these the list entry
alone is the change.

#### 3.1 `hip_centered_stepping` — $\mathcal{H}_{\mathbf{r}}(\boldsymbol{\Theta}) = \mathrm{PTP}\!\left(R(\boldsymbol{\Theta})\,\mathbf{r}_{hip}\right)$

The foot placed under its own hip: this leg's hip position in the base frame, rotated into the world and projected
onto the ground.

$$\Delta\mathbf{r}_i \;=\; R_z(\psi_{meas}) \begin{bmatrix} s_x\, r_{hip,i,x} \\ s_y\, r_{hip,i,y} \end{bmatrix}$$

$R(\boldsymbol{\Theta})$ is the full body rotation in the dissertation and $R_z(\psi)$ here, which is the same
simplification the control model already makes for its own orientation dynamics (section 3.2.1: roll and pitch of a
walking base are small), and $\mathrm{PTP}$ would flatten their contribution anyway. $\mathbf{r}_{hip}$ is **derived
from the URDF at start-up** and is never a task-file key; $s_x, s_y$ are the two scales that stretch it.

This is the base case of figure 4-8, the panel every later heuristic is measured against. Alone it "stably stands and
takes a few steps forward" and falls as soon as it has speed, because a foot placed under the hip at lift-off is well
behind the robot by touch-down. **It is also the one heuristic here that moves the anchor** of the foothold reference,
from the stance foot to the measured base — a change `SwitchedModelReferenceManager.h` argues against for good
reasons (in single support the base sits roughly over the stance foot, so offsets taken from it give less separation
than intended). It is registered because Bledt never uses it alone: it is the term the Table C.2 velocity-dependent
heuristics are summed on top of. The layer warns when it is listed alone, and warns again when the others are listed
without it.

#### 3.2 `capture_point` — $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}}, \Phi) = \mathrm{PTP}\!\left(\sqrt{p_z/g}\,(\dot{\mathbf{p}} - \dot{\mathbf{p}}_d)\right)$

$$\Delta\mathbf{r}_i \;=\; k \sqrt{\frac{z_{com}}{g}} \left(\dot{\mathbf{p}}_{meas} - \dot{\mathbf{p}}_{cmd}\right), \qquad \|\Delta\mathbf{r}_i\| \le \Delta_{max}$$

$\sqrt{z/g} = 1/\omega$ is the time constant of a linear inverted pendulum of height $z$ — the same $\omega$ this
repository's DCM terminal cost and its ZMP/DCM viewer markers already use. This is Pratt's capture point applied to
the **velocity error** rather than to the velocity, which is what makes it the one heuristic in the family that is
silent while the robot is tracking its command and grows only when a push, a slip or a model error has opened a gap.

The clamp is on the **magnitude**, not per axis: a per-axis clamp would rotate the offset towards the diagonal
exactly when it is largest, which is when the robot most needs the foot to go where the push came from.

Under this controller this is the heuristic with the clearest mandate. With `zero_velocity` in `hard_constraints` the
stance foot is pinned by the schedule; the foot cost's $x,y$ weights are the only thing in the problem that can move a
foot sideways in response to a push, and nothing is currently writing them a target.

#### 3.3 `impulse_scaling` — $\mathcal{H}_{\mathbf{f}}(\Phi) = \dfrac{m g}{F \beta}$

Over one gait cycle of period $T$ the vertical impulse the feet deliver must equal the weight times the cycle. A foot
that is on the ground for a fraction $\beta$ of the cycle therefore has to average $mg/(F\beta)$ **while it is down**,
not $mg/F$. Bledt credits the idea to the vertical impulse scaling of the MIT Cheetah 2's bounding controller
(section 4.1).

**$F$ is the number of feet the robot HAS, a constant — not the number in contact at this instant.** That
distinction is the whole content of the heuristic. The reference this layer shapes is $W / n_{stance}(t)$, which sums
to exactly $W$ at *every* instant; Bledt's is $W/(F\beta)$, the same value on every stance foot whatever
$n_{stance}(t)$ happens to be, so its total varies over the cycle — below $W$ in single support, above it in double
support — and averages, over one cycle,

$$\langle n_{stance}\rangle \cdot \frac{W}{F\beta} \;=\; (F\beta)\cdot\frac{W}{F\beta} \;=\; W,$$

because the mean number of feet on the ground *is* $F$ times the duty factor. That is the impulse budget. Scaling
$W/n_{stance}(t)$ by $1/\beta$ instead would put $W/\beta$ on the ground at every instant and deliver $WT/\beta$ over
the cycle — it would regularize the solver towards accelerating the centre of mass upwards for ever.

The offset is therefore written as the difference from the existing reference, with `scale` $s$ blending:

$$\Delta f_{z,i} \;=\; \mathrm{clamp}\!\left(\frac{W}{n_{stance}} + s\left(\frac{W}{F\bar\beta_i} - \frac{W}{n_{stance}}\right),\; 0,\; \rho_{max}\frac{W}{n_{stance}}\right) - \frac{W}{n_{stance}}, \qquad \bar\beta_i = \max(\beta_i, \beta_{min})$$

so an empty list is exactly today's reference. Two sanity checks fall straight out: standing ($\beta = 1$,
$n_{stance} = F$) and a pure alternating single support ($\beta = 1/2$, $n_{stance} = 1$) both give back the existing
reference exactly, so the correction bites only where the two genuinely disagree — a gait with double support.

$\beta_i$ is measured by `stanceDutyFactor()` over a **fixed window of the MPC horizon's length**, latched once per
solve. Both the length and its being fixed matter: this mode schedule has no cycle of its own — the gait scheduler or
the contact planner may re-time it at any moment — and measuring instead over "whatever is left until the last
scheduled event" would shrink the window as the node index grows, making $\beta$ a function of *where in the horizon
it is asked* rather than a property of the gait. The lower clamp at zero and the upper clamp at $\rho_{max}$ times
weight compensation are load-bearing rather than defensive: $\beta \to 0$ at the onset of a flight phase and
$1/\beta$ is unbounded there, and a force reference that pulls the robot down through the floor is not one any foot
can track.

What the correction buys on a biped is anticipation: during double support it asks the feet to push a little harder
than static equilibrium requires, because a single-support phase is coming in which one of them will carry everything.

#### 3.4 `centripetal_acceleration` — $\mathcal{H}_{\mathbf{f}}(\boldsymbol{\Theta} \times \dot{\mathbf{p}}) = m\,\boldsymbol{\omega} \times \dot{\mathbf{p}}$

The horizontal force that holds the robot on a circular path. With $\boldsymbol{\omega} = \dot\psi \mathbf{e}_z$ and a
horizontal velocity,

$$\boldsymbol{\omega} \times \dot{\mathbf{p}} = \begin{bmatrix}0\\0\\\dot\psi\end{bmatrix} \times \begin{bmatrix}v_x\\v_y\\0\end{bmatrix} = \begin{bmatrix}-\dot\psi\, v_y\\ \dot\psi\, v_x\\ 0\end{bmatrix}, \qquad \Delta\mathbf{f}_i = \frac{s\,m}{n_{stance}} \left(\boldsymbol{\omega} \times \dot{\mathbf{p}}\right),$$

a force of magnitude $m|\dot\psi||\mathbf{v}|$ pointing towards the centre of the turn — the companion of
`high_speed_turning`: one says where to put the foot for a fast turn, the other which way to push once it is there.

**Two caveats, both worth reading before listing it.**

*It is the only heuristic of the ten whose force is horizontal,* so it is the only one whose expression depends on the
foot's orientation. Under `useContactBasisVectorInputs: true` the input lives in the local contact frame, so the whole
contact-force reference must be written through `setContactForceInWorldFrame()`, which costs one forward-kinematics
pass per shooting node per SQP iteration. The layer switches paths by itself when this name is listed and logs a
warning; the other nine never pay it.

*It may well be the wrong thing to want here.* Bledt's control model is a single rigid body whose force reference is
the only thing telling the QP how hard to push. These dynamics are **full centroidal** and the state cost already
tracks the commanded linear and angular momentum — so $m\,\boldsymbol{\omega} \times \mathbf{v}$ is the derivative of
a momentum this problem is already regulating: an *output* the optimizer must produce, not information the reference
lacks. Supplying it again at lower fidelity can only help where the optimizer would have been slow to find it, and
can fight the momentum cost where the two disagree, as they do in double support and wherever the friction cone binds.
It is implemented because it is one of Bledt's ten and the seam for it is clean; `scale` is the knob for deciding in
simulation whether it earns its place.

The clamp is a fraction of body weight by default rather than an absolute force, so it scales across the robots here,
whose masses differ by a factor of five; a foot cannot pull sideways harder than friction allows, and the wrench
cone's coefficient is around 0.5 on these robots.

### Table C.2 — the data-extracted heuristics

These were not designed. Chapter 4 describes the framework that produced them: run the RPC **offline**, without
real-time constraints, over a systematic sweep of commands; collect every state, input and command into a candidate
variable set $V$; fit simple models — polynomials up to degree 9 and sums of up to 8 sines — to every pair; and keep
the relationships with high $R^2$ as heuristic candidates, after validating them on a *different* exploration.

Two cautions from section 4.5 carry over to anyone extending this layer. **Overfitting**: a degree-9 polynomial fit
the pitch–velocity relationship with $R^2 = 0.94$, almost as well as the linear one at $0.95$, and then collapsed on
validation data — which is exactly why the shipped forms are affine and sinusoidal rather than flexible.
**False dependency**: an 8-sine model related forward velocity to *wall-clock time* with $R^2 = 0.99$, purely because
the exploration stepped the command on a fixed schedule. The framework is not a black box; it proposes, the engineer
disposes.

**None of Bledt's fitted numbers is a default here.** They were fitted to a 9 kg quadruped. Every coefficient below
ships at zero.

#### 3.5 `orientation_compensation` — $\mathcal{H}_{\boldsymbol{\Theta}}(\dot{\mathbf{p}}) = a_1 \dot{\mathbf{p}} + a_0$

$$\Delta\theta_{roll} = a_1^{\theta} v_y^{base} + a_0^{\theta}, \qquad \Delta\theta_{pitch} = a_1^{\phi} v_x^{base} + a_0^{\phi},$$

each clamped to $\pm\theta_{max}$. Roll follows the **lateral** command and pitch the **forward** one — the pairing the
regressions found, with $R^2 = 0.983$ and $0.953$ (equations 4.11 and 4.12, figure 4-5). The velocities are the
commanded CoM velocity in the *reference* base's yaw frame, so $a_1$ is a lean per unit of commanded speed and $a_0$ a
standing trim that survives at zero speed.

Bledt reports two distinct benefits (section 4.3, figure 4-12): the robot trots "in a much more controlled manner",
and — less obviously — the **simplified orientation dynamics become more valid**, because a body that is closer to
level at speed is closer to the small-roll-and-pitch approximation the control model is built on. That argument
transfers directly: this repository makes the same approximation, in `CentroidalMpcRobotModel`'s use of the yaw-only
rotation.

The clamp is not defensive trimming. The command this reads is filtered but not rate limited, and an affine law
applied to it has no bound of its own; a reference tilt the legs cannot reach is worse than no reference at all,
because the state cost pulls towards it at every node and the solver spends iterations on a target it cannot have.

Sign note: the base pose is Euler ZYX about a z-up world, in which a **positive** pitch rotates the body's $+x$ axis
towards $-z$ — it puts the nose **down**. Leaning into a forward command is therefore a *positive*
`pitchPerForwardVelocity`, which is also the sign of Bledt's own fitted $+0.0725$ rad per m/s.

#### 3.6 `periodic_orientation` — $\mathcal{H}_{\boldsymbol{\Theta}}(\Phi) = b_1 \sin(c_1 \Phi + d_1)$

$$\Delta\theta_{roll} = b_1^{\theta}\sin\!\left(c_1^{\theta}\Phi + d_1^{\theta}\right), \qquad \Delta\theta_{pitch} = b_1^{\phi}\sin\!\left(c_1^{\phi}\Phi + d_1^{\phi}\right).$$

The orientation limit cycle a walking robot falls into on its own. This is the clearest illustration of what the
extraction framework is for: the steady-state pitch during a trot is genuinely complicated, and Bledt makes the point
that it is *"currently too difficult to analyze without data-driven methods"* — yet a single sine at the step
frequency fits it with $R^2 = 0.835$ (equation 4.13, figure 4-6), and summed with the velocity-dependent term of
§3.5 it reproduces the measured pitch closely (figure 4-7):

$$\mathcal{H}_{\phi}(\dot{p}_x, \Phi) \;=\; \underbrace{a_1^{\phi}\dot{p}_x + a_0^{\phi}}_{\text{§3.5}} \;+\; \underbrace{b_1^{\phi}\sin(c_1^{\phi}\Phi + d_1^{\phi})}_{\text{§3.6}}$$

The value of the fit is not its accuracy but that its four numbers *mean* something an engineer can turn: an
amplitude, a frequency and a phase. Bledt's own $c_1 \approx 12.566 \approx 4\pi$ recovers the trot's two
complementary diagonal pairs per gait cycle, which is a check that the fit found the physics rather than the data.

**This is the heuristic that could not be expressed by shaping the target trajectory**, whose three knots over a
one-second horizon would sample two or three whole periods and interpolate linearly between them. It only works at the
per-node seam. $\Phi$ comes from `getPhaseVariable(t)` and runs over $[0,1)$ for one left–right cycle, so
$c_1 = 2\pi$ is one period per cycle and $c_1 = 4\pi$ one per step; it **freezes** at 0 or 0.5 through double support
rather than advancing, so the offset is held constant there instead of continuing its cycle. Phase $[0, 0.5)$ is the
`LF` mode and $[0.5, 1)$ the `RF` mode, and those name the foot **in contact**, not the one swinging
(`modeNumber2StanceLeg` maps `LF` to contact flags `{true, false}`) — so the first half of the cycle is left stance
with the right foot in the air.

#### 3.7 `height_compensation` — $\mathcal{H}_z(\dot{p}_x) = a_2 v^2 + a_1 v + a_0$

$$\Delta z = \mathrm{clamp}\!\left(a_2 \|\mathbf{v}_{cmd}\|^2 + a_1 \|\mathbf{v}_{cmd}\| + a_0,\; \pm \Delta z_{max}\right)$$

Taken in the **magnitude** of the commanded horizontal velocity, because crouching to walk forwards and rising to walk
backwards is not something any legged system does, and an odd polynomial in a signed speed would do exactly that.

Bledt reports this one as having made the motion "smoother and more consistent" and as having marginally reduced the
solve time, without by itself enlarging the viable operating region (table 4.1) — a fair description of what a
well-chosen regularization does when it is not fixing a failure: it shapes the cost space so the optimizer converges
sooner on something it was going to find anyway. On a humanoid it also buys something a quadruped needs less of:
lowering the base shortens the leg and gives the swing more vertical room within the same joint range, at the cost of
knee torque headroom.

Applied as an **offset** and never as an absolute height: the same channel is where
`adaptToCurrentGroundHeight()` writes the terrain, and an absolute value would silently become a terrain override.

#### 3.8 `translational_stepping` — $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}}) = a_1 \dot{\mathbf{p}} + a_0$

$$\Delta\mathbf{r}_i^{base} = \begin{bmatrix} a_1^{x} v_x^{base} + a_0^{x} \\ a_1^{y} v_y^{base} + \mathrm{sgn}_i\, a_0^{y} \end{bmatrix}, \qquad \Delta\mathbf{r}_i = R_z(\psi_{meas})\,\Delta\mathbf{r}_i^{base}$$

Step in the direction of travel. The largest single effect in the dissertation (section 4.3, figure 4-9): a foot placed
under the hip at lift-off has fallen behind the robot by touch-down, so the forces over the stance that follows are
biased into tipping the robot the way it is already going, and the leg reaches its kinematic limit mid-stance. Placing
the step *ahead* of the hip by roughly half of what the robot will travel during the stance "gives it twice as much
stance time capability", and the maximum velocity rose from 1.6 m/s to 3.5 m/s.

The affine law is applied **in the base's yaw frame and rotated back**, which is not the same as applying it in the
world: the forward and lateral coefficients differ, and the robot's forward is not the world's $x$. The lateral
constant is signed per foot, so one number in the task file widens the stance rather than shifting the whole robot
sideways; the velocity-proportional part is not, because both feet should step the same way when the robot is asked to
move sideways.

#### 3.9 `in_place_turning` — $\mathcal{H}_{\mathbf{r}}(\dot{\psi}) = a_1 \dot{\psi} + a_0$

$$\Delta\mathbf{r}_i^{base} = \begin{bmatrix} \mathrm{sgn}_i\, a_1^{x} \dot\psi_{cmd} + a_0^{x} \\ a_1^{y} \dot\psi_{cmd} + \mathrm{sgn}_i\, a_0^{y} \end{bmatrix}$$

The rotational counterpart of §3.8, found the same way and for the same reason: with only the translational term the
feet lagged the hips through a fast spin until they reached the end of their workspace, exactly as they lagged the
body through a fast walk (figure 4-10). Adding it let the feet **lead** the hips into the turn and raised the
achievable yaw rate from 1.8 rad/s to 4.5 rad/s.

Note that the **forward** term is signed per foot here and the lateral one is not — the opposite of §3.8, and that is
what makes this a rotation. The direction is worth getting right: a foot at $\mathbf{r}$ relative to the base moves at
$\boldsymbol{\omega}\times\mathbf{r}$, so the *left* foot at $(0, +d, 0)$ moves at
$(0,0,\dot\psi)\times(0,d,0) = (-\dot\psi d, 0, 0)$ — **backwards** for a counter-clockwise turn, while the right
foot moves forwards. That is how a turn in place is actually done: to turn left you swing the right foot forward and
around it, and the left foot back.

The implementation carries that minus sign, so a **positive** `forwardPerYawRate` means "place each foot further along
the direction its own hip is travelling", i.e. lead the hips into the turn. Without it, a positive coefficient would
drive the feet to *trail* the hips by twice the intended amount — precisely the workspace-exhaustion failure the
heuristic exists to remove. The lateral rate term is unsigned because it translates the whole stance sideways, which
is what a turn with a radius does; the lateral *constant* is signed, and that is the one that opens or closes the
stance.

$\dot\psi$ is the **commanded** yaw rate, recovered by inverting the angular channel of the target's momentum,
$h_z = I_{zz}\dot\psi / m$, with the composite yaw inertia latched per solve. The target's base *yaw* is not usable:
it blends the measured heading over its first stretch and the planned-heading override rewrites it, so differentiating
it would recover the reference's own smoothing rather than the operator's command.

#### 3.10 `high_speed_turning` — $\mathcal{H}_{\mathbf{r}}(\dot{\mathbf{p}} \times \boldsymbol{\omega}) = a_1 (\dot{\mathbf{p}} \times \boldsymbol{\omega}) + a_0$

The result the dissertation is most pleased with, and the best argument for its framework. With both §3.8 and §3.9 in
place the robot still fell when asked to do both at once: it *"tended to fall outwards along the turning radius as if
it were an object slipping off a spinning plate"*, the outer leading foot running out of workspace with no contact
foot left able to stabilise the body. The framework surfaced the cross term as statistically significant, and only
**afterwards** did Bledt recognise what it was — equation 4.31, the foot placement that lines up with the resultant of
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

---

## 4. Switching one on

Every list ships empty on every robot, deliberately: each of these changes the closed loop, none has been validated in
simulation under hardware-like conditions on these robots, and Bledt's coefficients were fitted to a quadruped one
fifth the mass. The dissertation's own procedure is the right one to copy — figure 4-13 adds one heuristic per panel
and measures the viable operating region after each.

1. **Uncomment one name.** Its block is already at zero, so this on its own is still a no-op; launch and confirm that
   nothing changed and that the start-up banner lists the heuristic.
2. **Sweep its coefficients** in the tuning GUI. The sliders are generated from the YAML, so they appear with no code
   change. *The slider range is derived from the value, so a coefficient at 0 gets a $(0, 1)$ slider and cannot be
   dragged negative* — type the negative value into the entry box, which widens the bounds. Several of these are
   physically negative.
3. **Measure something.** The dissertation's metric is the viable operating region: the set of $(\dot p_x, \dot p_y,
   \dot\psi)$ commands the robot survives for several gait cycles.
4. **Only then add the next name.**

### Deriving the coefficients: `derive_parameters.py`

Most of these numbers are not free. They follow from the robot's mass and geometry, from the cadence of the gait it
is walking with, and from the command limits the operator can reach — so they are derived by a script rather than
maintained by hand, and the derivation of each is printed beside it:

<!-- LINT.IfChange(derive_parameters_usage) -->
```bash
make derive-heuristic-parameters ROBOT=drc_atlas
```

`ROBOT` takes `drc_atlas`, `engineai_sa01`, `unitree_g1` or `unitree_r1` — the centroidal packages; the whole-body one
has no layer to configure. Calling the script directly gives the rest of its options: `--gait walk` sizes the stepping
coefficients for another cadence, `--check` compares the shipped block against the derivation instead of printing
YAML, and `--task-file` / `--urdf` / `--reference-file` take an unregistered robot.

```bash
python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas --check
make test-heuristic-parameters
```

Both run **inside the dev container**, under its system interpreter rather than as Bazel targets: Pinocchio reaches
Python through the ROS install (`/opt/ros/jazzy`, Python 3.12) and Bazel's toolchain is a hermetic 3.11 that cannot
see it. The same is true of `tools/hooks/format_code.py`.
<!-- LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots) -->

It reads the URDF, the task file's `initialState`, `reference.yaml` and `gait.yaml`, and reproduces the C++
`deriveLocomotionHeuristicModelParameters()` — including the walk up the kinematic tree to the last joint before the
floating base, and the contact frames, which are not in the URDF but are added at run time from
`contacts.contact_frame_translation`. Its output splits into three groups:

| Group | Coefficients | Status |
| --- | --- | --- |
| **Closed form** | `capture_point`, `high_speed_turning`, `hip_centered_stepping`, `impulse_scaling`, `centripetal_acceleration`, every clamp | The script is the authority; there is nothing to fit |
| **Sized from the command limits** | `orientation_compensation.pitchPerForwardVelocity`, `height_compensation.heightPerSpeed`, `translational_stepping`, `in_place_turning` | A real choice (how much lean, how much crouch), taken from an assumption the script prints |
| **Genuinely fitted** | the rest of Table C.2 | Left at **zero** — this is what chapter 4's extraction framework produces from data, and a number invented here would be a guess wearing a derivation's clothes |

**It does not write the task file, and `--check` is not a test.** A coefficient that has been swept in simulation
*should* differ from its starting point; `--check` exists so that a gait or a geometry moving out from under a number
is noticed, and so the differences read as a list of what has been tuned.

`make test-heuristic-parameters` covers the machinery underneath rather than the values, for the same reason: that
each robot's URDF still parses, that `initialState`'s joint block still lines up with the reduced model, that the
contact frames are still reconstructible, that the gait table still yields a cadence, that the derived block is one
`LocomotionHeuristicConfig::validate()` would accept, and that the coefficients which need *data* are still left at
zero rather than quietly filled in with something plausible.

The script also reports the three configuration facts that decide whether any of this reaches the solver at all —
`useComAndAcomTracking`, `useContactPlanning`, and `task_space_foot_cost_weights.pos_x`/`pos_y` — because each makes
a listed heuristic silently inert.

### What it found on the DRC Atlas

Atlas is the only robot whose block ships with derived values rather than zeros. Two things the derivation surfaced
are worth carrying into any tuning of it:

* **The controller's LIP height disagrees with the robot's.** `dcm_terminal_cost.comHeight` and the contact planner's
  `shared.comHeight` are both `0.85`, but at `initialState` the centre of mass sits **1.0805 m** above the feet — 21%
  higher. So $\omega$ is 13% too large everywhere it is used, and every capture point and DCM is correspondingly
  eager. The heuristics use the configured 0.85 so that the three agree with each other, but one of the two numbers is
  wrong and it is not this block's to fix.
* **Atlas's nominal stance has no outward room.** `nominal_foothold.stepWidth` is `0.45 m`, which is exactly
  `foot_separation.maxStepWidth`, while the legs naturally splay to `0.223 m`. A capture step pushed outward therefore
  runs past a bound the planner would have enforced, and `hip_centered_stepping` needs a `lateralScale` of **2.528**
  just to reproduce the shipped stance.

Also note that Atlas ships `useComAndAcomTracking: true`, so **its three `base_pose` heuristics are inert as
shipped** — their values are derived and ready, and they do nothing until base-pose tracking is selected.

### Preconditions the layer checks for you

| Situation | What happens |
| --- | --- |
| Unknown name | **Error** at start-up, listing the valid names of that kind |
| A name in the wrong list | **Error**, naming the list it belongs in |
| A name listed twice | **Error** — the offsets are summed, so a repeat would silently double it |
| Foothold heuristic with `useContactPlanning: true` | **Error** — the planner supplies footholds itself and never consults these, so it would be silently inert; they are the analytic *alternative* to it |
| `hip_centered_stepping` listed alone, or omitted while others are listed | **Warning** — neither half of Bledt's sum is a configuration he validates |
| `centripetal_acceleration` with `useContactBasisVectorInputs: true` | **Warning** — the reference moves onto the forward-kinematics path; watch the solve time |
| A list written as a scalar or a map instead of a sequence | **Error** — it used to read as "no heuristic", i.e. a typo produced a silently inert configuration |
| A value that cannot be parsed as a number | **Error** naming the file, rather than an exception out of a function that returns a `Status` |
| A clamp set to zero (`capture_point.maximumOffset`, or both `centripetal_acceleration` limits) | **Error** — every clamp here stops a bad measurement reaching the solver, so removing one is not a way to configure it |
| A negative `scale` on either wrench heuristic | **Error** — it would invert the heuristic rather than reduce it |

### Two things that will make a heuristic do nothing

* **The foothold family needs `task_space_foot_cost_weights.pos_x` and `pos_y` to be non-zero.** They are `0` on every
  robot shipped here, because with `zero_velocity` in `hard_constraints` the stance foot is pinned by the schedule and
  placement follows from it. A landing target multiplied by a zero weight is dead weight. (This trap is already live
  on the DRC Atlas, which ships `nominal_foothold.stepWidth: 0.45` alongside `pos_x: 0`.)

### One thing a foothold heuristic changes even with those weights at zero

`CentroidalMpcEndEffectorFootCost::getParameters` branches on whether `getSwingFootReference()` returns a value **at
all**. Its `else` branch — the live one today on every robot here — zeroes `pos_x`/`pos_y` *and* installs a swing
velocity reference (the commanded CoM velocity) with a hard-coded sqrt-weight of 1.5, added specifically to stop the
swing foot dragging behind the moving body. Listing any foothold heuristic produces a landing target, which flips the
cost into the `if` branch and drops that override. The anchor the heuristics are summed onto is the stance foot
carried forward at the commanded velocity, so the reference they replace it with is of the same kind — but this is a
behaviour change independent of the weights, and it is the reason to sweep a foothold heuristic with the weights
raised rather than to leave one listed "harmlessly" at zero.
* **The base-pose family is inert while `useComAndAcomTracking: true`,** because the cost factory zeroes `Q`'s
  base-pose block in that mode and `ComAndAcomTrackingCost` reads the target trajectory directly instead.

### Hot reload

The **coefficients** follow edits to `task.yaml`, like the cost weights beside them: `MpcParameterUpdaterModule`
re-reads the file and calls `LocomotionHeuristicLayer::reconfigure()` from the pre-solve hook. The **lists** are not
reloaded — which heuristics are listed decides how the reference manager and the two input costs are wired at
construction, so changing one under a running solver would be a different controller rather than a retuned one; the
layer says so once and keeps running.

### Centroidal only, for now

The layer is built and installed by `CentroidalMpcInterface::setupLocomotionHeuristics()`, and `WBMpcInterface` has no
counterpart — nothing in the **whole-body** path reads a `locomotion_heuristics` block. The whole-body task file
therefore carries a comment saying so rather than a block that would be silently inert. Two things stand between here
and supporting it: the commanded yaw rate the foothold and wrench heuristics read is recovered by inverting the
*centroidal* state's angular-momentum channel, and the whole-body state has no such channel; and
`MpcParameterUpdaterModule`, which hot-reloads the coefficients, lives in `humanoid_centroidal_mpc` only. The
base-pose seam itself is in the shared reference manager and would work unchanged.

---

## 5. Tests

| Target | Covers |
| --- | --- |
| `//humanoid_nmpc/humanoid_common_mpc:testLocomotionHeuristics` | the registry, the loader, each of the ten formulae, the rejected combinations, and the **parity property** that empty lists are an exact no-op |
| `//humanoid_nmpc/humanoid_common_mpc:testStanceDutyFactor` | $\beta$ over a re-timable mode schedule, including the default-constructed schedule that carries one `FLY` mode and no events |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testLocomotionHeuristicIntegration` | end to end on the DRC Atlas model: the shipped task file builds an empty layer, both shaped references are unchanged by it, an enabled name reaches the reference, $\mathbf{r}_{hip}$ comes from the URDF, and the two task-file errors are returned as a `Status` |

```bash
bazel test //humanoid_nmpc/humanoid_common_mpc:testLocomotionHeuristics //humanoid_nmpc/humanoid_common_mpc:testStanceDutyFactor //humanoid_nmpc/humanoid_centroidal_mpc:testLocomotionHeuristicIntegration
```

---

## 6. Where the code is

```
humanoid_nmpc/humanoid_common_mpc/{include/humanoid_common_mpc,src}/locomotion_heuristics/
  LocomotionHeuristic.h                 root abstract base: name(), configure(), describe()
  BasePoseHeuristic.h                   kind base (A) + BasePoseOffset + its context
  FootholdHeuristic.h                   kind base (B) + its context
  WrenchHeuristic.h                     kind base (C) + its context
  LocomotionHeuristicFormulation.{h,cpp}  the names, the three kinds, the lists, validate(), warnings()
  LocomotionHeuristicConfig.{h,cpp}       the ten parameter structs and the task-file loader
  LocomotionHeuristicModelParameters.{h,cpp}  m, g, z_com, r_hip, derived once from the URDF
  LocomotionHeuristicFactory.{h,cpp}      name -> instance
  LocomotionHeuristicLayer.{h,cpp}        the assembled layer: sums, hot reload, start-up banner
  base_pose/  OrientationCompensation, PeriodicOrientation, HeightCompensation
  foothold/   HipCenteredStepping, CapturePoint, TranslationalStepping, InPlaceTurning, HighSpeedTurning
  wrench/     ImpulseScaling, CentripetalAcceleration
```

**Adding an eleventh heuristic** is four edits, three of which are tied together by `LINT.IfChange` /
`LINT.ThenChange` so the linter catches a half-finished one: the name in `knownHeuristicNames()`, the line in
`LocomotionHeuristicFactory`, the parameter struct and its keys in `LocomotionHeuristicConfig`, and the block in each
robot's `task.yaml`. Plus the class itself — one class per header/cpp pair, in the subdirectory of its kind.

### A note on the procedural arm swing

`SwitchedModelReferenceManager::getDesiredState()` already contains one Bledt-style heuristic that predates this
subsystem: the procedural arm swing, $0.15\sin(2\pi(\Phi - 0.15))\,v_x^{base}$ added to four arm joints. It is a
data-extracted periodic heuristic on a *joint* channel, gated by a boolean rather than selected by name — which is
what this repository's own convention forbids, and it is the existence proof that `getDesiredState()` is the right
seam. Registering it as an eleventh name in a fourth kind would make the rule hold everywhere, but it would also
change behaviour on every robot with arms, so it is deliberately left for a separate change with its own parity test.
