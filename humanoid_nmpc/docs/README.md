# Online Contact Planning and the DCM Terminal Cost

This document describes two optional formulation features of the centroidal humanoid NMPC, both selected by name in the
robot's `config/mpc/task.textproto`:

| Selection | Effect |
| --- | --- |
| `dcm_terminal_cost` in `costs` | The horizon ends with a Divergent Component of Motion (capture point) viability cost, in place of the quadratic terminal cost `terminal_cost` on `final_state_weights`. The two are alternative ends of the horizon, and start-up refuses a list that names both. |
| `contact_schedule_source: "contact_planner"` | The contact sequence, the switching times and the footholds are planned online on a reduced model by the contact planner `planner.type` in `contact_planning.textproto` selects - the closed-form H-LIP stepper (`hlip`, both shipped robots) or the mixed-integer program of section 2 (`lip_miqp`) - and replace the clock-driven gait schedule. `gait_schedule`, the default and what every robot ships, is that gait schedule with the nominal footholds. |

Both replaced a top-level boolean, now the retired fields `use_dcm_terminal_cost` and `use_contact_planning` of the
task file's schema. A task file that still carries either is refused when it is read, whatever its value, with a message
that names the replacement, so that a stale file cannot silently run the other formulation. The whole-body MPC
implements neither and refuses both names.

Both were added because a fixed, pre-scheduled contact sequence is the first bottleneck of a switched-system MPC: under
pushes, on uneven ground or with changing speed commands the rigid schedule causes scuffing, early impacts and infeasible
terminal states.

The contact planner itself is chosen by name in `config/mpc/contact_planning.textproto` (`planner.type`), and the
whole-body formulation by the term lists of `task.textproto`. Two further documents cover the implementation of
[arXiv:2502.15630](https://arxiv.org/abs/2502.15630) in this repository:

* [hlip_contact_planner](hlip_contact_planner/README.md) — the closed-form H-LIP contact planner, which both shipped
  robots select (`planner.type: "hlip"`; a file that names no planner gets `lip_miqp`, the library default) and which
  replaces the mixed-integer planner described in section 2 below.
* [contact_implicit_mpc](contact_implicit_mpc/README.md) — the relaxed complementarity formulation of contact in the
  whole-body NMPC, which lets the solver depart from the planner's nominal contact sequence. Off by default.

Another covers how the contact inputs themselves are parameterized, selected by name with
`contact_input_parameterization` in `task.textproto`:

* [contact_basis_vectors](contact_basis_vectors/README.md) — the basis-vector parameterization of the contact inputs
  (`contact_input_parameterization: "basis_vectors"`, which the DRC Atlas and the EngineAI SA01 ship; `wrench` is the
  default): the conservative generator set that ships and the exact wrench-cone set, the full-diagonal and null-space
  regularizations of the input cost, the lambda >= 0 barrier that replaces the explicit cone, and how to select each.

A fourth covers a feature that is not a formulation toggle but a reference-shaping layer over the ones that already
exist:

* [locomotion_heuristics](locomotion_heuristics/README.md) — the ten regularization heuristics of Bledt's
  *Regularized Predictive Control Framework for Robust Dynamic Legged Locomotion* (MIT, 2020), selected by name in
  the `locomotion_heuristics` block of `task.textproto`. They shape the base-pose, foothold and contact-force references
  the existing quadratic costs already regularize against — the base's roll and pitch reference in particular is a
  hard zero today. Every list ships empty, so the layer is an exact no-op until a robot is opted in.

One covers a simulation tool rather than the controller:

* [dodgeball](dodgeball/README.md) — throwing a physical ball at the robot in simulation: the GUI tab that
  aims it, the ball MuJoCo compiles into the scene, and how restitution is expressed in a simulator that has
  no restitution parameter. A directed, repeatable push test, selected with `sim_projectile` in `task.textproto`.

And one covers what the controller does when the plant jumps:

* [mpc_reset](mpc_reset/README.md) — the full reset of the MPC (solver, reference manager, gait schedule, contact
  planner, command ramps, the MRT's policy buffer), when the MRT joint controllers reset and hold the robot, how
  repeated solver failures back off, the gantry catch and settle sequence of the MuJoCo sims (`sim_max_base_tilt_angle`,
  `sim_gantry_catch_lift`), and the swing-foot yaw residual that was not differentiable at the reset pose.

---

## 1. DCM terminal cost (`dcm_terminal_cost`)

### 1.1 Divergent Component of Motion

For a Linear Inverted Pendulum (LIP) of constant height $z_c$ the CoM dynamics in the horizontal plane are

$$\ddot{\mathbf{c}} = \omega^2 (\mathbf{c} - \mathbf{z}), \qquad \omega = \sqrt{g / z_c},$$

with $\mathbf{z}$ the zero moment point (ZMP). The change of variables

$$\boldsymbol{\xi} = \mathbf{c} + \frac{\dot{\mathbf{c}}}{\omega}$$

splits the dynamics into a stable part (the CoM converges to $\boldsymbol{\xi}$) and an unstable part

$$\dot{\boldsymbol{\xi}} = \omega (\boldsymbol{\xi} - \mathbf{z}),$$

so the DCM $\boldsymbol{\xi}$ (equivalently the instantaneous capture point) diverges away from the ZMP. The robot can come to
rest without stepping if and only if the ZMP can be placed on the DCM, i.e. if $\boldsymbol{\xi}$ lies inside the support
polygon. This is the classic *capturability* condition.

### 1.2 Why a terminal cost

The NMPC has a finite horizon of about one second. Whatever the running cost does, a horizon that ends with the DCM far
outside the terminal support is a state the robot cannot recover from once the horizon recedes: the problem becomes
infeasible a few cycles later. The quadratic `final_state_weights` cost regulates the full state (base pose, momentum, joints) towards a
reference and knows nothing about support; it also has to be re-tuned for every gait cadence because "the right" base pose
at the end of the horizon depends on where in the gait cycle the horizon ends.

The DCM terminal cost replaces it with the physically meaningful quantity. At the terminal time $T$ it penalizes

$$\ell_T(\mathbf{x}_T) = \tfrac{1}{2}\, \big(\boldsymbol{\xi}_T - \boldsymbol{\xi}^{\mathrm{ref}}_T\big)^\top W \big(\boldsymbol{\xi}_T - \boldsymbol{\xi}^{\mathrm{ref}}_T\big),
\qquad
\boldsymbol{\xi}^{\mathrm{ref}}_T = \mathbf{p}_{\mathrm{support}}(\mathbf{q}_T) + \beta \frac{\mathbf{v}_{\mathrm{cmd}}}{\omega},$$

Under an online contact plan this reference is replaced by the plan's own DCM, $\mathbf{c}(T) +
\dot{\mathbf{c}}(T)/\omega$ with the $\omega$ of the pendulum the plan was made on
(`SwitchedModelReferenceManager::getPlannedDcm`, `ContactPlan::omega`), and $\boldsymbol{\xi}_T$ is then taken with that
same $\omega$: the reduced-order model decides where the horizon should end, and a second capturability reference of its
own would fight the footholds the planner is placing. See [hlip_contact_planner](hlip_contact_planner/README.md).
Without a plan the support-center reference below stands.

where

* $\boldsymbol{\xi}_T = \mathbf{c}_{xy}(\mathbf{q}_T) + \mathbf{h}_{xy}(\mathbf{x}_T)/\omega$ is evaluated on the full model: the CoM position
  from the kinematics and the CoM velocity from the normalized linear momentum in the centroidal state;
* $\mathbf{p}_{\mathrm{support}}$ is the weighted center of the contact frames of the feet in contact at $T$ according to the
  mode schedule. The weights are continuous in time: a foot's weight is 1 in the middle of a contact phase and ramps
  linearly to 0 over `support_blend_time` before its lift-off and after its touch-down (both feet as a fallback in flight).
  Without the blending the reference would jump by half the step width every time the receding horizon end crosses a
  mode switch, which with a weight of several hundred is a visible disturbance at every solve;
* $\mathbf{v}_{\mathrm{cmd}}$ is the commanded CoM velocity from the target trajectories and $\beta$ (`velocity_offset_factor`)
  scales the offset. With $\beta = 0$ the cost demands pure capturability (the robot can stop over its feet). With
  $\beta = 1$ the reference is the DCM of a CoM that moves at the commanded speed over the support point
  ($\dot{\mathbf{c}} = \omega(\boldsymbol{\xi} - \mathbf{c}) = \mathbf{v}_{\mathrm{cmd}}$), so a steady walk is not decelerated by the
  terminal cost while any excess divergence is still penalized.

The cost is implemented as a CppAD residual with a Gauss-Newton Hessian $J^\top W J$, so its quadratic approximation is
positive semi-definite for every configuration. $W$, $z_c$ and $\beta$ are parameters of the compiled model and are
hot-reloadable from the `dcm_terminal_cost` block of `task.textproto` through the parameter updater (the block of the
same name on the GUI's MPC Parameters tab). The choice between `dcm_terminal_cost` and `terminal_cost` in `costs` is
not: which terminal cost the problem ends on is decided when it is built, and a reload applies `final_state_weights`
and the `dcm_terminal_cost` block to whichever of the two the running problem carries - the updater asks the running
problem, never the reloaded file - with a warning when the reloaded file selects the other one.

$z_c$ is the robot's own by default: a block without `com_height`, as both shipped robots write it, has the model's
center of mass above the mean height of its soles at the task file's `initial_state` (`computeComHeightAboveFeet`, the
one definition of the pendulum length that the contact planner's `shared` block without `com_height` and the locomotion
heuristics resolve too, so they cannot disagree). `DcmTerminalCost::Create` is given that height by
`CentroidalMpcInterface` and resolves the absent height to it there and on every hot reload of the block; the interface
logs the height and $\omega$ at start-up (1.0805 m and 3.01 rad/s on the DRC Atlas, 0.6124 m and 4.00 rad/s on the
SA01). A positive `com_height` is an explicit override. A `com_height` of 0 or less (0 meant "the model's" before the
field was optional), a non-positive `gravity` and a negative weight are refused with a Status naming the field, at
start-up and on a reload, which then keeps the running cost.

```textproto
costs: "state_quadratic_cost"  # ... the other entries of the list
costs: "dcm_terminal_cost"     # in place of terminal_cost; start-up refuses a list that names both
dcm_terminal_cost {
  # com_height [m]: z_c for omega; left out, the model's center of mass above its feet at initial_state
  gravity: 9.81
  weight_x: 400                # W = diag(weight_x, weight_y)
  weight_y: 400
  velocity_offset_factor: 1.0  # beta
  support_blend_time: 0.1      # [s] ramp of the support weights through lift-off / touch-down
}
```

Implementation: `humanoid_centroidal_mpc/cost/DcmTerminalCost.{h,cpp}`; wiring in `CentroidalMpcInterface.cpp`.

---

## 2. Mixed-integer contact planning (`contact_schedule_source: "contact_planner"`, `planner.type: "lip_miqp"`)

Online contact planning is switched on by name, with `contact_schedule_source: "contact_planner"` in the robot's
`config/mpc/task.textproto` (`gait_schedule`, the clock-driven gait schedule with the nominal footholds, is the default and
what every robot ships; the registry in `MpcFormulationConfig.h` lists the names and refuses any other, naming the valid
ones), and `planner.type` in `contact_planning.textproto` - the one place the planner is chosen - names the planner that
runs. The source decides which reference manager the problem is built on, so it takes effect at start-up only; the
parameter updater warns when a reloaded file names another one. It is implemented for the centroidal MPC only: the
whole-body MPC refuses `contact_planner`. This section is the mixed-integer one,
`lip_miqp`; both shipped robots select the closed-form H-LIP planner, `hlip`, which
[hlip_contact_planner](hlip_contact_planner/README.md) describes, and the start-up log names the one that is
configured. Everything either planner is tuned with
is the contact planner's own file, `config/mpc/contact_planning.textproto` (a `humanoid_mpc_config.ContactPlanningFile`),
which the interface looks for in the task file's directory; a robot without one plans with the library defaults, and a
task file that still carries a `contact_planning` block is refused with the name of that file. The file is hot-reloaded
whenever it is saved, like the task file, and the tuning GUI edits it in place.

### 2.1 Architecture

```
 velocity command ──► target trajectories ─────────────────────────────┐
                                                                       ▼
 MPC state ──► ContactPlannerModule ──► LipContactPlanner (MIQP) ──► ContactPlan
                (pre-solve hook,          B&B over HPIPM               │ contacts per node,
                 background thread)       + propagation + local search │ footholds, CoM, ZMP
                                                                       ▼
                        ContactPlanningReferenceManager ◄──────────────┘
                          ├─ applied schedule adapted to measured contacts (early / late touch-down, 2.8)
                          ├─ mode schedule  = committed window of the applied schedule + plan
                          ├─ swing trajectory planner (foot height)   ──► constraints / pre-computation
                          └─ swing-foot references (planned landing + DCM step adjustment) ──► task_space_foot_cost (pos_x, pos_y)
                                                                       ▼
                                                              SQP NMPC (unchanged)
```

The planner is not written as one problem. `LipContactPlanner` owns a `ContactPlanningProblem` that
`ContactPlanningTermFactory` assembles from the term lists of `contact_planning.textproto`, in the way `WBMpcInterface`
assembles the OCS2 `OptimalControlProblem` from the `costs` / `soft_constraints` / `hard_constraints` lists of the task
file: model blocks that compose the variable layout, costs, soft and hard constraints, logic rules on the contact
binaries and assignment costs, each a named term with its own parameter block, plus the search stages around the
branch-and-bound and the execution rules the reference manager applies between plans (section 2.10). Everything below
describes the terms of the shipped formulation.

The NMPC itself is unchanged: it still sees a mode schedule (which feet are in contact when) and constraints derived from
it. What changes is who produces the schedule. Without contact planning the `GaitSchedule` tiles a periodic template
selected by the velocity-based gait state machine. With contact planning the schedule is the output of a mixed-integer
optimization that is re-solved continuously from the current state.

The planner runs on a reduced model so that it can afford combinatorial search; the whole-body NMPC then tracks the
result with the full dynamics. The two are coupled through the mode schedule, the swing-foot landing references and, at
the end of the horizon, through the DCM terminal cost that both formulations share.

### 2.2 Reduced model and decision variables

The planner discretizes a horizon of $N$ nodes of duration $\Delta t$ (default $12 \times 0.1\,\mathrm{s}$). Per interval $k$:

| Symbol | Type | Meaning |
| --- | --- | --- |
| $\mathbf{c}_k, \dot{\mathbf{c}}_k \in \mathbb{R}^2$ | state | LIP CoM position and velocity |
| $\mathbf{p}_{i,k} \in \mathbb{R}^2$ | state | position of foot $i \in \{L, R\}$ (its planned landing spot while swinging) |
| $\mathbf{z}_k \in \mathbb{R}^2$ | input | ZMP during the interval |
| $\delta\mathbf{p}_{i,k} \in \mathbb{R}^2$ | input | foot displacement during the interval |
| $c_{i,k} \in \{0, 1\}$ | **binary** input | foot $i$ in contact during the interval |

Dynamics (exact zero-order-hold discretization of the LIP with constant ZMP per interval):

$$\begin{bmatrix}\mathbf{c}_{k+1}\\ \dot{\mathbf{c}}_{k+1}\end{bmatrix} =
\begin{bmatrix}\cosh(\omega\Delta t) & \sinh(\omega\Delta t)/\omega\\ \omega\sinh(\omega\Delta t) & \cosh(\omega\Delta t)\end{bmatrix}
\begin{bmatrix}\mathbf{c}_k\\ \dot{\mathbf{c}}_k\end{bmatrix} +
\begin{bmatrix}1-\cosh(\omega\Delta t)\\ -\omega\sinh(\omega\Delta t)\end{bmatrix}\mathbf{z}_k,
\qquad \mathbf{p}_{i,k+1} = \mathbf{p}_{i,k} + \delta\mathbf{p}_{i,k}.$$

All geometric constraints are written in the yaw-aligned frame of the base at planning time (unit vectors $\mathbf{e}_x$,
$\mathbf{e}_y$), so the linear model stays valid for any heading.

### 2.3 Constraints

**Contact disjunctions (big-M, $M$ = `shared.big_m`).** A foot only moves while it is not in contact:

$$|\mathbf{e}_j^\top \delta\mathbf{p}_{i,k}| \le M\,(1 - c_{i,k}).$$

The ZMP must lie in the support region. In single support that is a box of half-widths $(r_x, r_y)$
(`zmp_support_region.half_width_x` / `.half_width_y`) around the supporting foot; each foot's box is relaxed unless that foot is the
*only* contact:

$$\pm \mathbf{e}_j^\top(\mathbf{z}_k - \mathbf{p}_{i,k}) \le r_j + M(1 - c_{i,k}) + M c_{\bar i,k}.$$

In double support the region is the convex hull of the two boxes. Laterally the feet never cross, so the hull is exactly
$\mathbf{e}_y^\top(\mathbf{p}_{R,k}) - r_y \le \mathbf{e}_y^\top \mathbf{z}_k \le \mathbf{e}_y^\top(\mathbf{p}_{L,k}) + r_y$, both rows
relaxed by $M(1-c_{L,k}) + M(1-c_{R,k})$ so that they bind only in double support (in single support the stance foot's
own box states the same bound, and a second copy of it in a soft row made the lateral slack penalty count twice on the
outward side). Along the heading the order of the feet is not known in advance and the exact hull would need one more
binary per node; the planner uses the box of half-width $r_x$ around the midpoint of the feet instead, relaxed the same
way.

That box and the lateral strip are each a valid *projection* of the hull onto one axis, but what the solver sees is their
**intersection**, and the intersection of a convex set's projections is its bounding box rather than the set. The two
coincide only while the feet are level along the heading. Earlier revisions of this section called the result "a
conservative inner approximation that costs nothing"; that is wrong, and wrong in the unsafe direction. In the transition
double support of every step the feet are a step apart along the heading, and the admitted region then strictly *contains*
the hull, over-admitting the far corners by about half the heading offset — up to half a step length. With
$\mathbf{p}_L=(0.2,0.1)$, $\mathbf{p}_R=(0,-0.1)$, $r_x=0.08$, $r_y=0.04$ the point $(0.18,-0.14)$ satisfies both rows
while the hull's support along $(1,-1)/\sqrt2$ is $0.22$ against that point's $0.32$.

Closing it means cutting the two missing hull edges, whose normal depends on $\mathbf{p}_L-\mathbf{p}_R$ — decision
variables — so the row is bilinear and has to be linearized about a nominal separation. That changes the feasible set of a
*soft* constraint the whole-body MPC re-solves against the true wrench cone, so it is a conservatism choice to validate in
simulation behind its own field, not a correctness repair. See `ZmpSupportRegionConstraint.h`. These constraints are
*soft* (HPIPM slacks with a quadratic and a linear penalty, `shared.slack_penalty.quadratic` / `.linear`, which a term's
own `slack` block, e.g. `zmp_support_region.slack.quadratic`, overrides) so that the relaxations always stay feasible and
an unavoidable violation shows up as cost instead of as a solver failure.

**Kinematics (soft).** Reachability of every foot with respect to the CoM (`reachability.reach_x`, `.reach_y_inner`,
`.reach_y_outer`), step length $|\mathbf{e}_x^\top(\mathbf{p}_L - \mathbf{p}_R)| \le$ `foot_separation.max_step_length` and step width
`foot_separation.min_step_width` $\le \mathbf{e}_y^\top(\mathbf{p}_L - \mathbf{p}_R) \le$ `foot_separation.max_step_width`
(self-collision margin). These
rows involve the state alone and are imposed at every node $k = 0 \dots N$, the terminal node included: the foothold of a
swing that ends at the horizon, $\mathbf{p}_{i,N} = \mathbf{p}_{i,N-1} + \delta\mathbf{p}_{i,N-1}$, is produced by an
input of node $N-1$ but is a state of node $N$, and it is the landing target the controller tracks when a step ends at
the horizon. (An earlier version skipped the terminal node along with its inputs and dynamics, which left that foothold
held only by the step-width and regularization costs.)

**Logic on the binaries (exact, outside the QP).** The combinatorial rules are the `logic_rules` list. They are enforced
by a propagation hook that the branch-and-bound calls on every partial assignment (fixing implied values, rejecting
contradictions) and on every candidate incumbent. On a partial assignment a rule may only reject what no completion can
satisfy and only fix what every completion shares; a preference, such as which of two overdue feet should step first,
is never a reason to fix a binary and belongs in the objective. A rule that fixes more prunes feasible subtrees silently
while the search still reports `optimal`, so `testContactPlanningRegression` walks every partial assignment the search
can reach on a short horizon and checks both properties against the complete enumeration:

* no flight phase (`no_flight`): $c_{L,k} + c_{R,k} \ge 1$ (also a row of the QP, the `no_flight` hard constraint);
* `phase_durations` (the `shared.gait_limits`): minimum and maximum swing duration, minimum (and optionally maximum)
  contact duration, counted in nodes from the
  start of the phase, including the time already spent in the current phase before the planning instant. That elapsed
  time is a non-integer number of nodes; it is rounded down when checked against a minimum and up against a maximum, so
  that the grid never violates either limit (rounding to the nearest node allowed a 0.16 s old swing to end after one
  more node, 0.26 s against a 0.3 s minimum). If the two limits are less than a node apart the nearest node decides.
  A foot may also not lift off in the last `minSwingNodes - 1` nodes of the horizon, where a minimum-length swing does
  not fit. Without that rule the plan ended mid-swing and `toModeSchedule()` closed it with a touch-down at `endTime()`,
  so the controller received a swing far shorter than `min_swing_duration` (0.1 s against a 0.3 s minimum, measured over a
  receding-horizon walk), and the swing trajectory planner scaled its height down in proportion to that duration. The
  step is simply deferred to the next plan.
  The maximum contact duration yields to the rules that can make lifting impossible: a foot that is overdue lifts at
  the first node where the other foot supports it and no minimum double support holds it, and when both feet are
  overdue at once (a standing start with `shared.gait_limits.max_contact_duration` set) the one that did not swing last, or that has stood
  longer, goes first. Forcing both to lift at the same node used to contradict the no-flight rule and left the planner
  without a single feasible assignment. On a partial assignment the lift is only fixed once the other foot's binaries
  up to that node are fixed, because until then the other foot may still step first and excuse this one;
* foot alternation (`alternating_feet`): a foot may not swing twice without the other foot swinging in between;
* minimum double support (`minimum_double_support`, `shared.gait_limits.min_double_support_duration`): after a touch-down
  the other foot stays down for at least that
  long, so weight transfer is never asked to happen in a single node. This includes the node of the touch-down itself:
  a lift-off at the very node the other foot lands would be an instantaneous switch with no double support at all.
  Set to exactly 0 the hold is disabled and that instantaneous exchange becomes admissible; any value above 0 is
  quantized up to a whole node, so at `dt: 0.1` anything in (0, 0.1] still yields a 0.1 s double support. Permitting the
  exchange is not the same as choosing it: with the hold at zero and no further incentive the planner still keeps a
  double support at walking speed, because it buys the ZMP freedom the support-region rows charge for. The
  `double_support_penalty` assignment cost is that incentive. Because it prices *every* double-support node it prices
  standing on two feet as well, and above roughly 0.2 (Atlas gait limits) stepping in place becomes cheaper than
  standing, so the robot marches at a zero velocity command; `testLipContactPlanner` pins both ends of that trade;
* the committed window: contacts up to the *commit boundary* are fixed to the schedule the NMPC is already executing.
  The boundary is `planner.commit_time` ahead of the planning instant, extended to the touch-down of any swing that has
  started or starts within that window. A swing in flight is therefore not re-timed or cut short by a later plan - with
  one exception: `planner.max_commit_extension` (0, the default, is no cap) caps that extension, and a capped boundary
  can fall inside a swing in flight and hand a later plan the authority to re-time it, which is the price of keeping the
  boundary finite in a gait that exchanges support in an instant (`commitBoundaryForSchedule`). `planner.commit_time`
  must cover the planner latency. A plan is merged into the executed schedule exactly at its own boundary, `planner.commit_time` after the snapshot it was planned from; merging it any later (for instance at the boundary of the solve that activates it) cut its first lift-off short by the plan's age and the controller executed swings shorter than `min_swing_duration`. A plan whose boundary has already passed when it is activated is stale and dropped with a rate-limited warning (the planner latency exceeded `commit_time`, or a touch-down the boundary was extended to happened while the plan was computed); so is a plan that has a foot down where the executed schedule already has it in flight at the merge point, which means a swing was activated between the plan's snapshot and its activation. In both cases the executed schedule keeps running until a fresh plan arrives. Shifting a late plan forward onto the current boundary instead, whole, was tried and is wrong: a plan made just before a touch-down and activated just after it was delayed by a whole swing, and the foot that had just landed was lifted again at once, which threw the robot. The planner's
  node grid is not aligned with the executed events, so every node that *starts* before the boundary is committed: a
  node entirely inside the window takes the executed contacts at its midpoint, the last committed node (straddling the
  boundary or ending exactly on it) takes the contacts the executed schedule hands over at the boundary itself. Sampling
  that node at its midpoint let the plan contradict the executed schedule inside it (a touch-down after the midpoint
  read as "still swinging"), and the merge then delayed an in-flight touch-down by up to half a node or re-lifted a foot
  that had just landed. A phase that begins inside a committed node (a touch-down
  between two node boundaries) is counted from the executed event, not from the node start: the reference manager hands
  the planner the phase start times of the committed nodes, and the propagation counts the nodes spent in such a phase
  rounded down against a minimum and up against a maximum, exactly like the elapsed time of the phase active at the
  planning instant. Counting from the node start credited a touch-down at 0.97 s inside the node [0.9, 1.0) with a full
  node of contact at 1.0 s, so the minimum double support, counted in whole nodes, let the other foot lift at 1.0 s and
  the merged schedule contained a double support of 0.03 s against a 0.1 s minimum (and contacts of 0.13 s against
  0.15 s). The merge itself is exact; those spurious short phases were the planner's grid accounting, not the stitching;
* plan consistency (the `plan_consistency` assignment cost, `plan_consistency.cost`): every node whose contact differs
  from the previous plan, shifted to the current time, is charged, which gives the anytime search hysteresis between
  cycles; the continuous footholds are likewise pulled towards the previous plan (the `previous_foothold_consistency`
  cost, `previous_foothold_consistency.weight`) so that the landing target tracked by the foot cost does not jitter from
  plan to plan.

Keeping the duration logic out of the QP keeps the relaxations small (8 states, 8 inputs, 27 rows per stage) and costs
nothing in accuracy, because the linear relaxation of such disjunctions is too weak to prune anyway (a foot that is "half
in contact" may move half a step at almost no cost). The search is therefore an enumeration of admissible contact
sequences with strong propagation, made anytime by node and time limits.

### 2.4 Objective

The objective is the `costs` list (quadratic, in the QP) plus the `assignment_costs` list (on the binaries, through the
logical cost hook). With the terms the shipped files list, and the heading costs of section 2.9 left out,

$$\begin{aligned}
J = {} & \sum_{k=0}^{N} \Big( w_v \|\dot{\mathbf{c}}_k - \mathbf{v}_{\mathrm{cmd}}\|^2 + w_w\big(\mathbf{e}_y^\top(\mathbf{p}_{L,k}-\mathbf{p}_{R,k}) - w_{\mathrm{nom}}\big)^2
 + w_f \sum_i \|\mathbf{p}_{i,k} - \mathbf{p}^{\mathrm{prev}}_{i,k}\|^2 \Big) \\
& + \sum_{k=0}^{N-1} \Big( w_z\|\mathbf{z}_k - \mathbf{c}_k\|^2 + w_p \sum_i \|\delta\mathbf{p}_{i,k}\|^2
 + w_\ell \sum_i \big\|\delta\mathbf{p}_{i,k} - \mathbf{d}_{\mathrm{nom}} (1 - c_{i,k})\big\|^2 \Big)
 + w_T\, \big\|\boldsymbol{\xi}_N - \mathbf{z}_{N-1} - \mathbf{r}\big\|^2 \\
& + w_s\, n_{\mathrm{switch}} + w_c\, n_{\mathrm{changed}} + w_d\, n_{\mathrm{double}} + \text{regularization},
\end{aligned}$$

with, term by term: `velocity_tracking` ($w_v$), `step_width` ($w_w$, `step_width.nominal_step_width`),
`previous_foothold_consistency` ($w_f$, the previous plan's footholds shifted to the current time, only when a previous
plan is usable), `zmp_regularization` ($w_z$), `foothold_regularization` ($w_p$), `step_length` ($w_\ell$, the
displacement $\mathbf{d}_{\mathrm{nom}} = \mathbf{v}_{\mathrm{cmd}}\,\Delta t\, T_{\mathrm{stride}} / T_{\mathrm{swing}}$ a cyclic gait at the
commanded speed needs per swing node, section 2.7), `terminal_dcm` ($w_T$), and the assignment costs `contact_switch`
($n_{\mathrm{switch}}$ counts lift-off and touch-down events), `plan_consistency` ($n_{\mathrm{changed}}$ counts the decided
nodes whose contact differs from the previous plan) and `double_support_penalty` ($n_{\mathrm{double}}$ counts the nodes
with both feet down). `regularization` adds a small multiple of the identity to every stage's Hessian.

The terminal term is the same terminal capturability idea as in section 1, on the reduced model. Its target is
$\mathbf{r} = 0$, a DCM that comes to rest over the last ZMP (`terminal_dcm.target: "rest"`), or with
`terminal_dcm.target: "commanded_velocity"` $\mathbf{r} = \mathbf{v}_{\mathrm{cmd}}/\omega$, the DCM offset of a CoM that keeps moving at the commanded velocity over the foot.
The terminal node has no ZMP, so the term is written on the last running node: with
$\boldsymbol{\xi}_N - \mathbf{z}_{N-1} = e^{\omega \Delta t}(\boldsymbol{\xi}_{N-1} - \mathbf{z}_{N-1})$ it is exactly
$w_T e^{2\omega\Delta t} \|\boldsymbol{\xi}_{N-1} - \mathbf{z}_{N-1} - e^{-\omega\Delta t}\mathbf{r}\|^2$, a quadratic function of
the last stage. (The code used to apply $\mathbf{r}$ to $\boldsymbol{\xi}_{N-1}$ instead, which put the target of
$\boldsymbol{\xi}_N$ at $e^{\omega \Delta t}\,\mathbf{v}_{\mathrm{cmd}}/\omega$, 40 % further ahead on the Atlas pendulum at
$\Delta t = 0.1$ s.)

### 2.5 Solver

`MixedIntegerOcpQp` is a branch-and-bound over OCP-structured QP relaxations, with the `search` list's stages around it:

1. Every relaxation is solved with HPIPM (Riccati-based interior point on the stage structure; the one-sided rows are
   masked instead of using loose finite bounds). A relaxation costs well under a millisecond for the planner's sizes.
2. Search order: best-bound first among the open nodes, with depth-first diving into the rounding direction after each
   branching. Branching picks the first fractional binary in time order, so early decisions are settled first; free
   binaries whose relaxation is already integral are skipped, so any binary can be the next one fixed.
3. Incumbents come early from (a) the previous plan shifted by the elapsed time (`warm_start_previous_plan`), (b) a
   diving heuristic at the root (`diving`: fix every integral binary, round the first fractional one, propagate,
   re-solve) and (c) the integral relaxations found while diving.
4. After the branch-and-bound an event-shift local search (`event_shift_local_search`) moves every lift-off /
   touch-down of the incumbent one node earlier or later, evaluates the fixed-assignment QP and keeps improvements
   (`event_shift_local_search.iterations`, `.max_time`). This refines the timing at a few QPs per round.
5. `cadence_stretch` (listed in neither shipped file, and a no-op while `cadence_stretch.samples` is 0) re-solves the
   incumbent's contact pattern on a grid of node duration $s\,\Delta t$ for `samples` stretches up to
   `cadence_stretch.max_stretch` and keeps the best, recovering the cadences the grid's quantization cannot express. The
   stretch is bounded so that no phase exceeds the gait limits (the elapsed part of the phase in flight included) and
   so that the last committed node is still the one live at the commit boundary: the boundary is not stretched, and on
   a grid stretched further an earlier committed node, sampled before the boundary, was live at the merge and landed a
   swing in flight late. A local search listed after it searches the stretched grid and only admits candidates whose
   phases fit the gait limits at the stretched node duration.
6. `planner.max_branch_and_bound_nodes` and `planner.max_solve_time` bound the effort; the best plan found so far is returned
   (anytime).

### 2.6 Runtime integration

* `ContactPlannerModule` (a solver synchronized module) snapshots the planner input before every MPC solve: CoM position
  and velocity from the state, foot positions from the kinematics, the contact phases and their elapsed time from the
  applied schedule, the committed window and the commanded velocity from the target trajectories. With
  `planner.threading: "background_thread"` the snapshot is posted to a worker thread (latest wins, rate-limited by
  `planner.planning_frequency`); with `"pre_solve_hook"` the plan is computed inside the pre-solve hook. After a plan is published the pending
  snapshot is discarded, so the next plan always starts from a schedule that already contains the previous plan, unless
  that snapshot follows a contact event (section 2.8): the event has already re-timed the schedule the finished plan was
  built on, so the post-event snapshot is kept and planned next. Switching `planner.threading` at runtime starts or
  joins the worker; the stop flag is published under the worker's mutex so a toggle or a shutdown can never lose the
  wake-up and hang the solver thread, and a snapshot or throttle state left over from the previous worker is cleared.
* `ContactPlanningReferenceManager` merges every new plan into the schedule the NMPC is executing (applied schedule up to
  the plan's commit boundary, plan afterwards, never cutting a swing that is in flight or imminent, always starting and ending in double support so that the swing trajectory planner
  finds a lift-off and a touch-down for every swing), updates the foot-height swing planner and exposes the planned
  landing spot of every swing foot as a task-space reference: the xy position interpolates from the lift-off position to
  the landing spot with the cubic $b(\tau) = -\tau^3 + \tau^2 + \tau$ of the normalized swing time, the height follows the
  swing trajectory planner. The cubic has $b'(0) = 1$ and $b'(1) = 0$: the foot leaves at the step's average velocity,
  displacement over duration, rather than from rest, so that it does not kick backward relative to a body that is
  already moving, and it arrives at zero velocity. It is not a smooth-step, and the xy reference is not at rest at
  lift-off. The lift-off position is
  the latched measured foot position for the swing that ends the foot's current contact phase; a later swing of the same
  foot inside the horizon starts from the planned foot position at its own lift-off, i.e. from where the earlier step
  lands, since the latched position would be a step behind.
* `CentroidalMpcEndEffectorFootCost` tracks that reference through `task_space_foot_cost.weights.pos_x` / `.pos_y`. Without a
  plan the xy position error is switched off inside the cost, so the weights are harmless when planning is disabled.
* Until the first valid plan arrives (or if the planner stalls) the gait schedule / the last applied schedule is used.
* The whole contact-planning file is hot-reloadable through the parameter updater (its blocks on the GUI's MPC
  Parameters tab): every term re-reads its parameter block, and a changed term list re-assembles the problem, the
  search stages and the execution rules before the planner's next run (the warm start survives unless the grid or the
  variable layout changed). The assembled formulation is logged at start-up and after every such reload.

### 2.7 Tuning notes

* Cyclic gait at speed. `velocity_tracking` penalizes the CoM velocity error at every node and is indifferent between a
  few long steps and many short ones at the same average speed; from rest it favors the short, quick steps that
  accelerate the pendulum fastest, and a step change in the command (the command filter,
  `velocity_command_filter_break_frequency` in `command/reference.textproto`, ships off) makes every replan re-decide the
  pattern. Three pieces make the gait progressive and cyclic instead: `max_linear_acceleration` /
  `max_angular_acceleration` in `command/reference.textproto` rate-limit the velocity reference the MPC target and the planner
  follow (0 = off); the `step_length` cost draws every swing to the displacement a cyclic gait at the commanded speed
  needs, `d_nom = v_cmd dt T_stride / T_swing` at the nominal cadence of the gait limits, with the residual affine in
  the foot displacement and the relaxed contact binary; and `terminal_dcm.target: "commanded_velocity"` moves the terminal
  target of the terminal DCM $\boldsymbol{\xi}_N$ from the last ZMP (a stop at the end of the horizon, which shortens the
  steps at speed) to `zmp + v_cmd / omega`, a CoM over the foot that keeps walking. `planner.log_plans` prints one line per plan with the
  phase durations, the step lengths and whether the branch-and-bound hit its node or time limit, which is the first
  thing to check when steps come out irregular: a truncated search hands out a different near-optimal plan each time.

* `planner.dt` / `planner.num_nodes`: the horizon must cover `mpc.time_horizon`; coarser nodes make the search cheaper but
  quantize the switching times.
* `planner.commit_time`: at least the planner latency (solve time plus one planning period) plus the time the NMPC needs to
  anticipate a switch; too small a value lets a fresh plan re-time switches the controller is already preparing, which
  shows as feet that stutter or barely lift.
* `shared.gait_limits.min_swing_duration` / `max_swing_duration` / `min_contact_duration` are the main shape parameters of
  the gait. `shared.gait_limits.max_contact_duration` forces stepping even without a command (leave at 0 to allow
  standing).
* `zmp_support_region.half_width_x` / `.half_width_y` should stay inside the physical foot (the NMPC enforces the real wrench cone); a
  smaller box makes the planner step earlier under disturbances. 0 takes the sole's footprint from the model.
* `contact_switch.cost` trades stepping against ankle strategy; `terminal_dcm.weight` makes plans end capturable.
* Raise `shared.slack_penalty.quadratic` (or a term's own `slack.quadratic`) if plans exploit the soft support region.

Implementation: `humanoid_common_mpc/contact_planning/` (`OcpQpHpipm`, `MixedIntegerOcpQp`, the terms under
`problem/`, `model/`, `cost/`, `constraint/`, `logic/`, `search/`, `execution/`, `ContactPlanningTermFactory`,
`LipContactPlanner`, `ContactPlan`, `ContactPlanningReferenceManager`, `ContactPlannerModule`), the Bazel target
`contact_planning_core` for everything that needs no robot model; tests in `humanoid_common_mpc/test/` (including the
fixture-based regression tests of the assembled formulation) and `humanoid_centroidal_mpc/test/testContactPlanningIntegration.cpp`.

### 2.8 Adaptive execution between plans

The planner decides the contact sequence on a coarse grid (0.1 s) and re-plans at about 10 Hz, and the commit window
protects a swing in flight from being re-timed by a later plan. On its own that makes the executed schedule rigid at
exactly the moments that matter for disturbance rejection: a swing foot that hits the ground early is still commanded to
fly (zero-wrench constraint against the ground), a foot that misses the ground is switched to stance in mid-air, and a
push during a swing cannot move the landing target before the next plan arrives. The reference manager therefore adapts
the schedule it executes between plans, using only the measured contact state and the closed-form LIP. The features are
implemented as `ExecutionRule` terms under `humanoid_common_mpc/contact_planning/execution/` (`PhaseResettingRule`,
`EnergyCadenceModulationRule`, `DcmStepAdjustmentRule`; the same list also holds the two reference overrides,
`PlannedComOverride` and the `PlannedHeadingOverride` of the heading model, which are plumbing rather than corrections) over the
pure schedule queries and edits of `ContactScheduleAdaptation.{h,cpp}`, written for any number of feet (`kNumContacts`),
and driven from `ContactPlanningReferenceManager` through the pipeline of `ScheduleAdaptationPipeline.h`.

**None of the three corrections is enabled in the shipped files**: `phase_resetting`, `energy_cadence_modulation` and
`dcm_step_adjustment` are all commented out of both robots' `execution` lists, which list only `planned_com_override`.
Each correction changes the closed loop, and with none listed the reference manager merges plans exactly as it did
before they existed, which is the behavior every gait is tuned against. Enable one at a time and validate it in
simulation. They are execution rules of the `execution` list of `contact_planning.textproto`, each with a parameter block of
its own (`phase_resetting` must precede `energy_cadence_modulation` when both are listed, because an early touch-down
ends a swing before the cadence rule may re-time it). The snippet below shows `phase_resetting` enabled, as an example
of opting one in; it is not the shipped list:

```textproto
# ... planner, shared, term lists ...
execution: "planned_com_override"  # the plan's center of mass replaces the horizontal CoM reference of the target
execution: "phase_resetting"       # early touch-down: switch the foot to contact at once; late: extend the swing
# execution: "energy_cadence_modulation"  # re-time the touch-down of the swing in flight by the LIP orbital energy error
# execution: "dcm_step_adjustment"        # move the landing target by the DCM error propagated to touch-down
phase_resetting {
  early_touchdown_min_swing_ratio: 0.25       # contact during this initial fraction of the nominal swing is ignored (scuffing)
  early_touchdown_min_contact_duration: 0.02  # [s] contact must persist this long before the swing is ended (debounce)
  early_touchdown_min_advance: 0.04           # [s] a touch-down less than this ahead of schedule is executed as planned
  max_late_touchdown_extension: 0.15          # [s] total extension budget of a swing past its planned touch-down
  late_touchdown_extension_step: 0.05         # [s] the touch-down is pushed this far ahead of the current time per cycle
  late_touchdown_search_velocity: 0.05        # [m/s] descent rate of the foot height target while searching for the ground
}
energy_cadence_modulation {
  gain: 0.01                                  # [s/J] touch-down shift = -gain * (E - E_pred)
  deadband: 0.0                               # [J]
}
dcm_step_adjustment {
  gain: 0.5                                   # 1 = exact LIP compensation of the DCM error at touch-down
  max_offset: 0.05                            # [m] bound on the landing target offset (also clipped to the reachable region)
}
```

Every query of the schedule in this layer treats an event at exactly the query time as already passed. That is the
convention of the SQP (after a post-event node the first interval starts an epsilon after the event), but the opposite of
`ocs2::ModeSchedule::modeAtTime`, so the two must not be mixed; the helpers `modeIndexAtTime` / `contactFlagsAtTime` are
used throughout the reference manager for this reason.

#### 2.8.1 Phase resetting on measured contact events (`phase_resetting`)

The measured contact flags reach the MPC as the observation mode. The MRT joint controller asks its contact estimator
(`robot_model/ContactEstimator.h`) once per control cycle; the answer is the observation mode and, in the same cycle,
the gate of the inverse dynamics: the feedforward torques project only the planned contact wrenches of the feet that are
measured in contact (`ContactWrenchGate`), since a foot in the air cannot transmit a wrench whatever
the executed plan expects there. The estimator is selected by name in the task file, `contact_estimator: "<name>"`, the
way the costs and constraints of the formulation are: the `ContactEstimatorRegistry` resolves the name and rejects an
unknown one listing the available names. `robot_state` hands back the flags of the `RobotState`, `always_in_contact`
reports every point as touching (the executed schedule then is the measured contact state), and the MuJoCo simulator
registers `cheater_sim`. A controller built without an estimator uses `robot_state`. The robot process of either
formulation hot-reloads the name: from the task file the GUI publishes on `operator/mpc_parameters` and from its watched
task file. Its operator mailbox resolves the estimator on the IO thread (`OperatorCommandMailbox::postControllerSettings`),
and the realtime thread swaps it in; the GUI's Base Controller tab has a drop-down for it. The measured flags are
compared with the schedule at the start of every solve.

The gate itself can shape the load onset (`ContactWrenchGate`, task file block `contact_wrench_gate`, a block of the GUI's
MPC Parameters tab): `debounce_time` withholds a foot's planned wrench until its measured contact has persisted that
long, a bounce across the detection threshold restarting it, and `ramp_time` then raises the wrench linearly from zero
to the planned value. Both default to zero, the instantaneous gate, in which the full planned wrench is projected
from the first cycle a heel or toe strike trips the contact detection. Lift-off is never shaped. The shaping acts only
on the feedforward torques; the MPC observation mode still switches at the first measured contact. Per foot the
manager keeps a small latch of the swing currently in flight (its lift-off, its nominal touch-down, and any re-timing
applied so far).

In the MuJoCo simulation `contact_estimator: "cheater_sim"` (the default when the field is absent) selects the
`CheaterSimContactEstimator` (`mujoco_sim_interface/CheaterSimContactEstimator.h`), which reports a contact point as
touching when the physics carries more than `sim_contact_force_threshold` newtons of normal force between it and anything
outside the robot (a contact point without a resolvable MuJoCo body keeps reading as touching). With
`always_in_contact` every swing reads as an early touch-down at its scuffing window, so phase resetting must not be
enabled in simulation with it (the simulator logs a warning), and every planned contact wrench reaches the inverse
dynamics as it historically did. The viewer's contact timeline (`b`, `contact_timeline` in `sim_visualizations`) shows
the contact state the executed policy plans against that ground truth, which is the quickest way to see early or late
touch-downs and foot scuffing.

The viewer also draws the **target contact patches** (`g` toggles them; like every viewer marker they are enabled by
listing them in `sim_visualizations`, on in the shipped task files): the `contacts.contact_rectangle` of every foot placed at the pose the contact planner wants the
foot on the ground, so the planned position and yaw of every step can be checked against the robot. A bright filled
patch is the landing pose of the swing in flight (the plan's foothold at its touch-down plus any DCM step adjustment),
a translucent one is the landing pose of the foot's next swing, and a faint outline marks a foot with no upcoming swing
at its current placement. The arrow is the patch's x axis, i.e. its yaw: the planned landing yaw with the heading model
(`heading_double_integrator` in `dynamics`) and the measured foot yaw without it. The poses are the ones the reference manager computed at its
last solve (`ContactPlanningReferenceManager::getTargetContactPoses`), so they move whenever a new plan or a contact
event re-times the schedule.

Three centroidal markers put the reduced model the planner reasons with next to the physics: `center_of_mass` (`o`)
draws the whole-body center of mass as a sphere with a vertical down to its shadow on the ground, `zmp` (`z`) the zero
moment point of the physical ground reaction as a disc on the ground (hidden while the robot carries no weight, e.g. on
the gantry), and `dcm` (`d`) the divergent component of motion of the measured centroidal state,
`com_xy + v_xy / omega` with `omega = sqrt(g / z_com)`, with a line from the CoM's shadow to it. The DCM ahead of the
support polygon is what the next step has to catch; the ZMP leaving the sole is what the wrench cone would not allow.
They are on in the shipped DRC Atlas task file and listed, off, in the others.

**Early touch-down.** A foot that is scheduled to swing but is measured in contact after the first
`early_touchdown_min_swing_ratio` of the *nominal* swing duration (scuffing right after lift-off is ignored), and whose
contact persists for `early_touchdown_min_contact_duration` (a debounce against a single chattering sensor sample), is
switched to contact at the current time, provided the scheduled touch-down is still more than
`early_touchdown_min_advance` (0.04 s by default) away. A touch-down closer to its schedule than that is executed as
planned: in a gait that exchanges support in a single instant the scheduled touch-down is the other foot's lift-off, so
ending the swing a few milliseconds early would open a double support shorter than the MPC's own time step on nearly
every step. The guard does not apply to a swing that has already been extended past its planned touch-down (a foot
searching for the ground lands at once). A switched swing ends at the current time: the phase containing the current time is split there and the foot is in
contact from then until its old touch-down, which disappears. The detection is level-triggered, so a contact that
started inside the ignored window and persists past it is a landing too; the debounce timer only runs past the window.
Nothing else moves: the other feet and every later event keep their timing, because those were planned consistently
with the plan that is about to be merged (shifting the tail would make the applied schedule disagree with the plan at
the merge point and produce phantom micro-swings). The touch-down now lies before the commit window, so the boundary
shrinks back to `commit_time` and the planner is free to re-time the following phases; the reference manager additionally
raises a re-plan request that makes `ContactPlannerModule` skip its rate limiter once. The commit window itself is not
shortened below `commit_time`, because it covers the planner latency and a shorter window would let the fresh plan re-time
a phase the controller has already started. The lift-off position latch records the landed position on the next cycle,
so the foot reference does not jitter.

**Late touch-down.** A foot whose swing has reached its planned touch-down without measured contact would be switched to
stance in the air. Instead its touch-down is pushed to `now + late_touchdown_extension_step` and every later event is delayed
by the same amount, so that the following double support and the other feet's phases keep their durations (no flight
phase can appear). The extension is repeated every cycle while contact is missing, in total at most
`max_late_touchdown_extension` past the planned touch-down (measured from the planned touch-down, not from the last
extension); when the budget is used up the contact phase proceeds as scheduled. During the extension the xy foot
reference holds the landing target (its interpolation is timed on the swing without the extension), and the height
reference is the planned swing's up to the planned touch-down, continued from there as a straight descent at
`late_touchdown_search_velocity` from the planned touch-down height (`SwingTrajectoryPlanner::GroundSearch`); the pitch and
impact-proximity references hold their touch-down values. Re-fitting the height spline over the extended swing, as an
earlier version did, moved its apex, raised the reference at the current time by several millimeters at every extension
step and then drove the foot down at the spline's slope of about 0.35 m/s instead of at the search velocity. Contact measured at any time during the extension ends the swing at once
through the early touch-down path.

Because a late extension moves later events, the active plan is shifted by the same amount (`ContactPlan::shiftInTime`,
start time and commit boundary), which keeps the merge consistent, and the shift is logged so that a plan that was still
being computed from the pre-shift schedule is shifted too when it arrives (a plan is shifted by every logged shift that
is younger than its start time; the planner snapshot is taken after the reference manager ran in the same cycle).

#### 2.8.2 DCM step adjustment (`dcm_step_adjustment`)

Between plans the landing target of every swing foot is corrected with the capture point. The reference the correction
is measured against is the whole-body NMPC's **own predicted trajectory**: after every solve `ContactPlannerModule`
hands the primal solution to the reference manager, which interpolates it at the start of the next solve and evaluates
the predicted CoM position and velocity there. With $\boldsymbol{\xi} = \mathbf{c} + \dot{\mathbf{c}}/\omega$ the DCM of the
measured state and $\boldsymbol{\xi}^{\mathrm{pred}}(t)$ that of the prediction, the error
$\Delta\boldsymbol{\xi}(t) = \boldsymbol{\xi}(t) - \boldsymbol{\xi}^{\mathrm{pred}}(t)$ is zero while the robot does what the controller
expects and non-zero only under a real disturbance. On the LIP the error grows until the touch-down at $t_{\mathrm{TD}}$ as
$e^{\omega (t_{\mathrm{TD}} - t)}$ (the error dynamics do not depend on the nominal ZMP as long as both trajectories share
it), and moving the foothold by exactly that amount restores the planned DCM offset with respect to the new support:

$$\mathbf{p}^{\mathrm{adj}}_{\mathrm{land}} = \mathbf{p}^{\mathrm{nom}}_{\mathrm{land}} + K_{\mathrm{dcm}}\, e^{\omega (t_{\mathrm{TD}} - t)}\, \Delta\boldsymbol{\xi}(t),$$

with $K_{\mathrm{dcm}} = 1$ the exact LIP compensation (`dcm_step_adjustment.gain`). The offset is limited to
`dcm_step_adjustment.max_offset` in norm and the adjusted foothold is clipped to the planner's reachable region around the
planned CoM at touch-down, in the planner's frame at touch-down (`reachability.reach_x`, `.reach_y_inner`, `.reach_y_outer`; with the heading
model that is the planned heading at the touch-down time, not the heading at the plan's snapshot, which is stale by the
plan's age while the robot turns), where the side of the foot (left or right of the CoM) is the foot's own, as in the
planner's reachability rows, so that a foot is never moved across the body. The
adjustment is recomputed at every solve (it is a function of the current state, so it cannot run faster than the MPC)
and blended into the swing reference with the same cubic profile as the step itself, so it is zero in position at
lift-off and fully applied at touch-down; like the step, it is commanded with a non-zero velocity from lift-off on (the
offset over the swing duration), not faded in from rest. The prediction is refreshed after every solve, so the error is the deviation that
appeared over the last solver period and the correction acts on that increment once: the next cycle measures against a
prediction that already starts from the disturbed state, and the offset fades on its own. That is what makes the loop
safe (it cannot grow with the age of a plan) and also what bounds it: it damps impulses and slips, and it does not
integrate a sustained push, whose persistent displacement is left to the planner, which re-plans from the measured state
every planning period. A skipped solve makes the next increment correspondingly larger. The adjustment only modulates
the landing target; it never triggers an early touch-down. Without a prediction covering the current time (first solve,
solver failure) no correction is applied.

**Why the reference is the NMPC prediction and not the planner's LIP.** An earlier version compared against the
mixed-integer planner's LIP trajectory. The whole-body controller chooses a different ZMP than the reduced model *by
design*, so that comparison reported the design difference as a disturbance. The LIP is unstable in exactly the
direction the correction acts: the mismatch present when a plan is made had grown by $e^{\omega\,t_{\mathrm{age}}}$ by the
time the plan was applied (1.4x to 2.3x for a plan 0.1-0.25 s old) and the correction multiplied it by
$e^{\omega (t_{\mathrm{TD}} - t)}$ again (up to about 4x over a swing), so a ZMP mismatch of a few centimeters drove the
offset to its bound on every step and, with a forward velocity command, pulled the swing foot backwards. Against the
controller's own prediction that term is absent. The feature remains opt-in like the others: enable it in simulation
first and check that the offset is not sitting at its bound.

#### 2.8.3 Energy-based cadence modulation (`energy_cadence_modulation`, off by default)

The orbital energy of the LIP along the planned heading at the current time (with the heading model the plan turns over
its horizon; without it this is the plan's yaw), relative to the planned ZMP,

$$E = \tfrac{1}{2} m \big(\dot{x}^2 - \omega^2 x^2\big), \qquad x = \mathbf{e}_x^\top(\mathbf{c} - \mathbf{z}),$$

is conserved between contact switches. A CoM that carries more energy than the plan passes over the support earlier and
should step earlier; less energy should delay the step. The touch-down of the swing in flight is moved by
$\Delta t_{\mathrm{TD}} = -K_E\,(E - E^{\mathrm{ref}})$ relative to its *nominal* touch-down (not cumulatively), clipped to
`[shared.gait_limits.min_swing_duration, shared.gait_limits.max_swing_duration]` after lift-off. A request earlier than what is still feasible brings the
touch-down forward to 20 ms from now at the earliest, and never pushes a touch-down that is already imminent further out
(flooring at "now + margin" on every cycle would drag the touch-down along with the clock and the foot would never
land); every later event moves with it and the plan is shifted alongside, exactly as for a late touch-down. Shifts
logged while a plan was being computed are applied to that plan when it arrives, summed against the plan's original
snapshot time so that shifts of opposite sign cancel. It is not applied while a foot is
searching for the ground. `energy_cadence_modulation.gain` is in seconds per joule of the full robot mass, so the same value acts differently
on robots of different mass; the default 0.01 s/J moves the touch-down by about 0.1 s for a 0.15 m/s forward velocity
deviation of a 150 kg robot walking at 0.4 m/s, if that deviation appears within one solver period (like the step
adjustment, the energy is compared against the prediction refreshed at every solve, so it is the per-period increment
that is corrected). `energy_cadence_modulation.deadband` (joules, 0 by default) ignores deviations within the band and measures the
shift from its edge: without it the CoM velocity noise re-times the touch-down, and every event after it, at every solve
(5 mm/s at 0.4 m/s on 150 kg is 0.3 J, about 3 ms). The feature is
a heuristic that overlaps with the DCM step adjustment and the planner's own re-timing; it is disabled by default and
should be enabled only with simulation tests. Like the step adjustment it measures the energy error against the NMPC's
own predicted CoM state, relative to the planned support point, not against the planner's LIP.

#### 2.8.4 What the tests cover

`humanoid_common_mpc/test/testContactPhaseResetting.cpp` exercises the schedule layer without a robot model, for every
foot: the event conventions, in-place truncation (also for swings spanning several phases), tail shifting, early
touch-down acceptance / rejection / level triggering, late extension in steps up to the budget, contact during an
extension, cadence clamping, an alternating-gait simulation with random early and late landings (schedule stays
consistent, no flight phase), the LIP closed form, energy conservation, the step adjustment propagation and bound, the
reach clipping in rotated frames, plan time shifting and the configuration fields.
`humanoid_centroidal_mpc/test/testContactPlanningIntegration.cpp` runs the same scenarios on the DRC Atlas model through
the interface: a forward velocity error moves the landing reference forward within the bound, an early contact switches
the foot to stance in place and yields a consistent planner input, a missing contact extends the swing in steps up to
the budget while the xy reference holds and the height keeps descending, and extra energy shortens the swing in flight.

Implementation: `humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.{h,cpp}` (pure schedule and LIP
functions), `ContactPlanningReferenceManager` (event handling, plan shifting, DCM adjustment, swing height search),
`ContactPlannerModule` (immediate re-plan on a contact event). Note that the mixed-integer planner itself
(`LipContactPlanner`) is formulated for two feet; the execution layer described here is not.

The closed forms of the adaptive execution are pinned by `humanoid_centroidal_mpc/test/testContactPlanningIntegration.cpp`
on the DRC Atlas model: the cadence shift against the orbital-energy increment (velocity, position, lateral, with and
without the deadband), the DCM adjustment against the propagated DCM increment (no accumulation over cycles, vanishing
once the prediction agrees, applied to the swing in flight only), the descent of the height reference at the search
velocity during a late extension, and the counting of dropped plans. `humanoid_common_mpc/test/testSwingLandingVelocity.cpp`
covers the ground-search reference of the swing trajectory planner, `testContactPlannerModule.cpp` the snapshot throttle,
and `testLipContactPlanner.cpp` / `testContactPlan.cpp` the committed-node sampling and merge at a grid-aligned commit
boundary, the warm start surviving a hot reload, and the alternation rule not judging the committed prefix.

### 2.9 Heading model: ACoM dynamics in the planner (`heading_double_integrator`)

The point-mass LIP has no notion of heading. Its foothold frame is the base yaw at planning time, held fixed over the
horizon, and the planner only receives the commanded linear velocity, so a pure yaw command produces no step at all
and the robot cannot turn. The heading model is the `heading_double_integrator` block of the `dynamics` list together
with its costs (`heading_rate_tracking`, `heading_tracking`, `foot_yaw_tracking`, `yaw_torque_regularization`,
`foot_yaw_regularization`), its constraints (`hip_yaw_range`, `yaw_torque_budget`, `foot_yaw_pinned_in_contact`), the
`heading_relinearization` search stage and the `planned_heading_override` execution rule;
`ContactPlanningFormulation::setHeadingModel()` adds or removes them as a whole. It adds the missing coordinate as a
reduced dynamics of its own, expressed
in the angular center of mass (ACoM): the whole-body heading `theta` (the ACoM yaw when the robot has an ACoM network,
the base yaw otherwise) and its rate `omega = L_z / I_zz`, the angular momentum about the vertical through the center
of mass over the whole-body yaw inertia (taken from the model at every plan; no configuration field sets it). The ACoM
network is installed whenever the configuration uses the heading model - at start-up, and on a hot reload that switches
it on (`ContactPlanningReferenceManager::loadHeadingModelEvaluator`) - so a reload gives the same heading as a start
with the same file. A robot without a network keeps the base yaw, with a warning; a network trained on a joint vector
other than the MPC model's refuses the start, or the reload, with a `Status` naming the mismatched joints.

Per node, appended to the LIP block so that the contact binaries keep their place:

```
omega_{k+1} = omega_k + dt (tau_L + tau_R) / I_zz
theta_{k+1} = theta_k + dt omega_k + dt^2 (tau_L + tau_R) / (2 I_zz)     exact zero-order hold of the double integrator
psi_{i,k+1} = psi_{i,k} + dpsi_{i,k}                    foot yaw, one per foot
|tau_i| <= T_t c_i + (T_c - T_t) (c_L + c_R - 1) / 2    T_t alone, (T_t + T_c) / 2 per foot in double support
|dpsi_i| <= 2 pi (1 - c_i)                              a foot's yaw is pinned while it is in contact
lower_i <= psi_i - theta <= upper_i                     hip range (soft, hip_yaw_range, per foot from the model)
```

The heading is driven only by the ground: a point mass's horizontal contact force acts through the ZMP and produces no
yaw moment about the center of mass, so what remains is the torsional friction of the weight a stance foot carries
(`torsionalFrictionTorque`, `T_t`, for the whole weight on one foot) and, in double support, the friction couple of the
two feet (`doubleSupportYawCouple`, `T_c`). With both feet down each carries half the weight and half the couple, so the
pair has `T_t + T_c`; granting the full-weight torsion to both feet at once, as an earlier version did, over-estimated
the double-support budget by `T_t` (78 N m on the DRC Atlas). A turn therefore has to be stepped: the planner rotates
the foothold frame with the heading, the feet must follow it within hip range, and the yaw torques bound how fast the
heading can go. The costs track the commanded yaw rate and the commanded heading, keep the foot yaws at the nominal
heading (the linearization point, which is the plan's own heading under the successive linearization; measuring a pinned
stance foot's yaw against the heading *state* was a penalty on the heading alone and pulled it back towards the stance
feet), and regularize the torques and the foot yaw displacements.

The commanded yaw rate is read off the momentum channel of the MPC's target trajectory, where the command is carried
as the angular momentum of a rigid turn about the vertical, `h_z = I_zz omega / m`, with the same locked inertia the
planner uses. The target's base yaw is not used: its first stretch integrates the average of the measured and the
commanded yaw rate, so differentiating it commanded half the rate at the start of a turn and fed the measured rate back
in, and once the override below has rewritten it, it carries the previous plan's rate.

The constraint frame of node k is the planned heading of node k. That makes the step-width, reach and foot-separation
rows bilinear in the heading and the footholds, which the mixed-integer solver cannot take. They are linearized to
first order around a nominal heading trajectory, the previous plan shifted to the current time or, without one, the
commanded yaw integrated from the current heading, so that the relaxations stay convex and the branch-and-bound is
unchanged. The measured heading arrives wrapped to $[-\pi, \pi]$ while the previous plan's heading lives on whatever
branch that plan started on; the nominal is moved onto the measurement's branch by one $2\pi$ offset before use (across
a crossing of $\pm\pi$ the unshifted nominal made every first-order frame term worth $2\pi g$, over a meter on the step
width, and aimed the foot yaw tracking a full turn away). After the search the frame is re-linearized at the incumbent's own heading and footholds and the QP is
re-solved with the contacts fixed (the `heading_relinearization` search stage, `heading_relinearization.passes`,
successive linearization), so that the frame the constraints were written in is the frame the plan actually turns
through.

Downstream, the plan carries the heading and the foot yaws per node. The swing-foot reference turns each foot from
its lift-off yaw to the planned landing yaw with the same profile as the position, and the foot cost tracks that yaw
through `task_space_foot_cost.weights.orientation_z` (0 by default: not tracked). With the `planned_heading_override`
execution rule listed the plan's heading replaces the commanded yaw in the MPC's target trajectory, so that the whole-body controller is
asked for the turn the ground can support rather than the raw command; with ACoM tracking the reference base yaw is
set so that the reference ACoM heading equals the planned heading.

The model stays a reduced one: the angular momentum exchange with the swing leg and the arms, which is how a humanoid
actually turns between steps, is the whole-body controller's business over its own horizon. The planner decides where
and when the feet go and how fast the heading may change for the ground to carry it.

**Parameters from the model.** The quantities that are properties of the robot and of the ground rather than tuning
are not task-file fields: the yaw inertia (composite inertia about the vertical through the center of mass, at every
plan), `torsionalFrictionTorque` (the torsional friction coefficient of the wrench cone times the weight),
`doubleSupportYawCouple` (the friction coefficient times half the weight times `step_width.nominal_step_width`) and the
foot yaw bounds of `hip_yaw_range` (the hip yaw joint limits of every leg, found by walking the kinematic tree up from
the joint that carries the contact frame to the first revolute joint whose axis is vertical in the world frame at the
neutral configuration, and mirrored when that axis points down, since the foot yaw is then minus the joint angle; per
foot and asymmetric, which matters because a hip yaw range is typically much wider outward than inward;
`deriveHipYawRange`). They are derived once by the interface and applied to every
planner configuration, including the hot reloads of `contact_planning.textproto`, and logged at start-up. `shared.com_height` and the ZMP
box stay tunable, a `shared.com_height` left out and a ZMP half width of 0 meaning "from the model": the center of mass
height above the feet at `initial_state` (`computeComHeightAboveFeet`, the same pendulum a `dcm_terminal_cost` without
`com_height` resolves to; both shipped robots leave it out, which the conversion marks with the library default of 0 -
a configuration with no model to fill it in must set a positive height, which validation otherwise refuses naming
`shared.com_height`, and a file that gives 0 is refused as no height), and the sole's footprint from the wrench cone. The friction and footprint come from `contact_wrench_cone_soft_constraint`, so the
planner cannot assume more yaw torque than the whole-body constraint would ever allow. That block is read through
`contactWrenchConeConfigFromConfig`, which requires every one of its geometry fields, so a task file without it
(the Unitree G1 and R1 configure friction in `friction_force_cone_soft_constraint` only) is refused under
`contact_schedule_source: "contact_planner"` by the missing field rather than planned against a library default.

### 2.10 Assembling the planner from terms (`contact_planning.textproto`)

The planner is assembled the way the whole-body NMPC is: `ContactPlanningProblem` holds named collections of terms
(the analog of OCS2's `OptimalControlProblem`), `ContactPlanningTermFactory` fills them from the term lists of the
configuration (the analog of `HumanoidCostConstraintFactory` with the `costs` / `soft_constraints` /
`hard_constraints` lists of the task file), and every term reads its own parameter block, which is the hot-reload path.
The file `contact_planning.textproto` mirrors that structure:

<!-- LINT.IfChange(formulation_term_table) -->

| Field | What it holds |
| --- | --- |
| `planner` | properties of the planner itself: `type` (`hlip` or `lip_miqp`), grid (`dt`, `num_nodes`), `commit_time`, `max_commit_extension`, solver budget, threading, `log_plans` (one line per plan: search statistics, phase durations, step lengths) |
| `shared` | parameters read by more than one term: `gravity`, `com_height`, `big_m`, the default `slack_penalty` of the soft constraints, the `gait_limits` |
| `dynamics` | model blocks, in the order that fixes the variable layout: `lip_com`, `foothold_integrator` (mandatory, always first), `heading_double_integrator` |
| `costs` | in the accumulation order of the stage matrices: `regularization`, `previous_foothold_consistency`, `velocity_tracking`, `step_width`, the heading costs, `zmp_regularization`, `foothold_regularization`, `step_length`, `terminal_dcm` |
| `soft_constraints` | `zmp_support_region`, `reachability`, `foot_separation`, `hip_yaw_range`; each may carry a `slack` block that overrides the shared penalty |
| `hard_constraints` | `no_flight`, `foot_motion_in_swing_only`, `yaw_torque_budget`, `foot_yaw_pinned_in_contact` |
| `logic_rules` | propagation on the binaries: `phase_durations`, `no_flight`, `minimum_double_support`, `alternating_feet` |
| `assignment_costs` | `contact_switch`, `plan_consistency`, `double_support_penalty` |
| `search` | `warm_start_previous_plan`, `diving`, `event_shift_local_search`, `heading_relinearization`, `cadence_stretch` |
| `execution` | the reference manager's rules of section 2.8 (`phase_resetting`, `energy_cadence_modulation`, `dcm_step_adjustment`) and the reference overrides `planned_com_override` and `planned_heading_override` |
| one block per term | its parameters, named as the term (`velocity_tracking { weight: 50.0 }`, ...) |

<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names, //humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h:term_names) -->

A term listed without the block it needs (a heading cost without `heading_double_integrator`), an unknown name, a
duplicate, or an execution order the rules cannot honor is rejected by `contactPlanningConfigFromConfig()` /
`ContactPlanningConfig::validateStatus()` with an InvalidArgument whose message names the list or field to change and
lists the supported names; combinations documented to fall or to block are reported by
`ContactPlanningConfig::warnings()`. `hip_yaw_range` and `yaw_torque_budget` take their values from the robot model
(`ContactPlanningModelParameters`), not from the file. The planner logs the assembled formulation (the planner settings,
the variable layout, every term and search stage with a one-line description of its math and current values) at start-up
and after every reload that changes that text or the formulation, grid or planner type
(`ContactPlannerModule::reloadSummary`); `LipContactPlanner::formulationSummary()` returns the same text. The values are
checked once, by `ContactPlanningConfig::validateStatus()`, whose messages name the field to change; a term's
`configure()` only reads its block, and `makeContactPlanner`, `ContactPlannerModule::Create()` and
`ContactPlanningReferenceManager::Create()` return that Status for an invalid configuration instead of building from it,
as `ContactPlannerModule::setConfig()` does on a reload, keeping the running configuration.

The flat layout of the previous planner (every key directly under `contact_planning`, `useAcomDynamics`,
`enablePhaseResetting`, ... as booleans) is no longer read: its keys are not fields of the schema, and the parser
refuses the first of them with a hint that says how to migrate the file (the `retired_layout_hint` of
`ContactPlanningFile`: the flags became list entries, every weight a term block). The shipped DRC
Atlas file (`robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.textproto`) is the structured layout
with the values and the formulation of the previous planner. `testContactPlanningRegression.cpp` pins the assembled
problems, the propagation of complete assignments and receding-horizon plans to fixtures under
`test/data/contact_planning/`, and checks the propagation of the partial assignments the search reaches against the
complete enumeration (section 2.3); the fixtures were recorded at the
point where the term-assembled planner had been shown equivalent to the previous one; re-record them with
`REGENERATE_CONTACT_PLANNING_FIXTURES` after an intended change of the formulation and review the diff.
The refactor is described in [contact_planner_ocp_refactor/README.md](contact_planner_ocp_refactor/README.md).
