# Online Contact Planning and the DCM Terminal Cost

This document describes two optional formulation features of the centroidal humanoid NMPC, both toggled in the robot's
`config/mpc/task.yaml`:

| Toggle | Effect |
| --- | --- |
| `useDcmTerminalCost: true` | The horizon ends with a Divergent Component of Motion (capture point) viability cost. The quadratic `Q_final` terminal cost (`terminal_cost`) is ignored. |
| `useContactPlanning: true` | The contact sequence, the switching times and the footholds are planned online by a mixed-integer program on a reduced model and replace the clock-driven gait schedule. |

Both were added because a fixed, pre-scheduled contact sequence is the first bottleneck of a switched-system MPC: under
pushes, on uneven ground or with changing speed commands the rigid schedule causes scuffing, early impacts and infeasible
terminal states.

---

## 1. DCM terminal cost (`useDcmTerminalCost`)

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
infeasible a few cycles later. The quadratic `Q_final` cost regulates the full state (base pose, momentum, joints) towards a
reference and knows nothing about support; it also has to be re-tuned for every gait cadence because "the right" base pose
at the end of the horizon depends on where in the gait cycle the horizon ends.

The DCM terminal cost replaces it with the physically meaningful quantity. At the terminal time $T$ it penalises

$$\ell_T(\mathbf{x}_T) = \tfrac{1}{2}\, \big(\boldsymbol{\xi}_T - \boldsymbol{\xi}^{\mathrm{ref}}_T\big)^\top W \big(\boldsymbol{\xi}_T - \boldsymbol{\xi}^{\mathrm{ref}}_T\big),
\qquad
\boldsymbol{\xi}^{\mathrm{ref}}_T = \mathbf{p}_{\mathrm{support}}(\mathbf{q}_T) + \beta \frac{\mathbf{v}_{\mathrm{cmd}}}{\omega},$$

where

* $\boldsymbol{\xi}_T = \mathbf{c}_{xy}(\mathbf{q}_T) + \mathbf{h}_{xy}(\mathbf{x}_T)/\omega$ is evaluated on the full model: the CoM position
  from the kinematics and the CoM velocity from the normalised linear momentum in the centroidal state;
* $\mathbf{p}_{\mathrm{support}}$ is the weighted centre of the contact frames of the feet in contact at $T$ according to the
  mode schedule. The weights are continuous in time: a foot's weight is 1 in the middle of a contact phase and ramps
  linearly to 0 over `supportBlendTime` before its lift-off and after its touch-down (both feet as a fallback in flight).
  Without the blending the reference would jump by half the step width every time the receding horizon end crosses a
  mode switch, which with a weight of several hundred is a visible disturbance at every solve;
* $\mathbf{v}_{\mathrm{cmd}}$ is the commanded CoM velocity from the target trajectories and $\beta$ (`velocityOffsetFactor`)
  scales the offset. With $\beta = 0$ the cost demands pure capturability (the robot can stop over its feet). With
  $\beta = 1$ the reference is the DCM of a CoM that moves at the commanded speed over the support point
  ($\dot{\mathbf{c}} = \omega(\boldsymbol{\xi} - \mathbf{c}) = \mathbf{v}_{\mathrm{cmd}}$), so a steady walk is not decelerated by the
  terminal cost while any excess divergence is still penalised.

The cost is implemented as a CppAD residual with a Gauss-Newton Hessian $J^\top W J$, so its quadratic approximation is
positive semi-definite for every configuration. $W$, $z_c$ and $\beta$ are parameters of the compiled model and are
hot-reloadable from the `dcm_terminal_cost` section of `task.yaml` through the parameter updater (GUI tab
"Terminal Cost").

```yaml
useDcmTerminalCost: true
dcm_terminal_cost:
  comHeight: 0.85            # z_c for omega
  gravity: 9.81
  weight_x: 400              # W = diag(weight_x, weight_y)
  weight_y: 400
  velocityOffsetFactor: 1.0  # beta
  supportBlendTime: 0.1      # [s] ramp of the support weights through lift-off / touch-down
```

Implementation: `humanoid_centroidal_mpc/cost/DcmTerminalCost.{h,cpp}`; wiring in `CentroidalMpcInterface.cpp`.

---

## 2. Mixed-integer contact planning (`useContactPlanning`)

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

The NMPC itself is unchanged: it still sees a mode schedule (which feet are in contact when) and constraints derived from
it. What changes is who produces the schedule. Without contact planning the `GaitSchedule` tiles a periodic template
selected by the velocity-based gait state machine. With contact planning the schedule is the output of a mixed-integer
optimisation that is re-solved continuously from the current state.

The planner runs on a reduced model so that it can afford combinatorial search; the whole-body NMPC then tracks the
result with the full dynamics. The two are coupled through the mode schedule, the swing-foot landing references and, at
the end of the horizon, through the DCM terminal cost that both formulations share.

### 2.2 Reduced model and decision variables

The planner discretises a horizon of $N$ nodes of duration $\Delta t$ (default $12 \times 0.1\,\mathrm{s}$). Per interval $k$:

| Symbol | Type | Meaning |
| --- | --- | --- |
| $\mathbf{c}_k, \dot{\mathbf{c}}_k \in \mathbb{R}^2$ | state | LIP CoM position and velocity |
| $\mathbf{p}_{i,k} \in \mathbb{R}^2$ | state | position of foot $i \in \{L, R\}$ (its planned landing spot while swinging) |
| $\mathbf{z}_k \in \mathbb{R}^2$ | input | ZMP during the interval |
| $\delta\mathbf{p}_{i,k} \in \mathbb{R}^2$ | input | foot displacement during the interval |
| $c_{i,k} \in \{0, 1\}$ | **binary** input | foot $i$ in contact during the interval |

Dynamics (exact zero-order-hold discretisation of the LIP with constant ZMP per interval):

$$\begin{bmatrix}\mathbf{c}_{k+1}\\ \dot{\mathbf{c}}_{k+1}\end{bmatrix} =
\begin{bmatrix}\cosh(\omega\Delta t) & \sinh(\omega\Delta t)/\omega\\ \omega\sinh(\omega\Delta t) & \cosh(\omega\Delta t)\end{bmatrix}
\begin{bmatrix}\mathbf{c}_k\\ \dot{\mathbf{c}}_k\end{bmatrix} +
\begin{bmatrix}1-\cosh(\omega\Delta t)\\ -\omega\sinh(\omega\Delta t)\end{bmatrix}\mathbf{z}_k,
\qquad \mathbf{p}_{i,k+1} = \mathbf{p}_{i,k} + \delta\mathbf{p}_{i,k}.$$

All geometric constraints are written in the yaw-aligned frame of the base at planning time (unit vectors $\mathbf{e}_x$,
$\mathbf{e}_y$), so the linear model stays valid for any heading.

### 2.3 Constraints

**Contact disjunctions (big-M, $M$ = `bigM`).** A foot only moves while it is not in contact:

$$|\mathbf{e}_j^\top \delta\mathbf{p}_{i,k}| \le M\,(1 - c_{i,k}).$$

The ZMP must lie in the support region. In single support that is a box of half-widths $(r_x, r_y)$
(`zmpHalfWidthX/Y`) around the supporting foot; each foot's box is relaxed unless that foot is the *only* contact:

$$\pm \mathbf{e}_j^\top(\mathbf{z}_k - \mathbf{p}_{i,k}) \le r_j + M(1 - c_{i,k}) + M c_{\bar i,k}.$$

In double support the region is the convex hull of the two boxes. Laterally the feet never cross, so the hull is exactly
$\mathbf{e}_y^\top(\mathbf{p}_{R,k}) - r_y \le \mathbf{e}_y^\top \mathbf{z}_k \le \mathbf{e}_y^\top(\mathbf{p}_{L,k}) + r_y$ (each side relaxed by
$M(1-c_{i,k})$). Along the heading the order of the feet is not known in advance and the exact hull would need one more
binary per node; the planner uses the box of half-width $r_x$ around the midpoint of the feet instead, a conservative inner
approximation that costs nothing. These constraints are *soft* (HPIPM slacks with a quadratic and a linear penalty,
`constraintSlackWeight`, `constraintSlackLinearWeight`) so that the relaxations always stay feasible and an unavoidable
violation shows up as cost instead of as a solver failure.

**Kinematics (soft).** Reachability of every foot with respect to the CoM (`reachX`, `reachYInner`, `reachYOuter`), step
length $|\mathbf{e}_x^\top(\mathbf{p}_L - \mathbf{p}_R)| \le$ `maxStepLength` and step width
`minStepWidth` $\le \mathbf{e}_y^\top(\mathbf{p}_L - \mathbf{p}_R) \le$ `maxStepWidth` (self-collision margin).

**Logic on the binaries (exact, outside the QP).** The combinatorial rules are enforced by a propagation hook that the
branch-and-bound calls on every partial assignment (fixing implied values, rejecting contradictions) and on every
candidate incumbent:

* no flight phase: $c_{L,k} + c_{R,k} \ge 1$ (also present in the QP);
* minimum and maximum swing duration, minimum (and optionally maximum) contact duration, counted in nodes from the
  start of the phase, including the time already spent in the current phase before the planning instant. That elapsed
  time is a non-integer number of nodes; it is rounded down when checked against a minimum and up against a maximum, so
  that the grid never violates either limit (rounding to the nearest node allowed a 0.16 s old swing to end after one
  more node, 0.26 s against a 0.3 s minimum). If the two limits are less than a node apart the nearest node decides.
  A foot may also not lift off in the last `minSwingNodes - 1` nodes of the horizon, where a minimum-length swing does
  not fit. Without that rule the plan ended mid-swing and `toModeSchedule()` closed it with a touch-down at `endTime()`,
  so the controller received a swing far shorter than `minSwingDuration` (0.1 s against a 0.3 s minimum, measured over a
  receding-horizon walk), and the swing trajectory planner scaled its height down in proportion to that duration. The
  step is simply deferred to the next plan.
  The maximum contact duration yields to the rules that can make lifting impossible: a foot that is overdue lifts at
  the first node where the other foot is known to support it and no minimum double support holds it, and when both
  feet are overdue at once (a standing start with `maxContactDuration` set) the one that has stood longer, or that did
  not swing last, goes first. Forcing both to lift at the same node used to contradict the no-flight rule and left the
  planner without a single feasible assignment;
* foot alternation (`enforceAlternatingFeet`): a foot may not swing twice without the other foot swinging in between;
* minimum double support (`minDoubleSupportDuration`): after a touch-down the other foot stays down for at least that
  long, so weight transfer is never asked to happen in a single node. This includes the node of the touch-down itself:
  a lift-off at the very node the other foot lands would be an instantaneous switch with no double support at all;
* the committed window: contacts up to the *commit boundary* are fixed to the schedule the NMPC is already executing.
  The boundary is `commitTime` ahead of the planning instant, extended to the touch-down of any swing that has started or
  starts within that window. A swing in flight is therefore never re-timed or cut short by a later plan, and `commitTime`
  must cover the planner latency. A plan is merged into the executed schedule exactly at its own boundary, `commitTime` after the snapshot it was planned from; merging it any later (for instance at the boundary of the solve that activates it) cut its first lift-off short by the plan's age and the controller executed swings shorter than `minSwingDuration`. A plan whose boundary has already passed when it is activated is stale and dropped with a rate-limited warning (the planner latency exceeded `commitTime`, or a touch-down the boundary was extended to happened while the plan was computed); so is a plan that has a foot down where the executed schedule already has it in flight at the merge point, which means a swing was activated between the plan's snapshot and its activation. In both cases the executed schedule keeps running until a fresh plan arrives. Shifting a late plan forward onto the current boundary instead, whole, was tried and is wrong: a plan made just before a touch-down and activated just after it was delayed by a whole swing, and the foot that had just landed was lifted again at once, which threw the robot. The planner's
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
* plan consistency (`planConsistencyCost`): every node whose contact differs from the previous plan, shifted to the
  current time, is charged, which gives the anytime search hysteresis between cycles; the continuous footholds are
  likewise pulled towards the previous plan (`previousFootholdWeight`) so that the landing target tracked by the foot cost
  does not jitter from plan to plan.

Keeping the duration logic out of the QP keeps the relaxations small (8 states, 8 inputs, 27 rows per stage) and costs
nothing in accuracy, because the linear relaxation of such disjunctions is too weak to prune anyway (a foot that is "half
in contact" may move half a step at almost no cost). The search is therefore an enumeration of admissible contact
sequences with strong propagation, made anytime by node and time limits.

### 2.4 Objective

$$J = \sum_{k=0}^{N} w_v \|\dot{\mathbf{c}}_k - \mathbf{v}_{\mathrm{cmd}}\|^2 + w_w\big(\mathbf{e}_y^\top(\mathbf{p}_{L,k}-\mathbf{p}_{R,k}) - w_{\mathrm{nom}}\big)^2
+ \sum_{k=0}^{N-1} \Big( w_z\|\mathbf{z}_k - \mathbf{c}_k\|^2 + w_p \sum_i \|\delta\mathbf{p}_{i,k}\|^2 \Big)
+ w_s\, n_{\mathrm{switch}} + w_T\, \big\|\boldsymbol{\xi}_N - \mathbf{z}_{N-1}\big\|^2 .$$

The last term is the same terminal capturability idea as in section 1, on the reduced model: with
$\boldsymbol{\xi}_N - \mathbf{z}_{N-1} = e^{\omega \Delta t}(\boldsymbol{\xi}_{N-1} - \mathbf{z}_{N-1})$ it is a quadratic function of the
last stage and pulls the end of the plan into a capturable state. $n_{\mathrm{switch}}$ counts lift-off and touch-down
events and is charged through the logical cost hook (`contactSwitchCost`).

### 2.5 Solver

`MixedIntegerOcpQp` is a branch-and-bound over OCP-structured QP relaxations:

1. Every relaxation is solved with HPIPM (Riccati-based interior point on the stage structure; the one-sided rows are
   masked instead of using loose finite bounds). A relaxation costs well under a millisecond for the planner's sizes.
2. Search order: best-bound first among the open nodes, with depth-first diving into the rounding direction after each
   branching. Branching picks the first fractional binary in time order, so early decisions are settled first.
3. Incumbents come early from (a) the previous plan shifted by the elapsed time (warm start), (b) a diving heuristic at
   the root (fix every integral binary, round the first fractional one, propagate, re-solve) and (c) the integral
   relaxations found while diving.
4. After the branch-and-bound an event-shift local search moves every lift-off / touch-down of the incumbent one node
   earlier or later, evaluates the fixed-assignment QP and keeps improvements (`localSearchIterations`,
   `localSearchMaxTime`). This refines the timing at a few QPs per round.
5. `maxBranchAndBoundNodes` and `maxSolveTime` bound the effort; the best plan found so far is returned (anytime).

Typical numbers on the DRC Atlas configuration (12 nodes, 24 binaries, single core): about 0.7 ms per relaxation, a first
incumbent after roughly a dozen relaxations, a walking plan to proven optimality in 0.3-0.5 s, a push-recovery plan in
under 0.1 s. With the default budget of 200 relaxations / 0.1 s the receding-horizon planner returns near-optimal plans at
10 Hz.

### 2.6 Runtime integration

* `ContactPlannerModule` (a solver synchronized module) snapshots the planner input before every MPC solve: CoM position
  and velocity from the state, foot positions from the kinematics, the contact phases and their elapsed time from the
  applied schedule, the committed window and the commanded velocity from the target trajectories. With
  `runInBackgroundThread: true` the snapshot is posted to a worker thread (latest wins, rate-limited by
  `planningFrequency`); otherwise the plan is computed inside the pre-solve hook. After a plan is published the pending
  snapshot is discarded, so the next plan always starts from a schedule that already contains the previous plan, unless
  that snapshot follows a contact event (section 2.8): the event has already re-timed the schedule the finished plan was
  built on, so the post-event snapshot is kept and planned next. Switching `runInBackgroundThread` at runtime starts or
  joins the worker; the stop flag is published under the worker's mutex so a toggle or a shutdown can never lose the
  wake-up and hang the solver thread, and a snapshot or throttle state left over from the previous worker is cleared.
* `ContactPlanningReferenceManager` merges every new plan into the schedule the NMPC is executing (applied schedule up to
  the plan's commit boundary, plan afterwards, never cutting a swing that is in flight or imminent, always starting and ending in double support so that the swing trajectory planner
  finds a lift-off and a touch-down for every swing), updates the foot-height swing planner and exposes the planned
  landing spot of every swing foot as a task-space reference: the xy position interpolates from the lift-off position to
  the landing spot with a smooth-step profile, the height follows the swing trajectory planner. The lift-off position is
  the latched measured foot position for the swing that ends the foot's current contact phase; a later swing of the same
  foot inside the horizon starts from the planned foot position at its own lift-off, i.e. from where the earlier step
  lands, since the latched position would be a step behind.
* `CentroidalMpcEndEffectorFootCost` tracks that reference through `task_space_foot_cost_weights.pos_x / pos_y`. Without a
  plan the xy position error is switched off inside the cost, so the weights are harmless when planning is disabled.
* Until the first valid plan arrives (or if the planner stalls) the gait schedule / the last applied schedule is used.
* The whole `contact_planning` section is hot-reloadable through the parameter updater (GUI tab "Contact Planning");
  structural keys (`numNodes`, threading) are applied by the planner on its next run.

### 2.7 Tuning notes

* `dt`/`numNodes`: the horizon must cover `mpc.timeHorizon`; coarser nodes make the search cheaper but quantise the
  switching times.
* `commitTime`: at least the planner latency (solve time plus one planning period) plus the time the NMPC needs to
  anticipate a switch; too small a value lets a fresh plan re-time switches the controller is already preparing, which
  shows as feet that stutter or barely lift.
* `minSwingDuration`/`maxSwingDuration`/`minContactDuration` are the main shape parameters of the gait. `maxContactDuration`
  forces stepping even without a command (leave at 0 to allow standing).
* `zmpHalfWidthX/Y` should stay inside the physical foot (the NMPC enforces the real wrench cone); a smaller box makes the
  planner step earlier under disturbances.
* `contactSwitchCost` trades stepping against ankle strategy; `terminalDcmWeight` makes plans end capturable.
* Raise `constraintSlackWeight` if plans exploit the soft support region.

Implementation: `humanoid_common_mpc/contact_planning/` (`OcpQpHpipm`, `MixedIntegerOcpQp`, `LipContactPlanner`,
`ContactPlan`, `ContactPlanningReferenceManager`, `ContactPlannerModule`); tests in `humanoid_common_mpc/test/` and
`humanoid_centroidal_mpc/test/testContactPlanningIntegration.cpp`.

### 2.8 Adaptive execution between plans

The planner decides the contact sequence on a coarse grid (0.1 s) and re-plans at about 10 Hz, and the commit window
protects a swing in flight from being re-timed by a later plan. On its own that makes the executed schedule rigid at
exactly the moments that matter for disturbance rejection: a swing foot that hits the ground early is still commanded to
fly (zero-wrench constraint against the ground), a foot that misses the ground is switched to stance in mid-air, and a
push during a swing cannot move the landing target before the next plan arrives. The reference manager therefore adapts
the schedule it executes between plans, using only the measured contact state and the closed-form LIP. The features are
implemented in `humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.{h,cpp}` as pure functions of a
`ModeSchedule`, written for any number of feet (`N_CONTACTS`), and driven from `ContactPlanningReferenceManager`.

**All three are disabled by default.** Each changes the closed loop, and with them off the reference manager merges
plans exactly as it did before they existed, which is the behaviour every gait is tuned against. Enable one at a time
and validate it in simulation. They are configured in the `contact_planning` block of `task.yaml`:

```yaml
contact_planning:
  # ... planner keys ...
  enablePhaseResetting: false           # early touch-down: switch the foot to contact at once; late: extend the swing
  earlyTouchdownMinSwingRatio: 0.25     # contact during this initial fraction of the nominal swing is ignored (scuffing)
  maxLateTouchdownExtension: 0.15       # [s] total extension budget of a swing past its planned touch-down
  lateTouchdownExtensionStep: 0.05      # [s] the touch-down is pushed this far ahead of the current time per cycle
  lateTouchdownSearchVelocity: 0.05     # [m/s] descent rate of the foot height target while searching for the ground
  enableDcmStepAdjustment: false        # move the landing target by the DCM error propagated to touch-down
  dcmAdjustmentGain: 0.5                # 1 = exact LIP compensation of the DCM error at touch-down
  dcmAdjustmentMaxOffset: 0.05          # [m] bound on the landing target offset (also clipped to reachX / reachY*)
  enableEnergyCadenceModulation: false  # re-time the touch-down of the swing in flight by the LIP orbital energy error
  energyCadenceGain: 0.01               # [s/J] touch-down shift = -gain * (E - E_plan)
```

Every query of the schedule in this layer treats an event at exactly the query time as already passed. That is the
convention of the SQP (after a post-event node the first interval starts an epsilon after the event), but the opposite of
`ocs2::ModeSchedule::modeAtTime`, so the two must not be mixed; the helpers `modeIndexAtTime` / `contactFlagsAtTime` are
used throughout the reference manager for this reason.

#### 2.8.1 Phase resetting on measured contact events (`enablePhaseResetting`)

The measured contact flags reach the MPC as the observation mode (`RobotState::getContactFlags()` in the MRT controller,
MuJoCo contact sensors in simulation) and are compared with the schedule at the start of every solve. Per foot the
manager keeps a small latch of the swing currently in flight (its lift-off, its nominal touch-down, and any re-timing
applied so far).

In the MuJoCo simulation those flags are `true` for every contact point unless `simReportsGroundTruthContacts: true` is
set in the task file: the simulator then reports a contact point as touching when it carries more than
`simContactForceThreshold` newtons of normal force against anything outside the robot. With the default, every swing
reads as an early touch-down at its scuffing window, so phase resetting must not be enabled in simulation without it
(the simulator logs a warning). The viewer's contact timeline (`b`, `contact_timeline` in `simVisualizations`) shows
the contact state the executed policy plans against that ground truth, which is the quickest way to see early or late
touch-downs and foot scuffing.

The viewer also draws the **target contact patches** (`g` toggles them; like every viewer marker they are enabled by
listing them in `simVisualizations`, on in the shipped task files): the `contacts.contact_rectangle` of every foot placed at the pose the contact planner wants the
foot on the ground, so the planned position and yaw of every step can be checked against the robot. A bright filled
patch is the landing pose of the swing in flight (the plan's foothold at its touch-down plus any DCM step adjustment),
a translucent one is the landing pose of the foot's next swing, and a faint outline marks a foot with no upcoming swing
at its current placement. The arrow is the patch's x axis, i.e. its yaw: the planned landing yaw with the heading model
(`useAcomDynamics`) and the measured foot yaw without it. The poses are the ones the reference manager computed at its
last solve (`ContactPlanningReferenceManager::getTargetContactPoses`), so they move whenever a new plan or a contact
event re-times the schedule.

**Early touch-down.** A foot that is scheduled to swing but is measured in contact after the first
`earlyTouchdownMinSwingRatio` of the *nominal* swing duration (scuffing right after lift-off is ignored), and whose
contact persists for `earlyTouchdownMinContactDuration` (a debounce against a single chattering sensor sample), is
switched to contact at the current time: the phase containing the current time is split there and the foot is in
contact from then until its old touch-down, which disappears. The detection is level-triggered, so a contact that
started inside the ignored window and persists past it is a landing too; the debounce timer only runs past the window.
Nothing else moves: the other feet and every later event keep their timing, because those were planned consistently
with the plan that is about to be merged (shifting the tail would make the applied schedule disagree with the plan at
the merge point and produce phantom micro-swings). The touch-down now lies before the commit window, so the boundary
shrinks back to `commitTime` and the planner is free to re-time the following phases; the reference manager additionally
raises a re-plan request that makes `ContactPlannerModule` skip its rate limiter once. The commit window itself is not
shortened below `commitTime`, because it covers the planner latency and a shorter window would let the fresh plan re-time
a phase the controller has already started. The lift-off position latch records the landed position on the next cycle,
so the foot reference does not jitter.

**Late touch-down.** A foot whose swing has reached its planned touch-down without measured contact would be switched to
stance in the air. Instead its touch-down is pushed to `now + lateTouchdownExtensionStep` and every later event is delayed
by the same amount, so that the following double support and the other feet's phases keep their durations (no flight
phase can appear). The extension is repeated every cycle while contact is missing, in total at most
`maxLateTouchdownExtension` past the planned touch-down (measured from the planned touch-down, not from the last
extension); when the budget is used up the contact phase proceeds as scheduled. During the extension the xy foot
reference holds the landing target (its interpolation is timed on the swing without the extension), and the height
reference is the planned swing's up to the planned touch-down, continued from there as a straight descent at
`lateTouchdownSearchVelocity` from the planned touch-down height (`SwingTrajectoryPlanner::GroundSearch`); the pitch and
impact-proximity references hold their touch-down values. Re-fitting the height spline over the extended swing, as an
earlier version did, moved its apex, raised the reference at the current time by several millimetres at every extension
step and then drove the foot down at the spline's slope of about 0.35 m/s instead of at the search velocity. Contact measured at any time during the extension ends the swing at once
through the early touch-down path.

Because a late extension moves later events, the active plan is shifted by the same amount (`ContactPlan::shiftInTime`,
start time and commit boundary), which keeps the merge consistent, and the shift is logged so that a plan that was still
being computed from the pre-shift schedule is shifted too when it arrives (a plan is shifted by every logged shift that
is younger than its start time; the planner snapshot is taken after the reference manager ran in the same cycle).

#### 2.8.2 DCM step adjustment (`enableDcmStepAdjustment`)

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

with $K_{\mathrm{dcm}} = 1$ the exact LIP compensation (`dcmAdjustmentGain`). The offset is limited to
`dcmAdjustmentMaxOffset` in norm and the adjusted foothold is clipped to the planner's reachable region around the
planned CoM at touch-down, in the plan's yaw frame (`reachX`, `reachYInner`, `reachYOuter`), where the side of the foot
(left or right of the CoM) is taken from the nominal foothold so that a foot is never moved across the body. The
adjustment is recomputed at every solve (it is a function of the current state, so it cannot run faster than the MPC)
and blended into the swing reference with the same smooth-step profile as the step itself, so it is invisible at lift-off
and fully applied at touch-down. The prediction is refreshed after every solve, so the error is the deviation that
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
$e^{\omega (t_{\mathrm{TD}} - t)}$ again (up to about 4x over a swing), so a ZMP mismatch of a few centimetres drove the
offset to its bound on every step and, with a forward velocity command, pulled the swing foot backwards. Against the
controller's own prediction that term is absent. The feature remains opt-in like the others: enable it in simulation
first and check that the offset is not sitting at its bound.

#### 2.8.3 Energy-based cadence modulation (`enableEnergyCadenceModulation`, off by default)

The orbital energy of the LIP along the heading of the plan, relative to the planned ZMP,

$$E = \tfrac{1}{2} m \big(\dot{x}^2 - \omega^2 x^2\big), \qquad x = \mathbf{e}_x^\top(\mathbf{c} - \mathbf{z}),$$

is conserved between contact switches. A CoM that carries more energy than the plan passes over the support earlier and
should step earlier; less energy should delay the step. The touch-down of the swing in flight is moved by
$\Delta t_{\mathrm{TD}} = -K_E\,(E - E^{\mathrm{ref}})$ relative to its *nominal* touch-down (not cumulatively), clipped to
`[minSwingDuration, maxSwingDuration]` after lift-off. A request earlier than what is still feasible brings the
touch-down forward to 20 ms from now at the earliest, and never pushes a touch-down that is already imminent further out
(flooring at "now + margin" on every cycle would drag the touch-down along with the clock and the foot would never
land); every later event moves with it and the plan is shifted alongside, exactly as for a late touch-down. Shifts
logged while a plan was being computed are applied to that plan when it arrives, summed against the plan's original
snapshot time so that shifts of opposite sign cancel. It is not applied while a foot is
searching for the ground. `energyCadenceGain` is in seconds per joule of the full robot mass, so the same value acts differently
on robots of different mass; the default 0.01 s/J moves the touch-down by about 0.1 s for a 0.15 m/s forward velocity
deviation of a 150 kg robot walking at 0.4 m/s, if that deviation appears within one solver period (like the step
adjustment, the energy is compared against the prediction refreshed at every solve, so it is the per-period increment
that is corrected). `energyCadenceDeadband` (joules, 0 by default) ignores deviations within the band and measures the
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
reach clipping in rotated frames, plan time shifting and the configuration keys.
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

### 2.9 Heading model: ACoM dynamics in the planner (`useAcomDynamics`)

The point-mass LIP has no notion of heading. Its foothold frame is the base yaw at planning time, held fixed over the
horizon, and the planner only receives the commanded linear velocity, so a pure yaw command produces no step at all
and the robot cannot turn. The heading model adds the missing coordinate as a reduced dynamics of its own, expressed
in the angular centre of mass (ACoM): the whole-body heading `theta` (the ACoM yaw when the robot has an ACoM network,
the base yaw otherwise) and its rate `omega = L_z / I_zz`, the angular momentum about the vertical through the centre
of mass over the whole-body yaw inertia (taken from the model at every plan, `yawInertia` overrides it).

Per node, appended to the LIP block so that the contact binaries keep their place:

```
omega_{k+1} = omega_k + dt (tau_L + tau_R) / I_zz
theta_{k+1} = theta_k + dt omega_k + dt^2 (tau_L + tau_R) / (2 I_zz)     exact zero-order hold of the double integrator
psi_{i,k+1} = psi_{i,k} + dpsi_{i,k}                    foot yaw, one per foot
|tau_i| <= T_t c_i + (T_c - T_t) (c_L + c_R - 1) / 2    T_t alone, (T_t + T_c) / 2 per foot in double support
|dpsi_i| <= 2 pi (1 - c_i)                              a foot's yaw is pinned while it is in contact
|psi_i - theta| <= maxFootYawOffset                     hip range (soft)
```

The heading is driven only by the ground: a point mass's horizontal contact force acts through the ZMP and produces no
yaw moment about the centre of mass, so what remains is the torsional friction of the weight a stance foot carries
(`torsionalFrictionTorque`, `T_t`, for the whole weight on one foot) and, in double support, the friction couple of the
two feet (`doubleSupportYawCouple`, `T_c`). With both feet down each carries half the weight and half the couple, so the
pair has `T_t + T_c`; granting the full-weight torsion to both feet at once, as an earlier version did, over-estimated
the double-support budget by `T_t` (78 N m on the DRC Atlas). A turn therefore has to be stepped: the planner rotates
the foothold frame with the heading, the feet must follow it within hip range, and the yaw torques bound how fast the
heading can go. The costs track the commanded yaw rate and the commanded heading, keep the foot yaws at the nominal
heading (the linearisation point, which is the plan's own heading under the successive linearisation; measuring a pinned
stance foot's yaw against the heading *state* was a penalty on the heading alone and pulled it back towards the stance
feet), and regularise the torques and the foot yaw displacements.

The commanded yaw rate is read off the momentum channel of the MPC's target trajectory, where the command is carried
as the angular momentum of a rigid turn about the vertical, `h_z = I_zz omega / m`, with the same locked inertia the
planner uses. The target's base yaw is not used: its first stretch integrates the average of the measured and the
commanded yaw rate, so differentiating it commanded half the rate at the start of a turn and fed the measured rate back
in, and once the override below has rewritten it, it carries the previous plan's rate.

The constraint frame of node k is the planned heading of node k. That makes the step-width, reach and foot-separation
rows bilinear in the heading and the footholds, which the mixed-integer solver cannot take. They are linearised to
first order around a nominal heading trajectory, the previous plan shifted to the current time or, without one, the
commanded yaw integrated from the current heading, so that the relaxations stay convex and the branch-and-bound is
unchanged. After the search the frame is re-linearised at the incumbent's own heading and footholds and the QP is
re-solved with the contacts fixed (`headingLinearizationPasses`, successive linearisation), so that the frame the
constraints were written in is the frame the plan actually turns through.

Downstream, the plan carries the heading and the foot yaws per node. The swing-foot reference turns each foot from
its lift-off yaw to the planned landing yaw with the same profile as the position, and the foot cost tracks that yaw
through `task_space_foot_cost_weights.orientation_z` (0 by default: not tracked). With `planHeadingOverridesTarget`
the plan's heading replaces the commanded yaw in the MPC's target trajectory, so that the whole-body controller is
asked for the turn the ground can support rather than the raw command; with ACoM tracking the reference base yaw is
set so that the reference ACoM heading equals the planned heading.

The model stays a reduced one: the angular momentum exchange with the swing leg and the arms, which is how a humanoid
actually turns between steps, is the whole-body controller's business over its own horizon. The planner decides where
and when the feet go and how fast the heading may change for the ground to carry it.

**Parameters from the model.** The quantities that are properties of the robot and of the ground rather than tuning
are not task-file keys: the yaw inertia (composite inertia about the vertical through the centre of mass, at every
plan), `torsionalFrictionTorque` (the torsional friction coefficient of the wrench cone times the weight),
`doubleSupportYawCouple` (the friction coefficient times half the weight times `nominalStepWidth`) and the foot yaw
bounds (the hip yaw joint limits of every leg, found by walking the kinematic tree up from the joint that carries the
contact frame to the first revolute joint about the vertical; per foot and asymmetric, which matters because a hip yaw
range is typically much wider outward than inward). They are derived once by the interface and applied to every
planner configuration, including the hot reloads from the task file, and logged at start-up. `comHeight` and the ZMP
box stay tunable, with 0 meaning "from the model": the centre of mass height above the feet at `initialState`, and the
sole's footprint from the wrench cone. The friction and footprint come from `contactWrenchConeSoftConstraint`, so the
planner cannot assume more yaw torque than the whole-body constraint would ever allow.
