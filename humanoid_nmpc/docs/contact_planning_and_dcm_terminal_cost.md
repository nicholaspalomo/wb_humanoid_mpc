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
* $\mathbf{p}_{\mathrm{support}}$ is the centre of the contact frames of the feet that are in contact at $T$ according to the
  mode schedule (both feet in double support, the stance foot in single support, both as a fallback in flight);
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
                          ├─ mode schedule  = committed window of the applied schedule + plan
                          ├─ swing trajectory planner (foot height)   ──► constraints / pre-computation
                          └─ swing-foot references (planned landing)  ──► task_space_foot_cost (pos_x, pos_y)
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
  start of the phase, including the time already spent in the current phase before the planning instant;
* foot alternation (`enforceAlternatingFeet`): a foot may not swing twice without the other foot swinging in between;
* the committed window: contacts within `commitTime` of the planning instant are fixed to the schedule the NMPC is
  already executing, so that phases in flight are not rewritten under the controller.

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
  `planningFrequency`); otherwise the plan is computed inside the pre-solve hook.
* `ContactPlanningReferenceManager` merges every new plan into the schedule the NMPC is executing (applied schedule up to
  the commit time, plan afterwards, always starting and ending in double support so that the swing trajectory planner
  finds a lift-off and a touch-down for every swing), updates the foot-height swing planner and exposes the planned
  landing spot of every swing foot as a task-space reference: the xy position interpolates from the lift-off position to
  the landing spot with a smooth-step profile, the height follows the swing trajectory planner.
* `CentroidalMpcEndEffectorFootCost` tracks that reference through `task_space_foot_cost_weights.pos_x / pos_y`. Without a
  plan the xy position error is switched off inside the cost, so the weights are harmless when planning is disabled.
* Until the first valid plan arrives (or if the planner stalls) the gait schedule / the last applied schedule is used.
* The whole `contact_planning` section is hot-reloadable through the parameter updater (GUI tab "Contact Planning");
  structural keys (`numNodes`, threading) are applied by the planner on its next run.

### 2.7 Tuning notes

* `dt`/`numNodes`: the horizon must cover `mpc.timeHorizon`; coarser nodes make the search cheaper but quantise the
  switching times.
* `minSwingDuration`/`maxSwingDuration`/`minContactDuration` are the main shape parameters of the gait. `maxContactDuration`
  forces stepping even without a command (leave at 0 to allow standing).
* `zmpHalfWidthX/Y` should stay inside the physical foot (the NMPC enforces the real wrench cone); a smaller box makes the
  planner step earlier under disturbances.
* `contactSwitchCost` trades stepping against ankle strategy; `terminalDcmWeight` makes plans end capturable.
* Raise `constraintSlackWeight` if plans exploit the soft support region.

Implementation: `humanoid_common_mpc/contact_planning/` (`OcpQpHpipm`, `MixedIntegerOcpQp`, `LipContactPlanner`,
`ContactPlan`, `ContactPlanningReferenceManager`, `ContactPlannerModule`); tests in `humanoid_common_mpc/test/` and
`humanoid_centroidal_mpc/test/testContactPlanningIntegration.cpp`.
