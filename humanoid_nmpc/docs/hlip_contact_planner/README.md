# The H-LIP Contact Planner

An implementation of the reduced-order half of

> S. A. Esteban, V. Kurtz, A. B. Ghansah, A. D. Ames,
> *Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion*,
> [arXiv:2502.15630](https://arxiv.org/abs/2502.15630).

The paper's claim is architectural: the reduced-order model should propose a **nominal** gait and nothing more,
because the whole-body controller downstream is free to depart from it. Once that is true the reduced-order layer can
be trivial — a Hybrid Linear Inverted Pendulum with a fixed single-support duration and a closed-form deadbeat step,
with no optimization anywhere. That is `HlipContactPlanner`, selected by `planner.type: hlip` in the robot's
`config/mpc/contact_planning.yaml`, which is the default.

The other half of the paper, the whole-body controller that is free to depart, is
[contact_implicit_mpc](../contact_implicit_mpc/README.md).

---

## 1. Where it sits

```
 joypad / target trajectories
    v_cmd, omega_z,cmd
            |
            v
  +---------------------------+          ContactPlannerModule (its own thread, or the pre-solve hook)
  |    HlipContactPlanner     |          ContactPlannerFactory resolves planner.type
  |                           |
  |  1. cadence      fixed alternating single supports of sspDuration
  |  2. footholds    u = u* + K (x - x*), K closed form (HlipModel)
  |  3. stand/walk   alpha(phi) (HlipStandingBlend)
  +---------------------------+
            |  ContactPlan: contacts per interval, footholds, CoM / ZMP / heading per node
            v
  ContactPlanningReferenceManager
      merge with the executed schedule (commit window)
      mode schedule  ---------------------> whole-body NMPC (switched dynamics)
      swing-foot references  -------------> foot tracking cost
            |
            v
     OCS2 SQP whole-body NMPC  ----------->  MRT joint controller  ----->  robot
```

Everything downstream of `ContactPlan` is shared with the mixed-integer planner and is unchanged.

## 2. The model

One planar pendulum of fixed height `z0`, state `x = [p, v]` of the point mass **relative to the stance foot**. A step
is a single support phase of `T_ssp` in which the pendulum is unactuated, an instantaneous impact that hands the
stance over to the foot placed a step length `u` ahead, and a double support phase of `T_dsp` in which the mass is
assumed to drift at constant velocity. With `w = sqrt(g / z0)`:

```
A_ssp = [ cosh(w T_ssp)      sinh(w T_ssp) / w ]     A_dsp = [ 1  T_dsp ]
        [ w sinh(w T_ssp)    cosh(w T_ssp)     ]             [ 0  1     ]
```

Composing impact, drift and flow gives the **step-to-step (S2S) dynamics** on the pre-impact state:

```
x_{k+1} = A x_k + B u_k,     A = A_ssp A_dsp,     B = -A e_1
```

Three-dimensional walking is the orthogonal composition of two of these, one along the heading and one lateral
(`HlipModel`, `contact_planning/hlip/HlipModel.h`).

## 3. The step law has no gain to tune

`A + B K` is nilpotent — both eigenvalues at zero, so any error is gone after two steps — for

```
K = [ 1,  T_dsp + coth(w T_ssp) / w ]
```

*Derivation.* Write `c = cosh(w T_ssp)`, `s = sinh(w T_ssp)`, so `A = [[c, c T_dsp + s/w], [w s, w s T_dsp + c]]` and
`B = -[c, w s]^T`. With the `K` above,

```
row 1 of A + B K = [ c - c,    (c T_dsp + s/w) - c (T_dsp + c/(w s)) ] = [ 0,  (s^2 - c^2) / (w s) ] = [ 0, -1/(w s) ]
row 2 of A + B K = [ w s - w s, (w s T_dsp + c) - w s (T_dsp + c/(w s)) ] = [ 0,  0 ]
```

using `c^2 - s^2 = 1`. The square of that matrix is zero, which `testHlipModel.cpp` asserts numerically. Nothing here
is a weight, a horizon, or a tuning constant: given the cadence and the height, `K` is determined.

The step the law is measured against is the nominal orbit of the commanded velocity, and those are solved as linear
systems rather than approximated:

* **along the heading** — a period-one orbit with `u* = v_x,cmd (T_ssp + T_dsp)`, whose pre-impact fixed point solves
  `x* = A x* + B u*`;
* **laterally** — a period-two orbit alternating `u*_L = +w_step + v_y,cmd T` and `u*_R = -w_step + v_y,cmd T`, whose
  two pre-impact states solve the 4x4 system `x*_1 = A x*_2 + B u*_2`, `x*_2 = A x*_1 + B u*_1`.

The planner then rolls the measured centre-of-mass state forward to the pre-impact instant of each step in the
horizon and places the foot at `stance + R(heading) u` with `u = u* + K (x - x*)` per axis.

**The roll-out starts from the committed window, not from the planning instant.** The first few intervals of a plan are
not the planner's to choose: the reference manager merges the executed schedule over them, and a plan that disagreed
with it there would be dropped as inconsistent. They used to be stamped over the contact sequence at the very end,
*after* the centre of mass and the footholds had been rolled out against a gait built as if the window were free, so
the two described different gaits. The worst case is the one every walk starts from: out of a long stance the nominal
cadence lifted a foot at the first node while the commit window held both feet down for the first two intervals - a
lift-off 0.05 s out of a 0.25 s single support away from where the footholds assumed it. `buildGait()` now emits the
committed intervals first and continues the cadence from the state they leave behind, so the contact sequence, the
footholds and the centre-of-mass roll-out are one gait. A phase boundary is rounded to the nearest node exactly once,
and both the per-node quantities and the per-interval ones are filled from those same integer boundaries.

**A note on the first step.** From a standstill with a forward command the law places the first foot *behind* the
centre of mass. That is not a defect: stepping short is how an inverted pendulum accelerates, and the deadbeat gain
reaches the commanded orbit within two steps. The commanded velocity is ramped upstream, so the robot is not usually
asked to accelerate that hard in one step.

**Clipping costs the deadbeat property.** A step cut to `maxStepLength` or to the step-width bounds is no longer the
step the gain asked for, so the part of the error it was meant to cancel survives into the next step — and the next
step then asks for more, not less. That is a divergence, not a saturation. `ContactPlan::numClippedSteps` counts it
and `describe()` prints `CLIPPED-STEPS=n`; a plan that persistently clips is asking for more than the legs can deliver
at that cadence and command.

## 3b. The cadence is the lateral stability budget

The lateral pendulum amplifies an offset by `cosh(w T_ssp)` every single support: **1.38** at the shipped
`T_ssp = 0.25 s`, **1.79** at `0.35 s`, **2.82** at `0.5 s`, for `z0 = 0.85 m`. The deadbeat law cancels that, but only
if the step it asks for is reachable. Worked example, starting to walk from a standstill with the centre of mass still
between the feet, computed at **`T_dsp = 0.05 s`** and `stepWidth = 0.25 m`.

Naming `T_dsp` is not pedantry. It enters the deadbeat gain directly — `K = [1, T_dsp + coth(w T_ssp) / w]` — so the
last column moves with it (at `T_ssp = 0.35 s` the demand is 0.49 m with `T_dsp = 0`, 0.52 m at the shipped 0.05 s and
0.55 m at 0.1 s), and this table used to quote no `T_dsp` at all, with rows that could not all be reproduced from any
single value. `HlipContactPlanner::startUpLateralStep` computes that column, and the planner prints it at start-up, so
it can always be checked against the configuration that is actually running rather than against the numbers here.

| `T_ssp` | `cosh(w T_ssp)` | pre-impact lateral state | step the law asks for | reach clip |
| --- | --- | --- | --- | --- |
| 0.5 s | 2.82 | `p = 0.353 m`, `v = 1.12 m/s` | **0.79 m** | cut to 0.45 m — diverges |
| 0.35 s | 1.79 | `p = 0.224 m`, `v = 0.63 m/s` | **0.52 m** | cut to 0.45 m — this is the cadence that fell |
| **0.25 s (shipped)** | 1.38 | `p = 0.173 m`, `v = 0.41 m/s` | 0.42 m | inside the 0.45 m reach — fits |
| 0.35 s, after the centre of mass has moved over the stance foot | 1.79 | `p = 0.036 m`, `v = 0.10 m/s` | 0.12 m | within reach |

The last row is why `dspDuration` is not zero here although the paper sets it to zero: the reduced model's double
support only drifts at constant velocity and does not represent the weight transfer, so its only real job is to give
the whole-body MPC the time to move the centre of mass over the next stance foot before that foot has to carry the
robot alone. Starting a single support with the centre of mass still between the feet is what makes the lateral step
demand explode.

## 3c. The plan is a reference for the controller, not only a schedule

`ContactPlan` carries the reduced model's centre-of-mass trajectory as well as the contacts and the footholds, and
the `planned_com_override` execution rule writes it into the MPC's target trajectory: the reference horizontal centre
of mass becomes the planned one, and the reference linear momentum — which in the centroidal state *is* the centre of
mass velocity — becomes the planned centre-of-mass velocity.

This is the paper's equations 11–14, not a correction heuristic, and it is not optional. The H-LIP's period-two orbit
requires the centre of mass to be *falling towards the swing foot* at the pre-impact instant — 0.165 m/s at the
shipped `T_ssp = 0.25 s`, `T_dsp = 0.05 s`. A target trajectory built from the operator's command asks for the
opposite: a straight line with zero lateral velocity. With both in the cost, the whole-body MPC holds the centre of
mass laterally still, the planner reads that state back at the next cycle, concludes no lateral step is needed and
narrows the step towards `minStepWidth` — the support narrows, the next cycle starts further from the orbit, and the
robot sidesteps and falls. The rule is therefore listed in `execution` by default, and it is the only entry there;
everything else in that list is a heuristic and stays off. Section 5 below is about what `planner.type: hlip` gives
up, and this rule is the one thing in the `execution` list it does **not** give up.

### Every capturability reference must come from the plan too

The centre-of-mass reference is not the only one. The terminal DCM cost (`useDcmTerminalCost`) carries its own: the
centre of the terminal support plus `velocityOffsetFactor * v_cmd / omega`, i.e. "end the horizon able to come to rest
over the feet". That is the right reference for a gait whose footholds are decided elsewhere, and the wrong one here.
The H-LIP's lateral orbit puts the DCM *beyond* the stance foot, towards the foot about to land. At the shipped
cadence — `T_ssp = 0.25 s`, `T_dsp = 0.05 s`, `z0 = 0.85 m` so `omega = 3.397`, `stepWidth = 0.25 m`, which makes the
deadbeat gain `K = [1, T_dsp + coth(w T_ssp) / w] = [1, 0.476]` — the pre-impact orbit is `p = 0.121 m`,
`v = 0.165 m/s` relative to the stance foot, so

```
xi_orbit = p + v / omega = 0.121 + 0.165 / 3.397 = 0.169 m,   xi_support = 0
```

a 16.9 cm disagreement, and with `weight_x = weight_y = 400` and `velocityOffsetFactor: 0` the cost wins. The centre of
mass is then held over the stance foot with no lateral velocity, the planner reads that state back, and the deadbeat
law asks for

```
u_y = 0.25 + (0 - 0.121) + 0.476 (0 - 0.165) = 0.051 m
```

which the self-collision floor clips to `minStepWidth`. The feet come together, the support narrows, and the robot
falls sideways — at any commanded speed, including walking in place.

Every figure in the two blocks above is `HlipModel` evaluated at the cadence named at the top of this subsection. They
were previously written for the 0.35 s / 0.1 s cadence that was retired when `sspDuration` was shortened, and labelled
"the shipped cadence" rather than stated, so a reader who recomputed them from the configuration got `K = [1, 0.476]`
where the text said 0.455 and a 16.9 cm disagreement where it said 17.6 cm. The conclusion never depended on the
cadence — 0.051 m is as far below `minStepWidth: 0.15` as 0.040 m was — but a derivation that cannot be reproduced
from the running configuration is not usable, so the cadence is spelled out here and in section 3b instead.

So `SwitchedModelReferenceManager::getPlannedDcm(time, omega)` returns the plan's own DCM,
`com(t) + v(t) / omega`, and `DcmTerminalCost` blends its reference towards it whenever a plan is active. Without a
plan the accessor is empty and the support-centre reference stands exactly as before. The blend is arithmetic rather
than a branch, so one compiled model serves both cases; because that changes the term's parameter count, the generated
library is keyed to it, and the first start-up after this change regenerates that one model.

The general rule this is the third instance of: **under an online contact plan, every reference for a quantity the
plan also decides must come from the plan.** The centre of mass, the DCM and the operator's command each had a second
source, and each one fought the footholds until it was made to agree.

### The command channel

`planned_com_override` writes into the linear part of the target's momentum channel — and that channel is also where
several terms read *the operator's command* from, including the planner module itself. A target is only replaced when
a new one is published, so the rewrite survives into the next solver run: a term that reads the command back off the
live target is reading the planner's own output one cycle later. For the planner that is fatal in the obvious way —
at rest the plan's velocity is zero, so the command reads as zero, `alpha` never crosses its half point and the robot
never starts walking however far the operator pushes the slider.

`ContactPlanningReferenceManager` therefore keeps an untouched copy of the target as published
(`setTargetTrajectories` snapshots it) and answers `commandedVelocity()`, `commandedYawRate()` and the base class's
`getCommandedVelocity(time)` from that copy. Anything that means "what did the operator ask for" must go through
those, not through `getTargetTrajectories()`.

## 4. Standing and walking

Equations (15)–(17) of the paper, in `HlipStandingBlend`:

```
phi   = || [c_d ; v_b] ||^2_P          each component divided by the largest value it is expected to take
alpha = tanh(rho_1 (phi - rho_2)) / 2 + 1/2
```

Below `alpha = 0.5` the planner emits a standing plan and the feet stay exactly where they are — no stepping in place
at rest. Above it the commanded velocity used for `u*` is scaled by `alpha`, so the stride grows continuously out of
standing instead of jumping to full length at the first non-zero command. This one smooth scalar replaces what would
otherwise be a stepping trigger with a dead band and a hysteresis of its own.

Two deviations from the paper's Table II are deliberate:

* `phi` here omits the paper's commanded-height term `z_0,d / z_0,min`. That term is an *absolute* height, not a
  deviation, so on its own it exceeds one for any upright robot and would hold `alpha` at one always.
* Because `phi` therefore sits elsewhere, `rho_1` and `rho_2` are not the paper's `5.0` and `0.5`. The shipped values
  put the half point at about a tenth of the maximum forward command and saturate by a fifth of it.

## 5. What this removes

The counts below are the **registry** — every term the mixed-integer formulation can be assembled from, which is what
`knownTermNames()` in `ContactPlanningFormulation.cpp` lists — and not one robot's enabled lists, which are shorter.
Keep them in step with that function; nothing checks them automatically. They were previously
15 / 9 / 7 / 4, none of which matched the registry: 15 counted costs that no longer exist, 9 counted the headers in
`constraint/` including the abstract base class, 7 folded the three assignment costs into the logic rules, and the
search row silently dropped `cadence_stretch`.

| Removed with `planner.type: hlip` | What it was |
| --- | --- |
| `MixedIntegerOcpQp`, branch and bound over HPIPM relaxations | the contact sequence as a mixed-integer program |
| 13 cost terms (`velocity_tracking`, `step_width`, `zmp_regularization`, `terminal_dcm`, …) | the objective that shaped the footholds |
| 8 constraint terms — 4 soft (`zmp_support_region`, `reachability`, `foot_separation`, `hip_yaw_range`) and 4 hard (`no_flight`, `foot_motion_in_swing_only`, `yaw_torque_budget`, `foot_yaw_pinned_in_contact`) | the feasible region of that program |
| 4 logic rules (`phase_durations`, `no_flight`, `minimum_double_support`, `alternating_feet`) | combinatorial propagation on the contact binaries |
| 3 assignment costs (`contact_switch`, `plan_consistency`, `double_support_penalty`) | costs on the binaries themselves, scored outside the QP |
| 5 search stages (`warm_start_previous_plan`, `diving`, `event_shift_local_search`, `heading_relinearisation`, `cadence_stretch`) | the search around the branch and bound |
| 4 of the 5 execution rules (`phase_resetting`, `energy_cadence_modulation`, `dcm_step_adjustment`, `planned_heading_override`) | heuristic corrections applied between plans — left off in the shipped configuration, **not** disabled by `planner.type` |

**The execution list is not gated by `planner.type`.** That row used to say "5 execution rules", which was wrong twice
over. `ContactPlanningReferenceManager::setConfig` and `rebuildExecutionRules` build the rules from
`config.formulation.execution` whatever the planner is, and `preSolverRun` / `overrideTarget` apply them under either
one, so switching to `hlip` removes nothing from that list: the four heuristics above are off because the shipped
configurations comment them out. And the fifth rule, `planned_com_override`, is not a heuristic and is not off — it is
the paper's equations 11–14 reference plumbing, it is listed in `execution` in both shipped robots, and without it the
robot sidesteps and falls. See section 3c; do not empty the `execution` list when migrating a robot to `hlip`.

What is left as a heuristic, and is documented as such in the configuration: the clip of the deadbeat step to
`maxStepLength` / `[minStepWidth, maxStepWidth]`, which exists so that a foothold outside the leg's reach is clipped
rather than handed to the whole-body MPC — and, as the row above says, the one execution rule that stays on, which is
reference plumbing rather than a heuristic.

A plan costs microseconds instead of the mixed-integer planner's tens of milliseconds, so the node grid can be fine
(`planner.dt: 0.025`); the cadence is continuous but a mode schedule is not, and a phase boundary is rounded to the
nearest node.

## 6. Configuration

`config/mpc/contact_planning.yaml`, block `hlip` (and `planner.type`):

| Key | Meaning |
| --- | --- |
| `planner.type` | `hlip` or `lip_miqp`; the names `ContactPlannerFactory` knows |
| `hlip.sspDuration` | [s] single support; the whole cadence, and the lateral stability budget (section 3b) |
| `hlip.dspDuration` | [s] double support; the time the whole-body MPC has to transfer weight before single support |
| `hlip.stepWidth` | [m] lateral distance between the feet of the nominal period-two orbit |
| `hlip.maxStepLength`, `hlip.maxStepWidth`, `hlip.minStepWidth` | [m] the reachability clip, the only bound applied to the deadbeat step |
| `hlip.blend.sharpness`, `hlip.blend.threshold` | `rho_1`, `rho_2` of `alpha(phi)` |
| `hlip.blend.maxCommanded*` | the command ranges that normalise `phi` |
| `hlip.blend.maxComVelocity*` | the measured centre-of-mass velocity ranges that normalise `phi`. The paper writes `v_b` for the base velocity, but what the planner measures and feeds the blend is the centre-of-mass velocity, so the keys are named for that |

`planner.runInBackgroundThread` is **false** for this planner, and that is not an oversight. A plan solves nothing - a
fixed cadence and a closed-form step over 56 nodes - so it costs microseconds, and the worker thread buys nothing while
costing the one thing a feedback law cannot afford: the plan is posted at `planner.planningFrequency` and handed over a
cycle later, so at the 10 Hz that used to be configured, against a 50 Hz MPC, the footholds were computed from a state
up to 100 ms old. That is two fifths of the 0.25 s single support they exist to correct, in a law whose whole claim is
that it is deadbeat on the measured state. `lip_miqp` is the opposite case - a mixed-integer solve takes tens of
milliseconds and must not block the solver - so it needs `true`, and `ContactPlanningConfig::validate()` warns when the
two are mismatched.

Every numeric key of the `hlip` block is a slider in the remote control's MPC Parameters tab, under **Contact
Planning**, and is hot-reloadable: the parameter updater re-parses the file and hands the configuration to the planner,
which rebuilds its model from it. The tab renders every numeric leaf of the block generically, grouped by the
sub-block its key path sits in (`hlip`, `hlip.blend`, `planner`, …), and nothing in it is written per parameter.

That genericity has a consequence worth knowing before you tune: **the tab has no idea which planner is running.**
`mpc_params_tab.py` never reads `planner.type`, so the mixed-integer planner's per-term blocks are rendered as ordinary
live sliders while `hlip` is selected even though the planner never reads them, and moving one changes nothing. The
term lists themselves (`dynamics`, `costs`, `soft_constraints`, `search`, `execution`) do not appear at all, because
`yaml_param_tree.tunables` yields numeric scalars only and skips strings and lists — they are edited in the file. This
document previously claimed the header named the running planner and marked the blocks it does not read; it never has,
and re-introducing that would mean teaching the GUI about particular parameters, which is exactly what the generic
renderer exists to avoid. If the marking is wanted, it belongs in the file's own structure or in a generic rule, not in
a hand-written frame.

`shared.comHeight` and `shared.gravity` are the pendulum's; the per-term blocks and the `costs`, `soft_constraints`,
`hard_constraints`, `logic_rules`, `assignment_costs` and `search` lists belong to `lip_miqp` and are ignored. Two
lists are not ignored: whether the planner publishes a planned heading and planned foot yaws still follows
`dynamics: [... heading_double_integrator]`, because that is what the rest of the pipeline keys off, and `execution` is
read in full under either planner (section 5).

## 7. Tests

| Test | What it holds |
| --- | --- |
| `humanoid_common_mpc:testHlipModel` | `A + BK` nilpotent; the period-one fixed point; the period-two orbit; recovery in two steps; the flow and the S2S map agree |
| `humanoid_common_mpc:testHlipContactPlanner` | alternation, cadence, the committed window, steady-state step length `v T`, standing at rest, the reachability clip, plan well-formedness |
| `humanoid_common_mpc:testContactPlannerFactory` | both planner names resolve; an unknown one is rejected with the valid names |
| `humanoid_common_mpc:testHlipPlanConsistency` | the gait is built FROM the committed window rather than having it stamped over the contacts afterwards; the published contacts agree with the gait the roll-out used; a swing already in flight keeps the time it has already spent there; a foot's landing target moves on the same node its contact flag says it left the ground; a stepping plan publishes the unblended roll-out and starts at the measured state; a standing plan still asks for a centre of mass at rest over the feet |
| `humanoid_centroidal_mpc:testHlipPlanningIntegration` | the shipped configuration end to end: stands at rest, walks and alternates when commanded, advances at the commanded order of velocity, and the commanded yaw rate reaches the planner whether or not the heading model is on |
