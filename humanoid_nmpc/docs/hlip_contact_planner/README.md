# The H-LIP Contact Planner

An implementation of the reduced-order half of

> S. A. Esteban, V. Kurtz, A. B. Ghansah, A. D. Ames,
> *Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion*,
> [arXiv:2502.15630](https://arxiv.org/abs/2502.15630).

The paper's claim is architectural: the reduced-order model should propose a **nominal** gait and nothing more,
because the whole-body controller downstream is free to depart from it. Once that is true the reduced-order layer can
be trivial — a Hybrid Linear Inverted Pendulum with a fixed single-support duration and a closed-form deadbeat step,
with no optimization anywhere. That is `HlipContactPlanner`, selected by `planner.type: "hlip"` in the robot's
`config/mpc/contact_planning.textproto`, which both shipped robots select (a file that names no planner gets the library
default, `lip_miqp`).

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
  |  1. cadence      fixed alternating single supports of ssp_duration
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

The planner then rolls the measured center-of-mass state forward to the pre-impact instant of each step in the
horizon and places the foot at `stance + R(heading) u` with `u = u* + K (x - x*)` per axis. For the swing already in
flight that instant is the touch-down the robot executes, not a node near it (below).

**The roll-out starts from the committed window, not from the planning instant.** The first few intervals of a plan are
not the planner's to choose: the reference manager merges the executed schedule over them, and a plan that disagreed
with it there would be dropped as inconsistent. They used to be stamped over the contact sequence at the very end,
*after* the center of mass and the footholds had been rolled out against a gait built as if the window were free, so
the two described different gaits. The worst case is the one every walk starts from: out of a long stance the nominal
cadence lifted a foot at the first node while the commit window held both feet down for the first two intervals - a
lift-off 0.05 s out of a 0.25 s single support away from where the footholds assumed it. `buildGait()` now emits the
committed intervals first and continues the cadence from the state they leave behind, so the contact sequence, the
footholds and the center-of-mass roll-out are one gait.

**The committed window is replayed in continuous time.** The reference manager extends the commit boundary to the
touch-down of the swing in flight and samples the last committed node *at* that boundary, where the foot is already
down. Replaying the committed contacts as whole nodes therefore ended the swing at the start of the node its touch-down
falls in, and the deadbeat step was evaluated up to a node (25 ms) before the impact the robot executes - always early.
At the shipped cadence that asked a robot exactly on the 0.5 m/s orbit for a 0.230 m lateral and a 0.129 m sagittal
step instead of 0.25 m and 0.15 m, and the double support after the touch-down was counted from the node start, so the
next lift-off came up to a node early as well. `buildGait()` now places every switch of the committed contacts at the
executed schedule's own event time (`ContactPlannerInput::committedPhaseStartTimes`), so the swing in flight ends at its
executed touch-down and the double support is served out from there. Only the node each boundary is assigned to comes
from the grid (`GaitPhase::endNode`): inside the window the node where the committed contacts change, after it the
nearest node, assigned once. The contacts, the footholds and the ZMP follow those integer boundaries, so a foothold
changes on exactly the node its contact does; the deadbeat step and the center of mass follow the continuous times.
The mode schedule is built from the continuous times as well (`ContactPlan::phaseContacts` / `phaseStartTimes`, which
`ContactPlan::toModeSchedule()` reads), so every executed single support lasts `hlip.ssp_duration` and every executed
double support `hlip.dsp_duration` exactly - the durations the deadbeat gain `K = [1, T_dsp + coth(omega T_ssp) / omega]`
assumes. It used to be quantized to the grid of whichever plan last re-decided the lift-off (`dsp_duration +- dt / 2`).

**A note on the first step.** From a standstill with a forward command the law places the first foot *behind* the
center of mass. That is not a defect: stepping short is how an inverted pendulum accelerates, and the deadbeat gain
reaches the commanded orbit within two steps. The commanded velocity is ramped upstream, so the robot is not usually
asked to accelerate that hard in one step.

**Clipping costs the deadbeat property.** A step cut to `max_step_length` or to the step-width bounds is no longer the
step the gain asked for, so the part of the error it was meant to cancel survives into the next step, amplified by the
pendulum on the way — and the next step then asks for more, not less. Depending on how much was cut the gait either
recovers within a few steps or locks into a wide / narrow cycle it never leaves (section 3b works both cases through).
`ContactPlan::numClippedSteps` counts it and `describe()` prints `CLIPPED-STEPS=n`; a plan that persistently clips is
asking for more than the legs can deliver at that cadence and command.

## 3b. The cadence is the lateral stability budget

The lateral pendulum amplifies an offset by `cosh(w T_ssp)` every single support: **1.30** at the shipped
`T_ssp = 0.25 s`, **1.61** at `0.35 s`, **2.37** at `0.5 s`, on the Atlas's own pendulum, `z0 = 1.0805 m` (the model's
center of mass above its soles at `initial_state`, which a `shared` block without `com_height` resolves to; `w = 3.013 rad/s`). The
deadbeat law cancels that, but only if the step it asks for is reachable. Worked example, starting to walk from a
standstill with the center of mass still between the feet, computed at **`T_dsp = 0.05 s`** and `step_width = 0.25 m`.

Naming `T_dsp` and `z0` is not pedantry. Both enter the deadbeat gain directly — `K = [1, T_dsp + coth(w T_ssp) / w]`
— so the last column moves with them (at `T_ssp = 0.35 s` the demand is 0.450 m with `T_dsp = 0`, 0.472 m at the
shipped 0.05 s and 0.495 m at 0.1 s), and this table used to quote no `T_dsp` at all, with rows that could not all be
reproduced from any single value. It was also computed on the 0.85 m pendulum both Atlas files hand-set until the height
was derived from the model: 23 cm short, which made every demand larger (0.52 m instead of 0.47 m at 0.35 s) and the
gain 20 % smaller (`K[1]` 0.476 instead of 0.571 at the shipped cadence). `HlipContactPlanner::startUpLateralStep`
computes that column, and the planner prints it at start-up and again after every reload that changes it
(`ContactPlannerModule::reloadSummary`), so it can always be checked against the configuration that is actually
running rather than against the numbers here. `ContactPlanningConfig::warnings()` also reports a first step that does
not fit `hlip.max_step_width` whenever the configuration is validated, at start-up and on every reload.

Whether a step that does not fit is fatal depends on how much is cut, so the print does not stop at "does not fit": it
rolls the lateral reduced model forward from the standstill under the planner's own step law and clip
(`HlipContactPlanner::startUpLateralWidths`) and says whether the gait recovers or locks. At `T_dsp = 0.05 s` and a
0.45 m reach, `T_ssp = 0.5 s` locks into an alternating 0.45 / 0.15 m cycle, while `T_ssp = 0.35 s` is clipped from
0.47 m to 0.45 m and recovers (0.45, 0.253, 0.228, then 0.25 m); the shipped 0.25 s cadence fits a 0.40 m reach and
recovers against a 0.35 m one (0.35, 0.205, 0.207, then 0.25 m).

| `T_ssp` | `cosh(w T_ssp)` | pre-impact lateral state | step the law asks for | reach clip |
| --- | --- | --- | --- | --- |
| 0.5 s | 2.37 | `p = 0.296 m`, `v = 0.81 m/s` | **0.67 m** | cut to 0.45 m — locks into a 0.45 / 0.15 m cycle |
| 0.35 s | 1.61 | `p = 0.201 m`, `v = 0.48 m/s` | **0.47 m** | cut to 0.45 m — recovers. On the 0.85 m pendulum it asked 0.52 m and locked: this is the cadence that fell |
| **0.25 s (shipped)** | 1.30 | `p = 0.162 m`, `v = 0.31 m/s` | 0.39 m | inside the 0.45 m reach — fits |
| 0.35 s, after the center of mass has moved over the stance foot | 1.61 | `p = 0.032 m`, `v = 0.08 m/s` | 0.11 m | within reach |

The last row is why `dsp_duration` is not zero here although the paper sets it to zero: the reduced model's double
support only drifts at constant velocity and does not represent the weight transfer, so its only real job is to give
the whole-body MPC the time to move the center of mass over the next stance foot before that foot has to carry the
robot alone. Starting a single support with the center of mass still between the feet is what makes the lateral step
demand explode.

## 3c. The plan is a reference for the controller, not only a schedule

`ContactPlan` carries the reduced model's center-of-mass trajectory as well as the contacts and the footholds, and
the `planned_com_override` execution rule writes it into the MPC's target trajectory: the reference horizontal center
of mass becomes the planned one, and the reference linear momentum — which in the centroidal state *is* the center of
mass velocity — becomes the planned center-of-mass velocity.

This is the paper's equations 11–14, not a correction heuristic, and it is not optional. The H-LIP's period-two orbit
requires the center of mass to be *falling towards the swing foot* at the pre-impact instant — 0.132 m/s at the shipped
`T_ssp = 0.25 s`, `T_dsp = 0.05 s` on the Atlas's 1.0805 m pendulum. A target trajectory built from the operator's
command asks for the opposite: a straight line with zero lateral velocity. With both in the cost, the whole-body MPC
holds the center of mass laterally still, the planner reads that state back at the next cycle, concludes no lateral step
is needed and narrows the step towards `min_step_width` — the support narrows, the next cycle starts further from the
orbit, and the robot sidesteps and falls. The rule is therefore listed in `execution` in both shipped files, and it is
the only entry there (the library default formulation is lip_miqp's and lists no rules;
`ContactPlanningConfig::warnings()` reports an hlip configuration without it); everything else in that list is a
heuristic and stays off. Section 5 below is about what `planner.type: "hlip"` gives up, and this rule is the one thing in
the `execution` list it does **not** give up.

### Every capturability reference must come from the plan too

The center-of-mass reference is not the only one. The terminal DCM cost (`dcm_terminal_cost` in `costs`) carries its own: the
center of the terminal support plus `velocity_offset_factor * v_cmd / omega`, i.e. "end the horizon able to come to rest
over the feet". That is the right reference for a gait whose footholds are decided elsewhere, and the wrong one here.
The H-LIP's lateral orbit puts the DCM *beyond* the stance foot, towards the foot about to land. At the shipped
cadence — `T_ssp = 0.25 s`, `T_dsp = 0.05 s`, `z0 = 1.0805 m` (the model's) so `omega = 3.013`, `step_width = 0.25 m`,
which makes the deadbeat gain `K = [1, T_dsp + coth(w T_ssp) / w] = [1, 0.571]` — the pre-impact orbit is
`p = 0.122 m`, `v = 0.132 m/s` relative to the stance foot, so

```
xi_orbit = p + v / omega = 0.122 + 0.132 / 3.013 = 0.165 m,   xi_support = 0
```

a 16.5 cm disagreement, and with `weight_x = weight_y = 400` and `velocity_offset_factor: 0` the cost wins. The center of
mass is then held over the stance foot with no lateral velocity, the planner reads that state back, and the deadbeat
law asks for

```
u_y = 0.25 + (0 - 0.122) + 0.571 (0 - 0.132) = 0.053 m
```

which the self-collision floor clips to `min_step_width`. The feet come together, the support narrows, and the robot
falls sideways — at any commanded speed, including walking in place.

Every figure in the two blocks above is `HlipModel` evaluated at the cadence and pendulum named at the top of this
subsection. They were previously written for the 0.35 s / 0.1 s cadence that was retired when `ssp_duration` was
shortened, and then for the hand-set 0.85 m pendulum (`K = [1, 0.476]`, a 16.9 cm disagreement, `u_y = 0.051 m`)
until the height was derived from the model. The conclusion never depended on either - 0.053 m is as far below
`min_step_width: 0.15` as 0.051 m was - but a derivation that cannot be reproduced from the running configuration is not
usable, so the cadence and the pendulum are spelled out here and in section 3b instead.

So `SwitchedModelReferenceManager::getPlannedDcm(time)` returns the plan's own DCM, `com(t) + v(t) / omega`, with the
omega of the pendulum the plan was made on (`ContactPlan::omega`), and `DcmTerminalCost` blends its reference towards it
whenever a plan is active - and takes the robot's own DCM with that same omega, so that the residual compares two points
of one pendulum rather than mixing the plan's center of mass with the cost's own `com_height`. As shipped the two
pendulums are the same one anyway: `dcm_terminal_cost.com_height` and `shared.com_height` are both left out, which each
resolves with `computeComHeightAboveFeet` at `initial_state`. Without a
plan the accessor is empty and the support-center reference stands exactly as before. The blend is arithmetic rather
than a branch, so one compiled model serves both cases; because that changes the term's parameter count, the generated
library is keyed to it, and the first start-up after this change regenerates that one model.

The general rule this is the third instance of: **under an online contact plan, every reference for a quantity the
plan also decides must come from the plan.** The center of mass, the DCM and the operator's command each had a second
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

A standing plan still lands a swing that is committed in flight (its landing spot is the deadbeat step like any other),
and the static reference it blends towards is the center of the support the robot will stand on: the feet after that
landing, not the measured position of a foot still in the air.

**The lateral velocity is measured against the sway of stepping in place.** A biped cannot walk without swaying: at a
zero command the deadbeat law settles on the period-two orbit at `+-step_width`, which crosses every touch-down at 0.132
m/s at the shipped Atlas cadence on its 1.0805 m pendulum (0.151 m/s at the SA01's) - two to four times the half point
of the lateral center-of-mass term. Measured raw, that sway alone held `alpha` at one through every double support, the
next lift-off was planned and committed at every touch-down, and a robot that had walked never stood again at a zero
command. So the velocity the blend sees (`HlipContactPlanner::blendVelocity`) takes the lateral component only by how
far it lies outside the range between rest and the zero-command orbit's velocity at the phase the robot is in: in single
support the orbit flowed for the time the swing has been in flight, in double support the orbit's drift after the
touch-down of the last swung foot. The last swung foot is remembered past the schedule's one-horizon history
(`LiftOffHistory`), so the orbit reference survives a long stand. Stepping in place, and slowing down from it in a double support, read as no motion; a
push beyond the orbit's own sway, or against it, reads in full; a robot that has not stepped yet has no orbit and its
velocity counts as measured. Along the heading the zero-command orbit is rest, so the sagittal velocity is taken as
measured.

Two deviations from the paper's Table II are deliberate:

* `phi` here omits the paper's commanded-height term `z_0,d / z_0,min`. That term is an *absolute* height, not a
  deviation, so on its own it exceeds one for any upright robot and would hold `alpha` at one always.
* Because `phi` therefore sits elsewhere, `rho_1` and `rho_2` are not the paper's `5.0` and `0.5`. With the shipped
  `rho_1 = 40`, `rho_2 = 0.02` the half point sits at `sqrt(rho_2)`, 14 % of a component's range (0.099 m/s of Atlas's
  0.7 m/s forward command); `alpha` is about 0.83 at a fifth of the range and passes 0.99 only near 28 %. At rest it is
  0.17, not zero, so a standing plan's reference keeps that fraction of the rolled-out drift.

## 5. What this removes

The counts below are the **registry** — every term the mixed-integer formulation can be assembled from, which is what
`knownTermNames()` in `ContactPlanningFormulation.cpp` lists — and not one robot's enabled lists, which are shorter.
Keep them in step with that function; a LINT pair ties the table below to it. They were previously
15 / 9 / 7 / 4, none of which matched the registry: 15 counted costs that no longer exist, 9 counted the headers in
`constraint/` including the abstract base class, 7 folded the three assignment costs into the logic rules, and the
search row silently dropped `cadence_stretch`.

<!-- LINT.IfChange(hlip_term_registry_counts) -->
| Removed with `planner.type: "hlip"` | What it was |
| --- | --- |
| `MixedIntegerOcpQp`, branch and bound over HPIPM relaxations | the contact sequence as a mixed-integer program |
| 13 cost terms (`velocity_tracking`, `step_width`, `zmp_regularization`, `terminal_dcm`, …) | the objective that shaped the footholds |
| 8 constraint terms — 4 soft (`zmp_support_region`, `reachability`, `foot_separation`, `hip_yaw_range`) and 4 hard (`no_flight`, `foot_motion_in_swing_only`, `yaw_torque_budget`, `foot_yaw_pinned_in_contact`) | the feasible region of that program |
| 4 logic rules (`phase_durations`, `no_flight`, `minimum_double_support`, `alternating_feet`) | combinatorial propagation on the contact binaries |
| 3 assignment costs (`contact_switch`, `plan_consistency`, `double_support_penalty`) | costs on the binaries themselves, scored outside the QP |
| 5 search stages (`warm_start_previous_plan`, `diving`, `event_shift_local_search`, `heading_relinearization`, `cadence_stretch`) | the search around the branch and bound |
| 4 of the 5 execution rules (`phase_resetting`, `energy_cadence_modulation`, `dcm_step_adjustment`, `planned_heading_override`) | heuristic corrections applied between plans — left off in the shipped configuration, **not** disabled by `planner.type` |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names) -->

**The execution list is not gated by `planner.type`.** That row used to say "5 execution rules", which was wrong twice
over. `ContactPlanningReferenceManager::setConfigStatus` and its `Create()` (both through `buildExecutionRules`) build
the rules from `config.formulation.execution` whatever the planner is, and `preSolverRun` / `overrideTarget` apply them
under either one, so switching to `hlip` removes nothing from that list: the four heuristics above are off because the
shipped configurations comment them out. And the fifth rule, `planned_com_override`, is not a heuristic and is not off —
it is the paper's equations 11–14 reference plumbing, it is listed in `execution` in both shipped robots, and without it
the robot sidesteps and falls. See section 3c; do not empty the `execution` list when migrating a robot to `hlip`.

What is left as a heuristic, and is documented as such in the configuration: the clip of the deadbeat step to
`max_step_length` / `[min_step_width, max_step_width]`, which exists so that a foothold outside the leg's reach is clipped
rather than handed to the whole-body MPC — and, as the row above says, the one execution rule that stays on, which is
reference plumbing rather than a heuristic.

A plan costs microseconds instead of the mixed-integer planner's tens of milliseconds, so the node grid can be fine
(`planner.dt: 0.025`); the mode schedule takes the H-LIP's continuous event times, and only the per-node contacts,
footholds and ZMP are rounded to the nearest node.

## 6. Configuration

`config/mpc/contact_planning.textproto`, block `hlip` (and `planner.type`):

| Field | Meaning |
| --- | --- |
| `planner.type` | `hlip` or `lip_miqp`; the names `ContactPlannerFactory` knows |
| `hlip.ssp_duration` | [s] single support; the whole cadence, and the lateral stability budget (section 3b) |
| `hlip.dsp_duration` | [s] double support; the time the whole-body MPC has to transfer weight before single support |
| `hlip.step_width` | [m] lateral distance between the feet of the nominal period-two orbit |
| `hlip.max_step_length`, `hlip.max_step_width`, `hlip.min_step_width` | [m] the reachability clip, the only bound applied to the deadbeat step |
| `hlip.blend.sharpness`, `hlip.blend.threshold` | `rho_1`, `rho_2` of `alpha(phi)` |
| `hlip.blend.max_commanded_*` | the command ranges that normalize `phi` |
| `hlip.blend.max_com_velocity_*` | the measured center-of-mass velocity ranges that normalize `phi`. The paper writes `v_b` for the base velocity, but what the planner measures and feeds the blend is the center-of-mass velocity, so the fields are named for that. Its lateral component is taken relative to the sway of stepping in place (section 4, `HlipContactPlanner::blendVelocity`) |

`planner.threading` is **`"pre_solve_hook"`** for this planner, and that is not an oversight. A plan solves nothing - a
fixed cadence and a closed-form step over 56 nodes - so it costs microseconds, and the worker thread buys nothing while
costing the one thing a feedback law cannot afford: the plan is posted at most at `planner.planning_frequency`, so at
the 10 Hz that used to be configured, against a 50 Hz MPC, the footholds were computed from a state up to 100 ms old.
That is two fifths of the 0.25 s single support they exist to correct, in a law whose whole claim is that it is
deadbeat on the measured state. Planning synchronously makes a plan from every cycle's state. It does not make the plan
act in that same cycle: the solver runs the reference manager's pre-solve hook, which activates pending plans, before
the modules' hooks, so the plan made in `ContactPlannerModule::preSolverRun` is activated at the next cycle - one MPC
cycle (20 ms at 50 Hz) after its state was measured, on either path. `lip_miqp` is the opposite case - a mixed-integer
solve takes tens of milliseconds and must not block the solver - so it needs `true`. `ContactPlanningConfig::warnings()`
reports either mismatch (`hlip` on a background thread, `lip_miqp` planning synchronously, with the blocking bound
`planner.max_solve_time` + `event_shift_local_search.max_time`), and `validateStatus()` logs it at start-up and on every
reload.

Every numeric field of the `hlip` block is a slider in the remote control's MPC Parameters tab, under the block
selector's `contact_planning` entry, and is hot-reloadable: the parameter updater re-parses the file and hands the
configuration to the planner,
which rebuilds its model from it. A reload that changes anything the planner's start-up summary reports - the cadence,
the step width and its bounds, `rho_1` and `rho_2`, the grid - prints that summary again, start-up check included
(`ContactPlannerModule::reloadSummary`); a reload that changes none of it stays silent. The tab renders every field
of the file generically from its schema, grouped by the block its field path sits in (`hlip`, `hlip.blend`, `planner`,
…), and nothing in it is written per parameter.

That genericity has a consequence worth knowing before you tune: **the tab has no idea which planner is running.**
No block of the contact-planning schema says which planner reads it, so the mixed-integer planner's per-term blocks are
rendered as ordinary live sliders while `hlip` is selected even though the planner never reads them, and moving one
changes nothing. The term lists themselves (`dynamics`, `costs`, `soft_constraints`, `search`, `execution`) are shown
read-only — they are edited in the file. This document previously claimed the header named the running planner and
marked the blocks it does not read; it never has, and teaching the GUI about particular parameters is exactly what the
generic renderer exists to avoid. If the marking is wanted, it belongs in the schema: an `active_when` condition on
`planner.type` in the `tuning` options of the blocks one planner reads, which the tab already renders as
"not applicable".

`shared.com_height` and `shared.gravity` are the pendulum's - a `com_height` left out, as both shipped robots write it,
is the model's center of mass above its soles at `initial_state`, the same pendulum a `dcm_terminal_cost` without
`com_height` resolves to; a positive value is an explicit override - and the per-term blocks and the `costs`, `soft_constraints`,
`hard_constraints`, `logic_rules`, `assignment_costs` and `search` lists belong to `lip_miqp` and are ignored. Two
lists are not ignored: whether the planner publishes a planned heading and planned foot yaws still follows
`dynamics: "heading_double_integrator"`, because that is what the rest of the pipeline keys off, and `execution` is
read in full under either planner (section 5).

## 7. Tests

| Test | What it holds |
| --- | --- |
| `humanoid_common_mpc:testHlipModel` | `A + BK` nilpotent; the period-one fixed point; the period-two orbit; recovery in two steps; the flow and the S2S map agree |
| `humanoid_common_mpc:testHlipStandingBlend` | `phi` normalizes each of its five terms independently and is even in every sign; `alpha` is the documented sigmoid, monotone, on its asymptotes and exactly a half at `rho_2`; every term alone can start the gait; the shipped half point and the numbers section 4 quotes; non-positive ranges are rejected by the configuration |
| `humanoid_common_mpc:testHlipContactPlanner` | alternation, cadence, the committed window, steady-state step length `v T`; a lateral command shifts both steps of the lateral orbit by `v_y T` in its own direction; a yaw rate turns every foothold by the integrated heading (and the published heading and foot yaws with the heading model); the stride is `alpha v T`; a yaw command alone starts the gait; standing at rest; the reachability clip and its count; the start-up widths the summary rolls out are the planner's own, and the banner says "does not fit" and "locks" exactly when they do; and, closed around its own reduced model at the MPC rate: every executed step is the deadbeat step of its executed touch-down, every executed double support is `hlip.dsp_duration` and every swing `hlip.ssp_duration` exactly at any MPC period (with a node-grid control), the last swung foot outlives the executed schedule's history window, and a released walk comes back to standing |
| `humanoid_common_mpc:testContactPlannerFactory` | both planner names resolve; an unknown one is rejected with the valid names |
| `humanoid_common_mpc:testContactPlannerModule` | the background worker's snapshot throttle; a reload re-prints the formulation summary, start-up check included, exactly when something it reports changed or the formulation, grid or planner type did |
| `humanoid_common_mpc:testHlipPlanConsistency` | the gait is built FROM the committed window rather than having it stamped over the contacts afterwards; the swing in flight is planned for the touch-down the robot executes, on and off the node grid; the published contacts agree with the gait the roll-out used and a foot's landing target moves on the same node its contact flag says it left the ground, on off-grid inputs as well; a swing already in flight keeps the time it has already spent there; a stepping plan publishes the unblended roll-out and starts at the measured state; a standing plan asks for a center of mass at rest over the support it will stand on, even mid-swing; stepping in place reads as rest to the blend and a push does not |
| `humanoid_centroidal_mpc:testHlipPlanningIntegration` | the shipped configuration end to end: stands at rest, walks and alternates when commanded, advances at the commanded order of velocity, and a yaw command alone reaches the planner and starts the gait whether or not the heading model is on; the base-position channel and the reference configuration's CoM carry the plan; past the plan's end the reference is the operator's and the planned DCM is absent; the DCM cost and the planner run on the model's pendulum (computed independently in the test) and a plan carries its omega, which the DCM cost takes the planned DCM and the robot's DCM with, while explicit heights are used as given; weightless planned footholds are warned about at start-up and the log names the configured planner; a rejected reload names its field and keeps the running configuration; `swing_time_scale` is warned about on a contact_planning reload and, once per change, after a task.textproto reload |
| `humanoid_centroidal_mpc:testShippedContactPlanningFiles` | both shipped contact_planning.textproto files, on the pendulum the planner runs (a `com_height` left out, resolved from the model): they validate with no warnings, list planned_com_override, fit their first step and plan synchronously under hlip; no planned swing is shorter than task.textproto `swing_trajectory_config.swing_time_scale` under either planner; the planner pendulum equals the DCM terminal cost's, as each resolves it |
| `humanoid_centroidal_mpc:testNominalPendulum` | for every centroidal robot, the contact planner, the locomotion heuristics and the DCM cost derive the same pendulum (`computeComHeightAboveFeet`, checked against an independent Pinocchio computation), and every pendulum number a robot ships - a LIP-height override, `capture_point.com_height_override`, `high_speed_turning`'s `z / g` - is the model's |
| `humanoid_common_mpc:testContactPlanningFormulation` | every rejection names its field; the library defaults are one warning-free lip_miqp configuration; each warning appears exactly when its condition holds (missing planned_com_override, first step, background thread, synchronous lip_miqp, commit window, uncapped instantaneous exchange, sidestep clips, `swing_time_scale`) |
