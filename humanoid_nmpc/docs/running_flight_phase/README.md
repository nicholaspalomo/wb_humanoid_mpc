# Running with a flight phase: implementation proposal

Goal: let the DRC Atlas run like a human, with a flight phase in which no foot touches the ground, planned by the
mixed-integer contact planner and executed by the centroidal whole-body MPC. This document lays out what forbids flight
today, the reduced model that admits it, how a plan with flight reaches the MPC and the execution rules, the work in
phases with what each phase proves, and the risks that decide whether the robot can run at all.

Status: the planner side (sections 2 and 3, phases 2 and 3 of section 4) is implemented, unit-tested and **on in the
shipped DRC Atlas configuration**, gated on the commanded speed so that walking keeps its gait and its search. The
first attempt at enabling it produced no plans at all, and the cause was not the solver budget of phase 1 but a real
defect: with the heading model the yaw torque budget of a foot went negative with no foot down, which made every
flight node infeasible. With that fixed the shipped budget returns a usable plan at every speed. The hop in MuJoCo
(phase 4) and running on the robot (phase 5) are open, and the actuator question of section 5 is still unanswered.
What is built is summarised in [../README.md](../README.md) section 2.11.

## 1. What forbids flight today

| Where | What | Why it blocks running |
| --- | --- | --- |
| planner, `no_flight` logic rule and hard constraint | `c_L + c_R >= 1` at every node | a node without contact is infeasible before the QP is even assembled |
| planner, `lip_com` model block | constant-height linear inverted pendulum, `x'' = omega^2 (x - zmp)` | there is no ballistic mode; the ZMP must sit in a support region at every node |
| planner, `zmp_support_region` | big-M disjunction over the contact binaries | with both binaries at 0 the region is empty |
| MPC target trajectories | base or CoM height reference is `defaultBaseHeight`, constant | the MPC has no push-off to plan against and no landing height to expect |
| MPC `dcm_terminal_cost` | DCM reference blended between the stance feet (`supportBlendTime`) | with no stance foot the reference is undefined |
| swing planner | per-leg swing windows from the mode schedule | works for any sequence, but apex and touch-down velocity are walking values |
| execution, `phase_resetting` | early or late touch-down of one swinging foot against one stance foot | in flight both feet swing and the first touch-down ends the flight |
| inverse dynamics, `contact_wrench_gate` | 10 ms debounce, 40 ms ramp | a running stance lasts about 0.25 s; the ramp would cover a sixth of it |

The whole-body MPC itself already has a `FLY` mode (`MotionPhaseDefinition.h`): with both contacts open the
centroidal dynamics carry no wrench and the constraints of both feet switch to swing. The measured mode from the
contact estimator becomes `FLY` when neither foot touches. That layer needs references, not new physics.

## 2. Reduced model with flight

Human running at 3 to 4 m/s: stance about 0.25 s, flight 0.10 to 0.15 s, peak vertical ground reaction 2.5 times body
weight. For a 160 kg Atlas that is about 3.9 kN on one foot. The planner grid must resolve these phases, so the first
prerequisite is `planner.dt: 0.05` with `numNodes: 24` (the walking gait at this grid is a validation step of its own).

### 2.1 Vertical double integrator (new model block `vertical_double_integrator`)

The horizontal model stays the LIP; the vertical motion of the CoM joins the state so that a flight has a beginning and
an end the physics can produce:

    z_{k+1}  = z_k + vz_k dt + 1/2 a_k dt^2
    vz_{k+1} = vz_k + a_k dt

with `a_k` the vertical CoM acceleration as an input. The block appends `[z, vz]` to the state and `a` to the input, so
the variable layout changes and the regression fixtures are re-recorded (the walking configuration does not list the
block and keeps its layout).

### 2.2 Flight through the contact sum, no extra binary

Let `s_k = c_{L,k} + c_{R,k}` (0 in flight, 1 in single support, 2 in double support). Every flight condition is a big-M
row on `s_k`, so no new binary enters the search:

| Term | Row | Meaning |
| --- | --- | --- |
| `ballistic_in_flight` (hard) | `|a_k + g| <= M_a s_k` | in flight the CoM falls at g |
| `zmp_pinned_in_flight` (hard) | `|zmp_k - c_k| <= M s_k` | in flight the LIP has no horizontal acceleration (the ZMP at the CoM projection makes `x'' = 0`) |
| `vertical_thrust_limit` (hard) | `a_k + g <= a_thrust s_k` | each stance foot can push at most `F_max / m`, with `F_max` from the wrench cone or an explicit limit |
| `contact_height` (soft) | `|z_k - z_nom| <= tol + M (1 - c_{i,k})` per foot | a foot on the ground puts the CoM near the LIP height, so a landing is where the ballistic arc returns to it |
| `height_tracking` (cost) | `w_z (z_k - z_nom)^2` | keeps the stance height at the nominal, so the LIP frequency stays valid |
| `vertical_input_regularization` (cost) | `w_a a_k^2` | keeps the push-off as small as the flight allows |

The LIP frequency `omega = sqrt(g / z_nom)` stays fixed. During a push-off the true horizontal dynamics are
`x'' = (g + z'') / z (x - zmp)`, which the fixed-frequency model under-estimates by the ratio `(g + a) / g`, up to 2.5 at
peak thrust. That is the accepted approximation of this proposal; the successive-linearisation mechanism of the heading
model can later re-linearise `omega` per node from the previous plan's `a_k` if the error shows in the plans.

### 2.3 Logic rules and gait limits

`no_flight` leaves the walking lists when running is enabled; the running configuration lists instead:

- `flight_durations` (logic rule): a flight lasts between `minFlightDuration` and `maxFlightDuration` nodes, is entered
  from single support and left into single support of the other foot (running alternates), and never starts so late
  that it cannot end before the horizon does.
- `phase_durations` as today, with running values: `minContactDuration` about 0.2 s, `maxContactDuration` about 0.35 s
  while a velocity is commanded (the walking-only cap of the earlier discussion, so standing stays possible).
- `minDoubleSupportDuration: 0` and `maxSwingDuration` covering stance plus flight, since a foot swings through the
  other foot's stance and the flight.

The maximum flight duration is a bound the vertical model must be able to reach: with `a_thrust` and a 0.25 s stance,
`vz` at lift-off is at most `(a_thrust - g) * t_push`, and a 0.15 s flight needs `vz = g * 0.075 = 0.74 m/s`. The
loader checks this consistency and refuses a configuration whose flight the thrust cannot produce.

### 2.4 Costs that already exist

`velocity_tracking`, `step_length` (with `T_stride` now stance plus flight per foot), `terminal_dcm` with
`trackCommandedVelocity`, `contact_switch`, `plan_consistency` all apply unchanged. `step_length` is what makes the
stride grow with speed; in running the nominal displacement per swing node uses the running cadence.

## 3. From a plan with flight to the controller

| Consumer | Today | With flight |
| --- | --- | --- |
| mode schedule | `ContactPlan::toModeSchedule` maps `{false, false}` to `FLY` already | unchanged |
| CoM height reference | constant `defaultBaseHeight` | the plan's `z(t)`, `vz(t)` per node, interpolated, written into the target trajectories by `ContactPlanningReferenceManager` next to the heading override (`planned_height_override`, a new execution rule so it can be switched) |
| DCM terminal cost of the MPC | support weights blended between stance feet | in flight the reference is the landing foot of the next stance, weight ramped in over the flight; a new `flightReference` option of the cost |
| swing references | one foot per swing, apex `swingHeight`, touch-down velocity | both feet may swing at once (the planner already does per-leg windows); a running profile with higher apex and a touch-down velocity that matches the ground speed, selected by the stance-phase duration |
| wrench cone | friction and CoP limits | an explicit vertical force limit per foot in the task file, the same number the planner's `a_thrust` is derived from, so the two models agree on what a leg can push |
| execution, phase resetting | one foot in flight | flight: the first measured touch-down ends the flight and starts that foot's stance (early or late as today); the other foot's swing continues to its own touch-down; a late touch-down from flight cannot extend the ballistic arc, so the foot searches downward at the configured velocity and the plan is re-timed on contact |
| inverse dynamics gate | 10 / 40 ms | 5 / 15 ms for running stances; the ramp becomes speed-dependent through the task file only, no code change |
| viewer | contact timeline, CoM, ZMP, DCM markers | the timeline shows flight as a gap on both rows already; the plan log gains an `F` phase |

## 4. Work in phases, with what each proves

1. **Grid and walking at 50 ms nodes.** `planner.dt: 0.05`, `numNodes: 24`, double support 50 ms. Proves the MIQP
   budget holds at twice the nodes (watch `TIME-LIMIT` in the plan log) and that walking at full stick is clean before
   any flight is attempted. About a day, mostly tuning.
2. **Planner with flight.** The model block, the rules and constraints of section 2, configuration and registration,
   unit tests: the vertical block's stage matrices, the big-M rows against the contact sum, the flight-durations rule on
   an exhaustive small grid, a hop from standing (flight of one node, both feet land where they left), a running plan
   at 3 m/s whose phases alternate stance, flight, stance. Fixtures re-recorded for the running configuration only.
   About two days.
3. **References.** Height override rule, DCM flight reference, running swing profile, the vertical force limit in both
   models, plan log with flight phases. Tests on the reference manager: the height reference follows the plan, the DCM
   reference is defined at every time of a running schedule. About two days.
4. **Hop in place in MuJoCo.** The first closed-loop milestone: a commanded hop (a flight of one or two nodes with zero
   velocity command) shows whether the whole-body MPC can produce the push-off within the torque limits and land it.
   This is where the actuator question of section 5 is answered with the real model. One to two days, uncertain.
5. **Running.** Ramp the command past the walking cadence floor and let the planner choose flight; tune the gate,
   phase resetting and the swing profile against the plan log and the viewer markers. Open-ended.

Phases 1 to 3 are code with tests and can be finished regardless of the outcome of 4. Phase 4 is the go or no-go.

## 5. Risks and open questions

- **Actuators.** A 160 kg robot at 2.5 times body weight on one leg needs about 3.9 kN of vertical force. With the leg
  geometry of Atlas the knee torque at push-off is of the order of the 890 N m limit of the gains file. The planner can
  only ask for what `a_thrust` allows; if the hop of phase 4 saturates, running is not available to this model at these
  limits, and the answer is a lower flight target or none. This is the one risk that can stop the project, which is why
  the hop comes before any running.
- **The fixed LIP frequency during push-off** (section 2.2). Visible as the planner landing the foot short during the
  stance in which it pushes. Mitigation exists (per-node re-linearisation of `omega`) but is not in this proposal.
- **Search size.** Flight adds no binary, but the 50 ms grid doubles the nodes and the running gait limits allow more
  patterns. The event-shift local search and the warm start should carry it; the plan log will say.
- **Impacts.** A touch-down from 0.15 s of flight arrives at about 0.75 m/s. The contact detection, the gate and the
  phase resetting were tuned for walking touch-downs; the debounce and ramp need running values, and the swing
  planner's touch-down velocity must be reachable by the leg in the last 50 ms of the arc.
- **Standing.** A maximum contact duration applies while standing too. The walking-only cap is required before
  running values can be entered.

## 6. Decisions asked for

1. The vertical double integrator with a fixed LIP frequency as the flight model (section 2), rather than a
   variable-height pendulum. This is the choice that keeps the problem a convex-relaxed MIQP.
2. Flight through the contact sum with big-M rows, no additional binary (section 2.2).
3. `planner.dt: 0.05` as the grid for running, with walking validated on it first (phase 1).
4. Whether double support remains allowed while running or is forbidden by the running gait limits.
5. The hop in place as the go or no-go milestone before any running is attempted (phase 4).
