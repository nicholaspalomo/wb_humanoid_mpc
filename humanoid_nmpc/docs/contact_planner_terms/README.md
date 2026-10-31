# The Contact Planner as an OCP Assembled from Terms

The mixed-integer contact planner (`humanoid_common_mpc/contact_planning/`, `planner.type: "lip_miqp"`; section 2 of
[humanoid_nmpc/docs/README.md](../README.md)) is built the way the whole-body NMPC is: an optimal control problem
assembled from named terms, selected by the term lists of `config/mpc/contact_planning.textproto` and hot-reloaded term
by term. Section 2.10 of [humanoid_nmpc/docs/README.md](../README.md) describes the file; this document describes the
code that reads it.

## The mirror of the OCS2 structure

| Whole-body MPC (OCS2) | Contact planner |
| --- | --- |
| `OptimalControlProblem` with term collections | `ContactPlanningProblem` with `TermCollection`s (`problem/ContactPlanningProblem.h`) |
| `SystemDynamicsBase` | `LipModelBlock`: declares variables, fills the dynamics and the input bounds, sets `x0`, decodes the plan |
| `StateInputCost`, `StateCost` | `LipCost`, on a `NodeSet` (running, terminal, all nodes, last running node) |
| soft and hard constraints | `LipConstraint`, `Softness::kSoft` (with its own `slackPenalty()`, the shared one by default) or `Softness::kHard` |
| (no analog: the NMPC has no binaries) | `ContactLogicRule` and `AssignmentCost` over a shared `ContactLogicState` |
| `PreComputation` | `ContactPlanningContext`, built once per plan (and once per heading re-linearization pass) |
| `HumanoidCostConstraintFactory` and the task file's term lists | `ContactPlanningTermFactory` and `ContactPlanningFormulation` |
| the SQP settings | the `SearchStage` pipeline |
| the reference manager | the `ExecutionRule` pipeline of `ContactPlanningReferenceManager` |

Every term derives `ContactPlanningTerm` (`problem/ContactPlanningTerm.h`): `describe()` (one line of math for the
start-up print), `requiredBlocks()` (the model blocks it needs), `bind()` (its variable indices, resolved once) and
`configure()` (its parameter block; the hot-reload path).

## The problem

`ContactPlanningTermFactory::buildProblemStatus()` fills the collections from the term lists. `finalize()` composes
the variable layout from the model blocks in their order, which must start with `lip_com` and `foothold_integrator`
(so that `LipIndices` and the binaries keep their places), checks that every term finds the blocks it needs, and binds
and configures every term; a model or a term that fails is an InvalidArgument naming it. `assemble()` then writes the
OCP-QP of one plan node by node and term by term in collection order: the costs in their listed order, which is the
floating-point accumulation order of the stage matrices, and the general rows grouped by term, the hard constraints
before the soft ones. Assembly is index arithmetic over cached integers, a few microseconds per node, so the term lists
can change at runtime.

The binaries are the model blocks' (`binaryInputs()`), branched in time order. `propagate()` runs the listed logic
rules in order to a fixpoint (at most 4 N passes), and every rule reads the assignment as it is at the time of the
read, including what an earlier rule of the same pass has just fixed. A rule decides feasibility on a complete
assignment and fixes nothing; on a partial one it may only reject what no completion satisfies and fix what every
satisfying completion shares (`problem/ContactLogicRule.h`): a rule that fixes more prunes feasible subtrees while the
search still reports `optimal`. The `phase_durations` rule keeps the swing-fits-in-horizon and the max-contact yielding
logic inside it, because neither separates from its forward pass. `assignmentCost()` sums the listed assignment costs.

## The search and execution pipelines

`LipContactPlanner` owns the assembled problem and runs the listed `SearchStage`s around the branch-and-bound:
`beforeSearch()` for the warm start (`warm_start_previous_plan`) and the diving heuristic (`diving`), `afterSearch()`
for `event_shift_local_search`, `heading_relinearization` and `cadence_stretch`. The solver budget is a planner setting,
not a stage.

`ContactPlanningReferenceManager` keeps its core (activating, merging, the swing trajectories, the references, the
target poses) and runs the listed `ExecutionRule`s (`execution/`): `phase_resetting`, `energy_cadence_modulation`,
`dcm_step_adjustment`, `planned_heading_override` and `planned_com_override`. A rule implements the hooks it needs
(`adaptSwingingFoot()`, `adaptContactFoot()`, `groundSearch()`, `correctFootholds()`, `overrideTarget()`) on a shared
`ExecutionContext`; the schedule rules share the per-foot swing timing latch through it.
`adaptScheduleToContactEvents()` (`ContactScheduleAdaptation.h`) runs the same pipeline and is the entry point of the
rule tests.

## Runtime configurability

- On a **parameter reload**, `ContactPlannerModule::setConfig()` applies a new configuration between plans. Every term
  re-reads its block (`ContactPlanningProblem::configure()`), and the values are checked once, by
  `ContactPlanningConfig::validateStatus()`, so that an invalid edit leaves the running configuration in place.
- On a **structural reload**, a changed term list re-assembles the problem and the stages through the factory. The warm
  start (the previous plan and its assignment) survives unless the grid or the variable layout changed
  (`LipContactPlanner::setConfig()`).
- For **introspection**, `ContactPlanningProblem::summary()` and `LipContactPlanner::getFormulationSummary()` print the layout,
  every term with its description, the search stages and the planner settings; the module logs it at start-up and after
  every reload that changes it.

## Code layout

```
humanoid_common_mpc/include/humanoid_common_mpc/contact_planning/
  problem/     ContactPlanningProblem.h  ContactPlanningTerm.h  ContactPlanningContext.h  TermCollection.h  Layout.h
               LayoutBuilder.h  LipIndices.h  StageAccumulator.h  RowBuilder.h  InputBoundsBuilder.h  the term bases
  model/       LipComDynamics.h  FootholdIntegrator.h  HeadingDoubleIntegrator.h
  cost/        one header per cost, on LipWeightedCost.h
  constraint/  one header per constraint, on LipConstraintBase.h
  logic/       ContactLogicState.h, the four rules and the three assignment costs
  search/      SearchStage.h and the five stages
  execution/   ExecutionRule.h  ExecutionContext.h and the five rules
  ContactPlanningFormulation.h  ContactPlanningTermFactory.h  LipContactPlanner.h  ContactPlannerModule.h
  ContactPlanningReferenceManager.h  MixedIntegerOcpQp.h  OcpQpHpipm.h  ...
```

The Bazel target `//humanoid_nmpc/humanoid_common_mpc:contact_planning_core` holds the problem, the terms, the search
stages, the execution rules and the solver that need no robot model, so that the planner's tests build without the
rest of the library. `ContactPlannerModule`, the reference manager and the two override rules (`planned_com_override`,
`planned_heading_override`), which need the robot model, are in `//humanoid_nmpc/humanoid_common_mpc`.

## Tests

- `testContactPlanningFormulation`: the term names, the validation (required blocks, execution order) and the file
  layout.
- `testContactPlanningTerms`: the layout, row counts, node sets, required blocks, factory errors, hot reload by term
  name, structural reload and the warm-start rule, the search stage lists and the execution rules.
- `testContactPlanningRegression`: the assembled problems (per stage the dimensions and hashes of every matrix, of the
  input bounds and of the rows as a set), the feasible assignments of a six-node horizon under a grid of
  configurations, the propagation of every partial assignment the search reaches on a short horizon, and eight-step
  receding-horizon walks with and without the heading model, against the fixtures under `test/data/contact_planning/`.
  Re-record them with `REGENERATE_CONTACT_PLANNING_FIXTURES` after an intended change of the formulation and review
  the diff.
- `testLipContactPlanner`, `testContactPhaseResetting` and `testContactPlannerModule`: the planner, the phase
  resetting and the module on the structured configuration.
