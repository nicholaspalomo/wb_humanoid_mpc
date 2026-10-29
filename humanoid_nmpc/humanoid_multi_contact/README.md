# humanoid_multi_contact

Multi-contact planning for the humanoid NMPC: the terrain, the contact candidates, the stability certifier, the plan
file, the scenarios, the planners and the tools around them. The design is
[`humanoid_nmpc/docs/multi_contact_planning/README.md`](../docs/multi_contact_planning/README.md); this package is its
milestones 1 (the certifier and the scenarios), 3 (planner B, the stance graph) and 5 (planner A, contact-invariant
optimization on the centroidal MPC's model and OCS2's SQP).

| Directory | Bazel targets | What |
|---|---|---|
| `terrain/` | `:terrain`, `:terrain_scene_nproto` | `TerrainScene` (boxes and capsules) and `TerrainModel`: signed distance, normal, nearest point and sphere distance, templated on the scalar for CppAD; box face frames and the patch under a point |
| `contact_set/` | `:contact_set` | `ContactSet`, the runtime list of candidates (point or surface, parent joint, pad frame, geom radius, rectangle) and the bitmask of a contact mode |
| `stability/` | `:equilibrium`, `:robot_stance_model`, `:certifier` | static equilibrium and its margin, torque feasibility, the contact wrench cone check, and `StabilityCertifier`, all on HPIPM's dense QP |
| `collision/` | `:collision` | `CollisionModel`: the spheres of the robot's MJCF collision geoms (design section 5.4), their terrain clearance outside the neighborhood of the active pads, and the self-collision pairs |
| `planner/` | `:planner`, `:stance_graph_planner`, `:stance_motion`, `:search`, `:contact_invariant`, `:contact_invariant_core` | `MultiContactPlanner` and `MultiContactPlannerRegistry` (by name); `StanceGraphPlanner` ("stance_graph"): a lazy weighted A* over stances, the stance IK (an SQP on Pinocchio and HPIPM), the motions between stances, the contact sampler and the timing law; `ContactInvariantPlanner` ("contact_invariant"): the MPC's problem augmented with a contact variable per candidate (`contact_invariant/`: the wrappers, the contact features on a CppAD tape, the contact-invariant cost, the force gate, the continuation and the thresholding), and solve 3 on the MPC tracking the thresholded plan with the torque, sphere and support-polygon terms (`HardContactRefinement`); `StanceGraphThenContactInvariantPlanner` ("stance_graph_then_contact_invariant"): planner B's sequence retimed as the seed of solve 3 |
| `plans/` | `:plans` | the shipped scenarios' plans (`make plan-drc-atlas-<scenario>`), which the dummy sims and the tracking test read |
| `plan/` | `:plan_file` | `MultiContactPlan` and `PlanFile`: a binary plan beside a textproto summary, refused when made for another contact set or scene |
| `scenarios/` | `:scenarios`, `:scenario_files` | `Scenario` and the DRC Atlas's scenarios in `config/`: stairs, stairs_steep, box_climb, box_climb_high, car_egress, get_up_supine, get_up_prone |
| `app/` | `:plan_multi_contact`, `:certify_plan`, `:plan_playback` | the offline planner, the certifier on a plan file, and the kinematic playback of a plan in Rerun |

The DRC Atlas's contact set and collision model are `contact_set.textproto` and `collision_model.textproto` in
`robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/`.

## Planning a scenario

<!-- LINT.IfChange(multi_contact_plan_targets) -->
```bash
make plan-drc-atlas-box-climb       # plan_multi_contact: writes humanoid_nmpc/humanoid_multi_contact/plans/box_climb.binpb
make playback-drc-atlas-box-climb   # plan_playback of that plan in Rerun (RERUN_SINK=, default serve_web)
```

There are the same targets for `stairs`, `stairs-steep`, `box-climb-high`, `car-egress`, `get-up-supine` and
`get-up-prone`. Each runs the scenario's own planner (its `planner` field); `MULTI_CONTACT_PLANNER=<name>` overrides it
(`stance_graph`, `contact_invariant` or `stance_graph_then_contact_invariant`).
<!-- LINT.ThenChange(//Makefile:multi_contact_plan_targets) -->

```bash
bazel run //humanoid_nmpc/humanoid_multi_contact/app:plan_multi_contact -- \
    --scenario=humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto \
    --output=humanoid_nmpc/humanoid_multi_contact/plans/stairs.binpb [--planner=stance_graph] [--v=1]
```

Planner A (`contact_invariant`) plans the stairs in 8 s of motion, and so does the combined planner
(`stance_graph_then_contact_invariant`, the stairs scenario's own, whose quasi-static plan the dummy sim tracks within
the certifier's conditions). Planner B (`stance_graph`) plans the box climb and both get-ups through their waypoints,
and the dummy sim tracks each of those plans too; the other scenarios do not plan yet, and the design's sections 4.4
and 5.5 ("Status") say where each planner stops. `plan_multi_contact` prints the certifier's report and a summary (stances, contacts used, duration, the smallest
static margin, the largest torque utilization, the smallest sphere clearance) and exits 0 with a certified plan, 1 when
no plan is found or the certifier refuses it (the message names the stance nearest the goal and why its changes
failed, or the failing knots), 2 on an error. `--v=1` logs every stance the search reaches, `--v=2` every failed
change, `--v=3` every knot of every motion.

## Certifying and playing a plan

```bash
bazel run //humanoid_nmpc/humanoid_multi_contact/app:certify_plan -- \
    --scenario=humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto \
    --plan=humanoid_nmpc/humanoid_multi_contact/app/testdata/box_climb_sway.textproto \
    --output=/tmp/box_climb_sway.binpb
bazel run //humanoid_nmpc/humanoid_multi_contact/app:plan_playback -- \
    --scenario=humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto \
    --plan=/tmp/box_climb_sway.binpb [--rerun_sink=serve_web]
```

`certify_plan` prints one line per knot (stance, static margin, torque utilization and its joint, whether the momentum
rate is in the contact wrench cone, the pads' distance to the terrain, the failures) and exits 0 when every knot
passes, 1 when one fails, 2 on an error. `plan_playback` draws the terrain, the robot at every knot, the contact forces,
the CoM over the static equilibrium polygon and the margins; it takes the Rerun sinks of the Rerun bridge
(`humanoid_nmpc/humanoid_rerun_viewer/README.md`). The `make plan-…` and `make playback-…` targets of the design come
with the first planner (milestone 3).

## For the next milestones

- **TerrainModel on a tape.** Record `signedDistance`, `normal`, `nearestPoint` and `sphereDistance` with the tape's
  scalar (`ad_scalar_t`): every branch is a conditional expression, so one tape is valid everywhere, and the scene
  enters it as constants. A tape belongs to the scene it was recorded with. A scene of one primitive gives that
  primitive's exact distance (the flat-ground parity of design section 9).
- **The contact set** numbers the bits of a contact mode (`ContactMask`, `contactBit`, `ContactSet::maskOf` /
  `namesOf`); a surface candidate's rectangle is in its pad frame, whose z axis is the terrain normal when the pad
  rests on a surface (`ContactCandidate::padRotation`).
- **The certifier** takes a `MultiContactPlan` (`certify`) or knots built in code (`certifyKnots`). The static
  conditions (1 and 3) are judged only with a static margin (`CertifierOptions::staticMargin`), as decision 5 of the
  design says.
- **The MPC tracks a plan** (milestone 4): `humanoid_centroidal_mpc/multi_contact/` reads a scenario and a plan
  (`contact_schedule_source: "multi_contact_plan"`), maps the plan's contact bits to the MPC's contacts by name
  (`PlanContactOrder`: the MPC lists its point contacts first and its feet last) and refuses a plan whose hashes are
  not the scenario's. A plan is tracked from its first knot's state; the dummy sim of a scenario
  (`make launch-drc-atlas-<scenario>-dummy-sim`) reads `plans/<scenario>.binpb` of this package, which
  `make plan-drc-atlas-<scenario>` is to write.
