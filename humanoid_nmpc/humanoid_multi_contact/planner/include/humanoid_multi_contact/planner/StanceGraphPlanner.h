/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/ContactSampler.h"
#include "humanoid_multi_contact/planner/LazyWeightedAStar.h"
#include "humanoid_multi_contact/planner/MultiContactPlanner.h"
#include "humanoid_multi_contact/planner/PlanTiming.h"
#include "humanoid_multi_contact/planner/StanceInverseKinematics.h"
#include "humanoid_multi_contact/planner/StanceMotion.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

/** The stance-graph planner's registry name (kPlannerNames). */
inline constexpr absl::string_view kStanceGraphPlannerName = "stance_graph";

/** The defaults of StanceGraphPlannerOptions; its fields say what each is. */
inline constexpr double kDefaultSupportContactWeight = 0.5;
inline constexpr double kDefaultWallPenalty = 0.2;
inline constexpr double kDefaultStaticMarginBuffer = 0.02;
// [m] The certifier's margins kept by planner B as planner A's seed
// (StanceGraphThenContactInvariantOptions), which solve 3 re-solves.
inline constexpr double kSeedStaticMarginBuffer = 0.01;
inline constexpr double kDefaultDynamicMargin = 0.02;
inline constexpr int kDefaultSamplesPerCandidate = 6;
inline constexpr int kDefaultSamplesPerFace = 3;
inline constexpr double kDefaultSampleSpacing = 0.1;
inline constexpr double kDefaultPadSeparation = 0.03;
inline constexpr double kDefaultTrunkContactReach = 0.4;
inline constexpr double kDefaultLimbRootTravel = 0.4;
inline constexpr double kDefaultAnchorGoalFraction = 0.5;
inline constexpr double kDefaultInitialChangeDisplacement = 0.3;
inline constexpr double kDefaultGoalPostureBlend = 0.25;
inline constexpr double kDefaultRetimingSlowdown = 1.5;
inline constexpr int kDefaultMaxRetimings = 4;

/** When a plan of the stance graph breaks a contact (design section 5.5,
 * "Timing law"). */
enum class ContactBreak {
  // At the second knot of the motion that lifts its pad: the timing planner
  // A's solve 3 retimes when planner B seeds it (the combined planner).
  kAtSecondKnot,
  // At the knot that ends the dwell before that motion, the configuration that
  // broke it, before the pad rises: a plan the MPC tracks as it is, whose
  // knots are seconds apart, so that the MPC does not hold a pad down that the
  // plan already lifts.
  kAtDwellEnd,
};

/** Which segments a plan whose knots fail condition 2 alone is retimed in
 * (design section 5.5, "Timing law"). */
enum class RetimingScope {
  // Every segment, by the same factor: the timing planner A's solve 3
  // retimes when planner B seeds it (the combined planner).
  kWholePlan,
  // The segments of the failing knots, and the one before a failing first
  // knot of a segment: a fast fold of a leg slows its own motion only.
  kFailingSegments,
};

/** The settings of StanceGraphPlanner. */
struct StanceGraphPlannerOptions {
  StanceGraphPlannerOptions();

  LazySearchOptions search;
  StanceIkOptions ik;
  StanceMotionOptions motion;
  ContactSamplerOptions sampler;
  TimingOptions timing;
  // [m] Added to the scenario's static margin for the CoM constraint of the
  // IK, so that the certifier, which measures the polygon at the reached pads,
  // finds at least the scenario's margin, and the tracking MPC, which follows
  // the CoM within a centimeter or two, does too.
  double staticMarginBuffer = kDefaultStaticMarginBuffer;
  // [m] The CoM margin of a scenario without a static margin (the stairs):
  // room for the momentum rates of condition 2.
  double dynamicMargin = kDefaultDynamicMargin;
  // The samples tried per candidate and expansion, at least `sampleSpacing`
  // [m] apart and at most `samplesPerFace` on one face, so that a nearer face
  // is tried when the faces nearest the goal are out of reach.
  int samplesPerCandidate = kDefaultSamplesPerCandidate;
  int samplesPerFace = kDefaultSamplesPerFace;
  double sampleSpacing = kDefaultSampleSpacing;
  // [m] How far apart two pads on one surface stay (placementsOverlap()).
  double padSeparation = kDefaultPadSeparation;
  // [m] How far a contact of the trunk (the seat, the torso) may move in one
  // change: its "limb's" reach.
  double trunkContactReach = kDefaultTrunkContactReach;
  // [m] How far a limb's root (a hip, a shoulder) may move with the trunk in
  // one change, added to the limb's length for the reach of its samples: a
  // standing robot reaches the floor with its hands by bending over.
  double limbRootTravel = kDefaultLimbRootTravel;
  // How far a sample's anchor moves toward the goal: the fraction of the way
  // from the pad to the goal pose's pad, for a goal contact, or of the base's
  // displacement to the goal otherwise.
  double anchorGoalFraction = kDefaultAnchorGoalFraction;
  // What a contact outside the goal stance (a hand, a knee) counts in the
  // heuristic's number of changes: less than the one change that will remove
  // it, so that the stances a climb needs on the way are not all worse than
  // every stance on the floor.
  double supportContactWeight = kDefaultSupportContactWeight;
  // [m] Added to a sample's score per unit of 1 - n_z: level faces first.
  double wallPenalty = kDefaultWallPenalty;
  // The cost of one contact change, plus this weight per meter of the base's
  // path (design section 5.1).
  double basePathWeight = 1.0;
  // [m] The base displacement of one change the heuristic assumes until a
  // larger one is seen (design section 5.1).
  double initialChangeDisplacement = kDefaultInitialChangeDisplacement;
  // The fraction of the way from a node's configuration to the goal pose of
  // the first posture reference of a transition.
  double goalPostureBlend = kDefaultGoalPostureBlend;
  // Plans whose knots fail condition 2 are retimed this much slower, at most
  // this many times.
  double retimingSlowdown = kDefaultRetimingSlowdown;
  int maxRetimings = kDefaultMaxRetimings;
  RetimingScope retimingScope = RetimingScope::kWholePlan;
  ContactBreak contactBreak = ContactBreak::kAtDwellEnd;
};

/**
 * Planner B of the design (section 5): a lazy weighted A* (lazyWeightedAStar())
 * over the stance graph, whose nodes are stances, candidates on points of the
 * terrain's faces, and whose edges add or remove one contact.
 *
 * - A node is reached by a transition configuration: the IK of the larger of
 *   the two stances with the CoM in the static equilibrium polygon of the
 *   smaller with a margin, collision-free, and statically feasible on the
 *   smaller (torque and exact margin when the scenario sets a static margin).
 * - An edge is feasible when the motion inside the stance it leaves, from the
 *   node's configuration to the transition, is: each knot an IK projection of
 *   the interpolation, statically feasible, with the leaving and arriving pads
 *   swinging clear of the terrain (StanceMotion).
 * - The cost is the number of changes plus a weighted base path length; the
 *   heuristic, the number of contacts that differ from the goal stance or the
 *   base's distance to the goal over the largest displacement one change has
 *   achieved, whichever is larger.
 * - Contact targets are sampled on a 5 cm grid of every face within the
 *   reach of the limb (sampleContacts()), and the samples nearest to an
 *   anchor (the goal pose's pad, or the pad moved toward the goal) are tried.
 *
 * - A scenario's waypoints (stances on the way to the goal, hints) split the
 *   search into legs: one search from the initial stance to the first
 *   waypoint, from there to the next, and from the last to the goal, within
 *   one budget. A leg to a waypoint makes only the waypoint's contacts. A
 *   waypoint with a pose is where its leg samples the contacts (on the faces
 *   the pose's pads touch) and a posture its changes are solved toward and,
 *   when they fail from the node's configuration, solved from, as is the pose
 *   of the waypoint the leg starts at.
 *
 * The motion is timed by the timing law of PlanTiming.h (quasi-static,
 * minimum-jerk segments between rests, with a dwell at every contact change),
 * stamped with the scenario's hashes and certified by its StabilityCertifier;
 * knots that fail condition 2 are retimed slower.
 */
class StanceGraphPlanner : public MultiContactPlanner {
 public:
  /** InvalidArgument for inconsistent options. */
  static absl::StatusOr<std::unique_ptr<StanceGraphPlanner>> Create(
      const StanceGraphPlannerOptions& options);

  StanceGraphPlanner(const StanceGraphPlanner&) = delete;
  StanceGraphPlanner& operator=(const StanceGraphPlanner&) = delete;
  ~StanceGraphPlanner() override = default;

  absl::StatusOr<config::MultiContactPlan> plan(
      const LoadedScenario& loaded) override;
  absl::string_view name() const override { return kStanceGraphPlannerName; }

  const StanceGraphPlannerOptions& options() const { return options_; }

 private:
  explicit StanceGraphPlanner(const StanceGraphPlannerOptions& options);

  StanceGraphPlannerOptions options_;
};

}  // namespace ocs2::humanoid::multi_contact
