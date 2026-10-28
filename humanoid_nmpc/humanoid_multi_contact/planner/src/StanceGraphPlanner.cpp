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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "humanoid_multi_contact/stability/TorqueFeasibility.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The weighted A*'s epsilon by default: a path at most twice the
// optimum, and a search that commits to progress toward the goal.
constexpr double kPlannerHeuristicWeight = 2.0;
// The search's budgets by default.
constexpr int kPlannerMaxValidations = 6000;
// The fraction of a segment that sets a point pad down by which its posture
// reaches the end's (StanceMotionOptions::pointTouchdownPostureLead).
constexpr double kPointTouchdownPostureLead = 0.5;
constexpr int64_t kPlannerTimeBudgetSeconds = 2400;
// The --v level that logs every failed change (1 logs every reached stance).
constexpr int kFailureLogLevel = 2;
// How many failed edges a node keeps for the failure report.
constexpr size_t kKeptFailures = 4;

/** The limb of a candidate: the joint its chain hangs from and its reach. */
struct Limb {
  // The joint whose origin the reach is measured from; std::nullopt for a
  // contact of the trunk, which reaches from its own pad.
  std::optional<pinocchio::JointIndex> root;
  // [m]
  double reach = 0.0;
};

/** What a node changed relative to its parent. */
enum class Change {
  kStart,
  kAdd,
  kRemove,
};

/** One node of the stance graph. */
struct StanceNode {
  StancePlacements stance;
  // Validated: the transition configuration into this stance. Not yet: the
  // parent's.
  Eigen::VectorXd q;
  bool validated = false;
  int parent = -1;
  Change change = Change::kStart;
  // The candidate added or removed.
  int changed = 0;
  // The motion inside the parent's stance, from the parent's configuration to
  // `q` (its last knot).
  std::vector<Eigen::VectorXd> motion;
  // Of a goal: the motion inside the goal stance to the goal pose.
  std::vector<Eigen::VectorXd> finalMotion;
  // Why candidates of this node failed, for the failure report.
  std::vector<std::string> failures;
};

/** "foot_l_contact@step_1 (0.64, 0.11, 0.18)". */
std::string describe(const ContactPlacement& placement,
                     const ContactSet& contactSet,
                     const TerrainModel& terrain) {
  return absl::StrFormat("%s@%s (%.2f, %.2f, %.2f)",
                         contactSet.candidate(placement.candidate).name,
                         terrain.primitive(placement.primitive).name,
                         placement.position.x(), placement.position.y(),
                         placement.position.z());
}

std::string describe(const StancePlacements& stance,
                     const ContactSet& contactSet,
                     const TerrainModel& terrain) {
  std::vector<std::string> parts;
  for (const ContactPlacement& placement : stance) {
    parts.push_back(describe(placement, contactSet, terrain));
  }
  return absl::StrCat("[", absl::StrJoin(parts, ", "), "]");
}

/** The placement of the pad of `candidate` at the model's configuration on
 * the patch under it. */
ContactPlacement placementAt(const RobotStanceModel& model,
                             const TerrainModel& terrain,
                             int candidate) {
  const Eigen::Vector3d position = model.padPosition(candidate);
  const PatchFrame patch = terrain.patchFrameAt(position);
  return ContactPlacement{
      .candidate = candidate,
      .primitive = patch.primitiveIndex,
      .face = patch.face,
      .position = position,
      .normal = patch.normal(),
      .xAxis = tangentAxis(model.padRotation(candidate).col(kXAxisIndex),
                           patch.normal())};
}

/** Whether `joint` is of the trunk: the free flyer (or the universe), or a
 * joint with two or more children that lead to candidates. */
bool isTrunk(pinocchio::JointIndex joint,
             const std::vector<int>& leadingChildren) {
  constexpr pinocchio::JointIndex kFreeFlyer = 1;
  return joint <= kFreeFlyer || leadingChildren[joint] > 1;
}

/** The limbs of the candidates: a limb hangs from the first joint below the
 * trunk (isTrunk()). */
std::vector<Limb> limbsOf(const RobotStanceModel& model,
                          double trunkContactReach) {
  const pinocchio::Model& pinocchioModel = model.model();
  const int joints = pinocchioModel.njoints;
  // How many children of each joint lead to a candidate's joint.
  std::vector<int> leadingChildren(joints, 0);
  std::vector<bool> leads(joints, false);
  for (int index = 0; index < model.contactSet().size(); ++index) {
    for (pinocchio::JointIndex joint = model.padJoint(index); joint > 0;
         joint = pinocchioModel.parents[joint]) {
      if (leads[joint]) break;
      leads[joint] = true;
      ++leadingChildren[pinocchioModel.parents[joint]];
    }
  }
  std::vector<Limb> limbs;
  for (int index = 0; index < model.contactSet().size(); ++index) {
    pinocchio::JointIndex joint = model.padJoint(index);
    if (isTrunk(joint, leadingChildren)) {
      limbs.push_back(Limb{.root = std::nullopt, .reach = trunkContactReach});
      continue;
    }
    double reach =
        Eigen::Vector3d(model.contactSet().candidate(index).offset).norm();
    while (!isTrunk(pinocchioModel.parents[joint], leadingChildren)) {
      reach += pinocchioModel.jointPlacements[joint].translation().norm();
      joint = pinocchioModel.parents[joint];
    }
    // The limb's own first joint.
    limbs.push_back(Limb{.root = joint, .reach = reach});
  }
  return limbs;
}

/**
 * The largest distance two pads can be apart, per pair of candidates: the
 * lengths of the links on the tree path between their joints, plus their
 * offsets. A necessary condition of a stance that holds both.
 */
std::vector<std::vector<double>> padSeparationsOf(
    const RobotStanceModel& model) {
  const pinocchio::Model& pinocchioModel = model.model();
  const int count = model.contactSet().size();
  // The path length from each joint up to the universe.
  std::vector<double> depth(pinocchioModel.njoints, 0.0);
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(pinocchioModel.njoints);
       ++joint) {
    depth[joint] = depth[pinocchioModel.parents[joint]] +
                   pinocchioModel.jointPlacements[joint].translation().norm();
  }
  std::vector<std::vector<double>> separations(count,
                                               std::vector<double>(count, 0.0));
  for (int a = 0; a < count; ++a) {
    for (int b = 0; b < count; ++b) {
      // The common ancestor of the two pad joints.
      pinocchio::JointIndex first = model.padJoint(a);
      pinocchio::JointIndex second = model.padJoint(b);
      while (first != second) {
        if (first > second) {
          first = pinocchioModel.parents[first];
        } else {
          second = pinocchioModel.parents[second];
        }
      }
      separations[a][b] =
          depth[model.padJoint(a)] + depth[model.padJoint(b)] - depth[first] -
          depth[first] +
          Eigen::Vector3d(model.contactSet().candidate(a).offset).norm() +
          Eigen::Vector3d(model.contactSet().candidate(b).offset).norm();
    }
  }
  return separations;
}

}  // namespace

StanceGraphPlannerOptions::StanceGraphPlannerOptions() {
  search.heuristicWeight = kPlannerHeuristicWeight;
  search.maxValidations = kPlannerMaxValidations;
  search.timeBudget = absl::Seconds(kPlannerTimeBudgetSeconds);
  // A plan the MPC tracks as it is keeps every sphere within half the
  // certifier's 5 mm of the terrain, at its knots and between them, which the
  // MPC's reference interpolates: the tracked robot is a few millimeters off
  // the plan, and a thigh settling onto a box beside its knee dips a few
  // millimeters into it.
  ik.penetrationTolerance = kDefaultSweepTolerance;
  motion.sphereTolerance = kDefaultSweepTolerance;
  motion.sweep = SweepResolution();
  // A knee swinging onto a box folds before it reaches the box's edge.
  motion.pointTouchdownPostureLead = kPointTouchdownPostureLead;
  retimingScope = RetimingScope::kFailingSegments;
}

StanceGraphPlanner::StanceGraphPlanner(const StanceGraphPlannerOptions& options)
    : options_(options) {}

absl::StatusOr<std::unique_ptr<StanceGraphPlanner>> StanceGraphPlanner::Create(
    const StanceGraphPlannerOptions& options) {
  if (options.samplesPerCandidate < 1 || !(options.staticMarginBuffer >= 0.0) ||
      !(options.dynamicMargin >= 0.0) ||
      !(options.initialChangeDisplacement > 0.0) ||
      !(options.retimingSlowdown > 1.0) || options.maxRetimings < 0 ||
      !(options.sampler.gridSpacing > 0.0)) {
    return absl::InvalidArgumentError(
        "the stance graph planner's samples, margins, displacement and "
        "retiming must be positive");
  }
  return absl::WrapUnique(new StanceGraphPlanner(options));
}

namespace {

/** A contact of the goal stance. Passive data. */
struct GoalPlacement {
  int candidate = 0;
  int primitive = 0;
  // Of a waypoint with a pose: the face of the primitive its pose's pad is on,
  // the only face the leg samples the contact on (a knee onto a box's top, not
  // its front). std::nullopt: any face, or a capsule.
  std::optional<BoxFace> face;
  // Of a waypoint with a pose: the pose's pad on that face, tried first.
  std::optional<ContactSample> poseSample;
};

/** What a run of the planner shares: the scenario, the robot and the goal. */
struct PlanningContext {
  const LoadedScenario* absl_nonnull loaded;
  const StanceGraphPlannerOptions* absl_nonnull options;
  std::unique_ptr<StanceMotion> motion;
  std::vector<Limb> limbs;
  // The largest distance between two candidates' pads (padSeparationsOf()).
  std::vector<std::vector<double>> padSeparations;
  std::vector<int> allowed;
  // The goal stance: each candidate on its primitive.
  std::vector<GoalPlacement> goalContacts;
  Eigen::VectorXd initialQ;
  // The scenario's goal pose (the initial pose when it has none).
  Eigen::VectorXd finalQ;
  // The goal of the search that runs: the scenario's goal, or a waypoint on
  // the way to it (setLegGoal()).
  Eigen::VectorXd goalQ;
  Eigen::Vector3d goalBase = Eigen::Vector3d::Zero();
  // The pads of the goal pose, per candidate.
  std::vector<Eigen::Vector3d> goalPads;
  // The region the base must reach, if any.
  std::optional<config::Scenario::BaseRegion> goalRegion;
  // Whether the goal is the scenario's: the search then ends with the motion
  // to the goal pose. A waypoint is reached by its stance (and region) alone.
  bool finalLeg = true;
  // Whether the goal is a waypoint with a pose: its changes are then also
  // solved toward that pose itself.
  bool waypointPose = false;
  // The poses of the waypoints the leg runs between (the one it starts at and
  // the one it ends at, where they have poses): the starts a change's
  // transition is solved from when it fails from the node's configuration.
  std::vector<Eigen::VectorXd> poseSeeds;
};

/** The stance graph of one run, as a LazySearchProblem. */
class StanceGraphSearch : public LazySearchProblem {
 public:
  explicit StanceGraphSearch(PlanningContext& context) : context_(context) {}

  /** Adds the start node: `start`'s stance and configuration, and the change
   * that entered it (a contact it broke still lifts off in the first motion);
   * its id. */
  int addStart(const StanceNode& start) {
    nodes_.push_back(StanceNode{.stance = start.stance,
                                .q = start.q,
                                .validated = true,
                                .change = start.change,
                                .changed = start.changed});
    return 0;
  }

  const std::vector<StanceNode>& nodes() const { return nodes_; }
  int validations() const { return validations_; }

  std::vector<SearchCandidate> candidates(int node) override;
  absl::StatusOr<std::optional<double>> validate(int node) override;
  double heuristic(int node) override;
  std::string key(int node) override { return stanceKey(nodes_[node].stance); }
  absl::StatusOr<bool> isGoal(int node) override;

  /** The validated node nearest the goal by the heuristic. */
  int nearestToGoal();

 private:
  /** The contact changes between `stance` and the goal stance: one per goal
   * contact absent, two per goal contact on another primitive, and
   * StanceGraphPlannerOptions::supportContactWeight per other contact. */
  double goalMismatch(const StancePlacements& stance) const;
  /** Whether `placement` is where `goal` asks for its contact: on its
   * primitive and, for a waypoint with a pose, on the pose's face and within
   * the sample spacing of the pose's pad. */
  bool onGoal(const ContactPlacement& placement,
              const GoalPlacement& goal) const {
    if (placement.primitive != goal.primitive) return false;
    if (goal.face.has_value() && placement.face != goal.face) return false;
    return !goal.poseSample.has_value() ||
           (placement.position - goal.poseSample->position).norm() <=
               context_.options->sampleSpacing;
  }
  /** Whether `candidate` is a contact of the goal stance. */
  bool isGoalCandidate(int candidate) const {
    return std::any_of(
        context_.goalContacts.begin(), context_.goalContacts.end(),
        [&](const GoalPlacement& goal) { return goal.candidate == candidate; });
  }
  /** The goal stance's candidates exactly, each where the goal asks for it
   * (onGoal()). */
  bool isGoalStance(const StancePlacements& stance) const;
  /** Adds the candidate of `node` that places `candidate` at `sample` (its
   * score the tie-break), unless it is within the sample spacing of one
   * `picked`, its face has given its samples (`perFace`) or it overlaps a pad
   * of the stance; whether it was added. */
  bool tryAdd(int node,
              int candidate,
              const ContactSample& sample,
              double score,
              std::vector<Eigen::Vector3d>& picked,
              absl::flat_hash_map<std::pair<int, int>, int>& perFace,
              std::vector<SearchCandidate>& result);
  /** The add candidates of `node` for `candidate`. */
  void addCandidates(int node,
                     int candidate,
                     std::vector<SearchCandidate>& result);
  /** The posture references of a transition from `q`. */
  std::vector<Eigen::VectorXd> referencesFrom(const Eigen::VectorXd& q) const;
  /** Records a failure of a candidate of `parent`. */
  void recordFailure(int parent, int node, const std::string& failure);

  PlanningContext& context_;
  std::vector<StanceNode> nodes_;
  double largestChange_ = 0.0;
  int validations_ = 0;
};

double StanceGraphSearch::goalMismatch(const StancePlacements& stance) const {
  double count = 0.0;
  // A goal contact on another primitive must be broken and made again.
  constexpr double kMisplacedChanges = 2.0;
  for (const GoalPlacement& goal : context_.goalContacts) {
    const std::optional<ContactPlacement> placement =
        placementOf(stance, goal.candidate);
    if (!placement.has_value()) {
      count += 1.0;
    } else if (!onGoal(*placement, goal)) {
      count += kMisplacedChanges;
    }
  }
  for (const ContactPlacement& placement : stance) {
    if (!isGoalCandidate(placement.candidate)) {
      count += context_.options->supportContactWeight;
    }
  }
  return count;
}

bool StanceGraphSearch::isGoalStance(const StancePlacements& stance) const {
  if (stance.size() != context_.goalContacts.size()) return false;
  for (const GoalPlacement& goal : context_.goalContacts) {
    const std::optional<ContactPlacement> placement =
        placementOf(stance, goal.candidate);
    if (!placement.has_value() || !onGoal(*placement, goal)) return false;
  }
  return true;
}

double StanceGraphSearch::heuristic(int node) {
  const StanceNode& entry = nodes_[node];
  const double displacement =
      std::max(largestChange_, context_.options->initialChangeDisplacement);
  const double distance =
      (entry.q.head<kPositionDim>() - context_.goalBase).norm();
  return std::max(goalMismatch(entry.stance), distance / displacement);
}

std::vector<Eigen::VectorXd> StanceGraphSearch::referencesFrom(
    const Eigen::VectorXd& q) const {
  const pinocchio::Model& model = context_.motion->model().model();
  std::vector<Eigen::VectorXd> references = {pinocchio::interpolate(
      model, q, context_.goalQ, context_.options->goalPostureBlend)};
  // A waypoint's pose is a posture for its changes, which can be far from the
  // node's (a knee onto a box from standing).
  if (context_.waypointPose) references.push_back(context_.goalQ);
  references.push_back(q);
  return references;
}

void StanceGraphSearch::addCandidates(int node,
                                      int candidate,
                                      std::vector<SearchCandidate>& result) {
  RobotStanceModel& model = context_.motion->model();
  const StanceGraphPlannerOptions& options = *context_.options;
  const ContactCandidate& contact = model.contactSet().candidate(candidate);
  const Limb& limb = context_.limbs[candidate];
  const Eigen::Vector3d center =
      limb.root.has_value() ? model.jointPlacement(*limb.root).translation()
                            : model.padPosition(candidate);
  const Eigen::Vector3d xHint = model.padRotation(candidate).col(kXAxisIndex);
  // A limb's root moves with the trunk, which bends, crouches and turns.
  const double reach =
      limb.root.has_value() ? limb.reach + options.limbRootTravel : limb.reach;
  std::vector<ContactSample> samples =
      sampleContacts(context_.loaded->terrain, contact, candidate, center,
                     reach, xHint, options.sampler);
  // A waypoint's contact on the face of its pose.
  std::optional<ContactSample> poseSample;
  for (const GoalPlacement& goal : context_.goalContacts) {
    if (goal.candidate != candidate || !goal.face.has_value()) continue;
    samples.erase(std::remove_if(samples.begin(), samples.end(),
                                 [&goal](const ContactSample& sample) {
                                   return sample.primitive != goal.primitive ||
                                          sample.face != goal.face;
                                 }),
                  samples.end());
    poseSample = goal.poseSample;
  }
  // Two anchors, a part of the way and all the way: the pad toward the goal
  // pose's pad, for a goal contact, or by the base's way to the goal
  // otherwise. Their nearest samples are taken in turn, so that both a short
  // and a long change are tried.
  const Eigen::Vector3d pad = model.padPosition(candidate);
  const Eigen::Vector3d toGoal =
      isGoalCandidate(candidate)
          ? Eigen::Vector3d(context_.goalPads[candidate] - pad)
          : Eigen::Vector3d(context_.goalBase -
                            nodes_[node].q.head<kPositionDim>());
  std::vector<std::vector<std::pair<double, ContactSample>>> ranked;
  for (const double fraction : {options.anchorGoalFraction, 1.0}) {
    const Eigen::Vector3d anchor = pad + fraction * toGoal;
    // A sample the limb reaches without the trunk's help scores better: the
    // distance to the anchor, plus twice the distance beyond the limb's
    // length, plus the wall penalty.
    std::vector<std::pair<double, ContactSample>> scored;
    for (const ContactSample& sample : samples) {
      const bool apart = std::any_of(
          nodes_[node].stance.begin(), nodes_[node].stance.end(),
          [&](const ContactPlacement& other) {
            return (sample.position - other.position).norm() >
                   context_.padSeparations[candidate][other.candidate];
          });
      if (apart) continue;
      const double beyond =
          std::max(0.0, (sample.position - center).norm() - limb.reach);
      scored.emplace_back((sample.position - anchor).norm() + 2.0 * beyond +
                              options.wallPenalty * (1.0 - sample.normal.z()),
                          sample);
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [](const std::pair<double, ContactSample>& first,
                        const std::pair<double, ContactSample>& second) {
                       return first.first < second.first;
                     });
    ranked.push_back(std::move(scored));
  }
  std::vector<Eigen::Vector3d> picked;
  // How many samples each face gave, by primitive and face.
  absl::flat_hash_map<std::pair<int, int>, int> perFace;
  // The waypoint pose's own pad first: it scores below every sample.
  constexpr double kPoseSampleScore = -1.0;
  if (poseSample.has_value()) {
    tryAdd(node, candidate, *poseSample, kPoseSampleScore, picked, perFace,
           result);
  }
  std::vector<size_t> next(ranked.size(), 0);
  bool progress = true;
  while (progress &&
         static_cast<int>(picked.size()) < options.samplesPerCandidate) {
    progress = false;
    for (size_t anchor = 0; anchor < ranked.size(); ++anchor) {
      if (static_cast<int>(picked.size()) >= options.samplesPerCandidate) {
        break;
      }
      for (; next[anchor] < ranked[anchor].size(); ++next[anchor]) {
        const std::pair<double, ContactSample>& entry =
            ranked[anchor][next[anchor]];
        if (tryAdd(node, candidate, entry.second, entry.first, picked, perFace,
                   result)) {
          ++next[anchor];
          progress = true;
          break;
        }
      }
    }
  }
}

bool StanceGraphSearch::tryAdd(
    int node,
    int candidate,
    const ContactSample& sample,
    double score,
    std::vector<Eigen::Vector3d>& picked,
    absl::flat_hash_map<std::pair<int, int>, int>& perFace,
    std::vector<SearchCandidate>& result) {
  const StanceGraphPlannerOptions& options = *context_.options;
  const bool crowded = std::any_of(
      picked.begin(), picked.end(), [&](const Eigen::Vector3d& other) {
        return (other - sample.position).norm() < options.sampleSpacing;
      });
  if (crowded) return false;
  const std::pair<int, int> face(
      sample.primitive,
      sample.face.has_value() ? static_cast<int>(*sample.face) : -1);
  if (perFace[face] >= options.samplesPerFace) return false;
  const ContactPlacement placement{.candidate = candidate,
                                   .primitive = sample.primitive,
                                   .face = sample.face,
                                   .position = sample.position,
                                   .normal = sample.normal,
                                   .xAxis = sample.xAxis};
  const bool overlaps =
      std::any_of(nodes_[node].stance.begin(), nodes_[node].stance.end(),
                  [&](const ContactPlacement& other) {
                    return placementsOverlap(
                        placement, other, context_.motion->model().contactSet(),
                        options.padSeparation);
                  });
  if (overlaps) return false;
  picked.push_back(sample.position);
  ++perFace[face];
  nodes_.push_back(
      StanceNode{.stance = withPlacement(nodes_[node].stance, placement),
                 .q = nodes_[node].q,
                 .parent = node,
                 .change = Change::kAdd,
                 .changed = candidate});
  result.push_back(SearchCandidate{.node = static_cast<int>(nodes_.size()) - 1,
                                   .estimatedCost = 1.0,
                                   .tieBreak = score});
  return true;
}

std::vector<SearchCandidate> StanceGraphSearch::candidates(int node) {
  std::vector<SearchCandidate> result;
  RobotStanceModel& model = context_.motion->model();
  if (!model.setGeneralizedPosition(nodes_[node].q).ok()) return result;
  const StancePlacements stance = nodes_[node].stance;
  if (stance.size() > 1) {
    for (const ContactPlacement& placement : stance) {
      nodes_.push_back(
          StanceNode{.stance = withoutCandidate(stance, placement.candidate),
                     .q = nodes_[node].q,
                     .parent = node,
                     .change = Change::kRemove,
                     .changed = placement.candidate});
      result.push_back(SearchCandidate{
          .node = static_cast<int>(nodes_.size()) - 1, .estimatedCost = 1.0});
    }
  }
  for (const int candidate : context_.allowed) {
    if (placementOf(stance, candidate).has_value()) continue;
    // A leg to a waypoint makes only the waypoint's contacts: the hint says
    // which contacts the way there takes.
    if (!context_.finalLeg && !isGoalCandidate(candidate)) continue;
    // addCandidates() grows nodes_, so it reads the model, set above.
    addCandidates(node, candidate, result);
  }
  return result;
}

void StanceGraphSearch::recordFailure(int parent,
                                      int node,
                                      const std::string& failure) {
  const StanceNode& entry = nodes_[node];
  const ContactSet& contactSet = context_.motion->model().contactSet();
  std::string change;
  if (entry.change == Change::kAdd) {
    change =
        absl::StrCat("add ", describe(*placementOf(entry.stance, entry.changed),
                                      contactSet, context_.loaded->terrain));
  } else {
    change = absl::StrCat("remove ", contactSet.candidate(entry.changed).name);
  }
  std::vector<std::string>& failures = nodes_[parent].failures;
  if (failures.size() >= kKeptFailures) failures.erase(failures.begin());
  failures.push_back(absl::StrCat(change, ": ", failure));
}

absl::StatusOr<std::optional<double>> StanceGraphSearch::validate(int node) {
  ++validations_;
  const absl::Time started = absl::Now();
  const int parent = nodes_[node].parent;
  const StanceNode& from = nodes_[parent];
  const bool adds = nodes_[node].change == Change::kAdd;
  const StancePlacements& held = adds ? nodes_[node].stance : from.stance;
  const StancePlacements& supporting = adds ? from.stance : nodes_[node].stance;
  ASSIGN_OR_RETURN(MotionCheck transition,
                   context_.motion->transition(held, supporting, from.q,
                                               referencesFrom(from.q)));
  // A change far from the node's posture (the weight onto a knee on a box) is
  // solved from a waypoint's pose instead, and the motion goes there.
  for (const Eigen::VectorXd& seed : context_.poseSeeds) {
    if (transition.feasible) break;
    ASSIGN_OR_RETURN(
        const MotionCheck fromPose,
        context_.motion->transition(held, supporting, seed, {seed, from.q}));
    if (fromPose.feasible) {
      transition = fromPose;
    } else {
      VLOG(kFailureLogLevel)
          << "stance_graph: the transition from a waypoint's pose fails: "
          << fromPose.failure;
    }
  }
  // A pad that makes contact may have to go around the terrain (over a box's
  // edge) to reach its target, which the transition's IK cannot do from the
  // start: then the motion comes first, swinging the pad along its path, and
  // the transition is solved from its last knot.
  // Likewise a stance that breaks a contact may need the weight shifted far:
  // the motion moves the CoM toward the middle of the smaller stance's polygon
  // and the transition is solved from its last knot.
  const bool motionFirst = !transition.feasible;
  if (!transition.feasible && !motionFirst) {
    recordFailure(parent, node,
                  absl::StrCat("transition: ", transition.failure));
    VLOG(kFailureLogLevel) << "stance_graph: transition fails in "
                           << absl::ToDoubleSeconds(absl::Now() - started)
                           << " s: " << nodes_[parent].failures.back();
    return std::optional<double>();
  }
  SegmentContacts contacts;
  if (from.change == Change::kRemove) contacts.liftOff = from.changed;
  if (adds) {
    contacts.touchdown = placementOf(nodes_[node].stance, nodes_[node].changed);
  }
  contacts.endHeld = held;
  contacts.endSupporting = supporting;
  contacts.endMustBeSolved = motionFirst;
  // A climb's legs fold far: on the way to a waypoint's pose, a sole swings
  // level.
  contacts.levelTouchdownSole = context_.waypointPose;
  if (motionFirst && !adds) {
    ASSIGN_OR_RETURN(const std::vector<Eigen::Vector2d> polygon,
                     context_.motion->polygonOf(supporting));
    if (polygon.empty()) {
      recordFailure(parent, node,
                    absl::StrCat("transition: ", transition.failure));
      return std::optional<double>();
    }
    Eigen::Vector2d middle = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& vertex : polygon) middle += vertex;
    contacts.comGoal = middle / static_cast<double>(polygon.size());
  }
  ASSIGN_OR_RETURN(
      SegmentCheck segment,
      context_.motion->segment(from.stance, from.q,
                               motionFirst ? from.q : transition.q, contacts));
  if (!segment.feasible) {
    recordFailure(parent, node,
                  absl::StrCat(motionFirst ? "motion first (the transition: " +
                                                 transition.failure + "): "
                                           : "motion: ",
                               segment.failure));
    VLOG(kFailureLogLevel) << "stance_graph: motion fails in "
                           << absl::ToDoubleSeconds(absl::Now() - started)
                           << " s: " << nodes_[parent].failures.back();
    return std::optional<double>();
  }
  // The motion's end, solved from its last knot, is the transition.
  const Eigen::VectorXd& reached = segment.knots.back();
  const double displacement =
      (reached.head<kPositionDim>() - from.q.head<kPositionDim>()).norm();
  largestChange_ = std::max(largestChange_, displacement);
  nodes_[node].q = reached;
  nodes_[node].motion = std::move(segment.knots);
  nodes_[node].validated = true;
  VLOG(1) << "stance_graph: " << nodes_.size() << " nodes, " << validations_
          << " validations, " << absl::ToDoubleSeconds(absl::Now() - started)
          << " s, " << nodes_[node].motion.size() << " knots: "
          << describe(nodes_[node].stance,
                      context_.motion->model().contactSet(),
                      context_.loaded->terrain);
  return std::optional<double>(1.0 +
                               context_.options->basePathWeight * displacement);
}

/** Whether the base of `q` is in `region`. */
bool baseInRegion(const Eigen::VectorXd& q,
                  const config::Scenario::BaseRegion& region) {
  const Eigen::Vector3d offset =
      (q.head<kPositionDim>() - Eigen::Vector3d(region.center)).cwiseAbs();
  return (offset.array() <= Eigen::Vector3d(region.half_extents).array()).all();
}

absl::StatusOr<bool> StanceGraphSearch::isGoal(int node) {
  StanceNode& entry = nodes_[node];
  if (!isGoalStance(entry.stance)) return false;
  if (!context_.finalLeg) {
    // A waypoint: its stance, and its region when it has one.
    if (context_.goalRegion.has_value() &&
        !baseInRegion(entry.q, *context_.goalRegion)) {
      entry.failures.push_back(absl::StrFormat(
          "waypoint: the base (%.2f, %.2f, %.2f) is outside its region",
          entry.q.x(), entry.q.y(), entry.q.z()));
      return false;
    }
    return true;
  }
  // The goal pose's reference, shifted with the goal contacts' pads.
  Eigen::Vector3d reached = Eigen::Vector3d::Zero();
  Eigen::Vector3d planned = Eigen::Vector3d::Zero();
  for (const ContactPlacement& placement : entry.stance) {
    reached += placement.position;
    planned += context_.goalPads[placement.candidate];
  }
  Eigen::VectorXd reference = context_.goalQ;
  reference.head<kPositionDim>() +=
      (reached - planned) / static_cast<double>(entry.stance.size());
  ASSIGN_OR_RETURN(std::vector<Eigen::Vector2d> polygon,
                   context_.motion->polygonOf(entry.stance));
  StanceIkProblem problem =
      context_.motion->holdingProblem(entry.stance, std::move(polygon));
  problem.postureReference = reference;
  ASSIGN_OR_RETURN(const StanceIkResult result,
                   context_.motion->ik().solve(problem, entry.q));
  if (!result.feasible) {
    entry.failures.push_back(absl::StrCat("goal pose: IK: ", result.failure));
    return false;
  }
  const config::Scenario::Goal& goal = context_.loaded->scenario.goal;
  if (goal.base_region.has_value()) {
    if (!baseInRegion(result.q, *goal.base_region)) {
      entry.failures.push_back(absl::StrFormat(
          "goal pose: the base (%.2f, %.2f, %.2f) is outside the goal region",
          result.q.x(), result.q.y(), result.q.z()));
      return false;
    }
  }
  ASSIGN_OR_RETURN(const std::optional<std::string> statics,
                   context_.motion->checkStatics(result.q, entry.stance));
  if (statics.has_value()) {
    entry.failures.push_back(absl::StrCat("goal pose: statics: ", *statics));
    return false;
  }
  SegmentContacts contacts;
  if (entry.change == Change::kRemove) contacts.liftOff = entry.changed;
  ASSIGN_OR_RETURN(
      SegmentCheck segment,
      context_.motion->segment(entry.stance, entry.q, result.q, contacts));
  if (!segment.feasible) {
    entry.failures.push_back(
        absl::StrCat("goal pose: motion: ", segment.failure));
    return false;
  }
  entry.finalMotion = std::move(segment.knots);
  return true;
}

int StanceGraphSearch::nearestToGoal() {
  int best = 0;
  double bestHeuristic = std::numeric_limits<double>::infinity();
  for (int node = 0; node < static_cast<int>(nodes_.size()); ++node) {
    if (!nodes_[node].validated) continue;
    const double value = heuristic(node);
    if (value < bestHeuristic) {
      bestHeuristic = value;
      best = node;
    }
  }
  return best;
}

/** The planning context of `loaded`. */
absl::StatusOr<PlanningContext> contextOf(
    const LoadedScenario& loaded, const StanceGraphPlannerOptions& options) {
  if (!loaded.collisionModelConfig.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "scenario '", loaded.scenario.name,
        "': the stance graph planner needs a collision_model_file"));
  }
  ASSIGN_OR_RETURN(
      std::unique_ptr<RobotStanceModel> model,
      RobotStanceModel::Create(loaded.urdfPath, loaded.contactSet));
  ASSIGN_OR_RETURN(
      CollisionModel collision,
      CollisionModel::Create(*loaded.collisionModelConfig, *model));
  ASSIGN_OR_RETURN(const JointLimits jointLimits,
                   jointLimitsOf(loaded.scenario.joint_limits));
  if (relaxesPositionLimits(jointLimits)) {
    RETURN_IF_ERROR(
        model->relaxPositionLimits(loaded.scenario.joint_limit_relaxation));
  }
  PlanningContext context{.loaded = &loaded, .options = &options};
  context.limbs = limbsOf(*model, options.trunkContactReach);
  context.padSeparations = padSeparationsOf(*model);
  ASSIGN_OR_RETURN(const RobotConfiguration initial,
                   configurationOf(loaded.scenario.initial_pose));
  ASSIGN_OR_RETURN(context.initialQ, model->generalizedPositionOf(initial));
  context.finalQ = context.initialQ;
  if (loaded.scenario.goal.pose.has_value()) {
    ASSIGN_OR_RETURN(const RobotConfiguration goal,
                     configurationOf(*loaded.scenario.goal.pose));
    ASSIGN_OR_RETURN(context.finalQ, model->generalizedPositionOf(goal));
  }
  for (const std::string& name : loaded.scenario.allowed_contacts) {
    ASSIGN_OR_RETURN(const int index, loaded.contactSet.indexOf(name));
    context.allowed.push_back(index);
  }
  std::sort(context.allowed.begin(), context.allowed.end());
  ASSIGN_OR_RETURN(
      std::unique_ptr<StanceInverseKinematics> ik,
      StanceInverseKinematics::Create(std::move(model), loaded.terrain,
                                      std::move(collision), options.ik));
  StanceMotionOptions motion = options.motion;
  // Under the scenario's "position_and_effort", every planned configuration
  // holds its stance within the torque margin, with or without a static
  // margin: the robot cannot exceed its effort limits in a slow motion
  // either. Under "position_only" the torques are not checked.
  motion.jointLimits = jointLimits;
  if (loaded.scenario.static_margin.has_value()) {
    motion.comMargin =
        *loaded.scenario.static_margin + options.staticMarginBuffer;
    motion.exactStaticMargin = loaded.scenario.static_margin;
  } else {
    motion.comMargin = options.dynamicMargin;
    motion.exactStaticMargin = std::nullopt;
  }
  motion.torqueMargin = loaded.scenario.torque_margin;
  motion.clearedPads = context.allowed;
  ASSIGN_OR_RETURN(context.motion, StanceMotion::Create(std::move(ik), motion));
  return context;
}

/**
 * Sets the goal of the next search of `context` to `goal`: the scenario's goal
 * (`finalLeg`) or one of its waypoints, after the waypoint `previous` (null for
 * the first leg). The goal pose is the goal's pose, or the scenario's goal pose
 * when it has none; the goal base its region's center, or the pose's base. The
 * poses of a waypoint `goal` and of `previous` seed the leg's transitions.
 */
absl::Status setLegGoal(PlanningContext& context,
                        const config::Scenario::Goal* absl_nullable previous,
                        const config::Scenario::Goal& goal,
                        bool finalLeg) {
  const LoadedScenario& loaded = *context.loaded;
  RobotStanceModel& model = context.motion->model();
  context.finalLeg = finalLeg;
  context.waypointPose = !finalLeg && goal.pose.has_value();
  context.goalQ = context.finalQ;
  if (goal.pose.has_value()) {
    ASSIGN_OR_RETURN(const RobotConfiguration pose,
                     configurationOf(*goal.pose));
    ASSIGN_OR_RETURN(context.goalQ, model.generalizedPositionOf(pose));
  }
  context.poseSeeds.clear();
  if (context.waypointPose) context.poseSeeds.push_back(context.goalQ);
  if (previous != nullptr && previous->pose.has_value()) {
    ASSIGN_OR_RETURN(const RobotConfiguration pose,
                     configurationOf(*previous->pose));
    ASSIGN_OR_RETURN(Eigen::VectorXd seed, model.generalizedPositionOf(pose));
    context.poseSeeds.push_back(std::move(seed));
  }
  context.goalRegion = goal.base_region;
  context.goalBase = goal.base_region.has_value()
                         ? Eigen::Vector3d(goal.base_region->center)
                         : Eigen::Vector3d(context.goalQ.head<kPositionDim>());
  RETURN_IF_ERROR(model.setGeneralizedPosition(context.goalQ));
  context.goalPads.clear();
  for (int index = 0; index < loaded.contactSet.size(); ++index) {
    context.goalPads.push_back(model.padPosition(index));
  }
  context.goalContacts.clear();
  for (const config::Scenario::GoalContact& contact : goal.contacts) {
    ASSIGN_OR_RETURN(const int candidate,
                     loaded.contactSet.indexOf(contact.contact));
    ASSIGN_OR_RETURN(const int primitive,
                     loaded.terrain.primitiveIndex(contact.primitive));
    GoalPlacement placement{.candidate = candidate, .primitive = primitive};
    if (context.waypointPose) {
      const Eigen::Vector3d& pad = context.goalPads[candidate];
      const PatchFrame patch = loaded.terrain.patchFrameAt(pad);
      if (patch.primitiveIndex == primitive) {
        placement.face = patch.face;
        // The pose's pad, on the surface.
        const Eigen::Vector3d normal = patch.normal();
        placement.poseSample = ContactSample{
            .candidate = candidate,
            .primitive = primitive,
            .face = patch.face,
            .position = pad - normal.dot(pad - patch.origin) * normal,
            .normal = normal,
            .xAxis = tangentAxis(model.padRotation(candidate).col(kXAxisIndex),
                                 normal)};
      }
    }
    context.goalContacts.push_back(placement);
  }
  return absl::OkStatus();
}

/** The initial stance: the initial pose's contacts where its pads are. */
absl::StatusOr<StancePlacements> initialStance(PlanningContext& context) {
  RobotStanceModel& model = context.motion->model();
  RETURN_IF_ERROR(model.setGeneralizedPosition(context.initialQ));
  ASSIGN_OR_RETURN(const ContactMask mask,
                   contactsOf(context.loaded->scenario.initial_pose,
                              context.loaded->contactSet));
  ASSIGN_OR_RETURN(const std::vector<int> active,
                   context.loaded->contactSet.activeIndices(mask));
  StancePlacements stance;
  for (const int candidate : active) {
    stance.push_back(placementAt(model, context.loaded->terrain, candidate));
  }
  return stance;
}

/** The static contact wrenches of a knot, by the torque LP's forces. */
absl::StatusOr<std::vector<config::MultiContactPlan::ContactWrench>>
contactWrenches(RobotStanceModel& model,
                const TerrainModel& terrain,
                const config::MultiContactPlan::Knot& knot,
                const Eigen::VectorXd& q,
                const StaticEquilibriumOptions& equilibriumOptions) {
  std::vector<config::MultiContactPlan::ContactWrench> wrenches;
  RETURN_IF_ERROR(model.setGeneralizedPosition(q));
  ASSIGN_OR_RETURN(const StanceOnTerrain onTerrain,
                   stanceOnTerrain(model, terrain, knot.active_contacts));
  if (onTerrain.stance.empty()) return wrenches;
  ASSIGN_OR_RETURN(
      const StaticEquilibrium equilibrium,
      StaticEquilibrium::Create(onTerrain.stance, equilibriumOptions));
  const absl::StatusOr<TorqueFeasibilityResult> torques =
      minimumTorqueUtilization(model, equilibrium);
  if (!torques.ok()) return wrenches;
  const WrenchGenerators& generators = equilibrium.generators();
  const double weight = model.mass() * kGravitationalAcceleration;
  std::vector<Eigen::Vector3d> forces(onTerrain.stance.size(),
                                      Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector3d> torquesAbout(onTerrain.stance.size(),
                                            Eigen::Vector3d::Zero());
  for (int g = 0; g < generators.size(); ++g) {
    const int contact = generators.contactIndices[g];
    const Eigen::Vector3d force =
        torques->scalings(g) * weight * generators.directions[g];
    forces[contact] += force;
    torquesAbout[contact] +=
        (generators.points[g] - onTerrain.stance[contact].position)
            .cross(force);
  }
  for (size_t contact = 0; contact < onTerrain.stance.size(); ++contact) {
    config::MultiContactPlan::ContactWrench wrench;
    wrench.contact = onTerrain.stance[contact].name;
    wrench.force = forces[contact];
    if (onTerrain.stance[contact].kind == ContactKind::kSurface) {
      wrench.torque = torquesAbout[contact];
    }
    wrenches.push_back(std::move(wrench));
  }
  return wrenches;
}

/** Whether the motion after `node` breaks its contact at its start
 * (PathMotionBuilder::add()): under ContactBreak::kAtDwellEnd, when `node`
 * broke one. */
bool breaksAtDwellEnd(const StanceNode& node, ContactBreak contactBreak) {
  return contactBreak == ContactBreak::kAtDwellEnd &&
         node.change == Change::kRemove;
}

/** The segments of a path and the stance of each knot. */
struct PathMotion {
  std::vector<MotionSegment> segments;
  // Per segment and knot, the placements of the knot's stance.
  std::vector<std::vector<StancePlacements>> stances;
};

/** Builds a PathMotion segment by segment: each segment starts where the
 * previous one ended, with its mask. */
class PathMotionBuilder {
 public:
  explicit PathMotionBuilder(const StanceNode& start)
      : lastMask_(maskOf(start.stance)), lastStance_(start.stance) {}

  /** Adds the motion inside `inside` from `start`'s configuration through
   * `knots`, its last knot with the mask of `endStance`. With `breaksAtStart`
   * the first knot has the mask of `inside` already: the contact the stance
   * broke is broken at the start of the motion that lifts it (its knot at the
   * end of the dwell), not at the motion's second knot, which a sparse plan
   * puts seconds later while the pad already rises toward it. */
  void add(const StanceNode& start,
           const StancePlacements& inside,
           const std::vector<Eigen::VectorXd>& knots,
           const StancePlacements& endStance,
           bool breaksAtStart) {
    MotionSegment segment;
    std::vector<StancePlacements> stances;
    segment.knots.push_back(MotionKnot{
        .q = start.q, .mask = breaksAtStart ? maskOf(inside) : lastMask_});
    stances.push_back(breaksAtStart ? inside : lastStance_);
    for (size_t k = 0; k + 1 < knots.size(); ++k) {
      segment.knots.push_back(
          MotionKnot{.q = knots[k], .mask = maskOf(inside)});
      stances.push_back(inside);
    }
    segment.knots.push_back(
        MotionKnot{.q = knots.back(), .mask = maskOf(endStance)});
    stances.push_back(endStance);
    lastMask_ = maskOf(endStance);
    lastStance_ = endStance;
    motion_.segments.push_back(std::move(segment));
    motion_.stances.push_back(std::move(stances));
  }

  PathMotion take() { return std::move(motion_); }

 private:
  PathMotion motion_;
  ContactMask lastMask_ = 0;
  StancePlacements lastStance_;
};

/**
 * The motion of the path through `path` (node indices), with the masks of
 * design section 2's condition 3: a contact is made at the last knot of the
 * motion that brings it (the motion is inside the smaller stance), and broken
 * after the dwell at the end of the motion before the one that leaves it.
 */
PathMotion pathMotion(const std::vector<StanceNode>& nodes,
                      const std::vector<int>& path,
                      ContactBreak contactBreak) {
  PathMotionBuilder builder(nodes[path.front()]);
  for (size_t i = 1; i < path.size(); ++i) {
    const StanceNode& from = nodes[path[i - 1]];
    const StanceNode& to = nodes[path[i]];
    builder.add(from, from.stance, to.motion,
                to.change == Change::kAdd ? to.stance : from.stance,
                breaksAtDwellEnd(from, contactBreak));
  }
  const StanceNode& goal = nodes[path.back()];
  builder.add(goal, goal.stance, goal.finalMotion, goal.stance,
              breaksAtDwellEnd(goal, contactBreak));
  return builder.take();
}

/** The phases of the plan's contacts, from its knots and their stances: each
 * from the knot that makes the contact to the one that breaks it (or the
 * last). */
std::vector<config::MultiContactPlan::ContactPhase> phasesOf(
    const std::vector<config::MultiContactPlan::Knot>& knots,
    const std::vector<StancePlacements>& stances,
    const ContactSet& contactSet,
    const TerrainModel& terrain) {
  std::vector<config::MultiContactPlan::ContactPhase> phases;
  std::vector<int> open(contactSet.size(), -1);
  for (size_t k = 0; k < knots.size(); ++k) {
    for (int candidate = 0; candidate < contactSet.size(); ++candidate) {
      const bool active = isContactActive(knots[k].active_contacts, candidate);
      if (active && open[candidate] < 0) {
        const std::optional<ContactPlacement> placement =
            placementOf(stances[k], candidate);
        config::MultiContactPlan::ContactPhase phase;
        phase.contact = contactSet.candidate(candidate).name;
        phase.start_time = knots[k].time;
        phase.end_time = knots[k].time;
        if (placement.has_value()) {
          phase.primitive = terrain.primitive(placement->primitive).name;
          phase.face = placement->face.has_value()
                           ? static_cast<int>(*placement->face)
                           : 0;
          Eigen::VectorXd pose(kPoseTangentDim);
          pose << placement->position, placement->normal;
          phase.target_pad_pose = pose;
        }
        open[candidate] = static_cast<int>(phases.size());
        phases.push_back(std::move(phase));
      }
      // A phase lasts to the knot that breaks it, where the MPC's mode
      // schedule switches (PlannedMotion's events are the knots of the new
      // stance): ending it a knot earlier left the contact in contact with no
      // phase of its own in between, its patch the next phase's.
      if (active || open[candidate] >= 0) {
        phases[open[candidate]].end_time = knots[k].time;
      }
      if (!active) open[candidate] = -1;
    }
  }
  return phases;
}

/** The first failing knots of `report`, for messages. */
std::string firstFailure(const config::CertifierReport& report) {
  // How many failing knots a message names.
  constexpr int kNamedFailures = 3;
  std::vector<std::string> failures;
  for (size_t k = 0; k < report.knots.size(); ++k) {
    const config::CertifierReport::Knot& knot = report.knots[k];
    if (knot.passed) continue;
    failures.push_back(
        absl::StrFormat("knot %d, t = %.3f s [%s]: %s", static_cast<int>(k),
                        knot.time, absl::StrJoin(knot.active_contacts, ", "),
                        absl::StrJoin(knot.failures, "; ")));
    if (static_cast<int>(failures.size()) >= kNamedFailures) break;
  }
  return failures.empty() ? "no knot fails" : absl::StrJoin(failures, " | ");
}

/** Whether every failing knot of `report` fails condition 2 only. */
bool onlyDynamicFailures(const config::CertifierReport& report) {
  for (const config::CertifierReport::Knot& knot : report.knots) {
    if (knot.passed) continue;
    if (knot.static_condition.value_or(true) == false ||
        knot.transition_condition.value_or(true) == false ||
        knot.collision_free.value_or(true) == false ||
        knot.momentum_rate_feasible) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace {

/** The nodes of a path through the stance graph, in order, from the start to
 * the goal, and what finding it took. */
struct FoundPath {
  std::vector<StanceNode> nodes;
  int validations = 0;
  int expansions = 0;
};

/** "the goal" or "waypoint 2 of 5", for messages. */
std::string legName(size_t leg, size_t legs) {
  return leg + 1 == legs
             ? std::string("the goal")
             : absl::StrFormat("waypoint %d of %d", static_cast<int>(leg + 1),
                               static_cast<int>(legs - 1));
}

/**
 * The path from the initial stance of `context` through the scenario's
 * waypoints to its goal: one lazy weighted A* per leg, each from where the last
 * ended, within what is left of the search's budgets. NotFound names the leg,
 * the stance nearest its goal and why its last changes failed.
 */
absl::StatusOr<FoundPath> findPath(PlanningContext& context,
                                   StanceNode start,
                                   const LazySearchOptions& searchOptions) {
  const LoadedScenario& loaded = *context.loaded;
  std::vector<const config::Scenario::Goal* absl_nonnull> legs;
  for (const config::Scenario::Goal& waypoint : loaded.scenario.waypoints) {
    legs.push_back(&waypoint);
  }
  legs.push_back(&loaded.scenario.goal);
  const absl::Time deadline = searchOptions.timeBudget.has_value()
                                  ? absl::Now() + *searchOptions.timeBudget
                                  : absl::InfiniteFuture();
  FoundPath found;
  found.nodes.push_back(std::move(start));
  for (size_t leg = 0; leg < legs.size(); ++leg) {
    RETURN_IF_ERROR(setLegGoal(context,
                               /*previous=*/leg > 0 ? legs[leg - 1] : nullptr,
                               *legs[leg],
                               /*finalLeg=*/leg + 1 == legs.size()));
    StanceGraphSearch search(context);
    const int startNode = search.addStart(found.nodes.back());
    LazySearchOptions options = searchOptions;
    options.maxValidations =
        std::max(0, searchOptions.maxValidations - found.validations);
    if (searchOptions.timeBudget.has_value()) {
      options.timeBudget =
          std::max(absl::ZeroDuration(), deadline - absl::Now());
    }
    ASSIGN_OR_RETURN(const LazySearchResult result,
                     lazyWeightedAStar(search, startNode, options));
    found.validations += result.validations;
    found.expansions += result.expansions;
    if (!result.found) {
      const int nearest = search.nearestToGoal();
      const StanceNode& node = search.nodes()[nearest];
      return absl::NotFoundError(absl::StrFormat(
          "stance_graph: no plan for '%s' (stopped on the way to %s: %s, %d "
          "validations, %d expansions); the stance nearest it is %s, whose "
          "changes failed: %s",
          loaded.scenario.name, legName(leg, legs.size()), result.stopReason,
          found.validations, found.expansions,
          describe(node.stance, loaded.contactSet, loaded.terrain),
          node.failures.empty() ? "none tried"
                                : absl::StrJoin(node.failures, " | ")));
    }
    if (legs.size() > 1) {
      LOG(INFO) << absl::StrFormat(
          "stance_graph: '%s' reached %s in %d changes (%d validations so "
          "far): %s",
          loaded.scenario.name, legName(leg, legs.size()),
          static_cast<int>(result.path.size()) - 1, found.validations,
          describe(search.nodes()[result.path.back()].stance, loaded.contactSet,
                   loaded.terrain));
    }
    for (size_t i = 1; i < result.path.size(); ++i) {
      found.nodes.push_back(search.nodes()[result.path[i]]);
    }
    // A goal reached at the start of its search keeps the motion to its pose.
    if (result.path.size() == 1) {
      found.nodes.back().finalMotion =
          search.nodes()[result.path.front()].finalMotion;
    }
  }
  return found;
}

/** `slowdowns`, one per segment, after a retiming of a plan whose `report`
 * fails condition 2 alone (StanceGraphPlannerOptions::retimingScope): every
 * entry, or those of the failing knots' segments (`segmentOfKnot`, and the
 * segment before for a knot that is `firstOfSegment`), times the options'
 * retiming slowdown. */
std::vector<double> retimed(std::vector<double> slowdowns,
                            const config::CertifierReport& report,
                            const std::vector<int>& segmentOfKnot,
                            const std::vector<bool>& firstOfSegment,
                            const StanceGraphPlannerOptions& options) {
  std::vector<bool> slower(slowdowns.size(),
                           options.retimingScope == RetimingScope::kWholePlan);
  for (size_t k = 0; k < report.knots.size(); ++k) {
    if (report.knots[k].passed) continue;
    const int segment = segmentOfKnot[k];
    slower[segment] = true;
    if (firstOfSegment[k] && segment > 0) slower[segment - 1] = true;
  }
  for (size_t s = 0; s < slowdowns.size(); ++s) {
    if (slower[s]) slowdowns[s] *= options.retimingSlowdown;
  }
  return slowdowns;
}

/** Why the motion between two consecutive knots of `configurations` sweeps a
 * collision sphere into the terrain on the stance of the first
 * (StanceMotion::sweptClearance()), or std::nullopt when none does: the
 * segments check their steps, and this checks that the plan's knots are those
 * steps. */
absl::StatusOr<std::optional<std::string>> sweptCollision(
    StanceMotion& motion,
    const std::vector<Eigen::VectorXd>& configurations,
    const std::vector<StancePlacements>& stances) {
  for (size_t k = 0; k + 1 < configurations.size(); ++k) {
    ASSIGN_OR_RETURN(const std::optional<std::string> sweep,
                     motion.sweptClearance(configurations[k],
                                           configurations[k + 1], stances[k]));
    if (sweep.has_value()) {
      return std::optional<std::string>(
          absl::StrFormat("after knot %d: %s", static_cast<int>(k), *sweep));
    }
  }
  return std::optional<std::string>();
}

}  // namespace

absl::StatusOr<config::MultiContactPlan> StanceGraphPlanner::plan(
    const LoadedScenario& loaded) {
  const absl::Time started = absl::Now();
  ASSIGN_OR_RETURN(PlanningContext context, contextOf(loaded, options_));
  ASSIGN_OR_RETURN(StancePlacements start, initialStance(context));
  ASSIGN_OR_RETURN(const FoundPath found,
                   findPath(context,
                            StanceNode{.stance = std::move(start),
                                       .q = context.initialQ,
                                       .validated = true},
                            options_.search));
  const ContactSet& contactSet = loaded.contactSet;
  std::vector<int> path(found.nodes.size());
  for (size_t i = 0; i < path.size(); ++i) path[i] = static_cast<int>(i);
  PathMotion motion = pathMotion(found.nodes, path, options_.contactBreak);
  // timedPlanKnots() emits every knot of every segment, then a final hold.
  std::vector<StancePlacements> knotStances;
  std::vector<Eigen::VectorXd> configurations;
  for (size_t s = 0; s < motion.segments.size(); ++s) {
    for (size_t k = 0; k < motion.segments[s].knots.size(); ++k) {
      knotStances.push_back(motion.stances[s][k]);
      configurations.push_back(motion.segments[s].knots[k].q);
    }
  }
  knotStances.push_back(knotStances.back());
  configurations.push_back(configurations.back());
  std::vector<int> segmentOfKnot;
  std::vector<bool> firstOfSegment;
  for (size_t s = 0; s < motion.segments.size(); ++s) {
    for (size_t k = 0; k < motion.segments[s].knots.size(); ++k) {
      segmentOfKnot.push_back(static_cast<int>(s));
      firstOfSegment.push_back(k == 0);
    }
  }
  segmentOfKnot.push_back(segmentOfKnot.back());
  firstOfSegment.push_back(false);
  ASSIGN_OR_RETURN(
      const std::optional<std::string> sweep,
      sweptCollision(*context.motion, configurations, knotStances));
  if (sweep.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "stance_graph: the plan for '%s' sweeps through the terrain %s",
        loaded.scenario.name, *sweep));
  }
  ASSIGN_OR_RETURN(std::unique_ptr<StabilityCertifier> certifier,
                   makeCertifier(loaded));
  RobotStanceModel& model = context.motion->model();
  config::MultiContactPlan plan;
  plan.scenario = loaded.scenario.name;
  plan.planner = std::string(kStanceGraphPlannerName);
  for (const ActuatedJoint& joint : model.actuatedJoints()) {
    plan.joint_names.push_back(joint.name);
  }
  stampPlanHashes(loaded.contactSetConfig, loaded.scenario.terrain_scene, plan);
  std::vector<double> slowdowns(motion.segments.size(), 1.0);
  for (int retiming = 0; retiming <= options_.maxRetimings; ++retiming) {
    ASSIGN_OR_RETURN(plan.knots, timedPlanKnots(model, motion.segments,
                                                options_.timing, slowdowns));
    if (retiming == 0 && plan.knots.back().time < loaded.scenario.duration) {
      // Stretch the plan to the scenario's duration.
      const double stretch = loaded.scenario.duration / plan.knots.back().time;
      for (double& slowdown : slowdowns) slowdown = stretch;
      ASSIGN_OR_RETURN(plan.knots, timedPlanKnots(model, motion.segments,
                                                  options_.timing, slowdowns));
    }
    for (size_t k = 0; k < plan.knots.size(); ++k) {
      ASSIGN_OR_RETURN(plan.knots[k].contact_wrenches,
                       contactWrenches(model, loaded.terrain, plan.knots[k],
                                       configurations[k],
                                       context.motion->options().equilibrium));
    }
    plan.phases = phasesOf(plan.knots, knotStances, contactSet, loaded.terrain);
    ASSIGN_OR_RETURN(plan.report, certifier->certify(plan));
    if (plan.report->passed) break;
    if (!onlyDynamicFailures(*plan.report) ||
        retiming == options_.maxRetimings) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "stance_graph: the plan for '%s' fails the certifier at %d of %d "
          "knots; the first: %s",
          loaded.scenario.name, plan.report->failed_knots,
          static_cast<int>(plan.knots.size()), firstFailure(*plan.report)));
    }
    slowdowns = retimed(slowdowns, *plan.report, segmentOfKnot, firstOfSegment,
                        options_);
  }
  LOG(INFO) << absl::StrFormat(
      "stance_graph: '%s' planned in %.1f s: %d stances, %d knots, %.1f s of "
      "motion (%d validations, %d expansions)",
      loaded.scenario.name, absl::ToDoubleSeconds(absl::Now() - started),
      static_cast<int>(path.size()), static_cast<int>(plan.knots.size()),
      plan.knots.back().time, found.validations, found.expansions);
  return plan;
}

}  // namespace ocs2::humanoid::multi_contact
