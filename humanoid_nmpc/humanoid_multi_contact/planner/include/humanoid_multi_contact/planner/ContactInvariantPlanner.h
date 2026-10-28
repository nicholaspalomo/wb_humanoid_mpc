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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/MultiContactPlanner.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantContinuation.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantProblem.h"
#include "humanoid_multi_contact/planner/contact_invariant/HardContactRefinement.h"
#include "humanoid_multi_contact/planner/contact_invariant/RobotContactFeatures.h"
#include "humanoid_multi_contact/planner/contact_invariant/StanceThresholding.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

/** The contact-invariant planner's registry name (kPlannerNames). */
inline constexpr absl::string_view kContactInvariantPlannerName =
    "contact_invariant";

/** The defaults of ContactInvariantPlannerOptions; its fields say what each
 * is. */
inline constexpr double kDefaultGuessActivation = 0.5;
inline constexpr double kDefaultGuessTrackingScale = 0.01;
inline constexpr double kDefaultFrictionShare = 0.9;
inline constexpr double kDefaultConeSmoothing = 0.01;
inline constexpr double kDefaultMaxPointTilt = 0.5236;  // [rad], 30 degrees
inline constexpr double kDefaultContinuationDt = 0.05;
inline constexpr double kDefaultContactDistance = 0.02;

/** The settings of ContactInvariantPlanner. */
struct ContactInvariantPlannerOptions {
  ContactInvariantPlannerOptions();

  // The solves before the hard-contact one (design section 4.2,
  // "continuation"): solve 1 with the forces almost free, solve 2 with the
  // force gate and the contact-invariant weight raised.
  std::vector<ContinuationStage> stages;
  ContactInvariantBarriers barriers;
  // The friction of the point candidates' cones: this share of the scene's
  // least friction coefficient, inscribed in the certifier's eight-edged
  // pyramids (times cos(pi / 8)); and the cones' smoothing (in units of the
  // weight) and the free tilt of a point pad (design section 3.1).
  double frictionShare = kDefaultFrictionShare;
  double coneSmoothing = kDefaultConeSmoothing;
  double maxPointTilt = kDefaultMaxPointTilt;
  // The guess of the contact variables between the first and the last node,
  // and the share of the task file's state weights that pull solves 1 and 2
  // toward the guess (the posture regularization; the terminal cost, the
  // task, keeps its weights).
  double guessActivation = kDefaultGuessActivation;
  double guessTrackingScale = kDefaultGuessTrackingScale;
  ThresholdingOptions thresholding;
  // [m] A candidate whose pad is farther than this from the terrain is not
  // in contact whatever its c: the thresholding reads c only where the
  // contact-invariant cost has brought the pad to a surface.
  double contactDistance = kDefaultContactDistance;
  // [s] The node spacing of solves 1 and 2.
  double continuationDt = kDefaultContinuationDt;
  // Solve 3.
  RefinementOptions refinement;
};

/**
 * Planner A of the design (section 4): contact-invariant optimization
 * (Mordatch, Todorov and Popovic 2012) on OCS2's SQP and the centroidal MPC's
 * model (the scenario's mpc_task_file), over the scenario's duration.
 *
 *  1. The guess: the base and the joints interpolated from the initial pose
 *     to the goal pose, every allowed candidate's contact variable c at the
 *     guess's activation (its initial stance at the first node).
 *  2. Solves 1 and 2 (solveContinuation()): the MPC's problem with every
 *     allowed candidate in contact - its dynamics, posture and goal costs,
 *     joint limits, the surface candidates' wrench cones and the zero wrench
 *     of the candidates the scenario does not allow - augmented with c and its
 *     rate (augmentProblem()): the contact-invariant cost, the force gate, the
 *     quasi-static terms, the point candidates' friction cones in the blended
 *     patch frame and the pads' and the collision spheres' clearance.
 *  3. The thresholding of c into a stance sequence (thresholdActivations()),
 *     and solve 3, the hard-contact problem of that sequence with the torque
 *     limits and, with a static margin, the support polygons
 *     (HardContactRefinement).
 *
 * The plan is stamped with the scenario's hashes and certified.
 * FailedPrecondition naming the first failing knots when the certifier refuses
 * it, NotFound saying what of the goal is not reached, Internal when a solve
 * fails.
 *
 * Not thread-safe.
 */
class ContactInvariantPlanner final : public MultiContactPlanner {
 public:
  /** InvalidArgument for inconsistent options. */
  static absl::StatusOr<std::unique_ptr<ContactInvariantPlanner>> Create(
      const ContactInvariantPlannerOptions& options);

  ContactInvariantPlanner(const ContactInvariantPlanner&) = delete;
  ContactInvariantPlanner& operator=(const ContactInvariantPlanner&) = delete;
  ~ContactInvariantPlanner() override = default;

  absl::StatusOr<config::MultiContactPlan> plan(
      const LoadedScenario& loaded) override;
  absl::string_view name() const override {
    return kContactInvariantPlannerName;
  }

  const ContactInvariantPlannerOptions& options() const { return options_; }

 private:
  explicit ContactInvariantPlanner(
      const ContactInvariantPlannerOptions& options);

  ContactInvariantPlannerOptions options_;
};

/**
 * Solve 3 on `seed` and the checks every plan of planner A ends with: the
 * plan of `refinement` from the warm start `guess`, its goal reached
 * (checkGoalReached()) and its certificate. `planner` names the planner in
 * the messages.
 */
absl::StatusOr<config::MultiContactPlan> refinedPlan(
    HardContactRefinement& refinement,
    const config::MultiContactPlan& seed,
    const MpcTrajectory& guess,
    const LoadedScenario& loaded,
    absl::string_view planner);

}  // namespace ocs2::humanoid::multi_contact
