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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "support/CentroidalMpcLoop.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * A turn in place through 720 degrees on the quaternion layout (design README section 5(c)): the shipped EngineAI SA01's
 * centroidal MPC (SQP on the state manifold, its manifold feedback policy), with the yaw row of its base-orientation
 * weights weighed, in a lockstep loop whose plant is the projected RK4 rollout of the policy (support/CentroidalMpcLoop.h;
 * no simulator), commanded to turn at 1 rad/s until it has turned 4 pi. Every solve succeeds; the rotation part of the
 * initial state gap stays below 0.1 rad; no solve's cost or dynamics violation jumps beyond three times the median of the
 * turn; the heading the robot turns through grows monotonically across +-pi, twice round; and the feedback correction of
 * the policy stays bounded. On the Euler layout the heading's wrap at +-pi and the pitch singularity were the places this
 * could break. Generates the SA01's CppAD libraries on a cold cache.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kYawRate = 1.0;           // [rad/s]
constexpr scalar_t kStandingTime = 1.0;      // [s] before the command
constexpr scalar_t kTurn = 4.0 * M_PI;       // [rad]
constexpr scalar_t kLongestTurnTime = 25.0;  // [s] the turn must be done by
// The yaw weight of the base orientation, about half the shipped roll weight: the shipped SA01 weighs no yaw and does
// not follow a turn in place (CentroidalMpcLoopOptions); with the heading-tilt residual's yaw row weighed it does.
constexpr scalar_t kBaseYawWeight = 40.0;
// The sum of squared shooting gaps of a solve that counts as converged.
constexpr scalar_t kConvergedViolation = 1.0e-6;
// [rad] How far the heading may go back between two solves (the gait sways it): monotone up to that.
constexpr scalar_t kHeadingWobble = 1.0e-3;

/** The median of `values` (copied). */
scalar_t median(std::vector<scalar_t> values) {
  if (values.empty()) return 0.0;
  const std::vector<scalar_t>::iterator middle = values.begin() + static_cast<ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

TEST(TurnInPlace720, TheMpcTurnsTwiceRoundThroughPiWithoutAJumpOfItsSolves) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> created =
      CentroidalMpcLoop::Create(ShippedCentroidalRobot::kEngineAiSa01, CentroidalMpcLoopOptions{.baseYawWeight = kBaseYawWeight});
  ASSERT_TRUE(created.ok()) << created.status();
  CentroidalMpcLoop& loop = **created;
  const MpcRobotModelBase<scalar_t>& model = loop.interface().getMpcRobotModel();

  // Standing, then turning; the heading unwrapped by the steps between solves, each far below pi.
  scalar_t previousHeading = model.getBaseHeading(loop.observation().state);
  scalar_t turned = 0.0;
  scalar_t largestGapRotation = 0.0;
  scalar_t largestFeedbackShare = 0.0;
  scalar_t largestHeadingDecrease = 0.0;
  std::vector<scalar_t> costs;
  std::vector<scalar_t> dynamicsViolations;
  int crossingsOfPi = 0;
  loop.command(/*forward=*/0.0, /*lateral=*/0.0, /*yawRate=*/0.0);
  bool turning = false;
  std::string gaits;  // the gait at every second, for a failure's message
  while (turned < kTurn) {
    const scalar_t time = loop.observation().time;
    if (std::floor(time) != std::floor(time - loop.period())) {
      // The gait, the heading turned so far, and the headings the reference and the plan end the horizon at.
      const TargetTrajectories& targets = loop.interface().getSwitchedModelReferenceManagerPtr()->getTargetTrajectories();
      const scalar_t targetHeading = targets.empty() ? 0.0 : model.getBaseHeading(targets.stateTrajectory.back());
      const scalar_t plannedHeading =
          loop.hasPolicy() && !loop.policy().stateTrajectory_.empty() ? model.getBaseHeading(loop.policy().stateTrajectory_.back()) : 0.0;
      absl::StrAppend(&gaits, " t=", std::floor(time), ":", loop.motionManager().getCurrentGaitCommand(), " turned ", turned, " heading ",
                      model.getBaseHeading(loop.observation().state), " target ", targetHeading, " plan ", plannedHeading, ";");
    }
    ASSERT_LT(time, kStandingTime + kLongestTurnTime) << "the robot turned only " << turned << " rad; gaits:" << gaits;
    if (!turning && time >= kStandingTime) {
      loop.command(/*forward=*/0.0, /*lateral=*/0.0, kYawRate);
      turning = true;
    }
    // A small push of the momentum after every solve, so that the plant leaves the plan and the feedback acts.
    loop.push((vector6_t() << 0.002 * std::sin(1.7 * time), 0.002 * std::cos(1.3 * time), 0.0, 0.0005 * std::sin(0.9 * time),
               0.0005 * std::cos(0.7 * time), 0.0)
                  .finished());
    const absl::StatusOr<CentroidalMpcLoopStep> step = loop.step();
    ASSERT_TRUE(step.ok()) << "t = " << time << ": " << step.status();
    largestGapRotation = std::max(largestGapRotation, step->initialGapRotation);
    largestFeedbackShare = std::max(largestFeedbackShare, step->feedbackCorrection / std::max(1.0, step->feedforwardMagnitude));
    const scalar_t heading = model.getBaseHeading(loop.observation().state);
    const scalar_t change = std::remainder(heading - previousHeading, 2.0 * M_PI);
    if (std::abs(heading - previousHeading) > M_PI) ++crossingsOfPi;
    previousHeading = heading;
    if (!turning) continue;
    turned += change;
    largestHeadingDecrease = std::max(largestHeadingDecrease, -change);
    costs.push_back(step->performance.cost);
    dynamicsViolations.push_back(step->performance.dynamicsViolationSSE);
  }
  ASSERT_FALSE(costs.empty());

  EXPECT_LT(largestGapRotation, 0.1) << "the rotation part of the initial state gap";
  const scalar_t medianCost = median(costs);
  const scalar_t medianViolation = median(dynamicsViolations);
  EXPECT_LE(*std::max_element(costs.begin(), costs.end()), 3.0 * medianCost) << "a solve's cost jumped";
  // The dynamics violation after a solve's iteration is about 1e-10 on most solves, below which three times the median
  // is noise: a violation below kConvergedViolation counts as none.
  EXPECT_LE(*std::max_element(dynamicsViolations.begin(), dynamicsViolations.end()), std::max(3.0 * medianViolation, kConvergedViolation))
      << "a solve's dynamics violation jumped";
  EXPECT_LE(largestHeadingDecrease, kHeadingWobble) << "the heading went back by " << largestHeadingDecrease << " rad between two solves";
  EXPECT_GE(crossingsOfPi, 2) << "twice round crosses +-pi twice";
  EXPECT_LT(largestFeedbackShare, 0.5) << "the feedback correction of the policy, relative to its feedforward input";
  EXPECT_GT(largestFeedbackShare, 1.0e-6) << "the push must make the feedback act";
  ::testing::Test::RecordProperty("largest_initial_gap_rotation", absl::StrCat(largestGapRotation));
  ::testing::Test::RecordProperty("largest_feedback_share", absl::StrCat(largestFeedbackShare));
  ::testing::Test::RecordProperty("largest_heading_decrease", absl::StrCat(largestHeadingDecrease));
  ::testing::Test::RecordProperty("cost_max_over_median",
                                  absl::StrCat(*std::max_element(costs.begin(), costs.end()) / std::max(medianCost, 1.0e-300)));
  ::testing::Test::RecordProperty("dynamics_violation_median", absl::StrCat(medianViolation));
  ::testing::Test::RecordProperty("dynamics_violation_max",
                                  absl::StrCat(*std::max_element(dynamicsViolations.begin(), dynamicsViolations.end())));
  ::testing::Test::RecordProperty("turn_time", absl::StrCat(loop.observation().time - kStandingTime));
}

}  // namespace
}  // namespace ocs2::humanoid
