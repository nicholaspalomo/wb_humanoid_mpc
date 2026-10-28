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

// The timing law: the minimum-jerk profile and its inverse, the duration of a
// segment from its speed and acceleration limits, and the plan's knots: the
// dwells, the rests at the segments' ends and the momenta.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/PlanTiming.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasContactSet[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "contact_set.textproto";

std::unique_ptr<RobotStanceModel> atlasModel() {
  const absl::StatusOr<std::string> contactSetPath =
      robot::resolveResourcePath(kAtlasContactSet);
  EXPECT_TRUE(contactSetPath.ok()) << contactSetPath.status();
  absl::StatusOr<config::ContactSet> config =
      loadContactSetFile(*contactSetPath);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<ContactSet> contactSet = ContactSet::Create(*config);
  EXPECT_TRUE(contactSet.ok()) << contactSet.status();
  const absl::StatusOr<std::string> urdf =
      robot::resolveResourcePath(kAtlasUrdf);
  EXPECT_TRUE(urdf.ok()) << urdf.status();
  absl::StatusOr<std::unique_ptr<RobotStanceModel>> model =
      RobotStanceModel::Create(*urdf, *contactSet);
  EXPECT_TRUE(model.ok()) << model.status();
  return *std::move(model);
}

TEST(PlanTiming, MinimumJerkProfile) {
  EXPECT_DOUBLE_EQ(minimumJerk(0.0), 0.0);
  EXPECT_DOUBLE_EQ(minimumJerk(1.0), 1.0);
  EXPECT_DOUBLE_EQ(minimumJerk(0.5), 0.5);
  double previous = 0.0;
  for (int i = 1; i <= 100; ++i) {
    const double tau = i / 100.0;
    EXPECT_GT(minimumJerk(tau), previous);
    previous = minimumJerk(tau);
    EXPECT_NEAR(minimumJerk(inverseMinimumJerk(previous)), previous, 1.0e-12);
  }
  // At rest at both ends: the slope of a chord near each end vanishes.
  EXPECT_LT(minimumJerk(1.0e-3) / 1.0e-3, 1.0e-4);
  EXPECT_LT((1.0 - minimumJerk(1.0 - 1.0e-3)) / 1.0e-3, 1.0e-4);
}

/** A segment that lowers the base by `drop` [m] in `knots` equal steps. */
MotionSegment lowering(const RobotStanceModel& model, double drop, int knots) {
  MotionSegment segment;
  for (int k = 0; k <= knots; ++k) {
    absl::StatusOr<Eigen::VectorXd> neutral =
        model.generalizedPositionOf(RobotConfiguration());
    EXPECT_TRUE(neutral.ok()) << neutral.status();
    Eigen::VectorXd q = *neutral;
    q.z() = 0.9 - drop * k / knots;
    segment.knots.push_back(MotionKnot{.q = q, .mask = 0});
  }
  return segment;
}

TEST(PlanTiming, SegmentDurationFollowsTheLimits) {
  std::unique_ptr<RobotStanceModel> model = atlasModel();
  TimingOptions options;
  const MotionSegment segment = lowering(*model, /*drop=*/0.3, /*knots=*/6);
  const absl::StatusOr<std::vector<double>> times =
      segmentKnotTimes(*model, segment, options, /*slowdown=*/1.0);
  ASSERT_TRUE(times.ok()) << times.status();
  ASSERT_EQ(times->size(), 7U);
  EXPECT_DOUBLE_EQ(times->front(), 0.0);
  for (size_t k = 1; k < times->size(); ++k) {
    EXPECT_GT((*times)[k], (*times)[k - 1]);
  }
  // The CoM and the base move 0.3 m: the speed limits need 1.875 * 0.3 /
  // min(0.15, 0.2) s, the acceleration limit sqrt(5.774 * 0.3 / 0.15) s.
  const double expected =
      std::max(1.875 * 0.3 / options.maxComSpeed,
               std::sqrt(5.773502691896258 * 0.3 / options.maxComAcceleration));
  EXPECT_NEAR(times->back(), expected, 1.0e-6);
  // Twice as slow: twice as long.
  const absl::StatusOr<std::vector<double>> slower =
      segmentKnotTimes(*model, segment, options, /*slowdown=*/2.0);
  ASSERT_TRUE(slower.ok()) << slower.status();
  EXPECT_NEAR(slower->back(), 2.0 * expected, 1.0e-6);
}

TEST(PlanTiming, PlanKnotsRestAndDwell) {
  std::unique_ptr<RobotStanceModel> model = atlasModel();
  TimingOptions options;
  MotionSegment down = lowering(*model, /*drop=*/0.2, /*knots=*/4);
  MotionSegment up = down;
  std::reverse(up.knots.begin(), up.knots.end());
  const absl::StatusOr<std::vector<config::MultiContactPlan::Knot>> knots =
      timedPlanKnots(*model, {down, up}, options, /*slowdowns=*/{1.0, 1.0});
  ASSERT_TRUE(knots.ok()) << knots.status();
  // Both segments' knots and a final hold.
  ASSERT_EQ(knots->size(), 5U + 5U + 1U);
  for (size_t k = 1; k < knots->size(); ++k) {
    EXPECT_GT((*knots)[k].time, (*knots)[k - 1].time);
  }
  // The second segment starts a dwell after the first ends, at rest.
  EXPECT_NEAR((*knots)[5].time - (*knots)[4].time, options.dwell, 1.0e-9);
  for (const int rest : {0, 4, 5, 9, 10}) {
    EXPECT_EQ((*knots)[rest].linear_momentum.norm(), 0.0) << rest;
    EXPECT_EQ((*knots)[rest].joint_velocities.norm(), 0.0) << rest;
  }
  // In the middle of the descent, the momentum is m times the base's speed
  // (the joints are still), downward.
  const config::MultiContactPlan::Knot& middle = (*knots)[2];
  const double speed = 0.1 / ((*knots)[3].time - (*knots)[1].time);
  EXPECT_NEAR(middle.linear_momentum.z(), -model->mass() * speed, 1.0e-6);
  EXPECT_EQ(middle.joint_positions.size(),
            static_cast<Eigen::Index>(model->actuatedJoints().size()));
}

TEST(PlanTiming, EachSegmentHasItsOwnSlowdown) {
  std::unique_ptr<RobotStanceModel> model = atlasModel();
  TimingOptions options;
  MotionSegment down = lowering(*model, /*drop=*/0.2, /*knots=*/4);
  MotionSegment up = down;
  std::reverse(up.knots.begin(), up.knots.end());
  const absl::StatusOr<std::vector<config::MultiContactPlan::Knot>> even =
      timedPlanKnots(*model, {down, up}, options, /*slowdowns=*/{1.0, 1.0});
  const absl::StatusOr<std::vector<config::MultiContactPlan::Knot>> slower =
      timedPlanKnots(*model, {down, up}, options, /*slowdowns=*/{1.0, 2.0});
  ASSERT_TRUE(even.ok()) << even.status();
  ASSERT_TRUE(slower.ok()) << slower.status();
  // The first segment is as it was; the second, its dwell before it and the
  // final hold take twice as long.
  EXPECT_DOUBLE_EQ((*slower)[4].time, (*even)[4].time);
  EXPECT_NEAR((*slower)[5].time - (*slower)[4].time, 2.0 * options.dwell,
              1.0e-9);
  EXPECT_NEAR((*slower)[9].time - (*slower)[5].time,
              2.0 * ((*even)[9].time - (*even)[5].time), 1.0e-9);
  EXPECT_NEAR((*slower)[10].time - (*slower)[9].time, 2.0 * options.dwell,
              1.0e-9);
  EXPECT_EQ(timedPlanKnots(*model, {down, up}, options, /*slowdowns=*/{1.0})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
