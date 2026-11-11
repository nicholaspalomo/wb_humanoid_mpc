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
#include <memory>
#include <numbers>
#include <string>

#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "support/CentroidalMpcLoop.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * The shipped ANYbotics ANYmal B, the first shipped robot on point contacts,
 * on its centroidal MPC in the lockstep loop of support/CentroidalMpcLoop.h
 * with the quadruped gait file: standing, trotting forward and turning in
 * place, every solve succeeds and is finite, every planned and rolled-out
 * quaternion stays unit, the gait switching table of the gait file picks the
 * trot for the walk, and the plant goes forward and turns the way it is told.
 * The plant is the projected rollout of the policy, so this checks the
 * formulation and its configuration, not the robot's balance in a simulator.
 * Generates ANYmal B's CppAD libraries on a cold cache.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStandUntil = 2.0;                        // [s]
constexpr scalar_t kWalkUntil = 7.0;                         // [s]
constexpr scalar_t kTurnUntil = 11.0;                        // [s]
constexpr scalar_t kCmdLinvelBaseRtWorldEwrtHeadingX = 0.3;  // [m/s]
constexpr scalar_t kCmdRateYawBaseRtWorld = 0.4;             // [rad/s]
// The quaternion norm the projected rollout keeps (testUnitNorm's bound).
constexpr scalar_t kMaxNormError = 1.0e-12;
// [m] Half of the 1.5 m the walk asks for: the gait starts from rest, from
// stance straight to the trot, and the run with the shipped gait table went
// 1.18 m (0.75 m when static_walk came between stance and trot).
constexpr scalar_t kMinWalkDistance = 0.75;
// [rad] Half of the 1.6 rad the turn asks for; the run turned 1.39 rad.
constexpr scalar_t kMinTurnAngle = 0.8;
constexpr char kTrotGait[] = "trot";

/** The gaits the motion manager ran over a phase, in name order. */
using GaitNames = absl::btree_set<std::string>;

/** What a phase of the lockstep loop ran. */
struct PhaseResult {
  // The gaits the motion manager ran, in name order.
  GaitNames gaits;
  // The largest quaternion norm error of the phase's steps.
  scalar_t largestNormError = 0.0;
};

/**
 * Runs `loop` under the command (`cmdLinvelBaseRtWorldEwrtHeadingX`,
 * `cmdRateYawBaseRtWorld`) until the observation's time reaches `until`,
 * failing the calling test on a failed or non-finite solve. Returns the gaits
 * the motion manager ran and the steps' largest quaternion norm error, of the
 * steps up to the failure when one fails.
 */
PhaseResult runPhase(CentroidalMpcLoop& loop,
                     scalar_t until,
                     scalar_t cmdLinvelBaseRtWorldEwrtHeadingX,
                     scalar_t cmdRateYawBaseRtWorld) {
  PhaseResult result;
  loop.command(cmdLinvelBaseRtWorldEwrtHeadingX,
               /*cmdLinvelBaseRtWorldEwrtHeadingY=*/0.0, cmdRateYawBaseRtWorld);
  while (loop.observation().time < until) {
    const scalar_t time = loop.observation().time;
    const absl::StatusOr<CentroidalMpcLoopStep> step = loop.step();
    if (!step.ok()) {
      ADD_FAILURE() << "t = " << time << ": " << step.status();
      return result;
    }
    if (!std::isfinite(step->performance.merit) ||
        !loop.observation().state.allFinite()) {
      ADD_FAILURE() << "t = " << time << ": the solve or the plant diverged";
      return result;
    }
    result.largestNormError =
        std::max(result.largestNormError, step->unitQuaternionNormError);
    result.gaits.insert(loop.motionManager().getCurrentGaitCommand());
  }
  return result;
}

TEST(AnymalBLockstepLoop, StandsTrotsForwardAndTurnsInPlace) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> created =
      CentroidalMpcLoop::Create(ShippedCentroidalRobot::kAnymalB,
                                CentroidalMpcLoopOptions{});
  ASSERT_TRUE(created.ok()) << created.status();
  CentroidalMpcLoop& loop = **created;
  const MpcRobotModelBase<scalar_t>& model =
      loop.interface().getMpcRobotModel();
  ASSERT_EQ(loop.interface().modelSettings().contactNames.size(), 4U);

  const PhaseResult standing =
      runPhase(loop, kStandUntil, /*cmdLinvelBaseRtWorldEwrtHeadingX=*/0.0,
               /*cmdRateYawBaseRtWorld=*/0.0);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const vector3_t standingPosBaseRtWorldEwrtWorld =
      model.getPosBaseRtWorldEwrtWorld(loop.observation().state);

  const PhaseResult walking =
      runPhase(loop, kWalkUntil, kCmdLinvelBaseRtWorldEwrtHeadingX,
               /*cmdRateYawBaseRtWorld=*/0.0);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const vector3_t walkedPosBaseRtWorldEwrtWorld =
      model.getPosBaseRtWorldEwrtWorld(loop.observation().state);
  const scalar_t walkedYawBaseRtWorld =
      model.getYawBaseRtWorld(loop.observation().state);

  const PhaseResult turning =
      runPhase(loop, kTurnUntil, /*cmdLinvelBaseRtWorldEwrtHeadingX=*/0.0,
               kCmdRateYawBaseRtWorld);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const scalar_t turnedYawBaseRtWorld =
      model.getYawBaseRtWorld(loop.observation().state);

  const scalar_t largestNormError =
      std::max(standing.largestNormError,
               std::max(walking.largestNormError, turning.largestNormError));
  EXPECT_LE(largestNormError, kMaxNormError);
  EXPECT_TRUE(walking.gaits.contains(kTrotGait))
      << "the gaits of the walk: " << absl::StrJoin(walking.gaits, ", ");
  EXPECT_FALSE(standing.gaits.contains(kTrotGait))
      << "the gaits of the stand: " << absl::StrJoin(standing.gaits, ", ");
  EXPECT_GT(
      walkedPosBaseRtWorldEwrtWorld.x() - standingPosBaseRtWorldEwrtWorld.x(),
      kMinWalkDistance);
  EXPECT_GT(std::remainder(turnedYawBaseRtWorld - walkedYawBaseRtWorld,
                           2.0 * std::numbers::pi),
            kMinTurnAngle)
      << "the gaits of the turn: " << absl::StrJoin(turning.gaits, ", ");
  ::testing::Test::RecordProperty("largest_norm_error",
                                  absl::StrCat(largestNormError));
  ::testing::Test::RecordProperty(
      "walk_distance", absl::StrCat(walkedPosBaseRtWorldEwrtWorld.x() -
                                    standingPosBaseRtWorldEwrtWorld.x()));
  ::testing::Test::RecordProperty(
      "turn_angle", absl::StrCat(turnedYawBaseRtWorld - walkedYawBaseRtWorld));
}

}  // namespace
}  // namespace ocs2::humanoid
