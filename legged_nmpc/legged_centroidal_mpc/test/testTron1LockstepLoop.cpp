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

#include <cmath>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "legged_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "legged_mpc_validation/closed_loop/ClosedLoopScenario.h"
#include "legged_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "legged_mpc_validation/closed_loop/RobotConfiguration.h"
#include "legged_mpc_validation/closed_loop/TimeSeriesLabels.h"
#include "legged_mpc_validation/io/GoldenIo.h"
#include "legged_mpc_validation/io/JsonValue.h"

/*
 * The shipped LimX Tron1, the first shipped robot on rolling contacts, on its
 * centroidal MPC in the headless MuJoCo lockstep loop of
 * legged_mpc_validation (LockstepClosedLoop), with the wheeled gait file and
 * the ground-truth contact estimator of its task file: it stands, drives
 * forward on its wheels and turns in place. The robot survives, every solve
 * succeeds, every planned quaternion stays unit, the base goes forward and
 * turns by more than half of what it is told, and both wheels stay in the
 * simulator's measured contact throughout: the drive gait takes no step. Unlike
 * testAnymalBLockstepLoop, whose plant is the projected rollout of the policy,
 * this plant is the simulator, so it checks the wheels' balance and their
 * no-slip constraint against MuJoCo's contacts. Generates Tron1's CppAD
 * libraries on a cold cache.
 */

namespace ocs2::legged {
namespace {

using validation::ClosedLoopRunInfo;
using validation::ClosedLoopScenario;
using validation::CommandSegment;
using validation::golden_matrix_t;
using validation::GoldenFile;
using validation::JsonValue;
using validation::LockstepClosedLoop;
using validation::LockstepOptions;
using validation::LockstepResult;
using validation::RobotConfiguration;

constexpr double kStandingTime = 2.0;                      // [s]
constexpr double kDriveDuration = 4.0;                     // [s]
constexpr double kTurnDuration = 4.0;                      // [s]
constexpr double kRestDuration = 1.0;                      // [s]
constexpr double kCmdLinvelBaseRtWorldEwrtHeadingX = 0.5;  // [m/s]
constexpr double kCmdRateYawBaseRtWorld = 0.5;             // [rad/s]
// The quaternion norm the planned solutions keep (testUnitNorm's bound).
constexpr double kMaxNormError = 1.0e-12;
// [m] Half of the 2 m the drive asks for; the run went 2.03 m.
constexpr double kMinDriveDistance = 1.0;
// [rad] Half of the 2 rad the turn asks for; the run turned 1.36 rad: the yaw
// rate ramps up from rest, and on the flat 5 cm wide wheel cylinders of the
// MuJoCo scene the yaw rate settles at about 0.45 rad/s against the 0.5 rad/s
// commanded.
constexpr double kMinTurnAngle = 1.0;
// Tron1's two wheels, each in one stance phase that never ends.
constexpr double kNumWheels = 2.0;

/** The number at `path` of the metrics document; NaN after a failed
 * expectation. */
double numberAt(const JsonValue& metrics, absl::string_view path) {
  const JsonValue* absl_nullable value = metrics.findPath(path);
  EXPECT_NE(value, nullptr) << path;
  if (value == nullptr || !value->isNumber()) return std::nan("");
  return value->asNumber();
}

TEST(Tron1LockstepLoop, StandsDrivesForwardAndTurnsInPlaceOnItsWheels) {
  const absl::StatusOr<RobotConfiguration> configuration =
      validation::findRobotConfiguration("tron1");
  ASSERT_TRUE(configuration.ok()) << configuration.status();
  ClosedLoopScenario scenario;
  scenario.name = "stand_drive_turn";
  scenario.description =
      absl::StrCat(kCmdLinvelBaseRtWorldEwrtHeadingX, " m/s forward for ",
                   kDriveDuration, " s, ", kCmdRateYawBaseRtWorld,
                   " rad/s in place for ", kTurnDuration, " s, then at rest");
  scenario.standingTime = kStandingTime;
  scenario.segments = {CommandSegment{.duration = kDriveDuration,
                                      .cmdLinvelBaseRtWorldEwrtHeadingX =
                                          kCmdLinvelBaseRtWorldEwrtHeadingX},
                       CommandSegment{.duration = kTurnDuration,
                                      .yawRate = kCmdRateYawBaseRtWorld},
                       CommandSegment{.duration = kRestDuration}};
  const LockstepClosedLoop closedLoop(*configuration, LockstepOptions());
  ClosedLoopRunInfo info;
  info.label = "tron1_lockstep_loop";
  const absl::StatusOr<LockstepResult> run = closedLoop.run(scenario, info);
  ASSERT_TRUE(run.ok()) << run.status();
  const JsonValue& metrics = run->metrics;

  const JsonValue* absl_nullable survived =
      metrics.findPath("survival.survived");
  ASSERT_NE(survived, nullptr);
  ASSERT_TRUE(survived->isBool());
  EXPECT_TRUE(survived->asBool());
  EXPECT_EQ(numberAt(metrics, "failures.failed_solves"), 0.0);
  EXPECT_EQ(numberAt(metrics, "failures.simulator_resets"), 0.0);
  EXPECT_EQ(numberAt(metrics, "non_finite_values"), 0.0);
  EXPECT_LE(numberAt(metrics, "quaternion_norm.max_deviation"), kMaxNormError);

  // Both wheels in measured contact at every control cycle of the commands.
  EXPECT_EQ(run->cyclesWithALiftedContact, 0U);
  EXPECT_EQ(numberAt(metrics, "stance_foot_slip.stance_phases"), kNumWheels);

  const GoldenFile& series = run->timeSeries;
  const golden_matrix_t* absl_nullable posBaseRtWorldEwrtWorld =
      series.find(validation::time_series::kPosBaseRtWorldEwrtWorld);
  ASSERT_NE(posBaseRtWorldEwrtWorld, nullptr);
  ASSERT_GT(posBaseRtWorldEwrtWorld->rows(), 0);
  // The turn is in place, so the distance from the first to the last sample
  // is the drive's.
  const double driveDistance = (posBaseRtWorldEwrtWorld->bottomRows<1>() -
                                posBaseRtWorldEwrtWorld->topRows<1>())
                                   .leftCols<2>()
                                   .norm();
  EXPECT_GT(driveDistance, kMinDriveDistance);
  const double turnAngle = numberAt(metrics, "heading.cumulative_final_rad");
  EXPECT_GT(turnAngle, kMinTurnAngle);
  ::testing::Test::RecordProperty("drive_distance",
                                  absl::StrCat(driveDistance));
  ::testing::Test::RecordProperty("turn_angle", absl::StrCat(turnAngle));
  ::testing::Test::RecordProperty(
      "stance_foot_slip_max",
      absl::StrCat(numberAt(metrics, "stance_foot_slip.max_m")));
}

}  // namespace
}  // namespace ocs2::legged
