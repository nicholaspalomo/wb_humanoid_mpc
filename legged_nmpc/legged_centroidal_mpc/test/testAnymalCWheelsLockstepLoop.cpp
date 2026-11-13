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
#include <utility>
#include <vector>

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
 * The shipped wheeled ANYmal C, four rolling contacts, on its centroidal MPC in
 * the headless MuJoCo lockstep loop of legged_mpc_validation
 * (LockstepClosedLoop), with the wheeled quadrupeds' gait file and the
 * ground-truth contact estimator of its task file. It drives forward on its
 * four wheels with every wheel in the simulator's measured contact throughout,
 * and it turns in place in the gait file's turning_trot, since a yaw on its
 * four parallel axles would slide the wheels sideways
 * (legged_nmpc/docs/rolling_contact/README.md, section 8). Each run
 * survives, every solve succeeds and every planned quaternion stays unit.
 * Generates the robot's CppAD libraries on a cold cache.
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
// [m] Half of the 2 m the drive asks for; the run went 1.99 m.
constexpr double kMinDriveDistance = 1.0;
// [rad] Half of the 2 rad the turn asks for; the run turned 1.72 rad: the yaw
// rate ramps up from rest and the gait switches from drive to turning_trot
// first.
constexpr double kMinTurnAngle = 1.0;
// [rad] The base tilt the trot keeps below; the closed-loop turn of label
// 2026-11-12_anymal_c_wheels peaked at 0.012 rad, and the trot without
// four-wheel phases pitched the base by 0.7 rad.
constexpr double kMaxTurnTilt = 0.1;
// The four wheels, each in one stance phase that never ends while it drives.
constexpr double kNumWheels = 4.0;

/** Returns the number at `path` of the metrics document; NaN after a failed
 * expectation. */
double numberAt(const JsonValue& metrics, absl::string_view path) {
  const JsonValue* absl_nullable value = metrics.findPath(path);
  EXPECT_NE(value, nullptr) << path;
  if (value == nullptr || !value->isNumber()) return std::nan("");
  return value->asNumber();
}

/** Runs `segments` after the standing time on the shipped configuration. */
absl::StatusOr<LockstepResult> runScenario(
    absl::string_view name, std::vector<CommandSegment> segments) {
  const absl::StatusOr<RobotConfiguration> configuration =
      validation::findRobotConfiguration("anymal_c_wheels");
  if (!configuration.ok()) return configuration.status();
  ClosedLoopScenario scenario;
  scenario.name = std::string(name);
  scenario.description = std::string(name);
  scenario.standingTime = kStandingTime;
  scenario.segments = std::move(segments);
  const LockstepClosedLoop closedLoop(*configuration, LockstepOptions());
  ClosedLoopRunInfo info;
  info.label = absl::StrCat("anymal_c_wheels_lockstep_loop_", name);
  return closedLoop.run(scenario, info);
}

/** Expects the run to survive with every solve successful and finite. */
void expectHealthyRun(const JsonValue& metrics) {
  const JsonValue* absl_nullable survived =
      metrics.findPath("survival.survived");
  ASSERT_NE(survived, nullptr);
  ASSERT_TRUE(survived->isBool());
  EXPECT_TRUE(survived->asBool());
  EXPECT_EQ(numberAt(metrics, "failures.failed_solves"), 0.0);
  EXPECT_EQ(numberAt(metrics, "failures.simulator_resets"), 0.0);
  EXPECT_EQ(numberAt(metrics, "non_finite_values"), 0.0);
  EXPECT_LE(numberAt(metrics, "quaternion_norm.max_deviation"), kMaxNormError);
}

TEST(AnymalCWheelsLockstepLoop, DrivesForwardOnItsFourWheels) {
  const absl::StatusOr<LockstepResult> run = runScenario(
      "stand_drive", {CommandSegment{.duration = kDriveDuration,
                                     .cmdLinvelBaseRtWorldEwrtHeadingX =
                                         kCmdLinvelBaseRtWorldEwrtHeadingX},
                      CommandSegment{.duration = kRestDuration}});
  ASSERT_TRUE(run.ok()) << run.status();
  expectHealthyRun(run->metrics);

  // Every wheel in measured contact at every control cycle of the commands.
  EXPECT_EQ(run->cyclesWithALiftedContact, 0U);
  EXPECT_EQ(numberAt(run->metrics, "stance_foot_slip.stance_phases"),
            kNumWheels);

  const GoldenFile& series = run->timeSeries;
  const golden_matrix_t* absl_nullable posBaseRtWorldEwrtWorld =
      series.find(validation::time_series::kPosBaseRtWorldEwrtWorld);
  ASSERT_NE(posBaseRtWorldEwrtWorld, nullptr);
  ASSERT_GT(posBaseRtWorldEwrtWorld->rows(), 0);
  const double driveDistance = (posBaseRtWorldEwrtWorld->bottomRows<1>() -
                                posBaseRtWorldEwrtWorld->topRows<1>())
                                   .leftCols<2>()
                                   .norm();
  EXPECT_GT(driveDistance, kMinDriveDistance);
  ::testing::Test::RecordProperty("drive_distance",
                                  absl::StrCat(driveDistance));
}

TEST(AnymalCWheelsLockstepLoop, TurnsInPlaceInATrot) {
  const absl::StatusOr<LockstepResult> run = runScenario(
      "stand_turn", {CommandSegment{.duration = kTurnDuration,
                                    .yawRate = kCmdRateYawBaseRtWorld},
                     CommandSegment{.duration = kRestDuration}});
  ASSERT_TRUE(run.ok()) << run.status();
  expectHealthyRun(run->metrics);

  // The turn steps: the wheels leave the ground and land again.
  EXPECT_GT(run->cyclesWithALiftedContact, 0U);
  EXPECT_GT(numberAt(run->metrics, "stance_foot_slip.stance_phases"),
            kNumWheels);
  const double turnAngle =
      numberAt(run->metrics, "heading.cumulative_final_rad");
  EXPECT_GT(turnAngle, kMinTurnAngle);
  EXPECT_LT(numberAt(run->metrics, "tilt.max_rad"), kMaxTurnTilt);
  ::testing::Test::RecordProperty("turn_angle", absl::StrCat(turnAngle));
}

}  // namespace
}  // namespace ocs2::legged
