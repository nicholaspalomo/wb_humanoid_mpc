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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * The speed setters of TargetTrajectoriesCalculatorBase: a pose command reaches
 * its target in the time the distance takes at the set speed, starting level at
 * the measured heading. setTargetRotationVelocity assigned its own parameter to
 * itself and so set nothing; it had no caller, so fixing it changed no
 * behavior. Built on the G1 whole-body calculator, which needs no Pinocchio
 * model. Also the calculator's Create(): the typed reference file and its path
 * build the same calculator, and a reference file that does not parse or
 * convert is an InvalidArgument naming the file and the field.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;
// A pose command's first knot is at the initial time; at zero, the last knot's
// time is the time to the target exactly.
constexpr scalar_t kInitTime = 0.0;
// [deg] the turn of the pose command, which the calculator converts with the
// same expression.
constexpr scalar_t kTurnDegrees = 90.0;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

class TargetTrajectoriesCalculatorSettersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // LINT.IfChange(robot_files)
    const std::string taskFile = runfilePath(
        "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    const std::string urdfFile = runfilePath(
        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    referenceFile_ = runfilePath(
        "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:calculator_setters_test_data)
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile_.empty())
        << "the G1 whole-body files are not in the runfiles";
    modelSettings_ =
        std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
            taskFile, urdfFile, "wb_mpc_", /*verbose=*/false)));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    HUMANOID_ASSERT_OK_AND_ASSIGN(calculator_,
                                  WBMpcTargetTrajectoriesCalculator::Create(
                                      referenceFile_, *model_, kHorizon));
    HUMANOID_ASSERT_OK_AND_ASSIGN(reference_,
                                  loadReferenceFile(referenceFile_));
    // At the origin, level, facing along x: the identity orientation.
    initState_ = model_->getNeutralState();
  }

  /** [s] how long the pose command (dx, dy, dz, dyaw in degrees) is given to
   * reach its target. */
  scalar_t timeToTarget(const vector4_t& poseCommand) {
    const TargetTrajectories targets =
        calculator_->commandedPositionToTargetTrajectories(
            poseCommand, kInitTime, initState_);
    return targets.timeTrajectory.back() - kInitTime;
  }

  std::string referenceFile_;
  mpc_config::ReferenceFile reference_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
  vector_t initState_;
};

/**
 * Expects the reference of `targets` to turn the base by `turn` [rad] from its
 * first knot: knots at most kMaxKnotYawSpacing apart in heading, as many as
 * subdivideYawKnots() makes, and between them the robot model's slerp of the
 * orientation turning the commanded way at every step, level, by `turn` in all.
 */
void expectTheReferenceTurnsBy(const MpcRobotModelBase<scalar_t>& model,
                               const TargetTrajectories& targets,
                               scalar_t turn) {
  const size_t parts = std::max<size_t>(
      1, static_cast<size_t>(
             std::ceil(std::abs(turn) /
                       TargetTrajectoriesCalculatorBase::kMaxKnotYawSpacing)));
  ASSERT_EQ(targets.stateTrajectory.size(), parts + 1);
  for (size_t i = 1; i < targets.stateTrajectory.size(); ++i) {
    const scalar_t step =
        std::remainder(model.getBaseHeading(targets.stateTrajectory[i]) -
                           model.getBaseHeading(targets.stateTrajectory[i - 1]),
                       2.0 * M_PI);
    EXPECT_LE(std::abs(step),
              TargetTrajectoriesCalculatorBase::kMaxKnotYawSpacing + 1.0e-12)
        << "knot " << i;
  }
  constexpr int kSamples = 400;
  const scalar_t startTime = targets.timeTrajectory.front();
  const scalar_t duration = targets.timeTrajectory.back() - startTime;
  scalar_t turned = 0.0;
  scalar_t previousHeading =
      model.getBaseHeading(model.getDesiredState(targets, startTime));
  for (int sample = 1; sample <= kSamples; ++sample) {
    const vector_t desired = model.getDesiredState(
        targets,
        startTime + duration * static_cast<scalar_t>(sample) / kSamples);
    const scalar_t heading = model.getBaseHeading(desired);
    const scalar_t step = std::remainder(heading - previousHeading, 2.0 * M_PI);
    EXPECT_GT(step * turn, 0.0)
        << "the reference turns against the command at sample " << sample;
    const vector3_t eulerZyx = model.getBaseOrientationEulerZyx(desired);
    EXPECT_NEAR(eulerZyx(1), 0.0, 1.0e-12) << "sample " << sample;
    EXPECT_NEAR(eulerZyx(2), 0.0, 1.0e-12) << "sample " << sample;
    turned += step;
    previousHeading = heading;
  }
  EXPECT_NEAR(turned, turn, 1.0e-9);
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       APoseCommandTurnsAtTheRotationVelocitySet) {
  // A quarter turn is one segment; a turn of three quarters, or of three
  // eighths the other way, is cut into knots at most a quarter turn apart
  // (subdivideYawKnots()), so that the slerp of the orientation between them
  // turns the commanded way. Either way the time is the turn at the rotation
  // velocity.
  const scalar_t fileVelocity =
      reference_.target_rotation_velocity.value_or(0.0);
  ASSERT_GT(fileVelocity, 0.0);
  const absl::StatusOr<ReferenceSettings> settings =
      referenceSettingsFromConfig(reference_);
  ASSERT_TRUE(settings.ok()) << settings.status();
  for (const scalar_t turnDegrees : {kTurnDegrees, 270.0, -135.0}) {
    SCOPED_TRACE(turnDegrees);
    const vector4_t turn(0.0, 0.0, 0.0, turnDegrees);
    const scalar_t yaw = turnDegrees * M_PI / 180.0;
    EXPECT_DOUBLE_EQ(timeToTarget(turn), std::abs(yaw) / fileVelocity);
    expectTheReferenceTurnsBy(
        *model_,
        calculator_->commandedPositionToTargetTrajectories(turn, kInitTime,
                                                           initState_),
        yaw);

    // Speeds other than the file's, so that a setter that sets nothing cannot
    // pass.
    for (const scalar_t factor : {0.5, 2.0, 3.0}) {
      const scalar_t velocity = factor * fileVelocity;
      calculator_->setTargetRotationVelocity(velocity);
      EXPECT_DOUBLE_EQ(timeToTarget(turn), std::abs(yaw) / velocity)
          << "rotation velocity " << velocity;
      expectTheReferenceTurnsBy(
          *model_,
          calculator_->commandedPositionToTargetTrajectories(turn, kInitTime,
                                                             initState_),
          yaw);
    }

    // A reload of the reference file puts the file's speed back.
    calculator_->applyCommandLimits(*settings);
    EXPECT_DOUBLE_EQ(timeToTarget(turn), std::abs(yaw) / fileVelocity);
  }
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       APoseCommandStartsLevelAtTheTwistHeadingOfTheMeasuredAttitude) {
  // Measured at a stored yaw beyond pi and tilted: the first knot faces the
  // twist heading of that attitude, wrapped into
  // (-pi, pi], with no pitch or roll (design decision D9), and the turn takes
  // its time at the rotation velocity from there. The heading differs from the
  // stored yaw by the wrap and by -2 atan(tan(theta / 2) tan(phi / 2)).
  constexpr scalar_t kStoredYaw = 3.5;
  constexpr scalar_t kPitch = 0.1;
  constexpr scalar_t kRoll = -0.12;
  vector_t state = initState_;
  model_->setBaseOrientationEulerZyx(state,
                                     vector3_t(kStoredYaw, kPitch, kRoll));
  const TargetTrajectories targets =
      calculator_->commandedPositionToTargetTrajectories(
          vector4_t(0.0, 0.0, 0.0, kTurnDegrees), kInitTime, state);
  ASSERT_EQ(targets.stateTrajectory.size(), 2);
  const vector6_t first =
      model_->getBasePoseEulerZyx(targets.stateTrajectory.front());
  const vector6_t last =
      model_->getBasePoseEulerZyx(targets.stateTrajectory.back());
  EXPECT_EQ(first(3), model_->getBaseHeading(state));
  EXPECT_NEAR(
      first(3),
      std::remainder(kStoredYaw - 2.0 * std::atan(std::tan(kPitch / 2.0) *
                                                  std::tan(kRoll / 2.0)),
                     2.0 * M_PI),
      1.0e-12);
  EXPECT_EQ(first(4), 0.0);
  EXPECT_EQ(first(5), 0.0);
  const scalar_t yaw = kTurnDegrees * M_PI / 180.0;
  EXPECT_NEAR(last(3) - first(3), yaw, 1.0e-12);
  EXPECT_NEAR(targets.timeTrajectory.back() - kInitTime,
              yaw / reference_.target_rotation_velocity.value_or(0.0), 1.0e-12);
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       APoseCommandMovesAtTheDisplacementVelocitySet) {
  // One meter forward, no turn: the time is the distance at the displacement
  // velocity.
  const vector4_t step(1.0, 0.0, 0.0, 0.0);
  const scalar_t fileVelocity =
      reference_.target_displacement_velocity.value_or(0.0);
  ASSERT_GT(fileVelocity, 0.0);
  EXPECT_DOUBLE_EQ(timeToTarget(step), 1.0 / fileVelocity);
  for (const scalar_t factor : {0.5, 2.0, 3.0}) {
    const scalar_t velocity = factor * fileVelocity;
    calculator_->setTargetDisplacementVelocity(velocity);
    EXPECT_DOUBLE_EQ(timeToTarget(step), 1.0 / velocity)
        << "displacement velocity " << velocity;
  }
}

TEST_F(TargetTrajectoriesCalculatorSettersTest, TheTwoSpeedsAreIndependent) {
  // A turn's time depends on the rotation velocity only, a step's on the
  // displacement velocity only.
  const vector4_t turn(0.0, 0.0, 0.0, kTurnDegrees);
  const vector4_t step(1.0, 0.0, 0.0, 0.0);
  const scalar_t turnTime = timeToTarget(turn);
  const scalar_t stepTime = timeToTarget(step);
  calculator_->setTargetDisplacementVelocity(
      4.0 * reference_.target_displacement_velocity.value_or(0.0));
  EXPECT_EQ(timeToTarget(turn), turnTime);
  calculator_->setTargetRotationVelocity(
      4.0 * reference_.target_rotation_velocity.value_or(0.0));
  EXPECT_NE(timeToTarget(turn), turnTime);
  const absl::StatusOr<ReferenceSettings> settings =
      referenceSettingsFromConfig(reference_);
  ASSERT_TRUE(settings.ok()) << settings.status();
  calculator_->applyCommandLimits(*settings);
  calculator_->setTargetRotationVelocity(
      4.0 * reference_.target_rotation_velocity.value_or(0.0));
  EXPECT_EQ(timeToTarget(step), stepTime);
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       TheTypedReferenceFileBuildsWhatItsPathBuilds) {
  // The path form loads the file and builds the typed form: the same limits, so
  // the same times to every target.
  const std::unique_ptr<WBMpcTargetTrajectoriesCalculator> typed = checkedValue(
      WBMpcTargetTrajectoriesCalculator::Create(reference_, *model_, kHorizon));
  const vector4_t turn(0.0, 0.0, 0.0, kTurnDegrees);
  EXPECT_EQ(
      typed->commandedPositionToTargetTrajectories(turn, kInitTime, initState_)
          .timeTrajectory.back(),
      timeToTarget(turn));
  const vector4_t step(1.0, 0.5, 0.0, 0.0);
  EXPECT_EQ(
      typed->commandedPositionToTargetTrajectories(step, kInitTime, initState_)
          .timeTrajectory.back(),
      timeToTarget(step));
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       CreateRefusesACommandLimitThatDoesNotParse) {
  // The strict parser refuses a value that is not a number, naming the file and
  // the line and column of the value.
  std::ifstream in(referenceFile_);
  std::string reference((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::string key = "\nmax_rotation_velocity: ";
  const size_t value = reference.find(key);
  ASSERT_NE(value, std::string::npos)
      << "the shipped reference file no longer carries max_rotation_velocity";
  const size_t valueStart = value + key.size();
  reference.replace(valueStart,
                    reference.find_first_of(" \n", valueStart) - valueStart,
                    "fast");
  const std::string brokenReference =
      (std::filesystem::path(testing::TempDir()) /
       "testTargetTrajectoriesCalculatorSetters_reference.textproto")
          .string();
  std::ofstream(brokenReference) << reference;

  const absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>>
      created = WBMpcTargetTrajectoriesCalculator::Create(brokenReference,
                                                          *model_, kHorizon);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument)
      << created.status();
  int line = 1;
  for (size_t i = 0; i < valueStart; ++i) {
    if (reference[i] == '\n') ++line;
  }
  const int column =
      static_cast<int>(valueStart - reference.rfind('\n', valueStart));
  EXPECT_TRUE(absl::StrContains(
      created.status().message(),
      absl::StrCat(brokenReference, ":", line, ":", column, ":")))
      << created.status();
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       CreateRefusesADefaultPostureWithoutAJointNamingIt) {
  // The default posture names every joint of the model; one left out is refused
  // by name, not read as 0.
  mpc_config::ReferenceFile withoutAJoint = reference_;
  ASSERT_FALSE(withoutAJoint.default_joint_state.empty());
  const std::string joint = withoutAJoint.default_joint_state.back().joint;
  withoutAJoint.default_joint_state.pop_back();
  const absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>>
      created = WBMpcTargetTrajectoriesCalculator::Create(withoutAJoint,
                                                          *model_, kHorizon);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument)
      << created.status();
  EXPECT_TRUE(
      absl::StrContains(created.status().message(), "default_joint_state"))
      << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), joint))
      << created.status();
}

TEST_F(TargetTrajectoriesCalculatorSettersTest,
       TheErrorOfAConversionNamesTheReferenceFile) {
  // A command limit the file must give: the path form names the file before the
  // field.
  std::ifstream in(referenceFile_);
  std::string reference((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::string line = "\ndefault_base_height: ";
  const size_t start = reference.find(line);
  ASSERT_NE(start, std::string::npos)
      << "the shipped reference file no longer carries default_base_height";
  reference.erase(start + 1, reference.find('\n', start + 1) - start);
  const std::string withoutHeight =
      (std::filesystem::path(testing::TempDir()) /
       "testTargetTrajectoriesCalculatorSetters_no_height.textproto")
          .string();
  std::ofstream(withoutHeight) << reference;

  const absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>>
      created = WBMpcTargetTrajectoriesCalculator::Create(withoutHeight,
                                                          *model_, kHorizon);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument)
      << created.status();
  EXPECT_TRUE(absl::StartsWith(created.status().message(),
                               absl::StrCat(withoutHeight, ": ")))
      << created.status();
  EXPECT_TRUE(
      absl::StrContains(created.status().message(), "default_base_height"))
      << created.status();
}

}  // namespace
}  // namespace ocs2::humanoid
