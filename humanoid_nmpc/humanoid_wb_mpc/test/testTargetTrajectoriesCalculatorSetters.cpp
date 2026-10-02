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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/TargetTrajectories.h>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * The speed setters of TargetTrajectoriesCalculatorBase: a pose command reaches its target in the time the distance
 * takes at the set speed. setTargetRotationVelocity assigned its own parameter to itself and so set nothing; it had no
 * caller, so fixing it changed no behavior. Built on the G1 whole-body calculator, which needs no Pinocchio model.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;
// A pose command's first knot is at the initial time; at zero, the last knot's time is the time to the target exactly.
constexpr scalar_t kInitTime = 0.0;
// [deg] the turn of the pose command, which the calculator converts with the same expression.
constexpr scalar_t kTurnDegrees = 90.0;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
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
    const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:calculator_setters_test_data)
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    modelSettings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    calculator_ = std::make_unique<WBMpcTargetTrajectoriesCalculator>(referenceFile_, *model_, kHorizon);
    // At the origin, facing along x.
    initState_ = vector_t::Zero(model_->getStateDim());
  }

  /** [s] how long the pose command (dx, dy, dz, dyaw in degrees) is given to reach its target. */
  scalar_t timeToTarget(const vector4_t& poseCommand) {
    const TargetTrajectories targets = calculator_->commandedPositionToTargetTrajectories(poseCommand, kInitTime, initState_);
    return targets.timeTrajectory.back() - kInitTime;
  }

  /** The value of `key` in the reference file. */
  scalar_t referenceValue(const std::string& key) const {
    scalar_t value = 0.0;
    loadData::loadCppDataType(referenceFile_, key, value);
    return value;
  }

  std::string referenceFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
  vector_t initState_;
};

TEST_F(TargetTrajectoriesCalculatorSettersTest, APoseCommandTurnsAtTheRotationVelocitySet) {
  const vector4_t turn(0.0, 0.0, 0.0, kTurnDegrees);
  const scalar_t yaw = kTurnDegrees * M_PI / 180.0;
  const scalar_t fileVelocity = referenceValue("targetRotationVelocity");
  ASSERT_GT(fileVelocity, 0.0);
  EXPECT_DOUBLE_EQ(timeToTarget(turn), yaw / fileVelocity);

  // Speeds other than the file's, so that a setter that sets nothing cannot pass.
  for (const scalar_t factor : {0.5, 2.0, 3.0}) {
    const scalar_t velocity = factor * fileVelocity;
    calculator_->setTargetRotationVelocity(velocity);
    EXPECT_DOUBLE_EQ(timeToTarget(turn), yaw / velocity) << "rotation velocity " << velocity;
  }

  // A reload of the reference file puts the file's speed back.
  calculator_->reloadCommandLimits(referenceFile_);
  EXPECT_DOUBLE_EQ(timeToTarget(turn), yaw / fileVelocity);
}

TEST_F(TargetTrajectoriesCalculatorSettersTest, APoseCommandMovesAtTheDisplacementVelocitySet) {
  // One meter forward, no turn: the time is the distance at the displacement velocity.
  const vector4_t step(1.0, 0.0, 0.0, 0.0);
  const scalar_t fileVelocity = referenceValue("targetDisplacementVelocity");
  ASSERT_GT(fileVelocity, 0.0);
  EXPECT_DOUBLE_EQ(timeToTarget(step), 1.0 / fileVelocity);
  for (const scalar_t factor : {0.5, 2.0, 3.0}) {
    const scalar_t velocity = factor * fileVelocity;
    calculator_->setTargetDisplacementVelocity(velocity);
    EXPECT_DOUBLE_EQ(timeToTarget(step), 1.0 / velocity) << "displacement velocity " << velocity;
  }
}

TEST_F(TargetTrajectoriesCalculatorSettersTest, TheTwoSpeedsAreIndependent) {
  // A turn's time depends on the rotation velocity only, a step's on the displacement velocity only.
  const vector4_t turn(0.0, 0.0, 0.0, kTurnDegrees);
  const vector4_t step(1.0, 0.0, 0.0, 0.0);
  const scalar_t turnTime = timeToTarget(turn);
  const scalar_t stepTime = timeToTarget(step);
  calculator_->setTargetDisplacementVelocity(4.0 * referenceValue("targetDisplacementVelocity"));
  EXPECT_EQ(timeToTarget(turn), turnTime);
  calculator_->setTargetRotationVelocity(4.0 * referenceValue("targetRotationVelocity"));
  EXPECT_NE(timeToTarget(turn), turnTime);
  calculator_->reloadCommandLimits(referenceFile_);
  calculator_->setTargetRotationVelocity(4.0 * referenceValue("targetRotationVelocity"));
  EXPECT_EQ(timeToTarget(step), stepTime);
}

}  // namespace
}  // namespace ocs2::humanoid
