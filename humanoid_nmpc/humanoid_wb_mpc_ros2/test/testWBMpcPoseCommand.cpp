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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/SystemObservation.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc_ros2/WBMpcPoseCommand.h"

/**
 * The whole-body pose command node's target (WBMpcPoseCommand). The node used to build it by hand, with the base at
 * `defaultBaseHeight` in world coordinates and the xy displacement in the world frame: on a ground raised by the task
 * file's `terrainHeight` it asked the robot to crouch by the height of the ground. It is now the whole-body target
 * calculator's, as the centroidal node's is its calculator's. Every property is a difference or an identity, so none
 * depends on the tuned heights.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1e-12;

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

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

class WBMpcPoseCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    ASSERT_FALSE(taskFile_.empty() || urdfFile_.empty() || referenceFile_.empty()) << "the G1 whole-body files are not in the runfiles";
  }

  /** The shipped G1 whole-body task file on a ground at `terrainHeight`. */
  std::string writeTaskFileOnGround(scalar_t terrainHeight) const {
    const std::string shipped = readFile(taskFile_);
    EXPECT_EQ(shipped.find("\nterrainHeight:"), std::string::npos) << "the shipped file sets the ground; this test appends it";
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcPoseCommand_", terrainHeight, ".yaml")).string();
    std::ofstream out(path);
    out << shipped << "\nterrainHeight: " << terrainHeight << "\n";
    return path;
  }

  std::unique_ptr<WBMpcPoseCommand> create(const std::string& taskFile) const {
    absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> created = WBMpcPoseCommand::Create(taskFile, urdfFile_, referenceFile_);
    EXPECT_TRUE(created.ok()) << created.status();
    return created.ok() ? *std::move(created) : nullptr;
  }

  /** The task file's initial state at time 1, which is what the node's first observation is. */
  SystemObservation initialObservation(const WBMpcPoseCommand& poseCommand) const {
    SystemObservation observation;
    observation.time = 1.0;
    observation.state = vector_t::Zero(poseCommand.mpcRobotModel().getStateDim());
    loadData::loadEigenMatrix(taskFile_, "initialState", observation.state);
    observation.input = vector_t::Zero(poseCommand.mpcRobotModel().getInputDim());
    return observation;
  }

  std::string taskFile_, urdfFile_, referenceFile_;
};

}  // namespace

TEST_F(WBMpcPoseCommandTest, TheBaseHeightStandsOnTheTaskFilesGround) {
  const scalar_t raisedGround = 0.3;
  const std::unique_ptr<WBMpcPoseCommand> flat = create(taskFile_);
  const std::unique_ptr<WBMpcPoseCommand> raised = create(writeTaskFileOnGround(raisedGround));
  ASSERT_NE(flat, nullptr);
  ASSERT_NE(raised, nullptr);
  ASSERT_NEAR(raised->modelSettings().terrainHeight - flat->modelSettings().terrainHeight, raisedGround, kTol);

  const vector4_t command(0.2, -0.1, 0.05, 30.0);
  const SystemObservation observation = initialObservation(*flat);
  const TargetTrajectories onFlat = flat->toTargetTrajectories(command, observation);
  const TargetTrajectories onRaised = raised->toTargetTrajectories(command, observation);
  ASSERT_EQ(onFlat.stateTrajectory.size(), 2U);
  ASSERT_EQ(onRaised.stateTrajectory.size(), 2U);

  const vector6_t flatTarget = flat->mpcRobotModel().getBasePose(onFlat.stateTrajectory.back());
  const vector6_t raisedTarget = raised->mpcRobotModel().getBasePose(onRaised.stateTrajectory.back());
  // The whole ground and nothing else: the target is `defaultBaseHeight + deltaZ` above it on both grounds.
  EXPECT_NEAR(raisedTarget(2) - flatTarget(2), raisedGround, kTol);
  scalar_t defaultBaseHeight = 0.0;
  scalar_t maxDeltaPelvisHeight = 0.0;
  loadData::loadCppDataType(referenceFile_, "defaultBaseHeight", defaultBaseHeight);
  loadData::loadCppDataType(referenceFile_, "maxDeltaPelvisHeight", maxDeltaPelvisHeight);
  const scalar_t deltaZ = std::clamp(command(2), -maxDeltaPelvisHeight, maxDeltaPelvisHeight);
  EXPECT_NEAR(flatTarget(2), flat->modelSettings().terrainHeight + defaultBaseHeight + deltaZ, kTol);
  for (const int index : {0, 1, 3, 4, 5}) {
    EXPECT_NEAR(raisedTarget(index), flatTarget(index), kTol) << "base pose coordinate " << index << " does not depend on the ground";
  }
}

TEST_F(WBMpcPoseCommandTest, TheTargetIsTheWholeBodyTargetCalculators) {
  const std::unique_ptr<WBMpcPoseCommand> poseCommand = create(writeTaskFileOnGround(/*terrainHeight=*/0.1));
  ASSERT_NE(poseCommand, nullptr);
  WBMpcTargetTrajectoriesCalculator calculator(referenceFile_, poseCommand->mpcRobotModel(), /*mpcHorizon=*/1.0);
  const SystemObservation observation = initialObservation(*poseCommand);
  const vector4_t command(-0.3, 0.4, -0.02, -45.0);

  const TargetTrajectories expected = calculator.commandedPositionToTargetTrajectories(command, observation.time, observation.state);
  const TargetTrajectories actual = poseCommand->toTargetTrajectories(command, observation);
  ASSERT_EQ(actual.timeTrajectory.size(), expected.timeTrajectory.size());
  for (size_t i = 0; i < expected.timeTrajectory.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual.timeTrajectory[i], expected.timeTrajectory[i]);
    EXPECT_TRUE(actual.stateTrajectory[i].isApprox(expected.stateTrajectory[i])) << "knot " << i;
    EXPECT_EQ(actual.inputTrajectory[i].size(), expected.inputTrajectory[i].size());
  }
  // The target starts at the observation and takes time to reach: the calculator's estimate, not an instant jump.
  EXPECT_DOUBLE_EQ(actual.timeTrajectory.front(), observation.time);
  EXPECT_GT(actual.timeTrajectory.back(), observation.time);
}

TEST_F(WBMpcPoseCommandTest, TheDisplacementIsTakenInThePelvisFrame) {
  const std::unique_ptr<WBMpcPoseCommand> poseCommand = create(taskFile_);
  ASSERT_NE(poseCommand, nullptr);
  SystemObservation observation = initialObservation(*poseCommand);
  const vector6_t start = poseCommand->mpcRobotModel().getBasePose(observation.state);
  // The robot faces the world's +y: a step forward in its own frame is a step along +y.
  observation.state(3) = start(3) + M_PI / 2.0;
  const TargetTrajectories target = poseCommand->toTargetTrajectories(vector4_t(1.0, 0.0, 0.0, 0.0), observation);
  const vector6_t reached = poseCommand->mpcRobotModel().getBasePose(target.stateTrajectory.back());
  const scalar_t heading = start(3) + M_PI / 2.0;
  EXPECT_NEAR(reached(0) - start(0), std::cos(heading), 1e-9);
  EXPECT_NEAR(reached(1) - start(1), std::sin(heading), 1e-9);
  EXPECT_NEAR(reached(3), heading, 1e-9) << "no yaw was commanded";
}

TEST_F(WBMpcPoseCommandTest, AMissingFileIsNotFoundNamingItsPath) {
  const std::string missing = (std::filesystem::path(testing::TempDir()) / "testWBMpcPoseCommand_missing.yaml").string();
  ASSERT_FALSE(std::filesystem::exists(missing));
  const absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> noTask = WBMpcPoseCommand::Create(missing, urdfFile_, referenceFile_);
  const absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> noUrdf = WBMpcPoseCommand::Create(taskFile_, missing, referenceFile_);
  const absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> noReference = WBMpcPoseCommand::Create(taskFile_, urdfFile_, missing);
  for (const absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>>* created : {&noTask, &noUrdf, &noReference}) {
    EXPECT_EQ(created->status().code(), absl::StatusCode::kNotFound) << created->status();
    EXPECT_TRUE(absl::StrContains(created->status().message(), missing)) << created->status();
  }
  // Positive control: the shipped files build it.
  EXPECT_TRUE(WBMpcPoseCommand::Create(taskFile_, urdfFile_, referenceFile_).ok());
}

}  // namespace ocs2::humanoid
