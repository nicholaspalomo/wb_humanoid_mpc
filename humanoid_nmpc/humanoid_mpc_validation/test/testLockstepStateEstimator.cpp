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

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_mpc_validation/closed_loop/LockstepStateEstimator.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorNames.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * The lockstep closed loop builds the robot process's state estimator as
 * RobotProcess::Create() builds it: by the task file's state_estimator, from
 * the MPC's contact frames and their parent joints, the URDF, the robot
 * description and the kalman noise model. ground_truth hands the controller the
 * measured state itself, so a lockstep run with it is the run without an
 * estimator stage bit for bit; kalman replaces only the base's position and
 * linear velocity.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr char kTaskFile[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
/** [m] The base height of the probe state. */
constexpr double kBaseHeight = 0.75;

std::string runfilePath(absl::string_view relativePath) {
  const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path root =
      srcDir != nullptr ? std::filesystem::path(srcDir) / "_main"
                        : std::filesystem::current_path();
  return (root / std::string(relativePath)).string();
}

/** The G1's model settings, robot description and a standing state of it. */
struct G1 {
  G1()
      : taskFile(runfilePath(kTaskFile)),
        urdfFile(runfilePath(kUrdfFile)),
        modelSettings(ModelSettings::Create(taskFile,
                                            urdfFile,
                                            "lockstep_state_estimator_test_",
                                            /*verbose=*/false)
                          .value()),
        description(robot::model::RobotDescription::Create(urdfFile).value()),
        state(description) {
    state.setRootPositionInWorldFrame(Eigen::Vector3d(0.0, 0.0, kBaseHeight));
    for (size_t joint = 0; joint < description.getNumJoints(); ++joint) {
      state.setJointPosition(joint, 0.01 * static_cast<double>(joint));
    }
  }
  std::string taskFile;
  std::string urdfFile;
  ModelSettings modelSettings;
  robot::model::RobotDescription description;
  robot::model::RobotState state;
};

TEST(LockstepStateEstimator, TheContextIsTheRobotProcesss) {
  const G1 g1;
  RobotProcessSettings settings;
  settings.kalmanStateEstimator.contactHeight = 0.0123;
  const estimation::StateEstimatorContext context =
      robotProcessStateEstimatorContext(settings, g1.modelSettings, g1.urdfFile,
                                        &g1.description);
  // RobotProcess::Create() takes them from the robot binary's backend options,
  // which take them from the model settings.
  EXPECT_EQ(context.urdfFile, g1.urdfFile);
  EXPECT_EQ(context.contactFrameNames, g1.modelSettings.contactNames);
  EXPECT_EQ(context.contactParentJointNames,
            g1.modelSettings.contactParentJointNames);
  EXPECT_EQ(context.description, &g1.description);
  EXPECT_EQ(context.kalman.contactHeight, 0.0123);
}

TEST(LockstepStateEstimator, GroundTruthHandsOnTheMeasuredStateItself) {
  const G1 g1;
  const RobotProcessSettings settings;  // the default state_estimator
  ASSERT_EQ(settings.stateEstimator,
            estimation::kGroundTruthStateEstimatorName);
  absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      createRobotProcessStateEstimator(settings, g1.modelSettings, g1.urdfFile,
                                       g1.taskFile, &g1.description);
  ASSERT_TRUE(estimator.ok()) << estimator.status();
  EXPECT_EQ((*estimator)->name(), estimation::kGroundTruthStateEstimatorName);
  (*estimator)->reset(g1.state);
  EXPECT_EQ(&(*estimator)->estimate(g1.state), &g1.state)
      << "the controller reads the simulator's state, bit for bit";
}

TEST(LockstepStateEstimator, KalmanEstimatesTheBaseAndKeepsTheJoints) {
  const G1 g1;
  RobotProcessSettings settings;
  settings.stateEstimator = std::string(estimation::kKalmanStateEstimatorName);
  absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      createRobotProcessStateEstimator(settings, g1.modelSettings, g1.urdfFile,
                                       g1.taskFile, &g1.description);
  ASSERT_TRUE(estimator.ok()) << estimator.status();
  EXPECT_EQ((*estimator)->name(), estimation::kKalmanStateEstimatorName);
  (*estimator)->reset(g1.state);
  const robot::model::RobotState& estimate = (*estimator)->estimate(g1.state);
  EXPECT_NE(&estimate, &g1.state);
  for (size_t joint = 0; joint < g1.description.getNumJoints(); ++joint) {
    EXPECT_EQ(estimate.getJointPosition(joint),
              g1.state.getJointPosition(joint))
        << "joint " << joint;
    EXPECT_EQ(estimate.getJointVelocity(joint),
              g1.state.getJointVelocity(joint))
        << "joint " << joint;
  }
}

TEST(LockstepStateEstimator, AnUnknownNameIsRefusedNamingTheTaskFileField) {
  const G1 g1;
  RobotProcessSettings settings;
  settings.stateEstimator = "no_such_estimator";
  const absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      createRobotProcessStateEstimator(settings, g1.modelSettings, g1.urdfFile,
                                       g1.taskFile, &g1.description);
  ASSERT_FALSE(estimator.ok());
  EXPECT_EQ(estimator.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      absl::StartsWith(estimator.status().message(),
                       absl::StrCat(g1.taskFile, ": state_estimator: ")))
      << estimator.status();
  EXPECT_TRUE(
      absl::StrContains(estimator.status().message(), "no_such_estimator"))
      << estimator.status();
}

}  // namespace
}  // namespace ocs2::humanoid::validation
