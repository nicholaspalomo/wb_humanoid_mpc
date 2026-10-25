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
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/constraint/JointMimicDynamicsConstraint.h"

/**
 * JointMimicDynamicsConstraint::Create(): the two joints are resolved by name
 * among the MPC joints, and a name that is not one of them, or gains that do
 * not make the joint error a stable second-order system, are refused with a
 * Status. The joint indices used to be looked up by a Config constructor that
 * threw. Built on the G1 whole-body model, which needs no Pinocchio or CppAD
 * model.
 */
namespace ocs2::humanoid {
namespace {

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

class JointMimicDynamicsConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // LINT.IfChange(robot_files)
    const std::string taskFile = runfilePath(
        "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    const std::string urdfFile = runfilePath(
        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:joint_mimic_test_data)
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty())
        << "the G1 whole-body files are not in the runfiles";
    absl::StatusOr<ModelSettings> settings =
        ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    modelSettings_ = std::make_unique<ModelSettings>(*std::move(settings));
    ASSERT_GE(modelSettings_->mpcModelJointNames.size(), 2U);
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
  }

  const std::string& jointName(size_t index) const {
    return modelSettings_->mpcModelJointNames[index];
  }

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
};

TEST_F(JointMimicDynamicsConstraintTest, ResolvesTheJointsByName) {
  const absl::StatusOr<std::unique_ptr<JointMimicDynamicsConstraint>>
      constraint = JointMimicDynamicsConstraint::Create(
          *model_, jointName(0), jointName(1), /*multiplier=*/0.5,
          /*positionGain=*/20.0, /*velocityGain=*/5.0);
  ASSERT_TRUE(constraint.ok()) << constraint.status();
  const JointMimicDynamicsConstraint::Config& config =
      (*constraint)->getConfig();
  EXPECT_THAT(model_->findJointIndex(jointName(0)),
              absl_testing::IsOkAndHolds(config.parentJointIndex));
  EXPECT_THAT(model_->findJointIndex(jointName(1)),
              absl_testing::IsOkAndHolds(config.childJointIndex));

  // The child joint at half the parent's angle, velocity and acceleration: the
  // constraint is met.
  vector_t state = model_->getNeutralState();
  vector_t input = vector_t::Zero(model_->getInputDim());
  vector_t jointAngles = vector_t::Zero(model_->getJointDim());
  jointAngles[config.parentJointIndex] = 0.4;
  jointAngles[config.childJointIndex] = 0.2;
  model_->setJointAngles(state, jointAngles);
  const PreComputation preComputation;
  EXPECT_NEAR(
      (*constraint)->getValue(/*time=*/0.0, state, input, preComputation)[0],
      0.0, 1.0e-12);
  // And an error of the child's angle is weighted by the position gain.
  jointAngles[config.childJointIndex] = 0.1;
  model_->setJointAngles(state, jointAngles);
  EXPECT_NEAR(
      (*constraint)->getValue(/*time=*/0.0, state, input, preComputation)[0],
      20.0 * 0.1, 1.0e-12);
}

TEST_F(JointMimicDynamicsConstraintTest,
       AJointThatIsNotAnMpcJointIsNotFoundNamingIt) {
  const absl::StatusOr<std::unique_ptr<JointMimicDynamicsConstraint>> parent =
      JointMimicDynamicsConstraint::Create(
          *model_, "no_such_knee", jointName(1), /*multiplier=*/0.5,
          /*positionGain=*/20.0, /*velocityGain=*/5.0);
  EXPECT_EQ(parent.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(parent.status().message(), "no_such_knee"))
      << parent.status();
  const absl::StatusOr<std::unique_ptr<JointMimicDynamicsConstraint>> child =
      JointMimicDynamicsConstraint::Create(
          *model_, jointName(0), "no_such_knee", /*multiplier=*/0.5,
          /*positionGain=*/20.0, /*velocityGain=*/5.0);
  EXPECT_EQ(child.status().code(), absl::StatusCode::kNotFound);
}

/** Returns the largest real part of the roots of s^2 + velocityGain s +
 * positionGain, the joint error's poles. */
scalar_t largestPoleRealPart(scalar_t positionGain, scalar_t velocityGain) {
  // Complex roots (a negative discriminant) share the real part -velocityGain /
  // 2; real roots are (-Kd +- sqrt(d)) / 2.
  const scalar_t discriminant =
      velocityGain * velocityGain - 4.0 * positionGain;
  return (-velocityGain + std::sqrt(std::max(discriminant, 0.0))) * 0.5;
}

/** Simulates the error dynamics e'' = -Kp e - Kd e' from e = 1, e' = 0 for
 * `duration` and returns |e| + |e'| at the end. */
scalar_t errorAfter(scalar_t positionGain,
                    scalar_t velocityGain,
                    scalar_t duration) {
  constexpr int kSteps = 200'000;
  const scalar_t dt = duration / kSteps;
  scalar_t error = 1.0;
  scalar_t errorRate = 0.0;
  for (int step = 0; step < kSteps; ++step) {
    // Semi-implicit Euler, which keeps the undamped oscillator's energy
    // bounded.
    errorRate += dt * (-positionGain * error - velocityGain * errorRate);
    error += dt * errorRate;
  }
  return std::abs(error) + std::abs(errorRate);
}

TEST_F(JointMimicDynamicsConstraintTest,
       GainsThatDoNotMakeAStableErrorAreRefused) {
  constexpr scalar_t kNaN = std::numeric_limits<scalar_t>::quiet_NaN();
  constexpr scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
  for (const std::vector<scalar_t>& gains :
       {std::vector<scalar_t>{20.0, 0.0}, std::vector<scalar_t>{0.0, 5.0},
        std::vector<scalar_t>{-1.0, 5.0}, std::vector<scalar_t>{20.0, -1.0},
        std::vector<scalar_t>{20.0, kNaN}, std::vector<scalar_t>{kNaN, 5.0},
        std::vector<scalar_t>{kInfinity, 5.0},
        std::vector<scalar_t>{20.0, kInfinity}}) {
    const absl::StatusOr<std::unique_ptr<JointMimicDynamicsConstraint>>
        constraint = JointMimicDynamicsConstraint::Create(
            *model_, jointName(0), jointName(1), /*multiplier=*/0.5,
            /*positionGain=*/gains[0], /*velocityGain=*/gains[1]);
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument)
        << gains[0] << ", " << gains[1];
    // The finite refused gains have a pole on or right of the imaginary axis:
    // the error does not decay.
    if (std::isfinite(gains[0]) && std::isfinite(gains[1])) {
      EXPECT_GE(largestPoleRealPart(gains[0], gains[1]), 0.0)
          << gains[0] << ", " << gains[1];
    }
  }
}

TEST_F(JointMimicDynamicsConstraintTest,
       EveryPairOfPositiveGainsIsAcceptedAndMakesTheErrorDecay) {
  // Under-, critically and over-damped pairs, a velocity gain above the
  // position gain among them: stability needs only both gains positive.
  for (const std::vector<scalar_t>& gains :
       {std::vector<scalar_t>{20.0, 5.0}, std::vector<scalar_t>{1.0, 2.0},
        std::vector<scalar_t>{4.0, 4.0}, std::vector<scalar_t>{0.5, 30.0},
        std::vector<scalar_t>{100.0, 0.1}}) {
    const absl::StatusOr<std::unique_ptr<JointMimicDynamicsConstraint>>
        constraint = JointMimicDynamicsConstraint::Create(
            *model_, jointName(0), jointName(1), /*multiplier=*/0.5,
            /*positionGain=*/gains[0], /*velocityGain=*/gains[1]);
    ASSERT_TRUE(constraint.ok())
        << gains[0] << ", " << gains[1] << ": " << constraint.status();
    EXPECT_LT(largestPoleRealPart(gains[0], gains[1]), 0.0)
        << gains[0] << ", " << gains[1];
    // The slowest pole sets the time to decay; twenty of its time constants
    // shrink the error well below 1e-3.
    const scalar_t timeConstant =
        -1.0 / largestPoleRealPart(gains[0], gains[1]);
    EXPECT_LT(errorAfter(gains[0], gains[1], /*duration=*/20.0 * timeConstant),
              1.0e-3)
        << gains[0] << ", " << gains[1];
  }
}

}  // namespace
}  // namespace ocs2::humanoid
