/******************************************************************************
Copyright (c) 2025, Nicholas Palomo and Manuel Yves Galliker. All rights
reserved.

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

#include <absl/strings/str_cat.h>
#include <gtest/gtest.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_common_mpc/common/MockMpcRobotModel.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeLinearConstraint.h"
#include "humanoid_common_mpc/gait/MockGaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/MockSwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

constexpr scalar_t kFrictionCoefficient = 0.5;
constexpr scalar_t kMinimumNormalForce = 10.0;
constexpr size_t kNumBasisVectors = 4;
constexpr size_t kContactPointIndex = 0;
constexpr size_t kStateDim = 34;
constexpr std::string_view kRobotModelPackagePath = "drc_atlas_description";
constexpr std::string_view kRobotModelConfigPackagePath =
    "drc_atlas_centroidal_mpc";
constexpr std::string_view kUrdfFile = "/urdf/atlas.urdf";
constexpr std::string_view kTaskFile = "/config/mpc/task.info";

FrictionForceConeLinearConstraint::Config getConfig() {
  return FrictionForceConeLinearConstraint::Config(
      kFrictionCoefficient, kMinimumNormalForce, kNumBasisVectors);
}

PinocchioInterface createPinocchioInterface() {
  return createDefaultPinocchioInterface(
      ament_index_cpp::get_package_share_directory(
          std::string(kRobotModelPackagePath)) +
      std::string(kUrdfFile));
}

SwitchedModelReferenceManager createReferenceManager() {
  PinocchioInterface pinocchioInterface = createPinocchioInterface();
  const auto& taskFile =
      absl::StrCat(ament_index_cpp::get_package_share_directory(
                       std::string(kRobotModelConfigPackagePath)),
                   kTaskFile);
  const auto& urdfFile =
      absl::StrCat(ament_index_cpp::get_package_share_directory(
                       std::string(kRobotModelPackagePath)),
                   kUrdfFile);
  MockMpcRobotModel<scalar_t> mpcRobotModel(
      ModelSettings(taskFile, urdfFile, "test", "true"), kStateDim, kStateDim);
  auto gaitSchedulePtr = std::make_shared<MockGaitSchedule>();
  auto swingTrajectoryPtr = std::make_shared<MockSwingTrajectoryPlanner>();
  return SwitchedModelReferenceManager(std::move(gaitSchedulePtr),
                                       std::move(swingTrajectoryPtr),
                                       pinocchioInterface, mpcRobotModel);
}

class FrictionForceConeLinearConstraintTest : public ::testing::Test {
 protected:
  FrictionForceConeLinearConstraintTest() {}
  ~FrictionForceConeLinearConstraintTest() override = default;

  FrictionForceConeLinearConstraint::Config config_ = getConfig();
  FrictionForceConeLinearConstraint::PreComputationCallback
      preComputationCallback_ =
          [](const vector_t&, const vector_t&, const PreComputation&) {
            return matrix3_t::Identity();
          };
  std::unique_ptr<FrictionForceConeLinearConstraint> constraint_;
};

TEST_F(FrictionForceConeLinearConstraintTest, TestGetNumConstraints) {
  // TODO: Implement test
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetBasisVectors) {
  // TODO: Implement test
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetValue) {
  // TODO: Implement test
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetLinearApproximation) {
  // TODO: Implement test
}

}  // namespace ocs2::humanoid
