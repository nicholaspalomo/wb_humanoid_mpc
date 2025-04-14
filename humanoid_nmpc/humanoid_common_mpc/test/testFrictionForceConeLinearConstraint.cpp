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

MockMpcRobotModel<scalar_t> createMpcRobotModel() {
  const auto& taskFile =
      absl::StrCat(ament_index_cpp::get_package_share_directory(
                       std::string(kRobotModelConfigPackagePath)),
                   kTaskFile);
  const auto& urdfFile =
      absl::StrCat(ament_index_cpp::get_package_share_directory(
                       std::string(kRobotModelPackagePath)),
                   kUrdfFile);
  return MockMpcRobotModel<scalar_t>(
      ModelSettings(taskFile, urdfFile, "test", "true"), kStateDim, kStateDim);
}

SwitchedModelReferenceManager createReferenceManager() {
  const auto& pinocchioInterface = createPinocchioInterface();
  const auto& mpcRobotModel = createMpcRobotModel();
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
};

TEST_F(FrictionForceConeLinearConstraintTest, TestConstructConstraint) {
  auto mpcRobotModel = createMpcRobotModel();
  auto referenceManager = createReferenceManager();
  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);
  EXPECT_TRUE(constraint != nullptr);
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetNumConstraints) {
  auto mpcRobotModel = createMpcRobotModel();
  auto referenceManager = createReferenceManager();
  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);

  // Number of constraints should be numBasisVectors + 1 (for minimum normal
  // force)
  EXPECT_EQ(constraint->getNumConstraints(0.0), kNumBasisVectors + 1);

  // Test with a different number of basis vectors
  auto customConfig = FrictionForceConeLinearConstraint::Config(
      kFrictionCoefficient, kMinimumNormalForce, 8);
  auto customConstraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, customConfig, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);
  EXPECT_EQ(customConstraint->getNumConstraints(0.0), 8 + 1);
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetBasisVectors) {
  auto mpcRobotModel = createMpcRobotModel();
  auto referenceManager = createReferenceManager();
  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);

  // Get basis vectors
  const matrix_t basisVectors = constraint->getBasisVectors(0.0);

  // Check dimensions
  EXPECT_EQ(basisVectors.rows(), kNumBasisVectors + 1);
  EXPECT_EQ(basisVectors.cols(), 3);

  // Check that friction cone vectors follow the pattern:
  // [cos(theta), sin(theta), -mu]
  for (size_t i = 0; i < kNumBasisVectors; ++i) {
    scalar_t theta = i * 2 * M_PI / kNumBasisVectors;
    EXPECT_NEAR(basisVectors(i, 0), cos(theta), 1e-10);
    EXPECT_NEAR(basisVectors(i, 1), sin(theta), 1e-10);
    EXPECT_NEAR(basisVectors(i, 2), -kFrictionCoefficient, 1e-10);
  }

  // Check minimum normal force constraint
  EXPECT_NEAR(basisVectors(kNumBasisVectors, 0), 0.0, 1e-10);
  EXPECT_NEAR(basisVectors(kNumBasisVectors, 1), 0.0, 1e-10);
  EXPECT_NEAR(basisVectors(kNumBasisVectors, 2), -1.0, 1e-10);
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetValue) {
  auto mpcRobotModel = createMpcRobotModel();
  auto referenceManager = createReferenceManager();
  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);

  // Create state and input vectors
  vector_t state = vector_t::Zero(kStateDim);
  vector_t input = vector_t::Zero(kStateDim);

  // Set up a contact force
  vector3_t contactForce(10.0, 5.0, 20.0);  // Force with Fx=10, Fy=5, Fz=20
  mpcRobotModel.setContactForce(input, contactForce, kContactPointIndex);

  // Compute expected constraint values
  matrix_t basisVectors = constraint->getBasisVectors(0.0);
  vector_t expectedValue = vector_t::Zero(kNumBasisVectors + 1);
  expectedValue(kNumBasisVectors) =
      kMinimumNormalForce;  // Minimum normal force constraint
  expectedValue -= basisVectors * contactForce;

  // Get actual value
  vector_t actualValue =
      constraint->getValue(0.0, state, input, PreComputation());

  // Check results
  EXPECT_EQ(actualValue.size(), kNumBasisVectors + 1);
  for (size_t i = 0; i < actualValue.size(); ++i) {
    EXPECT_NEAR(actualValue(i), expectedValue(i), 1e-10);
  }

  // Test with a vertical force that slightly exceeds the friction cone
  vector3_t frictionConeForce(kFrictionCoefficient * 21 + 0.1, 0.0, 20.0);
  mpcRobotModel.setContactForce(input, frictionConeForce, kContactPointIndex);

  // Get values
  actualValue = constraint->getValue(0.0, state, input, PreComputation());

  // The first constraint should be violated (negative)
  EXPECT_LT(actualValue(0), 0);
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetLinearApproximation) {
  auto mpcRobotModel = createMpcRobotModel();
  auto referenceManager = createReferenceManager();
  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);

  // Create state and input vectors
  vector_t state = vector_t::Zero(kStateDim);
  vector_t input = vector_t::Zero(kStateDim);

  // Set up a contact force
  vector3_t contactForce(10.0, 5.0, 20.0);  // Force with Fx=10, Fy=5, Fz=20
  mpcRobotModel.setContactForce(input, contactForce, kContactPointIndex);

  // Get linear approximation
  VectorFunctionLinearApproximation linearApprox =
      constraint->getLinearApproximation(0.0, state, input, PreComputation());

  // Check dimensions
  EXPECT_EQ(linearApprox.f.size(), kNumBasisVectors + 1);
  EXPECT_EQ(linearApprox.dfdx.rows(), kNumBasisVectors + 1);
  EXPECT_EQ(linearApprox.dfdx.cols(), kStateDim);
  EXPECT_EQ(linearApprox.dfdu.rows(), kNumBasisVectors + 1);
  EXPECT_EQ(linearApprox.dfdu.cols(), kStateDim);

  // Check function value matches getValue()
  vector_t expectedValue =
      constraint->getValue(0.0, state, input, PreComputation());
  for (size_t i = 0; i < linearApprox.f.size(); ++i) {
    EXPECT_NEAR(linearApprox.f(i), expectedValue(i), 1e-10);
  }

  // Check Jacobian with respect to state is zero (constraint doesn't depend on
  // state)
  EXPECT_TRUE(linearApprox.dfdx.isZero(1e-10));

  // Check Jacobian with respect to input for contact force
  matrix_t basisVectors = constraint->getBasisVectors(0.0);
  size_t forceIndex =
      mpcRobotModel.getContactForceStartIndices(kContactPointIndex);

  // Only check the force part of the Jacobian
  Eigen::Matrix<scalar_t, Eigen::Dynamic, 3> forceJacobian =
      linearApprox.dfdu.block(0, forceIndex, kNumBasisVectors + 1, 3);

  // Should be -basisVectors (since we use identity rotation in the test)
  for (size_t i = 0; i < kNumBasisVectors + 1; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(forceJacobian(i, j), -basisVectors(i, j), 1e-10);
    }
  }
}

}  // namespace ocs2::humanoid
