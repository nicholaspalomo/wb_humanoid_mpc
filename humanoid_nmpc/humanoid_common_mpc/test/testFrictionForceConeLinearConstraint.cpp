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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

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

PinocchioInterface createPinocchioInterface() {
  return createDefaultPinocchioInterface(
      ament_index_cpp::get_package_share_directory(
          std::string(kRobotModelPackagePath)) +
      std::string(kUrdfFile));
}

class MockMpcRobotModel : public MpcRobotModelBase<scalar_t> {
 public:
  MockMpcRobotModel()
      : MpcRobotModelBase<scalar_t>(
            ModelSettings(ament_index_cpp::get_package_share_directory(
                              std::string(kRobotModelConfigPackagePath)) +
                              std::string(kTaskFile),
                          ament_index_cpp::get_package_share_directory(
                              std::string(kRobotModelPackagePath)) +
                              std::string(kUrdfFile),
                          "test", "true"),
            kStateDim, kStateDim) {}

  MOCK_METHOD(size_t, getContactForceStartIndices, (size_t), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getContactForce,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));

  // Other virtual methods need stubs
  MOCK_METHOD(MpcRobotModelBase<scalar_t>*, clone, (), (const, override));
  MOCK_METHOD(size_t, getBaseStartindex, (), (const, override));
  MOCK_METHOD(size_t, getJointStartindex, (), (const, override));
  MOCK_METHOD(size_t, getJointVelocitiesStartindex, (), (const, override));
  MOCK_METHOD(size_t, getContactWrenchStartIndices, (size_t),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getGeneralizedCoordinates,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getBasePose, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBasePosition, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBaseOrientationEulerZYX,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBaseComLinearVelocity,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getBaseComVelocity,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getJointAngles, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getJointVelocities,
              (const VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getGeneralizedVelocities,
              (const VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (override));
  MOCK_METHOD(void, setGeneralizedCoordinates,
              (VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBasePose,
              (VECTOR_T<scalar_t>&, const VECTOR6_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBasePosition,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBaseOrientationEulerZYX,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setJointAngles,
              (VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setJointVelocities,
              (VECTOR_T<scalar_t>&, VECTOR_T<scalar_t>&,
               const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, adaptBasePoseHeight, (VECTOR_T<scalar_t>&, scalar_t),
              (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getContactWrench,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getContactMoment,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));
  MOCK_METHOD(void, setContactWrench,
              (VECTOR_T<scalar_t>&, const VECTOR6_T<scalar_t>&, size_t),
              (const, override));
  MOCK_METHOD(void, setContactForce,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&, size_t),
              (const, override));
  MOCK_METHOD(void, setContactMoment,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&, size_t),
              (const, override));
};

MockMpcRobotModel createMpcRobotModel() {
  return MockMpcRobotModel();
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

// Mock classes for testing with GMock
class MockReferenceManager : public SwitchedModelReferenceManager {
 public:
  MockReferenceManager()
      : SwitchedModelReferenceManager(
            std::make_shared<MockGaitSchedule>(),
            std::make_shared<MockSwingTrajectoryPlanner>(),
            createPinocchioInterface(), createMpcRobotModel()) {}

  MOCK_METHOD(std::vector<bool>, getContactFlags, (scalar_t), (const));
};

// Mock callback for the PreComputation function
class MockPreComputationCallback {
 public:
  MOCK_METHOD(matrix3_t, Call,
              (const vector_t&, const vector_t&, const PreComputation&));

  auto AsStdFunction() {
    return [this](const vector_t& state, const vector_t& input,
                  const PreComputation& preComp) -> matrix3_t {
      return Call(state, input, preComp);
    };
  }
};

FrictionForceConeLinearConstraint::Config getConfig() {
  return FrictionForceConeLinearConstraint::Config(
      kFrictionCoefficient, kMinimumNormalForce, kNumBasisVectors);
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

// Test with mock objects to verify function calls
TEST_F(FrictionForceConeLinearConstraintTest, TestIsActiveWithMocks) {
  // Create mock reference manager
  auto mockReferenceManager =
      std::make_unique<::testing::NiceMock<MockReferenceManager>>();
  auto mpcRobotModel = createMpcRobotModel();

  // Set up the mock expectation
  std::vector<bool> contactFlags = {true, false, true, false};
  EXPECT_CALL(*mockReferenceManager, getContactFlags(::testing::_))
      .WillOnce(::testing::Return(contactFlags));

  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      *mockReferenceManager, config_, kContactPointIndex, mpcRobotModel,
      preComputationCallback_);

  // Test the isActive method (should call getContactFlags and return true)
  EXPECT_TRUE(constraint->isActive(0.0));

  // Test with inactive contact point
  EXPECT_CALL(*mockReferenceManager, getContactFlags(::testing::_))
      .WillOnce(::testing::Return(std::vector<bool>{false, true, true}));

  auto constraint2 = std::make_unique<FrictionForceConeLinearConstraint>(
      *mockReferenceManager, config_, 0, mpcRobotModel,
      preComputationCallback_);

  EXPECT_FALSE(constraint2->isActive(0.0));
}

TEST_F(FrictionForceConeLinearConstraintTest, TestGetValueWithMocks) {
  // Set up mock objects
  auto mockModel = std::make_unique<::testing::NiceMock<MockMpcRobotModel>>();
  auto mockCallback = std::make_unique<MockPreComputationCallback>();
  auto referenceManager = createReferenceManager();

  // Mock the force and index values
  vector3_t expectedForce(3.0, 1.5, 10.0);
  size_t expectedForceIndex = 42;

  // Set expectations
  EXPECT_CALL(*mockModel, getContactForce(::testing::_, kContactPointIndex))
      .WillOnce(::testing::Return(expectedForce));

  EXPECT_CALL(*mockModel, getContactForceStartIndices(kContactPointIndex))
      .WillRepeatedly(::testing::Return(expectedForceIndex));

  EXPECT_CALL(*mockCallback, Call(::testing::_, ::testing::_, ::testing::_))
      .WillOnce(::testing::Return(matrix3_t::Identity()));

  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, *mockModel,
      mockCallback->AsStdFunction());

  // Call and test
  vector_t state = vector_t::Zero(kStateDim);
  vector_t input = vector_t::Zero(kStateDim);

  // This should trigger the mock calls
  vector_t value = constraint->getValue(0.0, state, input, PreComputation());

  // Verify result size and some expected values
  EXPECT_EQ(value.size(), kNumBasisVectors + 1);

  // Test with rotated coordinate frame
  matrix3_t rotation;
  rotation << 0, 1, 0, -1, 0, 0, 0, 0, 1;

  EXPECT_CALL(*mockModel, getContactForce(::testing::_, kContactPointIndex))
      .WillOnce(::testing::Return(expectedForce));

  EXPECT_CALL(*mockCallback, Call(::testing::_, ::testing::_, ::testing::_))
      .WillOnce(::testing::Return(rotation));

  vector_t valueRotated =
      constraint->getValue(0.0, state, input, PreComputation());

  // The rotated value should be different from the original
  EXPECT_NE((valueRotated - value).norm(), 0.0);
}

TEST_F(FrictionForceConeLinearConstraintTest,
       TestLinearApproximationWithMocks) {
  // Set up mock objects
  auto mockModel = std::make_unique<::testing::NiceMock<MockMpcRobotModel>>();
  auto mockCallback = std::make_unique<MockPreComputationCallback>();
  auto referenceManager = createReferenceManager();

  // Mock the force and index values
  vector3_t expectedForce(3.0, 1.5, 10.0);
  size_t expectedForceIndex = 3;

  // Set expectations - getLinearApproximation calls getValue and more
  EXPECT_CALL(*mockModel, getContactForce(::testing::_, kContactPointIndex))
      .Times(2)  // Called once each by getValue and getLinearApproximation
      .WillRepeatedly(::testing::Return(expectedForce));

  EXPECT_CALL(*mockModel, getContactForceStartIndices(kContactPointIndex))
      .Times(::testing::AtLeast(3))  // Called multiple times
      .WillRepeatedly(::testing::Return(expectedForceIndex));

  EXPECT_CALL(*mockCallback, Call(::testing::_, ::testing::_, ::testing::_))
      .Times(2)  // Called once each by getValue and getLinearApproximation
      .WillRepeatedly(::testing::Return(matrix3_t::Identity()));

  auto constraint = std::make_unique<FrictionForceConeLinearConstraint>(
      referenceManager, config_, kContactPointIndex, *mockModel,
      mockCallback->AsStdFunction());

  // Call and test
  vector_t state = vector_t::Zero(kStateDim);
  vector_t input = vector_t::Zero(kStateDim);

  // First call getValue to verify it works
  vector_t value = constraint->getValue(0.0, state, input, PreComputation());

  // Then call getLinearApproximation which should use the same mocks
  VectorFunctionLinearApproximation linearApprox =
      constraint->getLinearApproximation(0.0, state, input, PreComputation());

  // Verify jacobian has entries only in the right spot
  EXPECT_TRUE(linearApprox.dfdx.isZero());

  // The Jacobian w.r.t input should be zero except at the force indices
  for (int col = 0; col < linearApprox.dfdu.cols(); ++col) {
    if (col < expectedForceIndex || col >= expectedForceIndex + 3) {
      for (int row = 0; row < linearApprox.dfdu.rows(); ++row) {
        EXPECT_EQ(linearApprox.dfdu(row, col), 0.0);
      }
    }
  }

  // The part corresponding to the force should be non-zero
  Eigen::Matrix<scalar_t, Eigen::Dynamic, 3> forceJacobian =
      linearApprox.dfdu.block(0, expectedForceIndex, kNumBasisVectors + 1, 3);
  EXPECT_FALSE(forceJacobian.isZero());
}

}  // namespace ocs2::humanoid
