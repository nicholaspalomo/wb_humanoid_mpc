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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

namespace ocs2::humanoid {
namespace {

static constexpr size_t kNumContacts = 2;

// Helper to create basis matrices with typical Atlas parameters.
std::array<ContactWrenchConeBasisMatrix, kNumContacts> makeTestBasisMatrices() {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = 4;
  config.frictionCoefficient = 0.7;
  config.torsionalFrictionCoefficient = 0.05;
  config.minNormalForce = 5.0;
  config.gripperForce = 0.0;

  // Approximate Atlas foot dimensions
  const PolygonBounds bounds(-0.10, 0.10, -0.05, 0.05);
  const ContactCenterPoint centerL("foot_l_contact", "l_leg_akx", vector3_t::Zero());
  const ContactCenterPoint centerR("foot_r_contact", "r_leg_akx", vector3_t::Zero());

  return {ContactWrenchConeBasisMatrix(config, ContactRectangle(bounds, centerL)),
          ContactWrenchConeBasisMatrix(config, ContactRectangle(bounds, centerR))};
}

class BasisInputsModelDecoratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Resolve config paths from the installed drc_atlas packages
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");

    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    const std::string referenceFile = configDir + "/config/command/reference.yaml";
    const std::string urdfFile = descriptionDir + "/urdf/atlas.urdf";

    // Create model settings and pinocchio interface — stored as members to avoid dangling references.
    // MpcRobotModelBase stores modelSettings as `const ModelSettings&`, so the object must outlive the model.
    modelSettings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "basis_decorator_test", "false");
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *modelSettings_));
    centroidalModelInfo_ = std::make_unique<CentroidalModelInfo>(centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF));

    // Create the wrapped model
    std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> wrappedModel =
        std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, *centroidalModelInfo_);
    wrenchInputDim_ = wrappedModel->getInputDim();
    stateDim_ = wrappedModel->getStateDim();
    jointDim_ = wrappedModel->getJointDim();

    basisMatrices_ = makeTestBasisMatrices();
    numBasisPerFoot_ = basisMatrices_[0].numBasis();

    decorator_ = std::make_unique<BasisInputsModelDecorator<scalar_t>>(std::move(wrappedModel), basisMatrices_);
  }

  // These must outlive the model due to const-reference semantics in MpcRobotModelBase.
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<CentroidalModelInfo> centroidalModelInfo_;

  std::array<ContactWrenchConeBasisMatrix, kNumContacts> basisMatrices_ = makeTestBasisMatrices();
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> decorator_;
  size_t numBasisPerFoot_ = 0;
  size_t wrenchInputDim_ = 0;
  size_t stateDim_ = 0;
  size_t jointDim_ = 0;
};

// ==================== Dimension tests ====================

TEST_F(BasisInputsModelDecoratorTest, InputDimIsCorrect) {
  const size_t expectedInputDim = numBasisPerFoot_ * kNumContacts + jointDim_;
  EXPECT_EQ(decorator_->getInputDim(), expectedInputDim);
  EXPECT_NE(decorator_->getInputDim(), wrenchInputDim_) << "Decorated input dim should differ from wrench input dim";
}

TEST_F(BasisInputsModelDecoratorTest, StateDimUnchanged) {
  EXPECT_EQ(decorator_->getStateDim(), stateDim_);
}

TEST_F(BasisInputsModelDecoratorTest, NumBasisPerFoot) {
  EXPECT_EQ(decorator_->getNumBasisPerFoot(), numBasisPerFoot_);
}

TEST_F(BasisInputsModelDecoratorTest, JointDimUnchanged) {
  EXPECT_EQ(decorator_->getJointDim(), jointDim_);
}

// ==================== Contact wrench start indices ====================

TEST_F(BasisInputsModelDecoratorTest, ContactWrenchStartIndices) {
  EXPECT_EQ(decorator_->getContactWrenchStartIndices(0), 0u);
  EXPECT_EQ(decorator_->getContactWrenchStartIndices(1), numBasisPerFoot_);
}

TEST_F(BasisInputsModelDecoratorTest, JointVelocitiesStartIndex) {
  EXPECT_EQ(decorator_->getJointVelocitiesStartindex(), numBasisPerFoot_ * kNumContacts);
}

// ==================== Wrench round-trip: set → get ====================

TEST_F(BasisInputsModelDecoratorTest, SetGetContactWrenchRoundTrip) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  // Set a known wrench for foot 0
  vector6_t wrench0;
  wrench0 << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5;
  decorator_->setContactWrench(input, wrench0, 0);

  // Get it back — should recover the wrench via B * B⁺ * W (projection onto column space)
  const matrix_t& B = basisMatrices_[0].getBasisMatrix();
  const matrix_t& B_pinv = basisMatrices_[0].getBasisMatrixPseudoInverse();
  vector6_t W_projected = B * (B_pinv * wrench0);

  vector6_t wrench0_recovered = decorator_->getContactWrench(input, 0);
  EXPECT_TRUE(wrench0_recovered.isApprox(W_projected, 1e-9))
      << "setContactWrench → getContactWrench should round-trip (modulo projection):\n"
      << "  set = " << wrench0.transpose() << "\n  got = " << wrench0_recovered.transpose()
      << "\n  expected (projected) = " << W_projected.transpose();
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactForceRoundTrip) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  vector3_t force1;
  force1 << 5.0, -3.0, 80.0;
  decorator_->setContactForce(input, force1, 1);

  vector3_t force1_recovered = decorator_->getContactForce(input, 1);

  vector6_t wrench_padded = vector6_t::Zero();
  wrench_padded.head<3>() = force1;
  const matrix_t& B = basisMatrices_[1].getBasisMatrix();
  const matrix_t& B_pinv = basisMatrices_[1].getBasisMatrixPseudoInverse();
  vector3_t force_projected = (B * (B_pinv * wrench_padded)).head<3>();

  EXPECT_TRUE(force1_recovered.isApprox(force_projected, 1e-9))
      << "setContactForce → getContactForce round-trip:\n"
      << "  set = " << force1.transpose() << "\n  got = " << force1_recovered.transpose();
}

// ==================== Independence of contacts ====================

TEST_F(BasisInputsModelDecoratorTest, ContactsAreIndependent) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  vector6_t wrench0;
  wrench0 << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5;
  decorator_->setContactWrench(input, wrench0, 0);

  vector6_t wrench1 = decorator_->getContactWrench(input, 1);
  EXPECT_TRUE(wrench1.isZero(1e-12)) << "Setting contact 0 wrench should not affect contact 1:\n  wrench1 = " << wrench1.transpose();
}

// ==================== Joint velocity pass-through ====================

TEST_F(BasisInputsModelDecoratorTest, JointVelocitiesPassThrough) {
  const size_t inputDim = decorator_->getInputDim();

  vector_t state = vector_t::Zero(stateDim_);
  vector_t input = vector_t::Zero(inputDim);

  vector_t jointVels = vector_t::LinSpaced(jointDim_, 0.1, 1.0);
  decorator_->setJointVelocities(state, input, jointVels);

  vector_t jointVels_recovered = decorator_->getJointVelocities(state, input);
  EXPECT_TRUE(jointVels_recovered.isApprox(jointVels, 1e-12)) << "Joint velocities should pass through unchanged";
}

// ==================== Clone test ====================

TEST_F(BasisInputsModelDecoratorTest, CloneProducesWorkingCopy) {
  MpcRobotModelBase<scalar_t>* cloned = decorator_->clone();
  ASSERT_NE(cloned, nullptr);

  EXPECT_EQ(cloned->getInputDim(), decorator_->getInputDim());
  EXPECT_EQ(cloned->getStateDim(), decorator_->getStateDim());
  EXPECT_EQ(cloned->getJointDim(), decorator_->getJointDim());

  const size_t inputDim = cloned->getInputDim();
  vector_t input = vector_t::Zero(inputDim);
  vector6_t wrench;
  wrench << 1.0, 2.0, 50.0, 0.1, -0.1, 0.01;
  cloned->setContactWrench(input, wrench, 0);

  vector6_t wrench_recovered = cloned->getContactWrench(input, 0);
  EXPECT_FALSE(wrench_recovered.isZero(1e-6));

  delete cloned;
}

// ==================== State delegate tests ====================

TEST_F(BasisInputsModelDecoratorTest, StateDelegatesMatchWrappedModel) {
  vector_t state = vector_t::Zero(stateDim_);

  vector3_t pos;
  pos << 1.0, 2.0, 0.95;
  decorator_->setBasePosition(state, pos);

  vector3_t pos_recovered = decorator_->getBasePosition(state);
  EXPECT_TRUE(pos_recovered.isApprox(pos, 1e-12)) << "Base position should delegate to wrapped model";

  vector_t jointAngles = vector_t::LinSpaced(jointDim_, -0.5, 0.5);
  decorator_->setJointAngles(state, jointAngles);
  vector_t jointAngles_recovered = decorator_->getJointAngles(state);
  EXPECT_TRUE(jointAngles_recovered.isApprox(jointAngles, 1e-12)) << "Joint angles should delegate to wrapped model";
}

// ==================== Basis matrix accessor ====================

TEST_F(BasisInputsModelDecoratorTest, BasisMatrixAccessor) {
  for (size_t i = 0; i < kNumContacts; ++i) {
    const matrix_t& B = decorator_->getBasisMatrix(i);
    EXPECT_EQ(B.rows(), 6);
    EXPECT_EQ(B.cols(), static_cast<int>(numBasisPerFoot_));

    EXPECT_TRUE(B.isApprox(basisMatrices_[i].getBasisMatrix(), 1e-12)) << "Accessor should return the same basis matrix for contact " << i;
  }
}

// ==================== Full wrench reconstruction ====================

TEST_F(BasisInputsModelDecoratorTest, NonNegativeLambdaProducesValidWrench) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  for (size_t k = 0; k < numBasisPerFoot_; ++k) {
    input(k) = 1.0;
  }

  vector6_t wrench = decorator_->getContactWrench(input, 0);
  EXPECT_GT(wrench(2), 0.0) << "Non-negative lambdas should produce positive Fz";
}

}  // namespace
}  // namespace ocs2::humanoid
