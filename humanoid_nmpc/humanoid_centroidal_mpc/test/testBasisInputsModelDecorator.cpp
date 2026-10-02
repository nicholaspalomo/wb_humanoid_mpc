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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

static constexpr size_t kNumContacts = 2;

/// Base yaw used by the frame tests: at +90deg the world x/y axes of the contact frames are swapped, which makes
/// any missing or transposed rotation in the world-frame accessors visible.
static constexpr scalar_t kYaw90 = M_PI / 2.0;
/// Nominal standing height of the Atlas pelvis; only there to keep the test configuration physically plausible,
/// the frame *orientations* do not depend on it.
static constexpr scalar_t kBaseHeight = 0.8952;
/// Relative tolerance for comparisons that are exact up to floating-point round-off (rotations, pseudoinverse).
static constexpr scalar_t kTol = 1e-9;
/// Index of the base yaw in the centroidal state [momentum(6), base position(3), base euler ZYX(3), joints].
static constexpr Eigen::Index kBaseYawStateIndex = 9;

/// Cone parameters of the test bases (approximately the Atlas foot).
ContactWrenchConeConstraint::Config makeTestConeConfig() {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = 4;
  config.frictionCoefficient = 0.7;
  config.torsionalFrictionCoefficient = 0.05;
  config.minNormalForce = 5.0;
  config.gripperForce = 0.0;
  return config;
}

ContactRectangle makeTestContactRectangle(size_t contactIndex) {
  const PolygonBounds bounds(-0.10, 0.10, -0.05, 0.05);
  return ContactRectangle(bounds, contactIndex == 0 ? ContactCenterPoint("foot_l_contact", "l_leg_akx", vector3_t::Zero())
                                                    : ContactCenterPoint("foot_r_contact", "r_leg_akx", vector3_t::Zero()));
}

/// The rows of the cone the test bases are built for, which the wrench-space formulation would enforce.
ContactWrenchConeRows makeTestConeRows() {
  return buildLocalWrenchConeRows(makeTestConeConfig(), makeTestContactRectangle(0));
}

// Helper to create basis matrices of the given generator set with typical Atlas parameters.
std::array<ContactWrenchConeBasisMatrix, kNumContacts> makeTestBasisMatrices(
    absl::string_view generatorSet = kConservativeInnerApproximationGeneratorSet) {
  absl::StatusOr<ContactWrenchConeBasisMatrix> left =
      ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle(0), generatorSet);
  absl::StatusOr<ContactWrenchConeBasisMatrix> right =
      ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle(1), generatorSet);
  if (!left.ok() || !right.ok()) {
    throw std::runtime_error(absl::StrCat("cannot build the test basis: ", left.status().ToString(), " / ", right.status().ToString()));
  }
  return {*std::move(left), *std::move(right)};
}

/**
 * Whether a local wrench is a non-negative combination of the columns of B, i.e. inside the cone the basis SPANS. That
 * is what an exact round trip through the setters needs. It is not the same as satisfying the wrench-space rows
 * (makeTestConeRows): the default conservative_inner_approximation set spans only part of that cone.
 */
bool isInConeOfBasis(const matrix_t& B, const vector6_t& wrench) {
  const vector_t lambda = solveNonNegativeLeastSquares(B, wrench);
  return lambda.minCoeff() >= 0.0 && (B * lambda - wrench).norm() <= kTol * std::max<scalar_t>(1.0, wrench.norm());
}

/// Applies the same rotation to the force and the moment part of a wrench: W_world = blkdiag(R, R) * W_local.
vector6_t rotateWrench(const matrix3_t& R, const vector6_t& wrench) {
  vector6_t rotated;
  rotated.head<3>() = R * wrench.head<3>();
  rotated.tail<3>() = R * wrench.tail<3>();
  return rotated;
}

class BasisInputsModelDecoratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The DRC Atlas files, from the test's runfiles.
    const std::string taskFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    const std::string referenceFile =
        robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    const std::string urdfFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

    // Create model settings and pinocchio interface — stored as members to avoid dangling references.
    // MpcRobotModelBase stores modelSettings as `const ModelSettings&`, so the object must outlive the model.
    modelSettings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "basis_decorator_test", /*verbose=*/false);
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

    // A second, undecorated wrench-space model for the tests that compare the two input parameterizations.
    wrenchModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, *centroidalModelInfo_);

    basisMatrices_ = makeTestBasisMatrices();
    numBasisPerFoot_ = basisMatrices_[0].numBasis();

    // The decorator needs the pinocchio interface to evaluate the contact frame orientation for the
    // world-frame accessors; it keeps its own copy.
    decorator_ = std::make_unique<BasisInputsModelDecorator<scalar_t>>(std::move(wrappedModel), basisMatrices_, *pinocchioInterface_);
  }

  /// Standing configuration with the given base yaw and pitch, zero roll and all joints at zero. With zero joint
  /// angles the Atlas leg chain has no rotational offsets, so the contact frames are aligned with the base and
  /// the base orientation alone determines their orientation in the world.
  vector_t makeState(scalar_t yaw, scalar_t pitch = 0.0) const {
    vector_t state = vector_t::Zero(stateDim_);
    decorator_->setBasePosition(state, vector3_t(0.0, 0.0, kBaseHeight));
    decorator_->setBaseOrientationEulerZYX(state, vector3_t(yaw, pitch, 0.0));
    decorator_->setJointAngles(state, vector_t::Zero(jointDim_));
    return state;
  }

  /// Contact frame orientation w_R_l computed with Pinocchio directly, independently of the decorator: fresh
  /// Data, forward kinematics on the generalized coordinates taken straight from the state layout, and the
  /// frame placement of the contact frame looked up by name.
  matrix3_t computeContactFrameRotationWithPinocchio(const vector_t& state, size_t contactIndex) const {
    const vector_t q = state.tail(6 + jointDim_);  // [base position, base euler ZYX, joint angles]
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    pinocchio::Data data(model);
    updateFramePlacements<scalar_t>(q, model, data);
    const pinocchio::FrameIndex frameId = model.getFrameId(modelSettings_->contactNames[contactIndex]);
    return data.oMf[frameId].rotation();
  }

  /// The basis-vector scalings λ of the given contact as stored in the input.
  vector_t getLambda(const vector_t& input, size_t contactIndex) const {
    return input.segment(decorator_->getContactWrenchStartIndices(contactIndex), numBasisPerFoot_);
  }

  // These must outlive the model due to const-reference semantics in MpcRobotModelBase.
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<CentroidalModelInfo> centroidalModelInfo_;

  std::array<ContactWrenchConeBasisMatrix, kNumContacts> basisMatrices_ = makeTestBasisMatrices();
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> decorator_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> wrenchModel_;
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
  EXPECT_EQ(decorator_->getWrenchInputDim(), wrenchInputDim_);
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

  // Set a known wrench for foot 0. It is inside the cone of the basis, so the round trip is exact. See
  // SetContactWrenchReproducesConeWrenchesWhoseMinimumNormScalingsAreNegative for one where the minimum-norm scalings
  // alone would not do.
  vector6_t wrench0;
  wrench0 << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
  decorator_->setContactWrench(input, wrench0, /*contactIndex=*/0);

  // Get it back — should recover the wrench via B * B⁺ * W (projection onto column space)
  const matrix_t& B = basisMatrices_[0].getBasisMatrix();
  const matrix_t& B_pinv = basisMatrices_[0].getBasisMatrixPseudoInverse();
  vector6_t W_projected = B * (B_pinv * wrench0);

  vector6_t wrench0_recovered = decorator_->getContactWrench(input, /*contactIndex=*/0);
  EXPECT_TRUE(wrench0_recovered.isApprox(W_projected, 1e-9))
      << "setContactWrench → getContactWrench should round-trip (modulo projection):\n"
      << "  set = " << wrench0.transpose() << "\n  got = " << wrench0_recovered.transpose()
      << "\n  expected (projected) = " << W_projected.transpose();
}

TEST_F(BasisInputsModelDecoratorTest, SetContactWrenchReproducesConeWrenchesWhoseMinimumNormScalingsAreNegative) {
  // This wrench is inside the cone of the basis, but its minimum-norm scalings have a negative entry. Clamping them at
  // zero - what setContactWrench used to do - returns a different wrench; the non-negative least-squares solve must
  // return this one exactly.
  vector6_t wrench;
  wrench << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5;
  const matrix_t& B = basisMatrices_[0].getBasisMatrix();
  const vector_t minimumNormLambda = basisMatrices_[0].getBasisMatrixPseudoInverse() * wrench;
  ASSERT_LT(minimumNormLambda.minCoeff(), 0.0) << "this test needs a wrench whose minimum-norm scalings are not all non-negative";
  ASSERT_GE(makeTestConeRows().evaluateCone(wrench).minCoeff(), 0.0) << "this test needs a wrench the wrench-space cone admits";
  ASSERT_TRUE(isInConeOfBasis(B, wrench)) << "this test needs a wrench the basis can reproduce, which is more than the rows admitting it";
  ASSERT_GT((B * minimumNormLambda.cwiseMax(0.0) - wrench).norm(), 1e-3 * wrench.norm()) << "the old clamp must be visibly wrong here";

  vector_t input = vector_t::Zero(decorator_->getInputDim());
  decorator_->setContactWrench(input, wrench, /*contactIndex=*/0);

  EXPECT_GE(getLambda(input, /*contactIndex=*/0).minCoeff(), 0.0) << "the scalings written into the input must be non-negative";
  EXPECT_TRUE(decorator_->getContactWrench(input, /*contactIndex=*/0).isApprox(wrench, kTol))
      << "set " << wrench.transpose() << ", got back " << decorator_->getContactWrench(input, /*contactIndex=*/0).transpose();
}

TEST_F(BasisInputsModelDecoratorTest, SetContactWrenchProjectsAWrenchOutsideTheConeIntoIt) {
  vector6_t outside;
  outside << 200.0, 0.0, 100.0, 0.0, 0.0, 0.0;  // well beyond the friction limit
  const ContactWrenchConeRows rows = makeTestConeRows();
  ASSERT_LT(rows.evaluateCone(outside).minCoeff(), 0.0);
  vector_t input = vector_t::Zero(decorator_->getInputDim());
  decorator_->setContactWrench(input, outside, /*contactIndex=*/0);
  EXPECT_GE(getLambda(input, /*contactIndex=*/0).minCoeff(), 0.0);
  EXPECT_GE(rows.evaluateCone(vector6_t(decorator_->getContactWrench(input, /*contactIndex=*/0))).minCoeff(), -1e-9 * outside.norm());
}

TEST_F(BasisInputsModelDecoratorTest, PitchedStanceFootWeightRoundTripsThroughTheWorldFrameSetter) {
  // The weight-compensation warm start and the MRT's fallback torques write the world-vertical weight of the robot
  // through setContactForceInWorldFrame. Seen from a stance foot pitched by 15 degrees (heel strike, toe-off) that force
  // has a tangential component and its minimum-norm scalings go negative; on the Atlas cone the old clamp returned a
  // wrench 7 % too large with a 6 % sideways push. It is inside the friction cone, so it has to come back exactly.
  const scalar_t totalGravitationalForce = centroidalModelInfo_->robotMass * 9.81;
  for (const scalar_t pitch : {-15.0 * M_PI / 180.0, 15.0 * M_PI / 180.0}) {
    const vector_t state = makeState(/*yaw=*/0.3, pitch);
    const vector3_t weight(0.0, 0.0, totalGravitationalForce / 2.0);
    for (size_t i = 0; i < kNumContacts; ++i) {
      vector6_t weightInWorld = vector6_t::Zero();
      weightInWorld.head<3>() = weight;
      const vector6_t weightInLocal = decorator_->rotateWrenchWorldToLocal(state, weightInWorld, i);
      ASSERT_LT((basisMatrices_[i].getBasisMatrixPseudoInverse() * weightInLocal).minCoeff(), 0.0)
          << "at this pitch the minimum-norm scalings of the weight must have a negative entry, or the test exercises nothing";

      vector_t input = vector_t::Zero(decorator_->getInputDim());
      decorator_->setContactForceInWorldFrame(state, input, weight, i);
      EXPECT_GE(getLambda(input, i).minCoeff(), 0.0);
      EXPECT_LE((decorator_->getContactForceInWorldFrame(state, input, i) - weight).norm(), kTol * weight.norm())
          << "pitch " << pitch << " contact " << i << ": got " << decorator_->getContactForceInWorldFrame(state, input, i).transpose();
      EXPECT_LE(decorator_->getContactMomentInWorldFrame(state, input, i).norm(), kTol * weight.norm());
    }
    // The same through the helper that builds the whole weight-compensating input.
    const vector_t compensating = weightCompensatingInput(*pinocchioInterface_, {true, true}, *decorator_, state);
    for (size_t i = 0; i < kNumContacts; ++i) {
      EXPECT_LE((decorator_->getContactForceInWorldFrame(state, compensating, i) - weight).norm(), 1e-6) << "pitch " << pitch;
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, ExactGeneratorSetDecoratesTheSameModel) {
  // The exact wrench cone set changes only the number of scalings per foot; every accessor keeps its meaning.
  const std::array<ContactWrenchConeBasisMatrix, kNumContacts> exactBases = makeTestBasisMatrices(kExactWrenchConeGeneratorSet);
  BasisInputsModelDecorator<scalar_t> exact(
      std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, *centroidalModelInfo_), exactBases,
      *pinocchioInterface_);
  const size_t numBasisPerFoot = 8 * makeTestConeConfig().numBasisVectors;
  EXPECT_EQ(exact.getNumBasisPerFoot(), numBasisPerFoot);
  EXPECT_EQ(exact.getInputDim(), numBasisPerFoot * kNumContacts + jointDim_);
  EXPECT_EQ(exact.getJointVelocitiesStartindex(), numBasisPerFoot * kNumContacts);

  // Every wrench of the cone round-trips, including an extreme ray, whose minimum-norm scalings are negative.
  const vector_t state = makeState(kYaw90, /*pitch=*/0.2);
  const ContactWrenchConeRows rows = makeTestConeRows();
  for (size_t i = 0; i < kNumContacts; ++i) {
    const vector6_t extremeRay = 300.0 * exactBases[i].getBasisMatrix().col(0);
    ASSERT_LT((exactBases[i].getBasisMatrixPseudoInverse() * extremeRay).minCoeff(), 0.0);
    vector6_t interior;
    interior << 30.0, -20.0, 300.0, 5.0, -10.0, 4.0;
    ASSERT_GT(rows.evaluateCone(interior).minCoeff(), 0.0);
    for (const vector6_t& wrenchLocal : {extremeRay, interior}) {
      const vector6_t wrenchWorld = rotateWrench(computeContactFrameRotationWithPinocchio(state, i), wrenchLocal);
      vector_t input = vector_t::Zero(exact.getInputDim());
      exact.setContactWrenchInWorldFrame(state, input, wrenchWorld, i);
      EXPECT_GE(input.segment(exact.getContactWrenchStartIndices(i), numBasisPerFoot).minCoeff(), 0.0);
      EXPECT_TRUE(exact.getContactWrench(input, i).isApprox(wrenchLocal, kTol)) << exact.getContactWrench(input, i).transpose();
      EXPECT_TRUE(exact.getContactWrenchInWorldFrame(state, input, i).isApprox(wrenchWorld, kTol));
      EXPECT_TRUE(input.tail(jointDim_).isZero(0.0));
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactForceRoundTrip) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  vector3_t force1;
  force1 << 5.0, -3.0, 80.0;
  decorator_->setContactForce(input, force1, /*contactIndex=*/1);

  vector3_t force1_recovered = decorator_->getContactForce(input, /*contactIndex=*/1);

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
  wrench0 << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
  decorator_->setContactWrench(input, wrench0, /*contactIndex=*/0);

  vector6_t wrench1 = decorator_->getContactWrench(input, /*contactIndex=*/1);
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
  cloned->setContactWrench(input, wrench, /*contactIndex=*/0);

  vector6_t wrench_recovered = cloned->getContactWrench(input, /*contactIndex=*/0);
  EXPECT_FALSE(wrench_recovered.isZero(1e-6));

  // The clone must carry its own pinocchio copy so that the world-frame accessors keep working after cloning.
  const vector_t state = makeState(kYaw90);
  EXPECT_TRUE(cloned->getContactWrenchInWorldFrame(state, input, /*contactIndex=*/0)
                  .isApprox(decorator_->getContactWrenchInWorldFrame(state, input, /*contactIndex=*/0), kTol));

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
    EXPECT_TRUE(decorator_->getBasisMatrices()[i].isApprox(B, 1e-12));
    EXPECT_TRUE(decorator_->getBasisMatrixPseudoInverse(i).isApprox(basisMatrices_[i].getBasisMatrixPseudoInverse(), 1e-12));
  }
}

// ==================== Full wrench reconstruction ====================

TEST_F(BasisInputsModelDecoratorTest, NonNegativeLambdaProducesValidWrench) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  for (size_t k = 0; k < numBasisPerFoot_; ++k) {
    input(k) = 1.0;
  }

  vector6_t wrench = decorator_->getContactWrench(input, /*contactIndex=*/0);
  EXPECT_GT(wrench(2), 0.0) << "Non-negative lambdas should produce positive Fz";
}

// ==================== Local basis-to-wrench map ====================

TEST_F(BasisInputsModelDecoratorTest, LocalBasisToWrenchMapHasBlockDiagonalStructure) {
  const matrix_t M = decorator_->getLocalBasisToWrenchMap();

  const Eigen::Index expectedRows = static_cast<Eigen::Index>(6 * kNumContacts + jointDim_);
  const Eigen::Index expectedCols = static_cast<Eigen::Index>(numBasisPerFoot_ * kNumContacts + jointDim_);
  ASSERT_EQ(M.rows(), expectedRows);
  ASSERT_EQ(M.cols(), expectedCols);
  ASSERT_EQ(static_cast<size_t>(M.rows()), wrenchInputDim_);
  ASSERT_EQ(static_cast<size_t>(M.cols()), decorator_->getInputDim());

  // Expected layout: M = blkdiag(B_0, B_1, I_joints), everything else zero.
  matrix_t expected = matrix_t::Zero(expectedRows, expectedCols);
  for (size_t i = 0; i < kNumContacts; ++i) {
    expected.block(6 * i, numBasisPerFoot_ * i, 6, numBasisPerFoot_) = basisMatrices_[i].getBasisMatrix();
  }
  expected.block(6 * kNumContacts, numBasisPerFoot_ * kNumContacts, jointDim_, jointDim_).setIdentity();
  EXPECT_TRUE((M - expected).isZero(0.0)) << "M must be exactly blkdiag(B_0, B_1, I):\nM =\n" << M << "\nexpected =\n" << expected;

  // Consistency with the accessors: mapping an arbitrary basis input reproduces the local-frame wrenches and
  // the pass-through joint velocities.
  const vector_t basisInput = vector_t::Random(expectedCols);
  const vector_t wrenchInput = M * basisInput;
  const vector_t state = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < kNumContacts; ++i) {
    EXPECT_TRUE(wrenchInput.segment(6 * i, 6).isApprox(decorator_->getContactWrench(basisInput, i), kTol));
  }
  EXPECT_TRUE(wrenchInput.tail(jointDim_).isApprox(decorator_->getJointVelocities(state, basisInput), kTol));
}

// ==================== World-frame accessors (frame correctness) ====================

TEST_F(BasisInputsModelDecoratorTest, WorldFrameAccessorsRotateLocalWrenchWithContactFrame) {
  const vector_t state = makeState(kYaw90);
  vector_t input = vector_t::Zero(decorator_->getInputDim());
  for (size_t i = 0; i < kNumContacts; ++i) {
    // Distinct, strictly positive scalings per foot so that both wrench blocks are non-trivial and inside the cone.
    input.segment(decorator_->getContactWrenchStartIndices(i), numBasisPerFoot_) =
        vector_t::LinSpaced(numBasisPerFoot_, 0.5 + static_cast<scalar_t>(i), 3.0 + static_cast<scalar_t>(i));
  }

  for (size_t i = 0; i < kNumContacts; ++i) {
    const vector6_t W_local_expected = basisMatrices_[i].getBasisMatrix() * getLambda(input, i);
    const vector6_t W_local = decorator_->getContactWrench(input, i);
    EXPECT_TRUE(W_local.isApprox(W_local_expected, kTol)) << "Input-only accessor must return B * lambda (local frame) for contact " << i;

    const matrix3_t w_R_l = computeContactFrameRotationWithPinocchio(state, i);
    EXPECT_TRUE(decorator_->getContactFrameRotationLocalToWorld(state, i).isApprox(w_R_l, kTol))
        << "Decorator contact frame rotation differs from an independent pinocchio evaluation for contact " << i;

    const vector6_t W_world_expected = rotateWrench(w_R_l, W_local_expected);
    const vector6_t W_world = decorator_->getContactWrenchInWorldFrame(state, input, i);
    EXPECT_TRUE(W_world.isApprox(W_world_expected, kTol))
        << "World-frame wrench must equal blkdiag(w_R_l, w_R_l) * B * lambda for contact " << i << ":\n  got      = " << W_world.transpose()
        << "\n  expected = " << W_world_expected.transpose();
    EXPECT_TRUE(W_world.isApprox(decorator_->rotateWrenchLocalToWorld(state, W_local, i), kTol));
    EXPECT_TRUE(decorator_->getContactForceInWorldFrame(state, input, i).isApprox(W_world_expected.head<3>(), kTol));
    EXPECT_TRUE(decorator_->getContactMomentInWorldFrame(state, input, i).isApprox(W_world_expected.tail<3>(), kTol));

    // Sanity: for a yawed base the tangential components genuinely differ between the two frames, i.e. the
    // world-frame accessor is not silently returning the local wrench.
    EXPECT_FALSE(W_world.head<3>().isApprox(W_local.head<3>(), 1e-6))
        << "Yawed contact frame should change the tangential force components for contact " << i;

    // Rotating back must recover the local wrench.
    EXPECT_TRUE(decorator_->rotateWrenchWorldToLocal(state, W_world, i).isApprox(W_local, kTol));
  }
}

TEST_F(BasisInputsModelDecoratorTest, YawedBaseRotatesLocalXForceIntoWorldY) {
  const vector_t stateZeroYaw = makeState(0.0);
  const vector_t stateYaw90 = makeState(kYaw90);
  EXPECT_DOUBLE_EQ(stateYaw90(kBaseYawStateIndex), kYaw90) << "Base yaw is expected at state index " << kBaseYawStateIndex;

  const matrix3_t Rz90 = Eigen::AngleAxis<scalar_t>(kYaw90, vector3_t::UnitZ()).toRotationMatrix();

  for (size_t i = 0; i < kNumContacts; ++i) {
    const matrix3_t R0 = computeContactFrameRotationWithPinocchio(stateZeroYaw, i);
    const matrix3_t R90 = computeContactFrameRotationWithPinocchio(stateYaw90, i);

    // With all joints at zero the Atlas leg chain has no rotational offsets (all URDF joint origins have zero rpy and
    // the contact frame is attached with identity rotation), so at zero yaw the contact frame is aligned with the world.
    EXPECT_TRUE(R0.isApprox(matrix3_t::Identity(), kTol)) << "Contact frame " << i << " should be world-aligned at zero yaw:\n" << R0;
    // The base yaw is the first ZYX Euler angle: a positive yaw is a right-handed rotation about world z of the whole chain.
    EXPECT_TRUE(R90.isApprox(Rz90 * R0, kTol)) << "Yaw = +90deg must rotate contact frame " << i << " about world z:\n" << R90;

    // Direction of the local +x axis in the world frame, taken from Pinocchio rather than assumed.
    const vector3_t localXInWorld = R90.col(0);
    EXPECT_NEAR(localXInWorld.x(), 0.0, kTol);
    EXPECT_NEAR(std::abs(localXInWorld.y()), 1.0, kTol);
    EXPECT_NEAR(localXInWorld.z(), 0.0, kTol);
    const scalar_t ySign = (localXInWorld.y() > 0.0) ? 1.0 : -1.0;
    EXPECT_GT(ySign, 0.0) << "A positive yaw maps local +x onto world +y (right-handed rotation about z)";

    // (1) A purely tangential force lies outside the friction cone, so exercise the rotation helper directly.
    vector6_t W_localX = vector6_t::Zero();
    W_localX(0) = 1.0;
    const vector6_t W_worldX = decorator_->rotateWrenchLocalToWorld(stateYaw90, W_localX, i);
    EXPECT_TRUE(W_worldX.head<3>().isApprox(vector3_t(0.0, ySign, 0.0), kTol))
        << "Local +x force must become a world " << (ySign > 0.0 ? "+" : "-") << "y force for contact " << i << ", got "
        << W_worldX.head<3>().transpose();
    EXPECT_TRUE(W_worldX.tail<3>().isZero(kTol));

    // (2) The same through the input parameterization with an in-cone wrench: local [10, 0, 100, 0, 0, 0].
    vector6_t W_inCone;
    W_inCone << 10.0, 0.0, 100.0, 0.0, 0.0, 0.0;
    vector_t input = vector_t::Zero(decorator_->getInputDim());
    decorator_->setContactWrench(input, W_inCone, i);
    ASSERT_GE(getLambda(input, i).minCoeff(), 0.0);
    ASSERT_TRUE(decorator_->getContactWrench(input, i).isApprox(W_inCone, kTol)) << "A wrench inside the cone must round-trip exactly";

    const vector3_t f_world = decorator_->getContactForceInWorldFrame(stateYaw90, input, i);
    EXPECT_TRUE(f_world.isApprox(vector3_t(0.0, ySign * 10.0, 100.0), kTol))
        << "Local force (10, 0, 100) should become (0, " << ySign * 10.0 << ", 100) in the world for contact " << i << ", got "
        << f_world.transpose();
    EXPECT_TRUE(f_world.isApprox(R90 * W_inCone.head<3>(), kTol));
    // The vertical component is invariant under yaw.
    EXPECT_NEAR(f_world.z(), W_inCone(2), kTol);
  }
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactWrenchInWorldFrameRoundTrip) {
  vector6_t W_local;
  W_local << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;

  for (const scalar_t yaw : {0.0, kYaw90, -0.7}) {
    const vector_t state = makeState(yaw);
    for (size_t i = 0; i < kNumContacts; ++i) {
      // Precondition of an exact round trip: the wrench is inside the cone the basis SPANS. Satisfying the wrench-space
      // rows is not enough, because the default generator set spans only part of that cone.
      ASSERT_TRUE(isInConeOfBasis(basisMatrices_[i].getBasisMatrix(), W_local)) << "Test wrench must be reproducible for contact " << i;

      const matrix3_t w_R_l = computeContactFrameRotationWithPinocchio(state, i);
      const vector6_t W_world_in = rotateWrench(w_R_l, W_local);

      vector_t input = vector_t::Zero(decorator_->getInputDim());
      decorator_->setContactWrenchInWorldFrame(state, input, W_world_in, i);

      EXPECT_GE(getLambda(input, i).minCoeff(), 0.0) << "All lambda must be non-negative (yaw = " << yaw << ", contact " << i << ")";
      EXPECT_TRUE(getLambda(input, 1 - i).isZero(0.0)) << "Setting contact " << i << " must not touch the other contact";
      EXPECT_TRUE(input.tail(jointDim_).isZero(0.0)) << "Setting a contact wrench must not touch the joint velocities";

      // The input-only accessor returns the wrench in the local contact frame, i.e. the un-rotated test wrench.
      EXPECT_TRUE(decorator_->getContactWrench(input, i).isApprox(W_local, kTol))
          << "Local wrench mismatch (yaw = " << yaw << ", contact " << i
          << "):\n  got = " << decorator_->getContactWrench(input, i).transpose() << "\n  expected = " << W_local.transpose();

      const vector6_t W_world_out = decorator_->getContactWrenchInWorldFrame(state, input, i);
      EXPECT_TRUE(W_world_out.isApprox(W_world_in, kTol))
          << "World-frame round trip failed (yaw = " << yaw << ", contact " << i << "):\n  set = " << W_world_in.transpose()
          << "\n  got = " << W_world_out.transpose();
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactForceInWorldFrameRoundTrip) {
  const vector_t state = makeState(kYaw90);
  const vector3_t f_local(5.0, -3.0, 80.0);

  for (size_t i = 0; i < kNumContacts; ++i) {
    const matrix3_t w_R_l = computeContactFrameRotationWithPinocchio(state, i);
    const vector3_t f_world_in = w_R_l * f_local;

    vector_t input = vector_t::Zero(decorator_->getInputDim());
    decorator_->setContactForceInWorldFrame(state, input, f_world_in, i);
    EXPECT_GE(getLambda(input, i).minCoeff(), 0.0);

    EXPECT_TRUE(decorator_->getContactForce(input, i).isApprox(f_local, kTol)) << "Input-only accessor must return the local force";
    EXPECT_TRUE(decorator_->getContactForceInWorldFrame(state, input, i).isApprox(f_world_in, kTol))
        << "World-frame force round trip failed for contact " << i << ":\n  set = " << f_world_in.transpose()
        << "\n  got = " << decorator_->getContactForceInWorldFrame(state, input, i).transpose();
    EXPECT_TRUE(decorator_->getContactMomentInWorldFrame(state, input, i).isZero(1e-9)) << "A pure force must not produce a moment";
  }
}

// ==================== Wrench-space model: world-frame accessors are the input-only accessors ====================

TEST_F(BasisInputsModelDecoratorTest, WrenchSpaceModelWorldFrameAccessorsEqualInputOnlyAccessors) {
  // Go through the base-class interface to exercise the virtual dispatch every consumer relies on.
  const MpcRobotModelBase<scalar_t>& model = *wrenchModel_;
  ASSERT_EQ(model.getInputDim(), wrenchInputDim_);

  for (const scalar_t yaw : {0.0, kYaw90, -1.1}) {
    // The wrench-space input already stores world-frame wrenches, so the state must be irrelevant: use a yawed base
    // *and* random joint angles.
    vector_t state = makeState(yaw);
    model.setJointAngles(state, vector_t::Random(jointDim_));
    const vector_t input = vector_t::Random(wrenchInputDim_);

    for (size_t i = 0; i < kNumContacts; ++i) {
      EXPECT_TRUE((model.getContactWrenchInWorldFrame(state, input, i) - model.getContactWrench(input, i)).isZero(0.0));
      EXPECT_TRUE((model.getContactForceInWorldFrame(state, input, i) - model.getContactForce(input, i)).isZero(0.0));
      EXPECT_TRUE((model.getContactMomentInWorldFrame(state, input, i) - model.getContactMoment(input, i)).isZero(0.0));

      vector6_t wrench;
      wrench << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
      vector_t inputSetWrench = vector_t::Zero(wrenchInputDim_);
      model.setContactWrenchInWorldFrame(state, inputSetWrench, wrench, i);
      EXPECT_TRUE((model.getContactWrench(inputSetWrench, i) - wrench).isZero(0.0));
      EXPECT_TRUE((model.getContactWrenchInWorldFrame(state, inputSetWrench, i) - wrench).isZero(0.0));

      const vector3_t force(5.0, -3.0, 80.0);
      vector_t inputSetForce = vector_t::Zero(wrenchInputDim_);
      model.setContactForceInWorldFrame(state, inputSetForce, force, i);
      EXPECT_TRUE((model.getContactForce(inputSetForce, i) - force).isZero(0.0));
      EXPECT_TRUE(model.getContactMoment(inputSetForce, i).isZero(0.0));
    }
  }
}

// ==================== Weight-compensating input ====================

TEST_F(BasisInputsModelDecoratorTest, WeightCompensatingInputIsVerticalInWorldFrameForYawedBase) {
  const vector_t state = makeState(kYaw90);
  const scalar_t totalGravitationalForce = centroidalModelInfo_->robotMass * 9.81;
  // Looser than kTol: the result is a least-squares solve of an O(1e3) N wrench, compare with an absolute tolerance.
  static constexpr scalar_t kForceTol = 1e-6;

  const vector_t inputDoubleContact = weightCompensatingInput(*pinocchioInterface_, {true, true}, *decorator_, state);
  ASSERT_EQ(static_cast<size_t>(inputDoubleContact.size()), decorator_->getInputDim());
  EXPECT_TRUE(inputDoubleContact.tail(jointDim_).isZero(0.0)) << "Weight compensation must not command joint velocities";

  const vector3_t expectedForcePerFoot(0.0, 0.0, totalGravitationalForce / 2.0);
  for (size_t i = 0; i < kNumContacts; ++i) {
    EXPECT_GE(getLambda(inputDoubleContact, i).minCoeff(), 0.0) << "All lambda must be non-negative for contact " << i;

    // The vertical force is invariant under a yaw of the base, so the world-frame force must be purely vertical
    // even though the input stores it in the (yawed) local contact frame.
    const vector3_t f_world = decorator_->getContactForceInWorldFrame(state, inputDoubleContact, i);
    EXPECT_LT((f_world - expectedForcePerFoot).norm(), kForceTol)
        << "World-frame weight-compensating force for contact " << i << " should be " << expectedForcePerFoot.transpose() << ", got "
        << f_world.transpose();
    EXPECT_TRUE(decorator_->getContactMomentInWorldFrame(state, inputDoubleContact, i).isZero(kForceTol))
        << "Weight compensation must not produce a contact moment for contact " << i;
  }

  // The CentroidalModelInfo variant must produce the identical input.
  const vector_t inputFromInfo = weightCompensatingInput(*centroidalModelInfo_, {true, true}, *decorator_, state);
  EXPECT_TRUE(inputFromInfo.isApprox(inputDoubleContact, kTol));

  // Single support puts the full weight on the stance foot and leaves the swing foot at zero.
  const vector_t inputLeftContact = weightCompensatingInput(*pinocchioInterface_, {true, false}, *decorator_, state);
  EXPECT_GE(getLambda(inputLeftContact, /*contactIndex=*/0).minCoeff(), 0.0);
  EXPECT_LT(
      (decorator_->getContactForceInWorldFrame(state, inputLeftContact, /*contactIndex=*/0) - vector3_t(0.0, 0.0, totalGravitationalForce))
          .norm(),
      kForceTol);
  EXPECT_TRUE(getLambda(inputLeftContact, /*contactIndex=*/1).isZero(0.0));
}

// ==================== Basis-input dynamics: validation before the CppAD build ====================

TEST_F(BasisInputsModelDecoratorTest, BasisDynamicsValidationNamesTheKeyToChangeBeforeBuildingAnything) {
  const std::array<matrix_t, kNumContacts> bases = decorator_->getBasisMatrices();
  // Positive control: the configuration the fixture's decorator runs on is accepted.
  EXPECT_TRUE(CentroidalDynamicsBasisInputsAD::validate(*pinocchioInterface_, *centroidalModelInfo_, *modelSettings_, bases).ok());

  // A contact frame the URDF does not have. Create() must report it instead of compiling a tape around a bad index.
  const std::string taskFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
  const std::string urdfFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();
  ModelSettings brokenSettings(taskFile, urdfFile, "basis_decorator_test", /*verbose=*/false);
  brokenSettings.contactNames[1] = "no_such_frame";
  const absl::StatusOr<std::unique_ptr<CentroidalDynamicsBasisInputsAD>> missingFrame =
      CentroidalDynamicsBasisInputsAD::Create(*pinocchioInterface_, *centroidalModelInfo_, "dynamics", brokenSettings, bases);
  ASSERT_FALSE(missingFrame.ok());
  EXPECT_EQ(missingFrame.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(missingFrame.status().message().find("model_settings.contactNames6DoF"), absl::string_view::npos) << missingFrame.status();
  EXPECT_NE(missingFrame.status().message().find("no_such_frame"), absl::string_view::npos) << missingFrame.status();

  // A contact count the basis-vector formulation does not support.
  CentroidalModelInfo oneContact = *centroidalModelInfo_;
  oneContact.numSixDofContacts = 1;
  const absl::Status wrongCount = CentroidalDynamicsBasisInputsAD::validate(*pinocchioInterface_, oneContact, *modelSettings_, bases);
  EXPECT_EQ(wrongCount.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(wrongCount.message().find("model_settings.contactNames6DoF"), absl::string_view::npos) << wrongCount;

  // Bases of different sizes are a programming error, not a configuration one.
  std::array<matrix_t, kNumContacts> mismatched = bases;
  mismatched[1] = makeTestBasisMatrices(kExactWrenchConeGeneratorSet)[1].getBasisMatrix();
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::validate(*pinocchioInterface_, *centroidalModelInfo_, *modelSettings_, mismatched).code(),
            absl::StatusCode::kInternal);
}

TEST_F(BasisInputsModelDecoratorTest, BasisDynamicsLibraryNameCarriesTheBasisKey) {
  const std::array<matrix_t, kNumContacts> conservative = decorator_->getBasisMatrices();
  const std::array<ContactWrenchConeBasisMatrix, kNumContacts> exactBases = makeTestBasisMatrices(kExactWrenchConeGeneratorSet);
  const std::array<matrix_t, kNumContacts> exact = {exactBases[0].getBasisMatrix(), exactBases[1].getBasisMatrix()};
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics", conservative),
            absl::StrCat("dynamics_", basisInputsLibraryKey(conservative)));
  EXPECT_NE(CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics", conservative),
            CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics", exact));
}

}  // namespace
}  // namespace ocs2::humanoid
