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
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/data.hpp"  // IWYU pragma: keep
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {
namespace {

/// Base yaw used by the frame tests: at +90deg the world x/y axes of the
/// contact frames are swapped, which makes any missing or transposed rotation
/// in the world-frame accessors visible.
constexpr scalar_t kQuarterTurnEulerzyxBaseRtWorldYaw = std::numbers::pi / 2.0;
/// Nominal standing height of the Atlas pelvis; only there to keep the test
/// configuration physically plausible, the frame *orientations* do not depend
/// on it.
constexpr scalar_t kPosBaseRtWorldEwrtWorldZ = 0.8952;
/// Relative tolerance for comparisons that are exact up to floating-point
/// round-off (rotations, pseudoinverse).
constexpr scalar_t kTol = 1.0e-9;
/// Index of the base quaternion in the centroidal state [momentum(6), base
/// position(3), base quaternion (x, y, z, w), joints].
constexpr Eigen::Index kBaseOrientationStateIndex = 9;

/// The fixture's contact set: the two Atlas feet.
constexpr size_t kNumFeet = static_cast<size_t>(kBipedFootCount);

/// Cone parameters of the test bases (approximately the Atlas foot).
ContactWrenchConeConstraint::Config makeTestConeConfig() {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = 4;
  config.frictionCoefficient = 0.7;
  config.torsionalFrictionCoefficient = 0.05;
  config.minForceContactEwrtContactZ = 5.0;
  config.gripForceContactEwrtContactZ = 0.0;
  return config;
}

ContactRectangle makeTestContactRectangle(size_t contactIndex) {
  const PolygonBounds bounds(-0.10, 0.10, -0.05, 0.05);
  return ContactRectangle(
      bounds,
      contactIndex == 0
          ? ContactCenterPoint("foot_l_contact", "l_leg_akx", vector3_t::Zero())
          : ContactCenterPoint("foot_r_contact", "r_leg_akx",
                               vector3_t::Zero()));
}

/// The rows of the cone the test bases are built for, which the wrench-space
/// formulation would enforce.
ContactWrenchConeRows makeTestConeRows() {
  return buildLocalWrenchConeRows(makeTestConeConfig(),
                                  makeTestContactRectangle(0));
}

// Helper to create basis matrices of the given generator set with typical Atlas
// parameters.
std::vector<ContactWrenchConeBasisMatrix> makeTestBasisMatrices(
    absl::string_view generatorSet =
        kConservativeInnerApproximationGeneratorSet) {
  absl::StatusOr<ContactWrenchConeBasisMatrix> left =
      ContactWrenchConeBasisMatrix::Create(
          makeTestConeConfig(), makeTestContactRectangle(0), generatorSet);
  absl::StatusOr<ContactWrenchConeBasisMatrix> right =
      ContactWrenchConeBasisMatrix::Create(
          makeTestConeConfig(), makeTestContactRectangle(1), generatorSet);
  if (!left.ok() || !right.ok()) {
    throw std::runtime_error(
        absl::StrCat("cannot build the test basis: ", left.status().ToString(),
                     " / ", right.status().ToString()));
  }
  return {*std::move(left), *std::move(right)};
}

/**
 * Whether a local wrench is a non-negative combination of the columns of B,
 * i.e. inside the cone the basis SPANS. That is what an exact round trip
 * through the setters needs. It is not the same as satisfying the wrench-space
 * rows (makeTestConeRows): the default conservative_inner_approximation set
 * spans only part of that cone.
 */
bool isInConeOfBasis(const matrix_t& B,
                     const vector6_t& wrenchContactEwrtContact) {
  const vector_t lambda =
      solveNonNegativeLeastSquares(B, wrenchContactEwrtContact);
  return lambda.minCoeff() >= 0.0 &&
         (B * lambda - wrenchContactEwrtContact).norm() <=
             kTol * std::max<scalar_t>(1.0, wrenchContactEwrtContact.norm());
}

/// Applies the same rotation to the force and the moment part of a wrench:
/// W_world = blkdiag(R, R) * W_local.
vector6_t rotateToWorld(const matrix3_t& rotmatContactRtWorld,
                        const vector6_t& wrenchContactEwrtContact) {
  vector6_t wrenchContactEwrtWorld;
  wrenchContactEwrtWorld.head<3>() =
      rotmatContactRtWorld * wrenchContactEwrtContact.head<3>();
  wrenchContactEwrtWorld.tail<3>() =
      rotmatContactRtWorld * wrenchContactEwrtContact.tail<3>();
  return wrenchContactEwrtWorld;
}

class BasisInputsModelDecoratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The typed DRC Atlas files, from the test's runfiles.
    const CentroidalRobotFiles files = atlasFiles();
    urdfFile_ = files.urdfFile;
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
    ASSERT_TRUE(config.ok()) << config.status();
    atlas_ = *std::move(config);

    // Create model settings and pinocchio interface - stored as members to
    // avoid dangling references. MpcRobotModelBase stores modelSettings as
    // `const ModelSettings&`, so the object must outlive the model.
    modelSettings_ = std::make_unique<ModelSettings>(checkedValue(
        ModelSettings::Create(atlas_.task, urdfFile_, "basis_decorator_test",
                              /*verbose=*/false)));
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(checkedValue(
        loadCustomPinocchioInterface(atlas_.task, urdfFile_, *modelSettings_,
                                     /*scaleTotalMass=*/false)));
    centroidalModelInfo_ = std::make_unique<CentroidalModelInfo>(checkedValue(
        centroidalModelInfoOf(atlas_, *pinocchioInterface_, *modelSettings_)));

    // Create the wrapped model
    std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> wrappedModel =
        std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
            *modelSettings_, *pinocchioInterface_, *centroidalModelInfo_);
    wrenchInputDim_ = wrappedModel->getInputDim();
    stateDim_ = wrappedModel->getStateDim();
    jointDim_ = wrappedModel->getJointDim();

    // A second, undecorated wrench-space model for the tests that compare the
    // two input parameterizations.
    wrenchModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
        *modelSettings_, *pinocchioInterface_, *centroidalModelInfo_);

    basisMatrices_ = makeTestBasisMatrices();
    numBasisPerFoot_ = basisMatrices_[0].numBasis();

    // The decorator needs the pinocchio interface to evaluate the contact frame
    // orientation for the world-frame accessors; it keeps its own copy.
    decorator_ = checkedValue(BasisInputsModelDecorator<scalar_t>::Create(
        std::move(wrappedModel), basisMatrices_, *pinocchioInterface_));
  }

  /// Standing configuration with the given base yaw and pitch, zero roll and
  /// all joints at zero. With zero joint angles the Atlas leg chain has no
  /// rotational offsets, so the contact frames are aligned with the base and
  /// the base orientation alone determines their orientation in the world.
  vector_t makeState(scalar_t eulerzyxBaseRtWorldYaw,
                     scalar_t eulerzyxBaseRtWorldPitch = 0.0) const {
    vector_t state = decorator_->getNeutralState();
    decorator_->setPosBaseRtWorldEwrtWorld(
        state, vector3_t(0.0, 0.0, kPosBaseRtWorldEwrtWorldZ));
    decorator_->setEulerzyxBaseRtWorld(
        state,
        vector3_t(eulerzyxBaseRtWorldYaw, eulerzyxBaseRtWorldPitch, 0.0));
    decorator_->setJointAngles(state, vector_t::Zero(jointDim_));
    return state;
  }

  /// Contact frame orientation w_R_l computed with Pinocchio directly,
  /// independently of the decorator: fresh Data, forward kinematics on the
  /// generalized coordinates taken straight from the state layout, and the
  /// frame placement of the contact frame looked up by name.
  matrix3_t computePinocchioRotmatContactRtWorld(const vector_t& state,
                                                 size_t contactIndex) const {
    // The centroidal state ends with the configuration q: [base position, base
    // quaternion (x, y, z, w), joint angles], the quaternion unit as the Euler
    // boundary writes it.
    const vector_t q = state.tail(
        static_cast<Eigen::Index>(decorator_->getConfigurationDim()));
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    pinocchio::Data data(model);
    updateFramePlacements<scalar_t>(q, model, data);
    const pinocchio::FrameIndex frameId =
        model.getFrameId(modelSettings_->contactNames[contactIndex]);
    return data.oMf[frameId].rotation();
  }

  /// The basis-vector scalings λ of the given contact as stored in the input.
  vector_t getLambda(const vector_t& input, size_t contactIndex) const {
    return input.segment(decorator_->getContactWrenchStartIndices(contactIndex),
                         numBasisPerFoot_);
  }

  std::string urdfFile_;
  // The typed DRC Atlas files.
  CentroidalMpcConfig atlas_;
  // These must outlive the model due to const-reference semantics in
  // MpcRobotModelBase.
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<CentroidalModelInfo> centroidalModelInfo_;

  std::vector<ContactWrenchConeBasisMatrix> basisMatrices_ =
      makeTestBasisMatrices();
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> decorator_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> wrenchModel_;
  size_t numBasisPerFoot_ = 0;
  size_t wrenchInputDim_ = 0;
  size_t stateDim_ = 0;
  size_t jointDim_ = 0;
};

// ==================== Dimension tests ====================

TEST_F(BasisInputsModelDecoratorTest, InputDimIsCorrect) {
  const size_t expectedInputDim = numBasisPerFoot_ * kNumFeet + jointDim_;
  EXPECT_EQ(decorator_->getInputDim(), expectedInputDim);
  EXPECT_NE(decorator_->getInputDim(), wrenchInputDim_)
      << "Decorated input dim should differ from wrench input dim";
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
  EXPECT_EQ(decorator_->getJointVelocitiesStartindex(),
            numBasisPerFoot_ * kNumFeet);
}

// ==================== Wrench round-trip: set → get ====================

TEST_F(BasisInputsModelDecoratorTest, SetGetContactWrenchRoundTrip) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  // Set a known wrench for foot 0. It is inside the cone of the basis, so the
  // round trip is exact. See
  // SetContactWrenchReproducesConeWrenchesWhoseMinimumNormScalingsAreNegative
  // for one where the minimum-norm scalings alone would not do.
  vector6_t wrenchContactEwrtParam;
  wrenchContactEwrtParam << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
  decorator_->setWrenchContactEwrtParam(input, wrenchContactEwrtParam,
                                        /*contactIndex=*/0);

  // Get it back - should recover the wrench via B * B⁺ * W (projection onto
  // column space)
  const matrix_t& B = basisMatrices_[0].getBasisMatrix();
  const matrix_t& B_pinv = basisMatrices_[0].getBasisMatrixPseudoInverse();
  vector6_t projected_wrench_contact_ewrt_param =
      B * (B_pinv * wrenchContactEwrtParam);

  vector6_t recovered_wrench_contact_ewrt_param =
      decorator_->getWrenchContactEwrtParam(input, /*contactIndex=*/0);
  EXPECT_TRUE(recovered_wrench_contact_ewrt_param.isApprox(
      projected_wrench_contact_ewrt_param, 1.0e-9))
      << "setWrenchContactEwrtParam → getWrenchContactEwrtParam should "
         "round-trip (modulo "
         "projection):\n"
      << "  set = " << wrenchContactEwrtParam.transpose()
      << "\n  got = " << recovered_wrench_contact_ewrt_param.transpose()
      << "\n  expected (projected) = "
      << projected_wrench_contact_ewrt_param.transpose();
}

TEST_F(
    BasisInputsModelDecoratorTest,
    SetContactWrenchReproducesConeWrenchesWhoseMinimumNormScalingsAreNegative) {
  // This wrench is inside the cone of the basis, but its minimum-norm scalings
  // have a negative entry. Clamping them at zero - what
  // setWrenchContactEwrtParam used to do - returns a different wrench; the
  // non-negative least-squares solve must return this one exactly.
  vector6_t wrenchContactEwrtParam;
  wrenchContactEwrtParam << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5;
  const matrix_t& B = basisMatrices_[0].getBasisMatrix();
  const vector_t minimumNormLambda =
      basisMatrices_[0].getBasisMatrixPseudoInverse() * wrenchContactEwrtParam;
  ASSERT_LT(minimumNormLambda.minCoeff(), 0.0)
      << "this test needs a wrench whose minimum-norm scalings are not all "
         "non-negative";
  ASSERT_GE(makeTestConeRows().evaluateCone(wrenchContactEwrtParam).minCoeff(),
            0.0)
      << "this test needs a wrench the wrench-space cone admits";
  ASSERT_TRUE(isInConeOfBasis(B, wrenchContactEwrtParam))
      << "this test needs a wrench the basis can reproduce, which is more than "
         "the rows admitting it";
  ASSERT_GT(
      (B * minimumNormLambda.cwiseMax(0.0) - wrenchContactEwrtParam).norm(),
      1.0e-3 * wrenchContactEwrtParam.norm())
      << "the old clamp must be visibly wrong here";

  vector_t input = vector_t::Zero(decorator_->getInputDim());
  decorator_->setWrenchContactEwrtParam(input, wrenchContactEwrtParam,
                                        /*contactIndex=*/0);

  EXPECT_GE(getLambda(input, /*contactIndex=*/0).minCoeff(), 0.0)
      << "the scalings written into the input must be non-negative";
  EXPECT_TRUE(decorator_->getWrenchContactEwrtParam(input, /*contactIndex=*/0)
                  .isApprox(wrenchContactEwrtParam, kTol))
      << "set " << wrenchContactEwrtParam.transpose() << ", got back "
      << decorator_->getWrenchContactEwrtParam(input, /*contactIndex=*/0)
             .transpose();
}

TEST_F(BasisInputsModelDecoratorTest,
       SetContactWrenchProjectsAWrenchOutsideTheConeIntoIt) {
  vector6_t outside;
  outside << 200.0, 0.0, 100.0, 0.0, 0.0,
      0.0;  // well beyond the friction limit
  const ContactWrenchConeRows rows = makeTestConeRows();
  ASSERT_LT(rows.evaluateCone(outside).minCoeff(), 0.0);
  vector_t input = vector_t::Zero(decorator_->getInputDim());
  decorator_->setWrenchContactEwrtParam(input, outside, /*contactIndex=*/0);
  EXPECT_GE(getLambda(input, /*contactIndex=*/0).minCoeff(), 0.0);
  EXPECT_GE(rows.evaluateCone(vector6_t(decorator_->getWrenchContactEwrtParam(
                                  input, /*contactIndex=*/0)))
                .minCoeff(),
            -1.0e-9 * outside.norm());
}

TEST_F(BasisInputsModelDecoratorTest,
       PitchedStanceFootWeightRoundTripsThroughTheWorldFrameSetter) {
  // The weight-compensation warm start and the MRT's fallback torques write the
  // world-vertical weight of the robot through setForceContactEwrtWorld.
  // Seen from a stance foot pitched by 15 degrees (heel strike, toe-off) that
  // force has a tangential component and its minimum-norm scalings go negative;
  // on the Atlas cone the old clamp returned a wrench 7 % too large with a 6 %
  // sideways push. It is inside the friction cone, so it has to come back
  // exactly.
  const scalar_t robotWeight = centroidalModelInfo_->robotMass * 9.81;
  for (const scalar_t pitch :
       {-15.0 * std::numbers::pi / 180.0, 15.0 * std::numbers::pi / 180.0}) {
    const vector_t state = makeState(/*eulerzyxBaseRtWorldYaw=*/0.3, pitch);
    const vector3_t forceContactEwrtWorld(0.0, 0.0, robotWeight / 2.0);
    for (size_t i = 0; i < kNumFeet; ++i) {
      vector6_t wrenchContactEwrtWorld = vector6_t::Zero();
      wrenchContactEwrtWorld.head<3>() = forceContactEwrtWorld;
      const vector6_t wrenchContactEwrtContact =
          decorator_->computeWrenchContactEwrtContact(
              state, wrenchContactEwrtWorld, i);
      ASSERT_LT((basisMatrices_[i].getBasisMatrixPseudoInverse() *
                 wrenchContactEwrtContact)
                    .minCoeff(),
                0.0)
          << "at this pitch the minimum-norm scalings of the weight must have "
             "a negative entry, or the test exercises nothing";

      vector_t input = vector_t::Zero(decorator_->getInputDim());
      decorator_->setForceContactEwrtWorld(state, input, forceContactEwrtWorld,
                                           i);
      EXPECT_GE(getLambda(input, i).minCoeff(), 0.0);
      EXPECT_LE((decorator_->getForceContactEwrtWorld(state, input, i) -
                 forceContactEwrtWorld)
                    .norm(),
                kTol * forceContactEwrtWorld.norm())
          << "pitch " << pitch << " contact " << i << ": got "
          << decorator_->getForceContactEwrtWorld(state, input, i).transpose();
      EXPECT_LE(decorator_->getTorqueContactEwrtWorld(state, input, i).norm(),
                kTol * forceContactEwrtWorld.norm());
    }
    // The same through the helper that builds the whole weight-compensating
    // input.
    const vector_t compensating = weightCompensatingInput(
        *pinocchioInterface_, {true, true}, *decorator_, state);
    for (size_t i = 0; i < kNumFeet; ++i) {
      EXPECT_LE((decorator_->getForceContactEwrtWorld(state, compensating, i) -
                 forceContactEwrtWorld)
                    .norm(),
                1.0e-6)
          << "pitch " << pitch;
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, ExactGeneratorSetDecoratesTheSameModel) {
  // The exact wrench cone set changes only the number of scalings per foot;
  // every accessor keeps its meaning.
  const std::vector<ContactWrenchConeBasisMatrix> exactBases =
      makeTestBasisMatrices(kExactWrenchConeGeneratorSet);
  const std::unique_ptr<BasisInputsModelDecorator<scalar_t>> exactPtr =
      checkedValue(BasisInputsModelDecorator<scalar_t>::Create(
          std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
              *modelSettings_, *pinocchioInterface_, *centroidalModelInfo_),
          exactBases, *pinocchioInterface_));
  const BasisInputsModelDecorator<scalar_t>& exact = *exactPtr;
  const size_t numBasisPerFoot =
      4 * makeTestConeConfig()
              .numBasisVectors;  // corner forces along the pyramid edges
  EXPECT_EQ(exact.getNumBasisPerFoot(), numBasisPerFoot);
  EXPECT_EQ(exact.getInputDim(), numBasisPerFoot * kNumFeet + jointDim_);
  EXPECT_EQ(exact.getJointVelocitiesStartindex(), numBasisPerFoot * kNumFeet);

  // Every wrench of the cone round-trips, including an extreme ray, whose
  // minimum-norm scalings are negative.
  const vector_t state = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw,
                                   /*eulerzyxBaseRtWorldPitch=*/0.2);
  const ContactWrenchConeRows rows = makeTestConeRows();
  for (size_t i = 0; i < kNumFeet; ++i) {
    const vector6_t extremeRay = 300.0 * exactBases[i].getBasisMatrix().col(0);
    ASSERT_LT(
        (exactBases[i].getBasisMatrixPseudoInverse() * extremeRay).minCoeff(),
        0.0);
    vector6_t interior;
    interior << 30.0, -20.0, 300.0, 5.0, -10.0, 4.0;
    ASSERT_GT(rows.evaluateCone(interior).minCoeff(), 0.0);
    for (const vector6_t& wrenchContactEwrtContact : {extremeRay, interior}) {
      const vector6_t wrenchContactEwrtWorld =
          rotateToWorld(computePinocchioRotmatContactRtWorld(state, i),
                        wrenchContactEwrtContact);
      vector_t input = vector_t::Zero(exact.getInputDim());
      exact.setWrenchContactEwrtWorld(state, input, wrenchContactEwrtWorld, i);
      EXPECT_GE(
          input.segment(exact.getContactWrenchStartIndices(i), numBasisPerFoot)
              .minCoeff(),
          0.0);
      EXPECT_TRUE(exact.getWrenchContactEwrtParam(input, i).isApprox(
          wrenchContactEwrtContact, kTol))
          << exact.getWrenchContactEwrtParam(input, i).transpose();
      EXPECT_TRUE(exact.getWrenchContactEwrtWorld(state, input, i)
                      .isApprox(wrenchContactEwrtWorld, kTol));
      EXPECT_TRUE(input.tail(jointDim_).isZero(0.0));
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactForceRoundTrip) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  vector3_t forceContactEwrtParam;
  forceContactEwrtParam << 5.0, -3.0, 80.0;
  decorator_->setForceContactEwrtParam(input, forceContactEwrtParam,
                                       /*contactIndex=*/1);

  vector3_t recovered_force_contact_ewrt_param =
      decorator_->getForceContactEwrtParam(input, /*contactIndex=*/1);

  vector6_t padded_wrench_contact_ewrt_param = vector6_t::Zero();
  padded_wrench_contact_ewrt_param.head<3>() = forceContactEwrtParam;
  const matrix_t& B = basisMatrices_[1].getBasisMatrix();
  const matrix_t& B_pinv = basisMatrices_[1].getBasisMatrixPseudoInverse();
  vector3_t projected_force_contact_ewrt_param =
      (B * (B_pinv * padded_wrench_contact_ewrt_param)).head<3>();

  EXPECT_TRUE(recovered_force_contact_ewrt_param.isApprox(
      projected_force_contact_ewrt_param, 1.0e-9))
      << "setForceContactEwrtParam → getForceContactEwrtParam round-trip:\n"
      << "  set = " << forceContactEwrtParam.transpose()
      << "\n  got = " << recovered_force_contact_ewrt_param.transpose();
}

// ==================== Independence of contacts ====================

TEST_F(BasisInputsModelDecoratorTest, ContactsAreIndependent) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  vector6_t wrenchContactEwrtParam;
  wrenchContactEwrtParam << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
  decorator_->setWrenchContactEwrtParam(input, wrenchContactEwrtParam,
                                        /*contactIndex=*/0);

  vector6_t otherWrenchContactEwrtParam =
      decorator_->getWrenchContactEwrtParam(input, /*contactIndex=*/1);
  EXPECT_TRUE(otherWrenchContactEwrtParam.isZero(1.0e-12))
      << "Setting contact 0 wrench should not affect contact 1:\n  "
         "otherWrenchContactEwrtParam = "
      << otherWrenchContactEwrtParam.transpose();
}

// ==================== Joint velocity pass-through ====================

TEST_F(BasisInputsModelDecoratorTest, JointVelocitiesPassThrough) {
  const size_t inputDim = decorator_->getInputDim();

  vector_t state = decorator_->getNeutralState();
  vector_t input = vector_t::Zero(inputDim);

  vector_t jointVels = vector_t::LinSpaced(jointDim_, 0.1, 1.0);
  decorator_->setJointVelocities(state, input, jointVels);

  vector_t jointVels_recovered = decorator_->getJointVelocities(state, input);
  EXPECT_TRUE(jointVels_recovered.isApprox(jointVels, 1.0e-12))
      << "Joint velocities should pass through unchanged";
}

// ==================== Clone test ====================

TEST_F(BasisInputsModelDecoratorTest, CloneProducesWorkingCopy) {
  MpcRobotModelBase<scalar_t>* absl_nonnull cloned = decorator_->clone();
  ASSERT_NE(cloned, nullptr);

  EXPECT_EQ(cloned->getInputDim(), decorator_->getInputDim());
  EXPECT_EQ(cloned->getStateDim(), decorator_->getStateDim());
  EXPECT_EQ(cloned->getJointDim(), decorator_->getJointDim());

  const size_t inputDim = cloned->getInputDim();
  vector_t input = vector_t::Zero(inputDim);
  vector6_t wrenchContactEwrtParam;
  wrenchContactEwrtParam << 1.0, 2.0, 50.0, 0.1, -0.1, 0.01;
  cloned->setWrenchContactEwrtParam(input, wrenchContactEwrtParam,
                                    /*contactIndex=*/0);

  vector6_t recovered_wrench_contact_ewrt_param =
      cloned->getWrenchContactEwrtParam(input, /*contactIndex=*/0);
  EXPECT_FALSE(recovered_wrench_contact_ewrt_param.isZero(1.0e-6));

  // The clone must carry its own pinocchio copy so that the world-frame
  // accessors keep working after cloning.
  const vector_t state = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw);
  EXPECT_TRUE(
      cloned->getWrenchContactEwrtWorld(state, input, /*contactIndex=*/0)
          .isApprox(decorator_->getWrenchContactEwrtWorld(state, input,
                                                          /*contactIndex=*/0),
                    kTol));

  delete cloned;
}

// ==================== State delegate tests ====================

TEST_F(BasisInputsModelDecoratorTest, StateDelegatesMatchWrappedModel) {
  vector_t state = decorator_->getNeutralState();

  vector3_t posBaseRtWorldEwrtWorld;
  posBaseRtWorldEwrtWorld << 1.0, 2.0, 0.95;
  decorator_->setPosBaseRtWorldEwrtWorld(state, posBaseRtWorldEwrtWorld);

  vector3_t recovered_pos_base_rt_world_ewrt_world =
      decorator_->getPosBaseRtWorldEwrtWorld(state);
  EXPECT_TRUE(recovered_pos_base_rt_world_ewrt_world.isApprox(
      posBaseRtWorldEwrtWorld, 1.0e-12))
      << "Base position should delegate to wrapped model";

  vector_t jointAngles = vector_t::LinSpaced(jointDim_, -0.5, 0.5);
  decorator_->setJointAngles(state, jointAngles);
  vector_t jointAngles_recovered = decorator_->getJointAngles(state);
  EXPECT_TRUE(jointAngles_recovered.isApprox(jointAngles, 1.0e-12))
      << "Joint angles should delegate to wrapped model";
}

// ==================== Basis matrix accessor ====================

TEST_F(BasisInputsModelDecoratorTest, BasisMatrixAccessor) {
  for (size_t i = 0; i < kNumFeet; ++i) {
    const matrix_t& B = decorator_->getBasisMatrix(i);
    EXPECT_EQ(B.rows(), 6);
    EXPECT_EQ(B.cols(), static_cast<int>(numBasisPerFoot_));

    EXPECT_TRUE(B.isApprox(basisMatrices_[i].getBasisMatrix(), 1.0e-12))
        << "Accessor should return the same basis matrix for contact " << i;
    EXPECT_TRUE(decorator_->getBasisMatrices()[i].isApprox(B, 1.0e-12));
    EXPECT_TRUE(decorator_->getBasisMatrixPseudoInverse(i).isApprox(
        basisMatrices_[i].getBasisMatrixPseudoInverse(), 1.0e-12));
  }
}

// ==================== Full wrench reconstruction ====================

TEST_F(BasisInputsModelDecoratorTest, NonNegativeLambdaProducesValidWrench) {
  const size_t inputDim = decorator_->getInputDim();
  vector_t input = vector_t::Zero(inputDim);

  for (size_t k = 0; k < numBasisPerFoot_; ++k) {
    input(k) = 1.0;
  }

  vector6_t wrenchContactEwrtParam =
      decorator_->getWrenchContactEwrtParam(input, /*contactIndex=*/0);
  EXPECT_GT(wrenchContactEwrtParam(2), 0.0)
      << "Non-negative lambdas should produce positive Fz";
}

// ==================== Local basis-to-wrench map ====================

TEST_F(BasisInputsModelDecoratorTest,
       LocalBasisToWrenchMapHasBlockDiagonalStructure) {
  const matrix_t M = decorator_->getLocalBasisGeneratorMap();

  const Eigen::Index expectedRows =
      static_cast<Eigen::Index>(6 * kNumFeet + jointDim_);
  const Eigen::Index expectedCols =
      static_cast<Eigen::Index>(numBasisPerFoot_ * kNumFeet + jointDim_);
  ASSERT_EQ(M.rows(), expectedRows);
  ASSERT_EQ(M.cols(), expectedCols);
  ASSERT_EQ(static_cast<size_t>(M.rows()), wrenchInputDim_);
  ASSERT_EQ(static_cast<size_t>(M.cols()), decorator_->getInputDim());

  // Expected layout: M = blkdiag(B_0, B_1, I_joints), everything else zero.
  matrix_t expected = matrix_t::Zero(expectedRows, expectedCols);
  for (size_t i = 0; i < kNumFeet; ++i) {
    expected.block(6 * i, numBasisPerFoot_ * i, 6, numBasisPerFoot_) =
        basisMatrices_[i].getBasisMatrix();
  }
  expected
      .block(6 * kNumFeet, numBasisPerFoot_ * kNumFeet, jointDim_, jointDim_)
      .setIdentity();
  EXPECT_TRUE((M - expected).isZero(0.0))
      << "M must be exactly blkdiag(B_0, B_1, I):\nM =\n"
      << M << "\nexpected =\n"
      << expected;

  // Consistency with the accessors: mapping an arbitrary basis input reproduces
  // the local-frame wrenches and the pass-through joint velocities.
  const vector_t basisInput = vector_t::Random(expectedCols);
  const vector_t wrappedInput = M * basisInput;
  const vector_t state = decorator_->getNeutralState();
  for (size_t i = 0; i < kNumFeet; ++i) {
    EXPECT_TRUE(wrappedInput.segment(6 * i, 6).isApprox(
        decorator_->getWrenchContactEwrtParam(basisInput, i), kTol));
  }
  EXPECT_TRUE(wrappedInput.tail(jointDim_).isApprox(
      decorator_->getJointVelocities(state, basisInput), kTol));
}

// ==================== World-frame accessors (frame correctness)
// ====================

TEST_F(BasisInputsModelDecoratorTest,
       WorldFrameAccessorsRotateLocalWrenchWithContactFrame) {
  const vector_t state = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw);
  vector_t input = vector_t::Zero(decorator_->getInputDim());
  for (size_t i = 0; i < kNumFeet; ++i) {
    // Distinct, strictly positive scalings per foot so that both wrench blocks
    // are non-trivial and inside the cone.
    input.segment(decorator_->getContactWrenchStartIndices(i),
                  numBasisPerFoot_) =
        vector_t::LinSpaced(numBasisPerFoot_, 0.5 + static_cast<scalar_t>(i),
                            3.0 + static_cast<scalar_t>(i));
  }

  for (size_t i = 0; i < kNumFeet; ++i) {
    const vector6_t expected_wrench_contact_ewrt_contact =
        basisMatrices_[i].getBasisMatrix() * getLambda(input, i);
    const vector6_t wrench_contact_ewrt_contact =
        decorator_->getWrenchContactEwrtParam(input, i);
    EXPECT_TRUE(wrench_contact_ewrt_contact.isApprox(
        expected_wrench_contact_ewrt_contact, kTol))
        << "Input-only accessor must return B * lambda (local frame) for "
           "contact "
        << i;

    const matrix3_t rotmat_contact_rt_world =
        computePinocchioRotmatContactRtWorld(state, i);
    EXPECT_TRUE(decorator_->getRotmatContactRtWorld(state, i).isApprox(
        rotmat_contact_rt_world, kTol))
        << "Decorator contact frame rotation differs from an independent "
           "pinocchio evaluation for contact "
        << i;

    const vector6_t expected_wrench_contact_ewrt_world = rotateToWorld(
        rotmat_contact_rt_world, expected_wrench_contact_ewrt_contact);
    const vector6_t wrench_contact_ewrt_world =
        decorator_->getWrenchContactEwrtWorld(state, input, i);
    EXPECT_TRUE(wrench_contact_ewrt_world.isApprox(
        expected_wrench_contact_ewrt_world, kTol))
        << "World-frame wrench must equal blkdiag(w_R_l, w_R_l) * B * lambda "
           "for contact "
        << i << ":\n  got      = " << wrench_contact_ewrt_world.transpose()
        << "\n  expected = " << expected_wrench_contact_ewrt_world.transpose();
    EXPECT_TRUE(wrench_contact_ewrt_world.isApprox(
        decorator_->computeWrenchContactEwrtWorld(
            state, wrench_contact_ewrt_contact, i),
        kTol));
    // The realtime accessor, with the contact frame's orientation handed in, is
    // the same wrench, through the base class.
    const MpcRobotModelBase<scalar_t>& base = *decorator_;
    EXPECT_TRUE(
        base.getWrenchContactEwrtWorld(input, rotmat_contact_rt_world, i)
            .isApprox(expected_wrench_contact_ewrt_world, kTol))
        << "contact " << i;
    EXPECT_TRUE(
        decorator_->getForceContactEwrtWorld(state, input, i)
            .isApprox(expected_wrench_contact_ewrt_world.head<3>(), kTol));
    EXPECT_TRUE(
        decorator_->getTorqueContactEwrtWorld(state, input, i)
            .isApprox(expected_wrench_contact_ewrt_world.tail<3>(), kTol));

    // Sanity: for a yawed base the tangential components genuinely differ
    // between the two frames, i.e. the world-frame accessor is not silently
    // returning the local wrench.
    EXPECT_FALSE(wrench_contact_ewrt_world.head<3>().isApprox(
        wrench_contact_ewrt_contact.head<3>(), 1.0e-6))
        << "Yawed contact frame should change the tangential force components "
           "for contact "
        << i;

    // Rotating back must recover the local wrench.
    EXPECT_TRUE(decorator_
                    ->computeWrenchContactEwrtContact(
                        state, wrench_contact_ewrt_world, i)
                    .isApprox(wrench_contact_ewrt_contact, kTol));
  }
}

TEST_F(BasisInputsModelDecoratorTest, YawedBaseRotatesLocalXForceIntoWorldY) {
  const vector_t stateZeroYaw = makeState(0.0);
  const vector_t stateYaw90 = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw);
  // The base quaternion (x, y, z, w) of the yaw, at the state's orientation
  // slots: (0, 0, sin(yaw / 2), cos(yaw / 2)).
  EXPECT_LT(
      (stateYaw90.segment<4>(kBaseOrientationStateIndex) -
       vector4_t(0.0, 0.0, std::sin(0.5 * kQuarterTurnEulerzyxBaseRtWorldYaw),
                 std::cos(0.5 * kQuarterTurnEulerzyxBaseRtWorldYaw)))
          .cwiseAbs()
          .maxCoeff(),
      1.0e-16)
      << "The base quaternion is expected at state index "
      << kBaseOrientationStateIndex;

  const matrix3_t Rz90 =
      Eigen::AngleAxis<scalar_t>(kQuarterTurnEulerzyxBaseRtWorldYaw,
                                 vector3_t::UnitZ())
          .toRotationMatrix();

  for (size_t i = 0; i < kNumFeet; ++i) {
    const matrix3_t unturned_rotmat_contact_rt_world =
        computePinocchioRotmatContactRtWorld(stateZeroYaw, i);
    const matrix3_t turned_rotmat_contact_rt_world =
        computePinocchioRotmatContactRtWorld(stateYaw90, i);

    // With all joints at zero the Atlas leg chain has no rotational offsets
    // (all URDF joint origins have zero rpy and the contact frame is attached
    // with identity rotation), so at zero yaw the contact frame is aligned with
    // the world.
    EXPECT_TRUE(
        unturned_rotmat_contact_rt_world.isApprox(matrix3_t::Identity(), kTol))
        << "Contact frame " << i << " should be world-aligned at zero yaw:\n"
        << unturned_rotmat_contact_rt_world;
    // The base yaw is the first ZYX Euler angle: a positive yaw is a
    // right-handed rotation about world z of the whole chain.
    EXPECT_TRUE(turned_rotmat_contact_rt_world.isApprox(
        Rz90 * unturned_rotmat_contact_rt_world, kTol))
        << "Yaw = +90deg must rotate contact frame " << i << " about world z:\n"
        << turned_rotmat_contact_rt_world;

    // Direction of the local +x axis in the world frame, taken from Pinocchio
    // rather than assumed.
    const vector3_t turned_x_axis_contact_ewrt_world =
        turned_rotmat_contact_rt_world.col(0);
    EXPECT_NEAR(turned_x_axis_contact_ewrt_world.x(), 0.0, kTol);
    EXPECT_NEAR(std::abs(turned_x_axis_contact_ewrt_world.y()), 1.0, kTol);
    EXPECT_NEAR(turned_x_axis_contact_ewrt_world.z(), 0.0, kTol);
    const scalar_t ySign =
        (turned_x_axis_contact_ewrt_world.y() > 0.0) ? 1.0 : -1.0;
    EXPECT_GT(ySign, 0.0) << "A positive yaw maps local +x onto world +y "
                             "(right-handed rotation about z)";

    // (1) A purely tangential force lies outside the friction cone, so exercise
    // the rotation helper directly.
    vector6_t x_wrench_contact_ewrt_contact = vector6_t::Zero();
    x_wrench_contact_ewrt_contact(0) = 1.0;
    const vector6_t x_wrench_contact_ewrt_world =
        decorator_->computeWrenchContactEwrtWorld(
            stateYaw90, x_wrench_contact_ewrt_contact, i);
    EXPECT_TRUE(x_wrench_contact_ewrt_world.head<3>().isApprox(
        vector3_t(0.0, ySign, 0.0), kTol))
        << "Local +x force must become a world " << (ySign > 0.0 ? "+" : "-")
        << "y force for contact " << i << ", got "
        << x_wrench_contact_ewrt_world.head<3>().transpose();
    EXPECT_TRUE(x_wrench_contact_ewrt_world.tail<3>().isZero(kTol));

    // (2) The same through the input parameterization with an in-cone wrench:
    // local [10, 0, 100, 0, 0, 0].
    vector6_t in_cone_wrench_contact_ewrt_param;
    in_cone_wrench_contact_ewrt_param << 10.0, 0.0, 100.0, 0.0, 0.0, 0.0;
    vector_t input = vector_t::Zero(decorator_->getInputDim());
    decorator_->setWrenchContactEwrtParam(input,
                                          in_cone_wrench_contact_ewrt_param, i);
    ASSERT_GE(getLambda(input, i).minCoeff(), 0.0);
    ASSERT_TRUE(decorator_->getWrenchContactEwrtParam(input, i).isApprox(
        in_cone_wrench_contact_ewrt_param, kTol))
        << "A wrench inside the cone must round-trip exactly";

    const vector3_t force_contact_ewrt_world =
        decorator_->getForceContactEwrtWorld(stateYaw90, input, i);
    EXPECT_TRUE(force_contact_ewrt_world.isApprox(
        vector3_t(0.0, ySign * 10.0, 100.0), kTol))
        << "Local force (10, 0, 100) should become (0, " << ySign * 10.0
        << ", 100) in the world for contact " << i << ", got "
        << force_contact_ewrt_world.transpose();
    EXPECT_TRUE(force_contact_ewrt_world.isApprox(
        turned_rotmat_contact_rt_world *
            in_cone_wrench_contact_ewrt_param.head<3>(),
        kTol));
    // The vertical component is invariant under yaw.
    EXPECT_NEAR(force_contact_ewrt_world.z(),
                in_cone_wrench_contact_ewrt_param(2), kTol);
  }
}

TEST_F(BasisInputsModelDecoratorTest,
       SetGetContactWrenchInWorldFrameRoundTrip) {
  vector6_t wrench_contact_ewrt_contact;
  wrench_contact_ewrt_contact << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;

  for (const scalar_t yaw : {0.0, kQuarterTurnEulerzyxBaseRtWorldYaw, -0.7}) {
    const vector_t state = makeState(yaw);
    for (size_t i = 0; i < kNumFeet; ++i) {
      // Precondition of an exact round trip: the wrench is inside the cone the
      // basis SPANS. Satisfying the wrench-space rows is not enough, because
      // the default generator set spans only part of that cone.
      ASSERT_TRUE(isInConeOfBasis(basisMatrices_[i].getBasisMatrix(),
                                  wrench_contact_ewrt_contact))
          << "Test wrench must be reproducible for contact " << i;

      const matrix3_t rotmat_contact_rt_world =
          computePinocchioRotmatContactRtWorld(state, i);
      const vector6_t in_wrench_contact_ewrt_world =
          rotateToWorld(rotmat_contact_rt_world, wrench_contact_ewrt_contact);

      vector_t input = vector_t::Zero(decorator_->getInputDim());
      decorator_->setWrenchContactEwrtWorld(state, input,
                                            in_wrench_contact_ewrt_world, i);

      EXPECT_GE(getLambda(input, i).minCoeff(), 0.0)
          << "All lambda must be non-negative (yaw = " << yaw << ", contact "
          << i << ")";
      EXPECT_TRUE(getLambda(input, 1 - i).isZero(0.0))
          << "Setting contact " << i << " must not touch the other contact";
      EXPECT_TRUE(input.tail(jointDim_).isZero(0.0))
          << "Setting a contact wrench must not touch the joint velocities";

      // The input-only accessor returns the wrench in the local contact frame,
      // i.e. the un-rotated test wrench.
      EXPECT_TRUE(decorator_->getWrenchContactEwrtParam(input, i).isApprox(
          wrench_contact_ewrt_contact, kTol))
          << "Local wrench mismatch (yaw = " << yaw << ", contact " << i
          << "):\n  got = "
          << decorator_->getWrenchContactEwrtParam(input, i).transpose()
          << "\n  expected = " << wrench_contact_ewrt_contact.transpose();

      const vector6_t out_wrench_contact_ewrt_world =
          decorator_->getWrenchContactEwrtWorld(state, input, i);
      EXPECT_TRUE(out_wrench_contact_ewrt_world.isApprox(
          in_wrench_contact_ewrt_world, kTol))
          << "World-frame round trip failed (yaw = " << yaw << ", contact " << i
          << "):\n  set = " << in_wrench_contact_ewrt_world.transpose()
          << "\n  got = " << out_wrench_contact_ewrt_world.transpose();
    }
  }
}

TEST_F(BasisInputsModelDecoratorTest, SetGetContactForceInWorldFrameRoundTrip) {
  const vector_t state = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw);
  const vector3_t force_contact_ewrt_contact(5.0, -3.0, 80.0);

  for (size_t i = 0; i < kNumFeet; ++i) {
    const matrix3_t rotmat_contact_rt_world =
        computePinocchioRotmatContactRtWorld(state, i);
    const vector3_t in_force_contact_ewrt_world =
        rotmat_contact_rt_world * force_contact_ewrt_contact;

    vector_t input = vector_t::Zero(decorator_->getInputDim());
    decorator_->setForceContactEwrtWorld(state, input,
                                         in_force_contact_ewrt_world, i);
    EXPECT_GE(getLambda(input, i).minCoeff(), 0.0);

    EXPECT_TRUE(decorator_->getForceContactEwrtParam(input, i).isApprox(
        force_contact_ewrt_contact, kTol))
        << "Input-only accessor must return the local force";
    EXPECT_TRUE(decorator_->getForceContactEwrtWorld(state, input, i)
                    .isApprox(in_force_contact_ewrt_world, kTol))
        << "World-frame force round trip failed for contact " << i
        << ":\n  set = " << in_force_contact_ewrt_world.transpose()
        << "\n  got = "
        << decorator_->getForceContactEwrtWorld(state, input, i).transpose();
    EXPECT_TRUE(
        decorator_->getTorqueContactEwrtWorld(state, input, i).isZero(1.0e-9))
        << "A pure force must not produce a moment";
  }
}

// ==================== Wrench-space model: world-frame accessors are the
// input-only accessors ====================

TEST_F(BasisInputsModelDecoratorTest,
       WrenchSpaceModelWorldFrameAccessorsEqualInputOnlyAccessors) {
  // Go through the base-class interface to exercise the virtual dispatch every
  // consumer relies on.
  const MpcRobotModelBase<scalar_t>& model = *wrenchModel_;
  ASSERT_EQ(model.getInputDim(), wrenchInputDim_);

  for (const scalar_t yaw : {0.0, kQuarterTurnEulerzyxBaseRtWorldYaw, -1.1}) {
    // The wrench-space input already stores world-frame wrenches, so the state
    // must be irrelevant: use a yawed base *and* random joint angles.
    vector_t state = makeState(yaw);
    model.setJointAngles(state, vector_t::Random(jointDim_));
    const vector_t input = vector_t::Random(wrenchInputDim_);

    for (size_t i = 0; i < kNumFeet; ++i) {
      EXPECT_TRUE((model.getWrenchContactEwrtWorld(state, input, i) -
                   model.getWrenchContactEwrtParam(input, i))
                      .isZero(0.0));
      EXPECT_TRUE((model.getForceContactEwrtWorld(state, input, i) -
                   model.getForceContactEwrtParam(input, i))
                      .isZero(0.0));
      EXPECT_TRUE((model.getTorqueContactEwrtWorld(state, input, i) -
                   model.getTorqueContactEwrtParam(input, i))
                      .isZero(0.0));
      // The realtime accessor ignores the contact frame's orientation as well.
      const matrix3_t anyRotmatContactRtWorld =
          Eigen::AngleAxisd(0.7, vector3_t(1.0, -2.0, 0.5).normalized())
              .toRotationMatrix();
      EXPECT_TRUE(
          (model.getWrenchContactEwrtWorld(input, anyRotmatContactRtWorld, i) -
           model.getWrenchContactEwrtParam(input, i))
              .isZero(0.0));

      vector6_t wrenchContactEwrtWorld;
      wrenchContactEwrtWorld << 10.0, 5.0, 100.0, 1.0, -1.0, 0.5;
      vector_t inputOfSetWrenchContactEwrtWorld =
          vector_t::Zero(wrenchInputDim_);
      model.setWrenchContactEwrtWorld(state, inputOfSetWrenchContactEwrtWorld,
                                      wrenchContactEwrtWorld, i);
      EXPECT_TRUE((model.getWrenchContactEwrtParam(
                       inputOfSetWrenchContactEwrtWorld, i) -
                   wrenchContactEwrtWorld)
                      .isZero(0.0));
      EXPECT_TRUE((model.getWrenchContactEwrtWorld(
                       state, inputOfSetWrenchContactEwrtWorld, i) -
                   wrenchContactEwrtWorld)
                      .isZero(0.0));

      const vector3_t forceContactEwrtWorld(5.0, -3.0, 80.0);
      vector_t inputOfSetForceContactEwrtWorld =
          vector_t::Zero(wrenchInputDim_);
      model.setForceContactEwrtWorld(state, inputOfSetForceContactEwrtWorld,
                                     forceContactEwrtWorld, i);
      EXPECT_TRUE(
          (model.getForceContactEwrtParam(inputOfSetForceContactEwrtWorld, i) -
           forceContactEwrtWorld)
              .isZero(0.0));
      EXPECT_TRUE(
          model.getTorqueContactEwrtParam(inputOfSetForceContactEwrtWorld, i)
              .isZero(0.0));
    }
  }
}

// ==================== Weight-compensating input ====================

TEST_F(BasisInputsModelDecoratorTest,
       WeightCompensatingInputIsVerticalInWorldFrameForYawedBase) {
  const vector_t state = makeState(kQuarterTurnEulerzyxBaseRtWorldYaw);
  const scalar_t robotWeight = centroidalModelInfo_->robotMass * 9.81;
  // Looser than kTol: the result is a least-squares solve of an O(1e3) N
  // wrench, compare with an absolute tolerance.
  static constexpr scalar_t kTolForceContactEwrtWorld = 1.0e-6;

  const vector_t inputDoubleContact = weightCompensatingInput(
      *pinocchioInterface_, {true, true}, *decorator_, state);
  ASSERT_EQ(static_cast<size_t>(inputDoubleContact.size()),
            decorator_->getInputDim());
  EXPECT_TRUE(inputDoubleContact.tail(jointDim_).isZero(0.0))
      << "Weight compensation must not command joint velocities";

  const vector3_t expectedForceFootEwrtWorld(0.0, 0.0, robotWeight / 2.0);
  for (size_t i = 0; i < kNumFeet; ++i) {
    EXPECT_GE(getLambda(inputDoubleContact, i).minCoeff(), 0.0)
        << "All lambda must be non-negative for contact " << i;

    // The vertical force is invariant under a yaw of the base, so the
    // world-frame force must be purely vertical even though the input stores it
    // in the (yawed) local contact frame.
    const vector3_t force_contact_ewrt_world =
        decorator_->getForceContactEwrtWorld(state, inputDoubleContact, i);
    EXPECT_LT((force_contact_ewrt_world - expectedForceFootEwrtWorld).norm(),
              kTolForceContactEwrtWorld)
        << "World-frame weight-compensating force for contact " << i
        << " should be " << expectedForceFootEwrtWorld.transpose() << ", got "
        << force_contact_ewrt_world.transpose();
    EXPECT_TRUE(
        decorator_->getTorqueContactEwrtWorld(state, inputDoubleContact, i)
            .isZero(kTolForceContactEwrtWorld))
        << "Weight compensation must not produce a contact moment for contact "
        << i;
  }

  // The CentroidalModelInfo variant must produce the identical input.
  const vector_t inputFromInfo = weightCompensatingInput(
      *centroidalModelInfo_, {true, true}, *decorator_, state);
  EXPECT_TRUE(inputFromInfo.isApprox(inputDoubleContact, kTol));

  // Single support puts the full weight on the stance foot and leaves the swing
  // foot at zero.
  const vector_t inputLeftContact = weightCompensatingInput(
      *pinocchioInterface_, {true, false}, *decorator_, state);
  EXPECT_GE(getLambda(inputLeftContact, /*contactIndex=*/0).minCoeff(), 0.0);
  EXPECT_LT((decorator_->getForceContactEwrtWorld(state, inputLeftContact,
                                                  /*contactIndex=*/0) -
             vector3_t(0.0, 0.0, robotWeight))
                .norm(),
            kTolForceContactEwrtWorld);
  EXPECT_TRUE(getLambda(inputLeftContact, /*contactIndex=*/1).isZero(0.0));
}

// ==================== Basis-input dynamics: validation before the CppAD build
// ====================

TEST_F(BasisInputsModelDecoratorTest,
       BasisDynamicsValidationNamesTheKeyToChangeBeforeBuildingAnything) {
  const ContactArray<matrix_t> bases = decorator_->getBasisMatrices();
  // Positive control: the configuration the fixture's decorator runs on is
  // accepted.
  EXPECT_TRUE(CentroidalDynamicsBasisInputsAD::validate(*pinocchioInterface_,
                                                        *centroidalModelInfo_,
                                                        *modelSettings_, bases)
                  .ok());

  // A contact frame the URDF does not have. Create() must report it instead of
  // compiling a tape around a bad index.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ModelSettings brokenSettings,
      ModelSettings::Create(atlas_.task, urdfFile_, "basis_decorator_test",
                            /*verbose=*/false));
  brokenSettings.contactNames[1] = "no_such_frame";
  const absl::StatusOr<std::unique_ptr<CentroidalDynamicsBasisInputsAD>>
      missingFrame = CentroidalDynamicsBasisInputsAD::Create(
          *pinocchioInterface_, *centroidalModelInfo_, "dynamics",
          brokenSettings, bases);
  ASSERT_FALSE(missingFrame.ok());
  EXPECT_EQ(missingFrame.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(
      missingFrame.status().message().find("model_settings.contact_names_6dof"),
      absl::string_view::npos)
      << missingFrame.status();
  EXPECT_NE(missingFrame.status().message().find("no_such_frame"),
            absl::string_view::npos)
      << missingFrame.status();

  // A contact count the basis-vector formulation does not support.
  CentroidalModelInfo oneContact = *centroidalModelInfo_;
  oneContact.numSixDofContacts = 1;
  const absl::Status wrongCount = CentroidalDynamicsBasisInputsAD::validate(
      *pinocchioInterface_, oneContact, *modelSettings_, bases);
  EXPECT_EQ(wrongCount.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(wrongCount.message().find("model_settings.contact_names_6dof"),
            absl::string_view::npos)
      << wrongCount;

  // A three-DoF contact: no generator set exists for a point contact.
  CentroidalModelInfo withPointContact = *centroidalModelInfo_;
  withPointContact.numThreeDofContacts = 1;
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::validate(
                *pinocchioInterface_, withPointContact, *modelSettings_, bases)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // One basis per contact.
  ContactArray<matrix_t> oneBasis(/*numContacts=*/1);
  oneBasis[0] = bases[0];
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::validate(*pinocchioInterface_,
                                                      *centroidalModelInfo_,
                                                      *modelSettings_, oneBasis)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // Bases of different sizes are a programming error, not a configuration one.
  ContactArray<matrix_t> mismatched = bases;
  mismatched[1] =
      makeTestBasisMatrices(kExactWrenchConeGeneratorSet)[1].getBasisMatrix();
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::validate(
                *pinocchioInterface_, *centroidalModelInfo_, *modelSettings_,
                mismatched)
                .code(),
            absl::StatusCode::kInternal);
}

TEST_F(BasisInputsModelDecoratorTest,
       BasisDynamicsLibraryNameCarriesTheBasisKey) {
  const ContactArray<matrix_t> conservative = decorator_->getBasisMatrices();
  const std::vector<ContactWrenchConeBasisMatrix> exactBases =
      makeTestBasisMatrices(kExactWrenchConeGeneratorSet);
  ContactArray<matrix_t> exact(exactBases.size());
  for (size_t i = 0; i < exactBases.size(); ++i) {
    exact[i] = exactBases[i].getBasisMatrix();
  }
  EXPECT_EQ(CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics",
                                                             conservative),
            absl::StrCat("dynamics_", basisInputsLibraryKey(conservative)));
  EXPECT_NE(
      CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics",
                                                       conservative),
      CentroidalDynamicsBasisInputsAD::uniqueModelName("dynamics", exact));
}

// ==================== Any number of 6-DoF contacts ====================

/**
 * The Atlas with `numContacts` 6-DoF contacts: the two feet, then the seat pad
 * and the back pad of the multi-contact Atlas, each around the shared contact
 * rectangle. The members outlive the models that keep references to them.
 */
struct SixDofContactAtlas {
  CentroidalMpcConfig config;
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<CentroidalModelInfo> info;
  std::vector<ContactWrenchConeBasisMatrix> bases;
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> decorator;
};

std::unique_ptr<SixDofContactAtlas> makeSixDofContactAtlas(size_t numContacts) {
  constexpr std::array<std::pair<absl::string_view, absl::string_view>, 2>
      kPads = {
          {{"seat_contact", "root_joint"}, {"torso_back_contact", "back_bkx"}}};
  const CentroidalRobotFiles files = atlasFiles();
  std::unique_ptr<SixDofContactAtlas> atlas =
      std::make_unique<SixDofContactAtlas>();
  atlas->config = checkedValue(loadConfigOf(files));
  mpc_config::ModelSettingsConfig& settings = atlas->config.task.model_settings;
  for (size_t i = kNumFeet; i < numContacts; ++i) {
    settings.contact_names_6dof.emplace_back(kPads[i - kNumFeet].first);
    settings.contact_parent_joint_names.emplace_back(
        kPads[i - kNumFeet].second);
  }
  atlas->modelSettings =
      std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
          atlas->config.task, files.urdfFile, "basis_decorator_contacts_test",
          /*verbose=*/false)));
  atlas->pinocchioInterface = std::make_unique<PinocchioInterface>(
      checkedValue(loadCustomPinocchioInterface(
          atlas->config.task, files.urdfFile, *atlas->modelSettings,
          /*scaleTotalMass=*/false)));
  atlas->info =
      std::make_unique<CentroidalModelInfo>(checkedValue(centroidalModelInfoOf(
          atlas->config, *atlas->pinocchioInterface, *atlas->modelSettings)));
  const std::vector<ContactWrenchConeBasisMatrix> feet =
      makeTestBasisMatrices();
  for (size_t i = 0; i < numContacts; ++i) {
    atlas->bases.push_back(feet[i % kNumFeet]);
  }
  atlas->decorator = checkedValue(BasisInputsModelDecorator<scalar_t>::Create(
      std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
          *atlas->modelSettings, *atlas->pinocchioInterface, *atlas->info),
      atlas->bases, *atlas->pinocchioInterface));
  return atlas;
}

/** Non-negative scalings, different for every contact and generator. */
vector_t positiveScalings(size_t numBasis, size_t contactIndex) {
  vector_t lambda(static_cast<Eigen::Index>(numBasis));
  for (Eigen::Index k = 0; k < lambda.size(); ++k) {
    lambda(k) = 1.0 + 0.5 * static_cast<scalar_t>(k) +
                3.0 * static_cast<scalar_t>(contactIndex);
  }
  return lambda;
}

TEST(BasisInputsModelDecoratorContacts, theLayoutCoversEveryContact) {
  for (const size_t numContacts : {size_t{3}, size_t{4}}) {
    const std::unique_ptr<SixDofContactAtlas> atlas =
        makeSixDofContactAtlas(numContacts);
    const BasisInputsModelDecorator<scalar_t>& decorator = *atlas->decorator;
    ASSERT_EQ(atlas->modelSettings->contactNames.size(), numContacts);
    const size_t numBasis = decorator.getNumBasisPerFoot();
    const size_t jointDim = atlas->modelSettings->mpc_joint_dim;
    EXPECT_EQ(decorator.getInputDim(), numBasis * numContacts + jointDim)
        << numContacts;
    EXPECT_EQ(decorator.getJointVelocitiesStartindex(), numBasis * numContacts);
    ASSERT_EQ(decorator.getBasisMatrices().size(), numContacts);
    for (size_t i = 0; i < numContacts; ++i) {
      EXPECT_EQ(decorator.getContactWrenchStartIndices(i), numBasis * i);
      EXPECT_EQ(decorator.getContactInputDim(i), numBasis);
      EXPECT_EQ(decorator.getBasisMatrix(i), atlas->bases[i].getBasisMatrix());
    }

    // M = blkdiag(B_0, ..., B_{n-1}, I): each B_i at its own wrench rows and
    // scaling columns, the identity on the joints, zero elsewhere.
    const matrix_t M = decorator.getLocalBasisGeneratorMap();
    const Eigen::Index wrenchInputDim =
        static_cast<Eigen::Index>(kContactWrenchDim * numContacts + jointDim);
    ASSERT_EQ(M.rows(), wrenchInputDim);
    ASSERT_EQ(M.cols(), static_cast<Eigen::Index>(decorator.getInputDim()));
    matrix_t expected = matrix_t::Zero(M.rows(), M.cols());
    for (size_t i = 0; i < numContacts; ++i) {
      expected.block(static_cast<Eigen::Index>(kContactWrenchDim * i),
                     static_cast<Eigen::Index>(numBasis * i),
                     static_cast<Eigen::Index>(kContactWrenchDim),
                     static_cast<Eigen::Index>(numBasis)) =
          atlas->bases[i].getBasisMatrix();
    }
    expected
        .bottomRightCorner(static_cast<Eigen::Index>(jointDim),
                           static_cast<Eigen::Index>(jointDim))
        .setIdentity();
    EXPECT_EQ(M, expected) << numContacts;
  }
}

TEST(BasisInputsModelDecoratorContacts, everyContactsWrenchRoundTrips) {
  for (const size_t numContacts : {size_t{3}, size_t{4}}) {
    const std::unique_ptr<SixDofContactAtlas> atlas =
        makeSixDofContactAtlas(numContacts);
    const BasisInputsModelDecorator<scalar_t>& decorator = *atlas->decorator;
    const size_t numBasis = decorator.getNumBasisPerFoot();
    const vector_t state = decorator.getNeutralState();
    vector_t localInput =
        vector_t::Zero(static_cast<Eigen::Index>(decorator.getInputDim()));
    vector_t worldInput = localInput;
    for (size_t i = 0; i < numContacts; ++i) {
      // A wrench inside the cone of B_i, so that the setters reproduce it.
      const vector6_t wrenchContactEwrtContact =
          atlas->bases[i].getBasisMatrix() * positiveScalings(numBasis, i);
      decorator.setWrenchContactEwrtParam(localInput, wrenchContactEwrtContact,
                                          i);
      const vector6_t wrenchContactEwrtWorld =
          decorator.computeWrenchContactEwrtWorld(state,
                                                  wrenchContactEwrtContact, i);
      decorator.setWrenchContactEwrtWorld(state, worldInput,
                                          wrenchContactEwrtWorld, i);
      EXPECT_TRUE(decorator.getWrenchContactEwrtParam(localInput, i)
                      .isApprox(wrenchContactEwrtContact, kTol))
          << numContacts << " contacts, contact " << i;
      EXPECT_TRUE(decorator.getWrenchContactEwrtWorld(state, worldInput, i)
                      .isApprox(wrenchContactEwrtWorld, kTol))
          << numContacts << " contacts, contact " << i;
    }
    // Writing one contact leaves the scalings of the others alone.
    for (size_t i = 0; i < numContacts; ++i) {
      EXPECT_TRUE(localInput
                      .segment(static_cast<Eigen::Index>(numBasis * i),
                               static_cast<Eigen::Index>(numBasis))
                      .isApprox(worldInput.segment(
                                    static_cast<Eigen::Index>(numBasis * i),
                                    static_cast<Eigen::Index>(numBasis)),
                                kTol))
          << numContacts << " contacts, contact " << i;
    }
    // The basis dynamics accept the set; the CppAD model is not built here.
    HUMANOID_ASSERT_OK(CentroidalDynamicsBasisInputsAD::validate(
        *atlas->pinocchioInterface, *atlas->info, *atlas->modelSettings,
        decorator.getBasisMatrices()));
  }
}

TEST(BasisInputsModelDecoratorContacts, theBasesMustMatchTheContactSet) {
  const std::unique_ptr<SixDofContactAtlas> atlas =
      makeSixDofContactAtlas(/*numContacts=*/3);
  HUMANOID_ASSERT_OK(BasisInputsModelDecorator<scalar_t>::validateBasisMatrices(
      atlas->bases, *atlas->modelSettings));
  const std::vector<ContactWrenchConeBasisMatrix> twoBases =
      makeTestBasisMatrices();
  const absl::Status tooFew =
      BasisInputsModelDecorator<scalar_t>::validateBasisMatrices(
          twoBases, *atlas->modelSettings);
  EXPECT_EQ(tooFew.code(), absl::StatusCode::kInvalidArgument) << tooFew;
  std::vector<ContactWrenchConeBasisMatrix> mixedSizes = atlas->bases;
  mixedSizes[2] = makeTestBasisMatrices(kExactWrenchConeGeneratorSet)[0];
  const absl::Status mixed =
      BasisInputsModelDecorator<scalar_t>::validateBasisMatrices(
          mixedSizes, *atlas->modelSettings);
  EXPECT_EQ(mixed.code(), absl::StatusCode::kInvalidArgument) << mixed;
  EXPECT_NE(mixed.message().find("seat_contact"), absl::string_view::npos)
      << mixed;

  // Create() refuses what validateBasisMatrices() refuses, and a null model.
  const absl::StatusOr<std::unique_ptr<BasisInputsModelDecorator<scalar_t>>>
      refused = BasisInputsModelDecorator<scalar_t>::Create(
          std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
              *atlas->modelSettings, *atlas->pinocchioInterface, *atlas->info),
          twoBases, *atlas->pinocchioInterface);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument)
      << refused.status();
  const absl::StatusOr<std::unique_ptr<BasisInputsModelDecorator<scalar_t>>>
      noModel = BasisInputsModelDecorator<scalar_t>::Create(
          /*wrappedModel=*/nullptr, atlas->bases, *atlas->pinocchioInterface);
  EXPECT_EQ(noModel.status().code(), absl::StatusCode::kInvalidArgument)
      << noModel.status();
  EXPECT_TRUE(
      BasisInputsModelDecorator<scalar_t>::Create(
          std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
              *atlas->modelSettings, *atlas->pinocchioInterface, *atlas->info),
          atlas->bases, *atlas->pinocchioInterface)
          .ok());
}

}  // namespace
}  // namespace ocs2::humanoid
