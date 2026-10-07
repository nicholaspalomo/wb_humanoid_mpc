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

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

namespace {

constexpr size_t kContactPointIndex = 0;
constexpr size_t kFourBasisVectors = 4;
constexpr size_t kEightBasisVectors = 8;
constexpr size_t kExpectedConstraintsFourBasis = 11;
constexpr size_t kExpectedConstraintsEightBasis = 15;

constexpr scalar_t kTestMu = 0.6;
constexpr scalar_t kTestMuRot = 0.1;
constexpr scalar_t kTestMinFz = 10.0;
constexpr scalar_t kTestFz = 100.0;
constexpr scalar_t kTestFx = 30.0;
constexpr scalar_t kTestFy = 10.0;
constexpr scalar_t kTestMz = 1.0;
constexpr scalar_t kTestBaseHeight = 0.8;
constexpr scalar_t kTolerance = 1.0e-4;
constexpr scalar_t kPrecisionTolerance = 1.0e-6;
constexpr scalar_t kFiniteDiffEps = 1.0e-7;
constexpr scalar_t kFiniteDiffTolerance = 1.0e-5;

constexpr size_t kYawConstraintPlusIdx = 9;
constexpr size_t kYawConstraintMinusIdx = 10;

/** Central differences of the constraint's value with respect to the state. */
matrix_t stateJacobianByFiniteDifferences(const ContactWrenchConeConstraint& constraint, const vector_t& state, const vector_t& input) {
  const PreComputation preComp;
  matrix_t jacobian(constraint.getNumConstraints(0.0), state.size());
  for (Eigen::Index index = 0; index < state.size(); ++index) {
    vector_t plus = state;
    plus(index) += kFiniteDiffEps;
    vector_t minus = state;
    minus(index) -= kFiniteDiffEps;
    jacobian.col(index) =
        (constraint.getValue(/*time=*/0.0, plus, input, preComp) - constraint.getValue(/*time=*/0.0, minus, input, preComp)) /
        (2.0 * kFiniteDiffEps);
  }
  return jacobian;
}

/**
 * A CentroidalMpcRobotModel that counts its live instances, so a test can see a clone that is never deleted. Its
 * clone() is built through the public constructor, from copies of the arguments, because the base's copy constructor
 * is private.
 */
class CountingRobotModel final : public CentroidalMpcRobotModel<scalar_t> {
 public:
  CountingRobotModel(const ModelSettings& modelSettings, const PinocchioInterface& pinocchioInterface, const CentroidalModelInfo& info)
      : CentroidalMpcRobotModel<scalar_t>(modelSettings, pinocchioInterface, info),
        modelSettings_(modelSettings),
        pinocchioInterface_(pinocchioInterface),
        info_(info) {
    ++liveInstances_;
  }
  ~CountingRobotModel() override { --liveInstances_; }
  CountingRobotModel(const CountingRobotModel&) = delete;
  CountingRobotModel& operator=(const CountingRobotModel&) = delete;
  CountingRobotModel* absl_nonnull clone() const override { return new CountingRobotModel(modelSettings_, pinocchioInterface_, info_); }

  static int liveInstances() { return liveInstances_; }

 private:
  const ModelSettings& modelSettings_;
  PinocchioInterface pinocchioInterface_;
  CentroidalModelInfo info_;
  static inline int liveInstances_ = 0;
};

}  // namespace

class TestContactWrenchConeConstraint : public ::testing::Test {
 protected:
  void SetUp() override {
    testingModelInterface_ = std::make_unique<CentroidalTestingModelInterface>();

    ModeSchedule initModeSchedule({0.0, 1.0}, {3});
    ModeSequenceTemplate initModeSequenceTemplate({0.5, 0.5}, {3, 3});
    std::shared_ptr<GaitSchedule> gaitSchedulePtr =
        std::make_shared<GaitSchedule>(initModeSchedule, initModeSequenceTemplate, /*phaseTransitionStanceTime=*/0.0);
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(gaitSchedulePtr, /*swingTrajectoryPtr=*/nullptr,
                                                                        testingModelInterface_->getPinocchioInterface(),
                                                                        testingModelInterface_->getMpcRobotModel());
  }

  std::unique_ptr<CentroidalTestingModelInterface> testingModelInterface_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

TEST_F(TestContactWrenchConeConstraint, NumberOfConstraintsAndBasisVectors) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  // Test with N = 4 basis vectors
  ContactWrenchConeConstraint::Config config4(kFourBasisVectors, /*frictionCoefficientParam=*/0.7,
                                              /*torsionalFrictionCoefficientParam=*/0.05, /*minNormalForceParam=*/5.0,
                                              /*gripperForceParam=*/0.0);
  ContactWrenchConeConstraint constraint4(*referenceManager_, contactRectangle, kContactPointIndex,
                                          testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                          config4);

  EXPECT_EQ(constraint4.getNumConstraints(0.0), kExpectedConstraintsFourBasis);

  // Test with N = 8 basis vectors
  ContactWrenchConeConstraint::Config config8(kEightBasisVectors, /*frictionCoefficientParam=*/0.7,
                                              /*torsionalFrictionCoefficientParam=*/0.05, /*minNormalForceParam=*/5.0,
                                              /*gripperForceParam=*/0.0);
  ContactWrenchConeConstraint constraint8(*referenceManager_, contactRectangle, kContactPointIndex,
                                          testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                          config8);

  EXPECT_EQ(constraint8.getNumConstraints(0.0), kExpectedConstraintsEightBasis);
}

TEST_F(TestContactWrenchConeConstraint, FrictionConeAndNormalForceValues) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  ContactWrenchConeConstraint::Config config(kFourBasisVectors, kTestMu, /*torsionalFrictionCoefficientParam=*/0.05, kTestMinFz,
                                             /*gripperForceParam=*/0.0);
  ContactWrenchConeConstraint constraint(*referenceManager_, contactRectangle, kContactPointIndex,
                                         testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                         config);

  const CentroidalMpcRobotModel<scalar_t>& robotModel = testingModelInterface_->getMpcRobotModel();
  vector_t state = vector_t::Zero(robotModel.getStateDim());
  state[2] = kTestBaseHeight;

  vector_t input = vector_t::Zero(robotModel.getInputDim());
  robotModel.setContactForce(input, vector3_t(kTestFx, 0.0, kTestFz), kContactPointIndex);

  PreComputation preComp;
  vector_t val = constraint.getValue(/*time=*/0.0, state, input, preComp);

  // For N=4, directions are (1,0), (0,1), (-1,0), (0,-1)
  // 1. mu*Fz - Fx = 0.6*100 - 30 = 30.0
  EXPECT_NEAR(val[0], 30.0, kTolerance);
  // 2. mu*Fz - Fy = 0.6*100 - 0 = 60.0
  EXPECT_NEAR(val[1], 60.0, kTolerance);
  // 3. mu*Fz + Fx = 0.6*100 + 30 = 90.0
  EXPECT_NEAR(val[2], 90.0, kTolerance);
  // 4. mu*Fz + Fy = 0.6*100 + 0 = 60.0
  EXPECT_NEAR(val[3], 60.0, kTolerance);
  // 5. Normal force: Fz - minFz = 100 - 10 = 90.0
  EXPECT_NEAR(val[4], 90.0, kTolerance);
}

TEST_F(TestContactWrenchConeConstraint, ContactPatchOffsetMoments) {
  PolygonBounds bounds(0.0, 0.2, -0.05, 0.05);
  ContactRectangle contactRectangle(bounds, ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  vector3_t explicitPatchOffset(0.1, 0.0, 0.0);
  ContactWrenchConeConstraint::Config config(kFourBasisVectors, kTestMu, kTestMuRot, /*minNormalForceParam=*/0.0, /*gripperForceParam=*/0.0,
                                             explicitPatchOffset);
  ContactWrenchConeConstraint constraint(*referenceManager_, contactRectangle, kContactPointIndex,
                                         testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                         config);

  const CentroidalMpcRobotModel<scalar_t>& robotModel = testingModelInterface_->getMpcRobotModel();
  vector_t state = vector_t::Zero(robotModel.getStateDim());
  state[2] = kTestBaseHeight;

  vector_t input = vector_t::Zero(robotModel.getInputDim());
  robotModel.setContactForce(input, vector3_t(0.0, kTestFy, kTestFz), kContactPointIndex);

  // Set contact moment at foot origin Mz = 1.0 Nm:
  // M_patch_z = Mz - (x_offset * Fy - y_offset * Fx) = 1.0 - (0.1 * 10.0 - 0.0) = 0.0 Nm
  robotModel.setContactMoment(input, vector3_t(0.0, 0.0, kTestMz), kContactPointIndex);

  PreComputation preComp;
  vector_t val = constraint.getValue(/*time=*/0.0, state, input, preComp);

  // Constraints 9 and 10 are yaw moment constraints: mu_rot*Fz +/- M_patch_z
  // mu_rot * Fz = 0.1 * 100 = 10.0 Nm
  // Since M_patch_z = 0.0, both should be 10.0
  EXPECT_NEAR(val[kYawConstraintPlusIdx], 10.0, kTolerance);
  EXPECT_NEAR(val[kYawConstraintMinusIdx], 10.0, kTolerance);
}

TEST_F(TestContactWrenchConeConstraint, LinearAndQuadraticApproximation) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  ContactWrenchConeConstraint::Config config(kFourBasisVectors, /*frictionCoefficientParam=*/0.7,
                                             /*torsionalFrictionCoefficientParam=*/0.05, /*minNormalForceParam=*/5.0,
                                             /*gripperForceParam=*/0.0);
  ContactWrenchConeConstraint constraint(*referenceManager_, contactRectangle, kContactPointIndex,
                                         testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                         config);

  const CentroidalMpcRobotModel<scalar_t>& robotModel = testingModelInterface_->getMpcRobotModel();
  vector_t state = vector_t::Zero(robotModel.getStateDim());
  state[2] = kTestBaseHeight;

  vector_t input = vector_t::Zero(robotModel.getInputDim());
  robotModel.setContactForce(input, vector3_t(10.0, 5.0, 50.0), kContactPointIndex);
  robotModel.setContactMoment(input, vector3_t(0.5, -0.2, 0.1), kContactPointIndex);

  PreComputation preComp;
  VectorFunctionLinearApproximation linApprox = constraint.getLinearApproximation(/*time=*/0.0, state, input, preComp);
  vector_t val = constraint.getValue(/*time=*/0.0, state, input, preComp);

  EXPECT_TRUE(linApprox.f.isApprox(val, kPrecisionTolerance));
  EXPECT_EQ(linApprox.dfdx.rows(), static_cast<Eigen::Index>(constraint.getNumConstraints(0.0)));
  EXPECT_EQ(linApprox.dfdx.cols(), static_cast<Eigen::Index>(robotModel.getStateDim()));
  // The rows read the wrench in the foot frame, so they depend on the foot's orientation; see
  // StateDerivativeGoesThroughTheFootOrientation.
  EXPECT_TRUE(linApprox.dfdx.isApprox(stateJacobianByFiniteDifferences(constraint, state, input), kFiniteDiffTolerance));

  // Finite difference test for dfdu
  matrix_t numDfdu = matrix_t::Zero(constraint.getNumConstraints(0.0), robotModel.getInputDim());
  for (size_t i = 0; i < robotModel.getInputDim(); ++i) {
    vector_t inputPlus = input;
    inputPlus[i] += kFiniteDiffEps;
    vector_t inputMinus = input;
    inputMinus[i] -= kFiniteDiffEps;
    vector_t valPlus = constraint.getValue(/*time=*/0.0, state, inputPlus, preComp);
    vector_t valMinus = constraint.getValue(/*time=*/0.0, state, inputMinus, preComp);
    numDfdu.col(i) = (valPlus - valMinus) / (2.0 * kFiniteDiffEps);
  }

  EXPECT_TRUE(linApprox.dfdu.isApprox(numDfdu, kFiniteDiffTolerance));

  VectorFunctionQuadraticApproximation quadApprox = constraint.getQuadraticApproximation(/*time=*/0.0, state, input, preComp);
  EXPECT_TRUE(quadApprox.f.isApprox(val, kPrecisionTolerance));
  EXPECT_TRUE(quadApprox.dfdu.isApprox(linApprox.dfdu, kPrecisionTolerance));
  EXPECT_EQ(quadApprox.dfdxx.size(), constraint.getNumConstraints(0.0));
  EXPECT_EQ(quadApprox.dfduu.size(), constraint.getNumConstraints(0.0));
  for (size_t k = 0; k < constraint.getNumConstraints(0.0); ++k) {
    EXPECT_TRUE(quadApprox.dfdxx[k].isZero());
    EXPECT_TRUE(quadApprox.dfduu[k].isZero());
  }
}

TEST_F(TestContactWrenchConeConstraint, StateDerivativeGoesThroughTheFootOrientation) {
  // Every row reads l_R_w(q) * W_world, so a friction row of a foot carrying F newtons changes by about F per radian of
  // pitch. The term used to report dfdx = 0, which the contact-implicit formulation - un-gated cones, and a loaded
  // foot's rocking rates deliberately left free - turns into a wrong gradient on exactly the motion it enables. Checked
  // at a foot pitched and rolled on a rotated base, gated and un-gated, against central differences of getValue().
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));
  const CentroidalMpcRobotModel<scalar_t>& robotModel = testingModelInterface_->getMpcRobotModel();
  const std::vector<std::string>& jointNames = testingModelInterface_->getModelSettings().mpcModelJointNames;

  vector_t state = vector_t::Zero(robotModel.getStateDim());
  state[robotModel.getBaseStartindex() + 2] = kTestBaseHeight;
  robotModel.setBaseOrientationEulerZYX(state, vector3_t(0.2, 0.1, -0.05));
  for (const std::string& jointName : {std::string("left_ankle_pitch_joint"), std::string("left_ankle_roll_joint")}) {
    const std::vector<std::string>::const_iterator joint = std::find(jointNames.begin(), jointNames.end(), jointName);
    ASSERT_NE(joint, jointNames.end()) << jointName;
    state[robotModel.getJointStartindex() + static_cast<size_t>(std::distance(jointNames.begin(), joint))] = 0.3;
  }

  vector_t input = vector_t::Zero(robotModel.getInputDim());
  robotModel.setContactForce(input, vector3_t(40.0, -25.0, 600.0), kContactPointIndex);
  robotModel.setContactMoment(input, vector3_t(5.0, -3.0, 2.0), kContactPointIndex);

  for (const bool scheduleGated : {true, false}) {
    const ContactWrenchConeConstraint constraint(
        *referenceManager_, contactRectangle, kContactPointIndex, testingModelInterface_->getPinocchioInterface(), robotModel,
        ContactWrenchConeConstraint::Config(kFourBasisVectors, kTestMu, kTestMuRot, /*minNormalForceParam=*/0.0, /*gripperForceParam=*/0.0),
        scheduleGated);
    const PreComputation preComp;
    const matrix_t analytic = constraint.getLinearApproximation(/*time=*/0.0, state, input, preComp).dfdx;
    const matrix_t numerical = stateJacobianByFiniteDifferences(constraint, state, input);
    // Positive control: under 600 N the orientation dependence is large, so a zero Jacobian cannot pass.
    ASSERT_GT(numerical.cwiseAbs().maxCoeff(), 100.0) << "gated: " << scheduleGated;
    for (Eigen::Index row = 0; row < numerical.rows(); ++row) {
      for (Eigen::Index col = 0; col < numerical.cols(); ++col) {
        EXPECT_NEAR(analytic(row, col), numerical(row, col), 1.0e-4)
            << "gated: " << scheduleGated << ", dfdx(" << row << ", " << col << ")";
      }
    }
  }
}

TEST_F(TestContactWrenchConeConstraint, RefusesABasisVectorModel) {
  // The term reads getContactForce/getContactMoment as a WORLD wrench and writes 3-column Jacobian blocks at the force
  // and moment start indices. Under BasisInputsModelDecorator both indices are the start of one block of scalings and
  // the accessors return a LOCAL wrench, so the value would be rotated twice and the moment block would overwrite the
  // force block. It must refuse the model rather than build that term.
  const ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                          ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));
  const ContactWrenchConeConstraint::Config coneConfig(kFourBasisVectors, /*frictionCoefficientParam=*/0.7,
                                                       /*torsionalFrictionCoefficientParam=*/0.05, /*minNormalForceParam=*/5.0,
                                                       /*gripperForceParam=*/0.0);

  // Positive control: the wrench-space model is accepted.
  const absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> wrenchSpace = ContactWrenchConeConstraint::Create(
      *referenceManager_, contactRectangle, kContactPointIndex, testingModelInterface_->getPinocchioInterface(),
      testingModelInterface_->getMpcRobotModel(), coneConfig);
  ASSERT_TRUE(wrenchSpace.ok()) << wrenchSpace.status();

  const PolygonBounds footBounds(-0.1, 0.1, -0.05, 0.05);
  const absl::StatusOr<ContactWrenchConeBasisMatrix> leftBasis = ContactWrenchConeBasisMatrix::Create(
      coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero())));
  const absl::StatusOr<ContactWrenchConeBasisMatrix> rightBasis = ContactWrenchConeBasisMatrix::Create(
      coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_r_contact", "right_ankle_roll_joint", vector3_t::Zero())));
  ASSERT_TRUE(leftBasis.ok()) << leftBasis.status();
  ASSERT_TRUE(rightBasis.ok()) << rightBasis.status();
  const BasisInputsModelDecorator<scalar_t> basisModel(
      std::unique_ptr<MpcRobotModelBase<scalar_t>>(testingModelInterface_->getMpcRobotModel().clone()),
      std::array<ContactWrenchConeBasisMatrix, kNumContacts>{*leftBasis, *rightBasis}, testingModelInterface_->getPinocchioInterface());

  const absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> onBasis = ContactWrenchConeConstraint::Create(
      *referenceManager_, contactRectangle, kContactPointIndex, testingModelInterface_->getPinocchioInterface(), basisModel, coneConfig);
  ASSERT_FALSE(onBasis.ok()) << "a wrench cone on the basis-vector model evaluates the wrench in the wrong frame";
  EXPECT_EQ(onBasis.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(onBasis.status().message(), R"(contact_input_parameterization: "basis_vectors")"))
      << onBasis.status().message();

  // And the constructor, for the callers not yet moved to Create(), refuses it the same way.
  EXPECT_DEATH(std::make_unique<ContactWrenchConeConstraint>(*referenceManager_, contactRectangle, kContactPointIndex,
                                                             testingModelInterface_->getPinocchioInterface(), basisModel, coneConfig),
               R"(contact_input_parameterization: "basis_vectors")");
}

TEST_F(TestContactWrenchConeConstraint, BorrowsItsRobotModelRatherThanLeakingAClone) {
  // Both constructors stored mpcRobotModel.clone() in a raw pointer that nothing deleted, so every construction and
  // every per-thread copy of the optimal control problem leaked a whole robot model, PinocchioInterface included.
  const ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                          ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));
  const int before = CountingRobotModel::liveInstances();
  {
    const CountingRobotModel robotModel(testingModelInterface_->getModelSettings(), testingModelInterface_->getPinocchioInterface(),
                                        testingModelInterface_->getCentroidalModelInfo());
    // Positive control: the counter sees a clone.
    const std::unique_ptr<MpcRobotModelBase<scalar_t>> probe(robotModel.clone());
    ASSERT_EQ(CountingRobotModel::liveInstances(), before + 2);

    const ContactWrenchConeConstraint constraint(*referenceManager_, contactRectangle, kContactPointIndex,
                                                 testingModelInterface_->getPinocchioInterface(), robotModel);
    const std::unique_ptr<ContactWrenchConeConstraint> copy(constraint.clone());
    EXPECT_EQ(CountingRobotModel::liveInstances(), before + 2) << "the term, or its copy, cloned the robot model";
  }
  EXPECT_EQ(CountingRobotModel::liveInstances(), before) << "a robot model outlived every owner: it was leaked";
}

}  // namespace ocs2::humanoid
