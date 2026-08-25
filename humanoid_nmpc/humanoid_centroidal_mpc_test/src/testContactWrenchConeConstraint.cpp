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
#include <cmath>
#include <vector>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

namespace {

static constexpr size_t kContactPointIndex = 0;
static constexpr size_t kFourBasisVectors = 4;
static constexpr size_t kEightBasisVectors = 8;
static constexpr size_t kExpectedConstraintsFourBasis = 11;
static constexpr size_t kExpectedConstraintsEightBasis = 15;

static constexpr scalar_t kTestMu = 0.6;
static constexpr scalar_t kTestMuRot = 0.1;
static constexpr scalar_t kTestMinFz = 10.0;
static constexpr scalar_t kTestFz = 100.0;
static constexpr scalar_t kTestFx = 30.0;
static constexpr scalar_t kTestFy = 10.0;
static constexpr scalar_t kTestMz = 1.0;
static constexpr scalar_t kTestBaseHeight = 0.8;
static constexpr scalar_t kTolerance = 1e-4;
static constexpr scalar_t kPrecisionTolerance = 1e-6;
static constexpr scalar_t kFiniteDiffEps = 1e-7;
static constexpr scalar_t kFiniteDiffTolerance = 1e-5;

static constexpr size_t kYawConstraintPlusIdx = 9;
static constexpr size_t kYawConstraintMinusIdx = 10;

}  // namespace

class TestContactWrenchConeConstraint : public ::testing::Test {
 protected:
  void SetUp() override {
    testingModelInterface_ = std::make_unique<CentroidalTestingModelInterface>();

    ModeSchedule initModeSchedule({0.0, 1.0}, {3});
    ModeSequenceTemplate initModeSequenceTemplate({0.5, 0.5}, {3, 3});
    std::shared_ptr<GaitSchedule> gaitSchedulePtr = std::make_shared<GaitSchedule>(initModeSchedule, initModeSequenceTemplate, 0.0);
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        gaitSchedulePtr, nullptr, testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel());
  }

  std::unique_ptr<CentroidalTestingModelInterface> testingModelInterface_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

TEST_F(TestContactWrenchConeConstraint, NumberOfConstraintsAndBasisVectors) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  // Test with N = 4 basis vectors
  ContactWrenchConeConstraint::Config config4(kFourBasisVectors, 0.7, 0.05, 5.0, 0.0);
  ContactWrenchConeConstraint constraint4(*referenceManager_, contactRectangle, kContactPointIndex,
                                          testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                          config4);

  EXPECT_EQ(constraint4.getNumConstraints(0.0), kExpectedConstraintsFourBasis);

  // Test with N = 8 basis vectors
  ContactWrenchConeConstraint::Config config8(kEightBasisVectors, 0.7, 0.05, 5.0, 0.0);
  ContactWrenchConeConstraint constraint8(*referenceManager_, contactRectangle, kContactPointIndex,
                                          testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                          config8);

  EXPECT_EQ(constraint8.getNumConstraints(0.0), kExpectedConstraintsEightBasis);
}

TEST_F(TestContactWrenchConeConstraint, FrictionConeAndNormalForceValues) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  ContactWrenchConeConstraint::Config config(kFourBasisVectors, kTestMu, 0.05, kTestMinFz, 0.0);
  ContactWrenchConeConstraint constraint(*referenceManager_, contactRectangle, kContactPointIndex,
                                         testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel(),
                                         config);

  const CentroidalMpcRobotModel<scalar_t>& robotModel = testingModelInterface_->getMpcRobotModel();
  vector_t state = vector_t::Zero(robotModel.getStateDim());
  state[2] = kTestBaseHeight;

  vector_t input = vector_t::Zero(robotModel.getInputDim());
  robotModel.setContactForce(input, vector3_t(kTestFx, 0.0, kTestFz), kContactPointIndex);

  PreComputation preComp;
  vector_t val = constraint.getValue(0.0, state, input, preComp);

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
  ContactWrenchConeConstraint::Config config(kFourBasisVectors, kTestMu, kTestMuRot, 0.0, 0.0, explicitPatchOffset);
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
  vector_t val = constraint.getValue(0.0, state, input, preComp);

  // Constraints 9 and 10 are yaw moment constraints: mu_rot*Fz +/- M_patch_z
  // mu_rot * Fz = 0.1 * 100 = 10.0 Nm
  // Since M_patch_z = 0.0, both should be 10.0
  EXPECT_NEAR(val[kYawConstraintPlusIdx], 10.0, kTolerance);
  EXPECT_NEAR(val[kYawConstraintMinusIdx], 10.0, kTolerance);
}

TEST_F(TestContactWrenchConeConstraint, LinearAndQuadraticApproximation) {
  ContactRectangle contactRectangle(PolygonBounds(-0.1, 0.1, -0.05, 0.05),
                                    ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()));

  ContactWrenchConeConstraint::Config config(kFourBasisVectors, 0.7, 0.05, 5.0, 0.0);
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
  VectorFunctionLinearApproximation linApprox = constraint.getLinearApproximation(0.0, state, input, preComp);
  vector_t val = constraint.getValue(0.0, state, input, preComp);

  EXPECT_TRUE(linApprox.f.isApprox(val, kPrecisionTolerance));
  EXPECT_EQ(linApprox.dfdx.rows(), static_cast<Eigen::Index>(constraint.getNumConstraints(0.0)));
  EXPECT_EQ(linApprox.dfdx.cols(), static_cast<Eigen::Index>(robotModel.getStateDim()));
  EXPECT_TRUE(linApprox.dfdx.isZero());

  // Finite difference test for dfdu
  matrix_t numDfdu = matrix_t::Zero(constraint.getNumConstraints(0.0), robotModel.getInputDim());
  for (size_t i = 0; i < robotModel.getInputDim(); ++i) {
    vector_t inputPlus = input;
    inputPlus[i] += kFiniteDiffEps;
    vector_t inputMinus = input;
    inputMinus[i] -= kFiniteDiffEps;
    vector_t valPlus = constraint.getValue(0.0, state, inputPlus, preComp);
    vector_t valMinus = constraint.getValue(0.0, state, inputMinus, preComp);
    numDfdu.col(i) = (valPlus - valMinus) / (2.0 * kFiniteDiffEps);
  }

  EXPECT_TRUE(linApprox.dfdu.isApprox(numDfdu, kFiniteDiffTolerance));

  VectorFunctionQuadraticApproximation quadApprox = constraint.getQuadraticApproximation(0.0, state, input, preComp);
  EXPECT_TRUE(quadApprox.f.isApprox(val, kPrecisionTolerance));
  EXPECT_TRUE(quadApprox.dfdu.isApprox(linApprox.dfdu, kPrecisionTolerance));
  EXPECT_EQ(quadApprox.dfdxx.size(), constraint.getNumConstraints(0.0));
  EXPECT_EQ(quadApprox.dfduu.size(), constraint.getNumConstraints(0.0));
  for (size_t k = 0; k < constraint.getNumConstraints(0.0); ++k) {
    EXPECT_TRUE(quadApprox.dfdxx[k].isZero());
    EXPECT_TRUE(quadApprox.dfduu[k].isZero());
  }
}

}  // namespace ocs2::humanoid
