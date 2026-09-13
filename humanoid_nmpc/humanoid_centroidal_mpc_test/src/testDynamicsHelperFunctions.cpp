/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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
#include "humanoid_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

#include "ocs2_centroidal_model/ModelHelperFunctions.h"

namespace ocs2::humanoid {

TEST(TestDynamicsHelperFunctions, computeContactCoP) {
  CentroidalTestingModelInterface testingModelInterface = CentroidalTestingModelInterface();

  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  size_t joint_dim = testingModelInterface.getMpcRobotModel().getJointDim();
  vector_t q = vector_t::Zero(6 + joint_dim);
  q[2] = 0.86566;
  q.tail(joint_dim) = vector_t::Random(joint_dim);

  vector_t input = vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setContactForce(input, vector3_t(0, 0, 100), 0);
  testingModelInterface.getMpcRobotModel().setContactForce(input, vector3_t(50, 30, 100), 1);

  std::vector<vector3_t> contactPositions =
      computeContactPositions<scalar_t>(q, pinocchioInterface, testingModelInterface.getMpcRobotModel());
  std::vector<vector3_t> contactCoPs = computeContactsCoP(input, pinocchioInterface, {1, 1}, testingModelInterface.getMpcRobotModel());

  EXPECT_TRUE(contactPositions[0].isApprox(contactCoPs[0]));
  EXPECT_TRUE(contactPositions[1].isApprox(contactCoPs[1]));
}

TEST(TestDynamicsHelperFunctions, computeContactCoPStateAwareOverloadMatchesInputOnlyForWrenchModel) {
  CentroidalTestingModelInterface testingModelInterface = CentroidalTestingModelInterface();
  CentroidalMpcRobotModel<scalar_t>& model = testingModelInterface.getMpcRobotModel();

  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const size_t joint_dim = model.getJointDim();
  vector_t q = vector_t::Zero(6 + joint_dim);
  q[2] = 0.86566;
  q[3] = M_PI / 2.0;  // yawed base
  q.tail(joint_dim) = vector_t::Random(joint_dim);
  updateFramePlacements<scalar_t>(q, pinocchioInterface);

  vector_t state = vector_t::Zero(model.getStateDim());
  model.setGeneralizedCoordinates(state, q);

  // Off-center wrenches so that the CoPs are not trivially the contact frame origins.
  vector_t input = vector_t::Zero(model.getInputDim());
  model.setContactWrench(input, (vector6_t() << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5).finished(), 0);
  model.setContactWrench(input, (vector6_t() << -20.0, 8.0, 150.0, -3.0, 4.0, -0.2).finished(), 1);

  // The wrench-space input already stores world-frame wrenches, so the state-aware overload must reduce to the
  // input-only one for every contact configuration.
  for (const contact_flag_t& flags : {contact_flag_t{true, true}, contact_flag_t{true, false}, contact_flag_t{false, true}}) {
    const std::vector<vector3_t> copsInputOnly = computeContactsCoP(input, pinocchioInterface, flags, model);
    const std::vector<vector3_t> copsStateAware = computeContactsCoP(state, input, pinocchioInterface, flags, model);
    ASSERT_EQ(copsInputOnly.size(), copsStateAware.size());
    for (size_t i = 0; i < copsInputOnly.size(); ++i) {
      EXPECT_TRUE((copsInputOnly[i] - copsStateAware[i]).isZero(1e-12))
          << "CoP mismatch for contact " << i << ": input-only = " << copsInputOnly[i].transpose()
          << ", state-aware = " << copsStateAware[i].transpose();
    }
  }
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    EXPECT_TRUE((computeContactCoP<scalar_t>(input, pinocchioInterface, i, model) -
                 computeContactCoP<scalar_t>(state, input, pinocchioInterface, i, model))
                    .isZero(1e-12));
  }
}

TEST(TestDynamicsHelperFunctions, weightCompensatingInput) {
  CentroidalTestingModelInterface testingModelInterface = CentroidalTestingModelInterface();

  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  CentroidalModelInfo centroidalModelInfo = testingModelInterface.getCentroidalModelInfo();

  const static scalar_t totalGravitationalForce = centroidalModelInfo.robotMass * 9.81;

  vector_t inputDoubleContact = vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setContactForce(inputDoubleContact, vector3_t(0, 0, totalGravitationalForce / 2), 0);
  testingModelInterface.getMpcRobotModel().setContactForce(inputDoubleContact, vector3_t(0, 0, totalGravitationalForce / 2), 1);

  EXPECT_TRUE(
      inputDoubleContact.isApprox(weightCompensatingInput(centroidalModelInfo, {true, true}, testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(
      inputDoubleContact.isApprox(weightCompensatingInput(pinocchioInterface, {true, true}, testingModelInterface.getMpcRobotModel())));

  vector_t inputLeftContact = vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setContactForce(inputLeftContact, vector3_t(0, 0, totalGravitationalForce), 0);

  EXPECT_TRUE(
      inputLeftContact.isApprox(weightCompensatingInput(centroidalModelInfo, {true, false}, testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(
      inputLeftContact.isApprox(weightCompensatingInput(pinocchioInterface, {true, false}, testingModelInterface.getMpcRobotModel())));

  vector_t inputRightContact = vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setContactForce(inputRightContact, vector3_t(0, 0, totalGravitationalForce), 1);
  EXPECT_TRUE(
      inputRightContact.isApprox(weightCompensatingInput(centroidalModelInfo, {false, true}, testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(
      inputRightContact.isApprox(weightCompensatingInput(pinocchioInterface, {false, true}, testingModelInterface.getMpcRobotModel())));

  EXPECT_TRUE(
      ocs2::getNormalizedCentroidalMomentumRate(pinocchioInterface, centroidalModelInfo, inputDoubleContact).isApprox(vector_t::Zero(6)));
}

TEST(TestDynamicsHelperFunctions, weightCompensatingInputStateAwareOverloadMatchesInputOnlyForWrenchModel) {
  CentroidalTestingModelInterface testingModelInterface = CentroidalTestingModelInterface();
  CentroidalMpcRobotModel<scalar_t>& model = testingModelInterface.getMpcRobotModel();

  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  CentroidalModelInfo centroidalModelInfo = testingModelInterface.getCentroidalModelInfo();

  // Fixture state: standing height, a yawed and slightly tilted base, and random joint angles. For the wrench-space
  // model the input stores world-frame wrenches directly, so none of this may influence the weight-compensating input.
  vector_t state = vector_t::Zero(model.getStateDim());
  model.setBasePosition(state, vector3_t(0.3, -0.2, 0.86566));
  model.setBaseOrientationEulerZYX(state, vector3_t(M_PI / 2.0, 0.1, -0.05));
  model.setJointAngles(state, vector_t::Random(model.getJointDim()));

  const scalar_t totalGravitationalForce = centroidalModelInfo.robotMass * 9.81;

  for (const contact_flag_t& flags :
       {contact_flag_t{true, true}, contact_flag_t{true, false}, contact_flag_t{false, true}, contact_flag_t{false, false}}) {
    const vector_t inputOnlyFromPinocchio = weightCompensatingInput(pinocchioInterface, flags, model);
    const vector_t stateAwareFromPinocchio = weightCompensatingInput(pinocchioInterface, flags, model, state);
    EXPECT_TRUE((stateAwareFromPinocchio - inputOnlyFromPinocchio).isZero(1e-12))
        << "PinocchioInterface overloads differ for flags {" << flags[0] << ", " << flags[1] << "}";

    const vector_t inputOnlyFromInfo = weightCompensatingInput(centroidalModelInfo, flags, model);
    const vector_t stateAwareFromInfo = weightCompensatingInput(centroidalModelInfo, flags, model, state);
    EXPECT_TRUE((stateAwareFromInfo - inputOnlyFromInfo).isZero(1e-12))
        << "CentroidalModelInfo overloads differ for flags {" << flags[0] << ", " << flags[1] << "}";

    // Both families must agree with each other and produce vertical world-frame forces that sum to the robot weight.
    EXPECT_TRUE((stateAwareFromInfo - stateAwareFromPinocchio).isZero(1e-9));
    vector3_t totalForce = vector3_t::Zero();
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      const vector3_t force = model.getContactForceInWorldFrame(state, stateAwareFromPinocchio, i);
      EXPECT_TRUE((force - model.getContactForce(stateAwareFromPinocchio, i)).isZero(0.0));
      if (!flags[i]) {
        EXPECT_TRUE(force.isZero(0.0)) << "Swing foot " << i << " must carry no force";
      }
      totalForce += force;
    }
    const scalar_t expectedTotal = (flags[0] || flags[1]) ? totalGravitationalForce : 0.0;
    EXPECT_TRUE((totalForce - vector3_t(0.0, 0.0, expectedTotal)).isZero(1e-9))
        << "Total weight-compensating force should be vertical and equal to the robot weight, got " << totalForce.transpose();
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace ocs2::humanoid
