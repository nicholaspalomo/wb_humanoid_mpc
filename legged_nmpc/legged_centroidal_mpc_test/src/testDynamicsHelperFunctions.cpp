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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "legged_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "legged_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::legged {

TEST(TestDynamicsHelperFunctions, computePosCopRtWorldEwrtWorld) {
  CentroidalTestingModelInterface testingModelInterface =
      CentroidalTestingModelInterface();

  PinocchioInterface pinocchioInterface =
      testingModelInterface.getPinocchioInterface();
  const CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface.getMpcRobotModel();
  // q = [p_W, xi, q_j] of a level base (the identity quaternion) at standing
  // height, the joints bent at random.
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.86566));
  model.setJointAngles(
      state, vector_t::Random(static_cast<Eigen::Index>(model.getJointDim())));
  const vector_t q = model.getGeneralizedCoordinates(state);

  vector_t input =
      vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      input, vector3_t(0, 0, 100), /*contactIndex=*/0);
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      input, vector3_t(50, 30, 100), /*contactIndex=*/1);

  std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          q, pinocchioInterface, testingModelInterface.getMpcRobotModel());
  // A pure force acts at the contact frame origin: the center of pressure is
  // the contact position.
  for (size_t contactIndex = 0;
       contactIndex < static_cast<size_t>(kBipedFootCount); ++contactIndex) {
    EXPECT_TRUE(posContactsRtWorldEwrtWorld[contactIndex].isApprox(
        computePosCopRtWorldEwrtWorld<scalar_t>(
            state, input, pinocchioInterface, contactIndex,
            testingModelInterface.getMpcRobotModel())))
        << "contact " << contactIndex;
  }
}

TEST(TestDynamicsHelperFunctions,
     computePosCopRtWorldEwrtWorldMatchesTheParamWrenchForWrenchModel) {
  CentroidalTestingModelInterface testingModelInterface =
      CentroidalTestingModelInterface();
  CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface.getMpcRobotModel();

  PinocchioInterface pinocchioInterface =
      testingModelInterface.getPinocchioInterface();
  // A base yawed by 90 degrees at standing height, the joints bent at random.
  vector_t state = model.getNeutralState();
  model.setPoseeulerzyxBaseRtWorld(state, (vector6_t() << 0.0, 0.0, 0.86566,
                                           std::numbers::pi / 2.0, 0.0, 0.0)
                                              .finished());
  model.setJointAngles(
      state, vector_t::Random(static_cast<Eigen::Index>(model.getJointDim())));
  const vector_t q = model.getGeneralizedCoordinates(state);
  updateFramePlacements<scalar_t>(q, pinocchioInterface);

  // Off-center wrenches so that the CoPs are not trivially the contact frame
  // origins.
  vector_t input = vector_t::Zero(model.getInputDim());
  model.setWrenchContactEwrtParam(
      input, (vector6_t() << 10.0, 5.0, 100.0, 1.0, -2.0, 0.5).finished(),
      /*contactIndex=*/0);
  model.setWrenchContactEwrtParam(
      input, (vector6_t() << -20.0, 8.0, 150.0, -3.0, 4.0, -0.2).finished(),
      /*contactIndex=*/1);

  // The wrench-space input already stores world-frame wrenches, so the CoP must
  // be the one of the input's wrench rotated into the contact frame.
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    const pinocchio::FrameIndex frameIndex =
        getContactFrameIndex(pinocchioInterface, model, i);
    const vector6_t wrenchContactEwrtContact =
        rotateVectorWorldToLocal<scalar_t>(
            model.getWrenchContactEwrtParam(input, i),
            pinocchioInterface.getData(), frameIndex);
    const vector3_t posCopRtContactEwrtContact(
        -wrenchContactEwrtContact[kWrenchTorqueYIndex] /
            wrenchContactEwrtContact[kWrenchForceZIndex],
        wrenchContactEwrtContact[kWrenchTorqueXIndex] /
            wrenchContactEwrtContact[kWrenchForceZIndex],
        /*z=*/0.0);
    EXPECT_TRUE(
        (transformPointLocalToWorld(posCopRtContactEwrtContact,
                                    pinocchioInterface.getData(), frameIndex) -
         computePosCopRtWorldEwrtWorld<scalar_t>(state, input,
                                                 pinocchioInterface, i, model))
            .isZero(1.0e-12))
        << "contact " << i;
  }
}

TEST(TestDynamicsHelperFunctions,
     thePendulumHeightsAreMeasuredFromTheirFrames) {
  CentroidalTestingModelInterface testingModelInterface =
      CentroidalTestingModelInterface();
  PinocchioInterface pinocchioInterface =
      testingModelInterface.getPinocchioInterface();
  const CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface.getMpcRobotModel();
  // Bent joints, so that the contact frames stand at different heights.
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.86566));
  model.setJointAngles(
      state, vector_t::Random(static_cast<Eigen::Index>(model.getJointDim())));
  const vector_t q = model.getGeneralizedCoordinates(state);

  // The independent computation: the center of mass and the contact origins.
  const scalar_t posComRtWorldEwrtWorldZ = pinocchio::centerOfMass(
      pinocchioInterface.getModel(), pinocchioInterface.getData(), q)(2);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface,
                                                   model);
  scalar_t posFeetmeanRtWorldEwrtWorldZ = 0.0;
  scalar_t minPosContactsRtWorldEwrtWorldZ =
      std::numeric_limits<scalar_t>::infinity();
  for (const vector3_t& posContactRtWorldEwrtWorld :
       posContactsRtWorldEwrtWorld) {
    posFeetmeanRtWorldEwrtWorldZ +=
        posContactRtWorldEwrtWorld(2) /
        static_cast<scalar_t>(posContactsRtWorldEwrtWorld.size());
    minPosContactsRtWorldEwrtWorldZ = std::min(minPosContactsRtWorldEwrtWorldZ,
                                               posContactRtWorldEwrtWorld(2));
  }

  const scalar_t posComRtFeetmeanEwrtWorldZ =
      computePosComRtFeetmeanEwrtWorldZ(q, pinocchioInterface, model);
  const scalar_t posComRtLowcontactEwrtWorldZ =
      computePosComRtLowcontactEwrtWorldZ(q, pinocchioInterface, model);
  EXPECT_NEAR(posComRtFeetmeanEwrtWorldZ,
              posComRtWorldEwrtWorldZ - posFeetmeanRtWorldEwrtWorldZ, 1.0e-12);
  EXPECT_NEAR(posComRtLowcontactEwrtWorldZ,
              posComRtWorldEwrtWorldZ - minPosContactsRtWorldEwrtWorldZ,
              1.0e-12);
  EXPECT_GE(posComRtLowcontactEwrtWorldZ, posComRtFeetmeanEwrtWorldZ)
      << "the lowest contact is never above the mean of the contacts";
  // The proxy is the height above the feet on the biped contact set, and the
  // height above the lowest contact on any other.
  EXPECT_EQ(
      computeProxyPosComRtFeetmeanEwrtWorldZ(q, pinocchioInterface, model),
      model.modelSettings.isBipedContactSet() ? posComRtFeetmeanEwrtWorldZ
                                              : posComRtLowcontactEwrtWorldZ);
}

TEST(TestDynamicsHelperFunctions, weightCompensatingInput) {
  CentroidalTestingModelInterface testingModelInterface =
      CentroidalTestingModelInterface();

  const PinocchioInterface& pinocchioInterface =
      testingModelInterface.getPinocchioInterface();
  CentroidalModelInfo centroidalModelInfo =
      testingModelInterface.getCentroidalModelInfo();

  const scalar_t robotWeight = centroidalModelInfo.robotMass * 9.81;

  vector_t inputDoubleContact =
      vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      inputDoubleContact, vector3_t(0, 0, robotWeight / 2),
      /*contactIndex=*/0);
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      inputDoubleContact, vector3_t(0, 0, robotWeight / 2),
      /*contactIndex=*/1);

  EXPECT_TRUE(inputDoubleContact.isApprox(
      weightCompensatingInput(centroidalModelInfo, {true, true},
                              testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(inputDoubleContact.isApprox(
      weightCompensatingInput(pinocchioInterface, {true, true},
                              testingModelInterface.getMpcRobotModel())));

  vector_t inputLeftContact =
      vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      inputLeftContact, vector3_t(0, 0, robotWeight),
      /*contactIndex=*/0);

  EXPECT_TRUE(inputLeftContact.isApprox(
      weightCompensatingInput(centroidalModelInfo, {true, false},
                              testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(inputLeftContact.isApprox(
      weightCompensatingInput(pinocchioInterface, {true, false},
                              testingModelInterface.getMpcRobotModel())));

  vector_t inputRightContact =
      vector_t::Zero(testingModelInterface.getMpcRobotModel().getInputDim());
  testingModelInterface.getMpcRobotModel().setForceContactEwrtParam(
      inputRightContact, vector3_t(0, 0, robotWeight),
      /*contactIndex=*/1);
  EXPECT_TRUE(inputRightContact.isApprox(
      weightCompensatingInput(centroidalModelInfo, {false, true},
                              testingModelInterface.getMpcRobotModel())));
  EXPECT_TRUE(inputRightContact.isApprox(
      weightCompensatingInput(pinocchioInterface, {false, true},
                              testingModelInterface.getMpcRobotModel())));

  EXPECT_TRUE(ocs2::getMnormRateCmomComRtWorldEwrtWorld(
                  pinocchioInterface, centroidalModelInfo, inputDoubleContact)
                  .isApprox(vector_t::Zero(6)));
}

TEST(TestDynamicsHelperFunctions,
     weightCompensatingInputStateAwareOverloadMatchesInputOnlyForWrenchModel) {
  CentroidalTestingModelInterface testingModelInterface =
      CentroidalTestingModelInterface();
  CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface.getMpcRobotModel();

  const PinocchioInterface& pinocchioInterface =
      testingModelInterface.getPinocchioInterface();
  CentroidalModelInfo centroidalModelInfo =
      testingModelInterface.getCentroidalModelInfo();

  // Fixture state: standing height, a yawed and slightly tilted base, and
  // random joint angles. For the wrench-space model the input stores
  // world-frame wrenches directly, so none of this may influence the
  // weight-compensating input.
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.3, -0.2, 0.86566));
  model.setEulerzyxBaseRtWorld(state,
                               vector3_t(std::numbers::pi / 2.0, 0.1, -0.05));
  model.setJointAngles(state, vector_t::Random(model.getJointDim()));

  const scalar_t robotWeight = centroidalModelInfo.robotMass * 9.81;

  for (const BipedFeetArray<bool>& flags :
       {BipedFeetArray<bool>{true, true}, BipedFeetArray<bool>{true, false},
        BipedFeetArray<bool>{false, true},
        BipedFeetArray<bool>{false, false}}) {
    const vector_t inputOnlyFromPinocchio =
        weightCompensatingInput(pinocchioInterface, flags, model);
    const vector_t stateAwareFromPinocchio =
        weightCompensatingInput(pinocchioInterface, flags, model, state);
    EXPECT_TRUE(
        (stateAwareFromPinocchio - inputOnlyFromPinocchio).isZero(1.0e-12))
        << "PinocchioInterface overloads differ for flags {" << flags[0] << ", "
        << flags[1] << "}";

    const vector_t inputOnlyFromInfo =
        weightCompensatingInput(centroidalModelInfo, flags, model);
    const vector_t stateAwareFromInfo =
        weightCompensatingInput(centroidalModelInfo, flags, model, state);
    EXPECT_TRUE((stateAwareFromInfo - inputOnlyFromInfo).isZero(1.0e-12))
        << "CentroidalModelInfo overloads differ for flags {" << flags[0]
        << ", " << flags[1] << "}";

    // Both families must agree with each other and produce vertical world-frame
    // forces that sum to the robot weight.
    EXPECT_TRUE((stateAwareFromInfo - stateAwareFromPinocchio).isZero(1.0e-9));
    vector3_t forceComEwrtWorld = vector3_t::Zero();
    for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
      const vector3_t forceContactEwrtWorld =
          model.getForceContactEwrtWorld(state, stateAwareFromPinocchio, i);
      EXPECT_TRUE((forceContactEwrtWorld -
                   model.getForceContactEwrtParam(stateAwareFromPinocchio, i))
                      .isZero(0.0));
      if (!flags[i]) {
        EXPECT_TRUE(forceContactEwrtWorld.isZero(0.0))
            << "Swing foot " << i << " must carry no force";
      }
      forceComEwrtWorld += forceContactEwrtWorld;
    }
    const scalar_t expectedTotal = (flags[0] || flags[1]) ? robotWeight : 0.0;
    EXPECT_TRUE(
        (forceComEwrtWorld - vector3_t(0.0, 0.0, expectedTotal)).isZero(1.0e-9))
        << "Total weight-compensating force should be vertical and equal to "
           "the robot weight, got "
        << forceComEwrtWorld.transpose();
  }
}

TEST(TestDynamicsHelperFunctions,
     weightCompensatingInputCarriesTheWeightOfTheModelItIsGiven) {
  // Two robots in one process, alternated: the weight used to be cached in a
  // function-static by whichever model reached weightCompensatingInput first,
  // and every later call carried that robot's weight.
  CentroidalTestingModelInterface g1;
  CentroidalTestingModelInterface atlas(
      CentroidalTestingModelInterface::Robot::kDrcAtlas);
  const std::array<CentroidalTestingModelInterface* absl_nonnull, 2> robots = {
      &g1, &atlas};
  const scalar_t g1Weight =
      9.81 * pinocchio::computeTotalMass(g1.getPinocchioInterface().getModel());
  const scalar_t atlasWeight =
      9.81 *
      pinocchio::computeTotalMass(atlas.getPinocchioInterface().getModel());
  ASSERT_GT(std::abs(atlasWeight - g1Weight), 100.0)
      << "positive control: the two robots differ in weight";

  for (int round = 0; round < 2; ++round) {
    for (CentroidalTestingModelInterface* absl_nonnull robot : robots) {
      CentroidalMpcRobotModel<scalar_t>& model = robot->getMpcRobotModel();
      const CentroidalModelInfo info = robot->getCentroidalModelInfo();
      const scalar_t weight =
          9.81 * pinocchio::computeTotalMass(
                     robot->getPinocchioInterface().getModel());
      EXPECT_NEAR(info.robotMass * 9.81, weight, 1.0e-9 * weight);
      vector_t state = model.getNeutralState();
      model.setEulerzyxBaseRtWorld(state, vector3_t(0.3, 0.0, 0.0));

      for (const BipedFeetArray<bool>& flags :
           {BipedFeetArray<bool>{true, true}, BipedFeetArray<bool>{true, false},
            BipedFeetArray<bool>{false, true}}) {
        SCOPED_TRACE(absl::StrCat(robot->taskFile, ", round ", round,
                                  ", contacts {", flags[0], ", ", flags[1],
                                  "}"));
        const std::array<vector_t, 4> inputs = {
            weightCompensatingInput(robot->getPinocchioInterface(), flags,
                                    model),
            weightCompensatingInput(robot->getPinocchioInterface(), flags,
                                    model, state),
            weightCompensatingInput(info, flags, model),
            weightCompensatingInput(info, flags, model, state)};
        for (const vector_t& input : inputs) {
          vector3_t forceComEwrtWorld = vector3_t::Zero();
          for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
            forceComEwrtWorld +=
                model.getForceContactEwrtWorld(state, input, i);
          }
          EXPECT_NEAR(forceComEwrtWorld.z(), weight, 1.0e-9 * weight);
          EXPECT_NEAR(forceComEwrtWorld.head<2>().norm(), 0.0, 1.0e-9 * weight);
        }
      }
    }
  }
}

}  // namespace ocs2::legged
