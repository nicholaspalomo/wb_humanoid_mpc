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

#include <cmath>
#include <functional>

#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/Types.h"

/*
 * The centroidal MPC's state and input against the whole-body kinematics the controllers run on (Pinocchio on the same
 * model), which is what the whole-body controller needs them to agree on. These are the checks of the former
 * humanoid_centroidal_mpc/test/testCentroidalConversions.cpp - never built, and pinned to the numbers of a 40-dimensional
 * robot that is not in this repository - as properties of the shipped G1 model.
 */

namespace ocs2::humanoid {
namespace {

/** A deterministic, well spread sequence in [-1, 1]. */
scalar_t spread(size_t i, scalar_t phase) {
  return std::sin(1.7 * static_cast<scalar_t>(i) + phase);
}

/** A walking-like state: momentum, a displaced and tilted base, and every joint away from zero. */
vector_t makeState(const CentroidalModelInfo& info) {
  vector_t state = vector_t::Zero(info.stateDim);
  centroidal_model::getNormalizedMomentum(state, info) << 0.4, -0.1, 0.05, 0.03, -0.02, 0.08;
  state.segment<6>(6) << 0.1, -0.2, 0.75, 0.4, 0.1, -0.05;
  for (size_t i = 0; i < info.actuatedDofNum; ++i) state(12 + static_cast<Eigen::Index>(i)) = 0.3 * spread(i, /*phase=*/0.7);
  return state;
}

/** Off-center contact wrenches on both feet, and every joint moving. */
vector_t makeInput(const CentroidalModelInfo& info) {
  vector_t input = vector_t::Zero(info.inputDim);
  for (size_t i = 0; i < info.numThreeDofContacts + info.numSixDofContacts; ++i) {
    const scalar_t sign = i % 2 == 0 ? 1.0 : -1.0;
    centroidal_model::getContactForces(input, i, info) << 20.0 * sign, -10.0, 200.0 + 30.0 * sign;
    if (i >= info.numThreeDofContacts) centroidal_model::getContactTorques(input, i, info) << 1.0, -2.0 * sign, 0.5;
  }
  vector_t jointVelocities(info.actuatedDofNum);
  for (Eigen::Index i = 0; i < jointVelocities.size(); ++i) jointVelocities(i) = 0.5 * spread(static_cast<size_t>(i), /*phase=*/1.9);
  centroidal_model::getJointVelocities(input, info) = jointVelocities;
  return input;
}

TEST(TestCentroidalConversions, theStateAndInputDimensionsFollowTheModel) {
  CentroidalTestingModelInterface testingModelInterface;
  const CentroidalModelInfo info = testingModelInterface.getCentroidalModelInfo();
  const pinocchio::Model& model = testingModelInterface.getPinocchioInterface().getModel();
  EXPECT_EQ(static_cast<Eigen::Index>(info.actuatedDofNum), model.nv - 6);
  // [normalized centroidal momentum (6), base pose (6), joint angles]
  EXPECT_EQ(info.stateDim, 12 + info.actuatedDofNum);
  // [contact forces (3 per point contact), contact wrenches (6 per foot), joint velocities]
  EXPECT_EQ(info.inputDim, 3 * info.numThreeDofContacts + 6 * info.numSixDofContacts + info.actuatedDofNum);
  EXPECT_EQ(testingModelInterface.getMpcRobotModel().getStateDim(), info.stateDim);
  EXPECT_EQ(testingModelInterface.getMpcRobotModel().getInputDim(), info.inputDim);
}

TEST(TestCentroidalConversions, theBaseTwistOfTheMappingCarriesTheMomentumOfTheState) {
  CentroidalTestingModelInterface testingModelInterface;
  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const pinocchio::Model& model = pinocchioInterface.getModel();
  const CentroidalModelInfo info = testingModelInterface.getCentroidalModelInfo();
  CentroidalModelPinocchioMapping mapping(info);
  mapping.setPinocchioInterface(pinocchioInterface);

  const vector_t state = makeState(info);
  const vector_t input = makeInput(info);
  const vector_t q = mapping.getPinocchioJointPosition(state);
  updateCentroidalDynamics(pinocchioInterface, info, q);
  const vector_t v = mapping.getPinocchioJointVelocity(state, input);

  // Pinocchio's centroidal momentum of the whole body at (q, v) is the momentum the state carries.
  const vector6_t momentum = info.robotMass * centroidal_model::getNormalizedMomentum(state, info);
  pinocchio::Data data(model);
  const vector6_t wholeBodyMomentum = pinocchio::computeCentroidalMomentum(model, data, q, v).toVector();
  EXPECT_LT((wholeBodyMomentum - momentum).norm(), 1.0e-9 * momentum.norm())
      << "state: " << momentum.transpose() << "\nwhole body: " << wholeBodyMomentum.transpose();
  EXPECT_TRUE(v.tail(info.actuatedDofNum) == centroidal_model::getJointVelocities(input, info)) << "the joints move as the input says";

  // Positive control: the base twist is what reconciles the joint motion with the momentum.
  vector_t withoutBaseTwist = v;
  withoutBaseTwist.head<6>().setZero();
  const vector6_t jointsOnly = pinocchio::computeCentroidalMomentum(model, data, q, withoutBaseTwist).toVector();
  EXPECT_GT((jointsOnly - momentum).norm(), 1.0e-2 * momentum.norm());
}

TEST(TestCentroidalConversions, theMomentumRateIsTheNewtonEulerSumOfTheContactWrenchesAndGravity) {
  CentroidalTestingModelInterface testingModelInterface;
  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const pinocchio::Model& model = pinocchioInterface.getModel();
  const CentroidalModelInfo info = testingModelInterface.getCentroidalModelInfo();
  CentroidalModelPinocchioMapping mapping(info);
  mapping.setPinocchioInterface(pinocchioInterface);

  const vector_t q = mapping.getPinocchioJointPosition(makeState(info));
  updateCentroidalDynamics(pinocchioInterface, info, q);

  // The center of mass and the contact points from a separate Pinocchio data.
  pinocchio::Data data(model);
  const vector3_t com = pinocchio::centerOfMass(model, data, q);
  pinocchio::framesForwardKinematics(model, data, q);
  const std::function<vector6_t(const vector_t&)> expectedRate = [&](const vector_t& input) {
    vector6_t rate;
    rate << vector3_t(0.0, 0.0, -9.81 * info.robotMass), vector3_t::Zero();
    for (size_t i = 0; i < info.numThreeDofContacts + info.numSixDofContacts; ++i) {
      const vector3_t force = centroidal_model::getContactForces(input, i, info);
      rate.head<3>() += force;
      rate.tail<3>() += (data.oMf[info.endEffectorFrameIndices[i]].translation() - com).cross(force);
      if (i >= info.numThreeDofContacts) rate.tail<3>() += centroidal_model::getContactTorques(input, i, info);
    }
    return rate;
  };

  const vector_t input = makeInput(info);
  const vector6_t rate = info.robotMass * getNormalizedCentroidalMomentumRate(pinocchioInterface, info, input);
  const vector6_t expected = expectedRate(input);
  EXPECT_LT((rate - expected).norm(), 1.0e-9 * expected.norm())
      << "rate: " << rate.transpose() << "\nNewton-Euler: " << expected.transpose();

  // Positive control: the moments depend on where each foot is, so exchanging the feet's forces changes the rate.
  vector_t exchanged = input;
  centroidal_model::getContactForces(exchanged, /*contactIndex=*/0, info) =
      centroidal_model::getContactForces(input, /*contactIndex=*/1, info);
  centroidal_model::getContactForces(exchanged, /*contactIndex=*/1, info) =
      centroidal_model::getContactForces(input, /*contactIndex=*/0, info);
  EXPECT_GT((info.robotMass * getNormalizedCentroidalMomentumRate(pinocchioInterface, info, exchanged) - rate).norm(), 1.0);
}

}  // namespace
}  // namespace ocs2::humanoid
