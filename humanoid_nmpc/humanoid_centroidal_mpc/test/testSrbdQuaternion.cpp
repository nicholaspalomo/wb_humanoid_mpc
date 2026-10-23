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
#include <cmath>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * The single rigid body dynamics of the centroidal model on the quaternion root (design README sections 2.5 and 5(c)):
 * its base block of the centroidal momentum matrix, A_b = [[m I, m [R r_nom]x R], [0, R I_nom]] in the base velocity
 * v_b = [dp_W, w_B], is the full centroidal dynamics' A_b at the nominal posture (the joints at the nominal joint angles
 * that r_nom and I_nom are computed at), at every attitude: level, turned, tilted, at a pitch of exactly 90 degrees and
 * with the negated quaternion; its joint columns are zero; and its CoM is the full model's. Away from the nominal
 * posture the two differ (the positive control). On every shipped configuration, in double precision; builds no CppAD
 * model.
 */

namespace ocs2::humanoid {
namespace {

/** The attitudes the models are compared at, (x, y, z, w). */
std::vector<vector4_t> attitudes() {
  std::vector<vector4_t> xis = {vector4_t(0.0, 0.0, 0.0, 1.0),
                                quaternionFromEulerZyx(vector3_t(2.0, 0.0, 0.0)),
                                quaternionFromEulerZyx(vector3_t(-2.7, 0.3, -0.2)),
                                quaternionFromEulerZyx(vector3_t(0.4, M_PI / 2.0, 0.1)),
                                quaternionFromEulerZyx(vector3_t(1.1, -M_PI / 2.0, -0.6)),
                                vector4_t(0.5, -0.5, 0.5, 0.5)};
  xis.push_back(-xis[2]);
  return xis;
}

/** The base block A_b and the joint columns A_j of the momentum matrix of `info` at `q`, and the CoM, on `interface`. */
struct MomentumMap {
  matrix_t baseBlock;
  matrix_t jointColumns;
  vector3_t com;
};

MomentumMap momentumMapOf(PinocchioInterface& interface, const CentroidalModelInfo& info, const vector_t& q) {
  updateCentroidalDynamics<scalar_t>(interface, info, q);
  const Eigen::Matrix<scalar_t, 6, Eigen::Dynamic>& momentumMatrix = getCentroidalMomentumMatrix<scalar_t>(interface);
  return MomentumMap{.baseBlock = momentumMatrix.leftCols<6>(),
                     .jointColumns = momentumMatrix.rightCols(momentumMatrix.cols() - 6),
                     .com = interface.getData().com[0]};
}

class SrbdQuaternion : public ::testing::TestWithParam<ShippedCentroidalRobot> {};

TEST_P(SrbdQuaternion, TheBaseBlockOfTheMomentumMapIsTheFullModelsAtTheNominalPostureAtEveryAttitude) {
  absl::StatusOr<LoadedCentroidalRobot> full = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"full_centroidal_dynamics");
  absl::StatusOr<LoadedCentroidalRobot> srbd = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"single_rigid_body_dynamics");
  ASSERT_TRUE(full.ok()) << full.status();
  ASSERT_TRUE(srbd.ok()) << srbd.status();
  ASSERT_EQ(full->info->centroidalModelType, CentroidalModelType::FullCentroidalDynamics);
  ASSERT_EQ(srbd->info->centroidalModelType, CentroidalModelType::SingleRigidBodyDynamics);
  const CentroidalModelInfo& srbdInfo = *srbd->info;
  const Eigen::Index joints = static_cast<Eigen::Index>(srbdInfo.actuatedDofNum);
  ASSERT_EQ(static_cast<size_t>(srbdInfo.qPinocchioNominal.size()), srbdInfo.configurationDim);
  // The nominal posture r_nom and I_nom are computed at: the base at the origin, level, the joints at their nominal angles.
  EXPECT_EQ(vector4_t(srbdInfo.qPinocchioNominal.segment<4>(3)), vector4_t(0.0, 0.0, 0.0, 1.0));
  const vector_t nominalJoints = srbdInfo.qPinocchioNominal.tail(joints);

  for (const vector4_t& xi : attitudes()) {
    SCOPED_TRACE(xi.transpose());
    vector_t q(srbdInfo.configurationDim);
    q << vector3_t(0.3, -1.2, 0.85), xi, nominalJoints;
    const MomentumMap fullMap = momentumMapOf(*full->pinocchioInterface, *full->info, q);
    const MomentumMap srbdMap = momentumMapOf(*srbd->pinocchioInterface, srbdInfo, q);
    const scalar_t scale = std::max(1.0, fullMap.baseBlock.cwiseAbs().maxCoeff());
    EXPECT_LT((srbdMap.baseBlock - fullMap.baseBlock).cwiseAbs().maxCoeff(), 1.0e-12 * scale) << "A_b:\n"
                                                                                              << srbdMap.baseBlock << "\nfull:\n"
                                                                                              << fullMap.baseBlock;
    EXPECT_EQ(srbdMap.jointColumns, matrix_t::Zero(6, joints)) << "the single rigid body has no joint columns";
    EXPECT_LT((srbdMap.com - fullMap.com).cwiseAbs().maxCoeff(), 1.0e-12);
    // The closed form of the port: A_b = [[m I, m [R r_nom]x R], [0, R I_nom]].
    const matrix3_t rotation = quaternion_t(xi).normalized().toRotationMatrix();
    const vector3_t offset = rotation * srbdInfo.comToBasePositionNominal;
    matrix3_t skew;
    skew << 0.0, -offset.z(), offset.y(), offset.z(), 0.0, -offset.x(), -offset.y(), offset.x(), 0.0;
    matrix_t closedForm = matrix_t::Zero(6, 6);
    closedForm.topLeftCorner<3, 3>() = srbdInfo.robotMass * matrix3_t::Identity();
    closedForm.topRightCorner<3, 3>() = srbdInfo.robotMass * skew * rotation;
    closedForm.bottomRightCorner<3, 3>() = rotation * srbdInfo.centroidalInertiaNominal;
    EXPECT_LT((srbdMap.baseBlock - closedForm).cwiseAbs().maxCoeff(), 1.0e-12 * scale);
    // Regular at a pitch of 90 degrees, where the Euler root's I_G T_W(Theta) is singular: R I_nom has I_nom's
    // condition number at every attitude.
    const Eigen::JacobiSVD<matrix3_t> angular(srbdMap.baseBlock.bottomRightCorner<3, 3>());
    const Eigen::JacobiSVD<matrix3_t> nominal(srbdInfo.centroidalInertiaNominal);
    const scalar_t angularCondition = angular.singularValues()(0) / angular.singularValues()(2);
    const scalar_t nominalCondition = nominal.singularValues()(0) / nominal.singularValues()(2);
    EXPECT_NEAR(angularCondition, nominalCondition, 1.0e-9 * nominalCondition);
  }
}

TEST_P(SrbdQuaternion, AwayFromTheNominalPostureTheFullModelDiffers) {
  absl::StatusOr<LoadedCentroidalRobot> full = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"full_centroidal_dynamics");
  absl::StatusOr<LoadedCentroidalRobot> srbd = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"single_rigid_body_dynamics");
  ASSERT_TRUE(full.ok()) << full.status();
  ASSERT_TRUE(srbd.ok()) << srbd.status();
  const CentroidalModelInfo& srbdInfo = *srbd->info;
  const Eigen::Index joints = static_cast<Eigen::Index>(srbdInfo.actuatedDofNum);
  vector_t q = srbdInfo.qPinocchioNominal;
  q.tail(joints) += vector_t::Constant(joints, 0.3);
  const MomentumMap fullMap = momentumMapOf(*full->pinocchioInterface, *full->info, q);
  const MomentumMap srbdMap = momentumMapOf(*srbd->pinocchioInterface, srbdInfo, q);
  EXPECT_GT((srbdMap.baseBlock - fullMap.baseBlock).cwiseAbs().maxCoeff(), 1.0e-3) << "the bent joints move the full model's inertia";
  EXPECT_GT(fullMap.jointColumns.cwiseAbs().maxCoeff(), 1.0e-3) << "and its joints carry momentum";
}

/**
 * The centroidal flow map f(x, u) = [hbar_dot; d/dt (p_W, xi, q_j)] of `info` on `interface`: the function
 * PinocchioCentroidalDynamicsAD tapes (getValueCppAd()), evaluated in double precision.
 */
vector_t centroidalFlowOf(PinocchioInterface& interface, const CentroidalModelInfo& info, const vector_t& state, const vector_t& input) {
  CentroidalModelPinocchioMapping mapping(info);
  mapping.setPinocchioInterface(interface);
  updateCentroidalDynamics<scalar_t>(interface, info, mapping.getPinocchioJointPosition(state));
  vector_t flow(info.stateDim);
  centroidal_model::getNormalizedMomentum(flow, info) = getNormalizedCentroidalMomentumRate<scalar_t>(interface, info, input);
  centroidal_model::configurationDerivative(flow, info) =
      computeConfigurationDerivative<scalar_t>(info, state, mapping.getPinocchioJointVelocity(state, input));
  return flow;
}

/** d f / d x by central differences of step 1e-4 (each ambient coordinate, the quaternion's included). */
matrix_t centroidalFlowStateJacobian(PinocchioInterface& interface,
                                     const CentroidalModelInfo& info,
                                     const vector_t& state,
                                     const vector_t& input) {
  constexpr scalar_t kStep = 1.0e-4;
  matrix_t jacobian(state.size(), state.size());
  for (Eigen::Index column = 0; column < state.size(); ++column) {
    vector_t up = state;
    vector_t down = state;
    up(column) += kStep;
    down(column) -= kStep;
    jacobian.col(column) = (centroidalFlowOf(interface, info, up, input) - centroidalFlowOf(interface, info, down, input)) / (2.0 * kStep);
  }
  return jacobian;
}

TEST_P(SrbdQuaternion, TheFlowMapIsTheFullModelsAtTheNominalPostureWithStillJointsAtEveryAttitude) {
  // The flow map of the single rigid body (selectable as single_rigid_body_dynamics; no shipped robot lists it) against
  // the full centroidal dynamics: at the nominal posture with the joints still (qd_j = 0), the momentum rate (O1) reads
  // the same CoM and contact positions, and the base velocity A_b^-1 m hbar the same A_b (the test above), so the flow is
  // the same at every attitude, a pitch of +-90 degrees and the negated quaternion included; so are its derivatives in
  // the momentum, the base position and the quaternion, where both A_b agree at every attitude. Its joint columns are
  // the single rigid body's own: the full model's inertia moves with the joints (the positive control).
  absl::StatusOr<LoadedCentroidalRobot> full = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"full_centroidal_dynamics");
  absl::StatusOr<LoadedCentroidalRobot> srbd = loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"single_rigid_body_dynamics");
  ASSERT_TRUE(full.ok()) << full.status();
  ASSERT_TRUE(srbd.ok()) << srbd.status();
  const CentroidalModelInfo& srbdInfo = *srbd->info;
  const Eigen::Index joints = static_cast<Eigen::Index>(srbdInfo.actuatedDofNum);
  const Eigen::Index stateDim = static_cast<Eigen::Index>(srbdInfo.stateDim);
  const Eigen::Index baseColumns = stateDim - joints;  // hbar (6), p_W (3), xi (4)
  std::mt19937 generator(/*sd=*/2026);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (const vector4_t& xi : attitudes()) {
    SCOPED_TRACE(xi.transpose());
    vector_t state(stateDim);
    state << 0.05 * vector3_t(uniform(generator), uniform(generator), uniform(generator)),
        0.02 * vector3_t(uniform(generator), uniform(generator), uniform(generator)), vector3_t(0.3, -1.2, 0.85), xi,
        srbdInfo.qPinocchioNominal.tail(joints);
    vector_t input = vector_t::Zero(static_cast<Eigen::Index>(srbdInfo.inputDim));
    for (Eigen::Index i = 0; i < 12; ++i) input(i) = 50.0 * uniform(generator);
    input(2) += 0.5 * srbdInfo.robotMass * 9.81;
    input(8) += 0.5 * srbdInfo.robotMass * 9.81;
    const vector_t fullFlow = centroidalFlowOf(*full->pinocchioInterface, *full->info, state, input);
    const vector_t srbdFlow = centroidalFlowOf(*srbd->pinocchioInterface, srbdInfo, state, input);
    ASSERT_TRUE(fullFlow.allFinite());
    const scalar_t scale = std::max(1.0, fullFlow.cwiseAbs().maxCoeff());
    EXPECT_LT((srbdFlow - fullFlow).cwiseAbs().maxCoeff(), 1.0e-12 * scale) << "f:\n"
                                                                            << srbdFlow.transpose() << "\nfull:\n"
                                                                            << fullFlow.transpose();

    const matrix_t fullJacobian = centroidalFlowStateJacobian(*full->pinocchioInterface, *full->info, state, input);
    const matrix_t srbdJacobian = centroidalFlowStateJacobian(*srbd->pinocchioInterface, srbdInfo, state, input);
    // Central differences of two flows equal to 1e-12 relative, at a step of 1e-4: 1e-8 of the flow's scale.
    EXPECT_LT((srbdJacobian.leftCols(baseColumns) - fullJacobian.leftCols(baseColumns)).cwiseAbs().maxCoeff(), 1.0e-7 * scale)
        << "df/d(hbar, p_W, xi)";
    EXPECT_GT((srbdJacobian.rightCols(joints) - fullJacobian.rightCols(joints)).cwiseAbs().maxCoeff(), 1.0e-6)
        << "the full model's joint columns are its own";
  }
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots,
                         SrbdQuaternion,
                         ::testing::Values(ShippedCentroidalRobot::kDrcAtlas,
                                           ShippedCentroidalRobot::kEngineAiSa01,
                                           ShippedCentroidalRobot::kUnitreeG1,
                                           ShippedCentroidalRobot::kUnitreeR1),
                         shippedCentroidalRobotTestName);

}  // namespace
}  // namespace ocs2::humanoid
