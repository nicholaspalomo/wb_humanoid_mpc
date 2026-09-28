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

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <ocs2_core/automatic_differentiation/Types.h>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

/**
 * The floating-base inverse dynamics the MPCs share (pinocchio_model/DynamicsHelperFunctions.h), checked against
 * Pinocchio's own recursive Newton-Euler algorithm rather than against a copy of their formulas:
 *
 *   - computeBaseAcceleration() solves the base rows of M qdd + nle = J^T F. With it, RNEA's base rows must equal the
 *     generalized contact force, and the solve must be exact for a base mass matrix that is not block diagonal.
 *   - It must be taped by CppAD without a single comparison between variables: CppADCodeGen cannot generate code for
 *     one, which is how the whole-body MPC stopped compiling its dynamics ("GreaterThanZero cannot be called for
 *     non-parameters").
 *
 * computeJointTorques() itself - the joint rows at this base acceleration, with the full mass matrix - is compared with
 * RNEA's joint rows in testJointTorqueInverseDynamics; the base-held torques of the gantry feedforward are
 * computeBaseHeldJointTorques().
 */
namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** A deterministic, well spread sequence in [-1, 1]. */
scalar_t spread(size_t i, scalar_t phase) {
  return std::sin(1.7 * static_cast<scalar_t>(i) + phase);
}

/**
 * A symmetric positive definite 6x6 base mass matrix with the structure of a floating base whose center of mass is
 * off the base origin: m I on the translation block and a non-zero coupling block, so that it is NOT block diagonal.
 */
matrix_t coupledBaseMassMatrix(size_t numJoints) {
  const scalar_t mass = 35.0;
  const vector3_t comOffset(0.04, -0.02, 0.21);
  matrix3_t skew;
  skew << 0.0, -comOffset.z(), comOffset.y(), comOffset.z(), 0.0, -comOffset.x(), -comOffset.y(), comOffset.x(), 0.0;
  matrix3_t inertiaAtCom;
  inertiaAtCom << 2.1, 0.05, -0.1, 0.05, 1.8, 0.02, -0.1, 0.02, 0.6;
  matrix_t M = matrix_t::Zero(6 + numJoints, 6 + numJoints);
  M.block<3, 3>(0, 0) = mass * matrix3_t::Identity();
  M.block<3, 3>(0, 3) = -mass * skew;
  M.block<3, 3>(3, 0) = mass * skew;
  M.block<3, 3>(3, 3) = inertiaAtCom - mass * skew * skew;
  for (size_t j = 0; j < numJoints; ++j) {
    for (Eigen::Index row = 0; row < 6; ++row) {
      M(row, 6 + static_cast<Eigen::Index>(j)) = 0.3 * spread(6 * j + static_cast<size_t>(row), /*phase=*/0.2);
    }
  }
  return M;
}

/**
 * Tapes f(M_bb, M_bj, nle_b, qdd_j, (J^T F)_b) = computeBaseAcceleration, with every input an independent variable, and
 * runs the operation sequence forward on CppADCodeGen variables - the step at which the code generator used to throw.
 * Returns the error message, empty when the sequence has no comparison between variables.
 */
std::string codeGenerationError(size_t numJoints, bool useLegacyPivotingInverse) {
  const Eigen::Index nj = static_cast<Eigen::Index>(numJoints);
  const Eigen::Index numInputs = 36 + 6 * nj + 6 + nj + 6;
  ad_vector_t x(numInputs);
  const matrix_t M = coupledBaseMassMatrix(numJoints);
  for (Eigen::Index i = 0; i < 36; ++i) x(i) = M(i % 6, i / 6);
  for (Eigen::Index i = 36; i < numInputs; ++i) x(i) = 0.1 * spread(static_cast<size_t>(i), /*phase=*/0.0);
  CppAD::Independent(x);
  ad_matrix_t adM = ad_matrix_t::Zero(6 + nj, 6 + nj);
  for (Eigen::Index i = 0; i < 36; ++i) adM(i % 6, i / 6) = x(i);
  for (Eigen::Index i = 0; i < 6 * nj; ++i) adM(i % 6, 6 + i / 6) = x(36 + i);
  const ad_vector_t nle = x.segment(36 + 6 * nj, 6);
  const ad_vector_t qddJoints = x.segment(36 + 6 * nj + 6, nj);
  ad_vector_t generalizedForce = ad_vector_t::Zero(6 + nj);
  generalizedForce.head(6) = x.tail(6);
  ad_vector_t y(6);
  if (useLegacyPivotingInverse) {
    // What computeBaseAcceleration did before: a pivoting inverse of the dynamic-size 6x6 block.
    const ad_matrix_t M_bb = adM.topLeftCorner(6, 6);
    const ad_matrix_t M_bj = adM.block(0, 6, 6, nj);
    const ad_vector_t intermediate = -nle - M_bj * qddJoints + generalizedForce.head(6);
    y = M_bb.inverse() * intermediate;
  } else {
    ad_vector_t fullNle = ad_vector_t::Zero(6 + nj);
    fullNle.head(6) = nle;
    y = computeBaseAcceleration<ad_scalar_t>(adM, fullNle, qddJoints, generalizedForce);
  }
  ad_fun_t fun(x, y);

  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(static_cast<size_t>(numInputs));
  handler.makeVariables(variables);
  try {
    fun.Forward(/*q=*/0, variables);
  } catch (const std::exception& error) {
    return error.what();
  }
  return std::string();
}

}  // namespace

TEST(FloatingBaseDynamicsTest, theBaseAccelerationSolvesACoupledBaseMassMatrixExactly) {
  constexpr size_t kNumJoints = 5;
  const matrix_t M = coupledBaseMassMatrix(kNumJoints);
  // The premise: the translational and rotational base coordinates are coupled, so a block-diagonal split is wrong.
  const scalar_t couplingNorm = M.block<3, 3>(0, 3).norm();
  ASSERT_GT(couplingNorm, 1.0);
  vector_t nle(6 + kNumJoints);
  vector_t qddJoints(kNumJoints);
  vector_t generalizedForce(6 + kNumJoints);
  for (size_t i = 0; i < 6 + kNumJoints; ++i) {
    nle(static_cast<Eigen::Index>(i)) = 20.0 * spread(i, /*phase=*/0.5);
    generalizedForce(static_cast<Eigen::Index>(i)) = 30.0 * spread(i, /*phase=*/1.1);
  }
  for (size_t i = 0; i < kNumJoints; ++i) qddJoints(static_cast<Eigen::Index>(i)) = 2.0 * spread(i, /*phase=*/2.3);

  const vector6_t baseAcceleration = computeBaseAcceleration<scalar_t>(M, nle, qddJoints, generalizedForce);
  const vector6_t residual =
      M.topLeftCorner(6, 6) * baseAcceleration + M.block(0, 6, 6, kNumJoints) * qddJoints + nle.head(6) - generalizedForce.head(6);
  EXPECT_LT(residual.norm(), 1e-9) << residual.transpose();
}

TEST(FloatingBaseDynamicsTest, theBaseAccelerationTapesWithoutAComparisonBetweenVariables) {
  // Positive control: the detector catches the pivoting 6x6 inverse the whole-body dynamics used to tape.
  const std::string legacy = codeGenerationError(/*numJoints=*/4, /*useLegacyPivotingInverse=*/true);
  ASSERT_FALSE(legacy.empty()) << "a pivoting 6x6 inverse was taped without a comparison; the check below proves nothing";
  EXPECT_NE(legacy.find("non-parameters"), std::string::npos) << legacy;

  EXPECT_EQ(codeGenerationError(/*numJoints=*/4, /*useLegacyPivotingInverse=*/false), "");
}

class FloatingBaseInverseDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty()) << "the G1 whole-body files are not in the runfiles";
    settings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *settings_));
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    q_ = vector_t(model.nq);
    v_ = vector_t(model.nv);
    for (Eigen::Index i = 0; i < model.nq; ++i) q_(i) = 0.3 * spread(static_cast<size_t>(i), /*phase=*/0.7);
    q_(2) = 0.75;  // the base height does not enter the dynamics, but keep the robot standing
    for (Eigen::Index i = 0; i < model.nv; ++i) v_(i) = 0.5 * spread(static_cast<size_t>(i), /*phase=*/1.9);
    qddJoints_ = vector_t(model.nv - 6);
    for (Eigen::Index i = 0; i < qddJoints_.size(); ++i) qddJoints_(i) = 1.5 * spread(static_cast<size_t>(i), /*phase=*/0.4);
    wrenches_ = {vector6_t(), vector6_t()};
    for (Eigen::Index i = 0; i < 6; ++i) {
      wrenches_[0](i) = (i == 2 ? 180.0 : 15.0) * spread(static_cast<size_t>(i), /*phase=*/0.1) + (i == 2 ? 200.0 : 0.0);
      wrenches_[1](i) = (i == 2 ? 180.0 : 15.0) * spread(static_cast<size_t>(i), /*phase=*/2.6) + (i == 2 ? 200.0 : 0.0);
    }
  }

  /** J_l^T F_l + J_r^T F_r with the LOCAL_WORLD_ALIGNED Jacobians of the sole frames, as computeJointTorques forms it. */
  vector_t generalizedContactForce() {
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    pinocchio::Data& data = pinocchioInterface_->getData();
    vector_t force = vector_t::Zero(model.nv);
    const std::array<std::string, 2> frames = {"foot_l_contact", "foot_r_contact"};
    for (size_t foot = 0; foot < frames.size(); ++foot) {
      matrix_t J = matrix_t::Zero(6, model.nv);
      pinocchio::computeFrameJacobian(model, data, q_, model.getFrameId(frames[foot]), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      force += J.transpose() * wrenches_[foot];
    }
    return force;
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  vector_t q_;
  vector_t v_;
  vector_t qddJoints_;
  std::array<vector6_t, 2> wrenches_;
};

TEST_F(FloatingBaseInverseDynamicsTest, rneaBalancesTheBaseAtTheSolvedAcceleration) {
  const pinocchio::Model& model = pinocchioInterface_->getModel();
  pinocchio::Data& data = pinocchioInterface_->getData();

  // The base rows of the crba() mass matrix, which is what the MPC's dynamics solve with: its root 6x6 block is filled
  // in full (the root joint's rows over its whole subtree), and its translational and rotational coordinates are
  // coupled on the real robot, so a block-diagonal split of it would fail the balance below.
  pinocchio::crba(model, data, q_);
  pinocchio::nonLinearEffects(model, data, q_, v_);
  const vector_t generalizedForce = generalizedContactForce();
  const vector6_t baseAcceleration = computeBaseAcceleration<scalar_t>(data.M, data.nle, qddJoints_, generalizedForce);
  vector_t qdd(model.nv);
  qdd << baseAcceleration, qddJoints_;
  const vector_t tau = pinocchio::rnea(model, data, q_, v_, qdd);

  // Positive control: the premise that the base block is coupled on the real model.
  const scalar_t couplingNorm = data.M.block<3, 3>(0, 3).norm();
  EXPECT_GT(couplingNorm, 1e-3);
  // The base is unactuated: at the solved acceleration the contact forces alone balance its rows.
  EXPECT_LT((tau.head(6) - generalizedForce.head(6)).norm(), 1e-6 * generalizedForce.head(6).norm())
      << "base rows: " << (tau.head(6) - generalizedForce.head(6)).transpose();
}

}  // namespace ocs2::humanoid
