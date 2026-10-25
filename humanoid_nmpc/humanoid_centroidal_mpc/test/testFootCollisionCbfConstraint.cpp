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

#include <array>
#include <memory>
#include <string>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/constraint/FootCollisionCbfConstraint.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/constraint/FootCollisionSpheres.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/DrcAtlasContactTestModel.h"
#include "support/FiniteDifferenceChecks.h"

/**
 * FootCollisionCbfConstraint on the DRC Atlas centroidal model, checked against
 * FootCollisionConstraint, whose rows are the distances |p_a - p_b| - 2 r of
 * the same pairs: at rest a row is gamma * h, h = |p_a - p_b|^2 - (2 r)^2; its
 * rate is the derivative of h along the joint motion of the input, also when
 * the base turns, which the row leaves out; and its Jacobians are those of its
 * value.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTime = DrcAtlasContactTestModel::kQueryTime;
// [s] The step of the central differences of h along a motion.
constexpr scalar_t kMotionStep = 1.0e-5;
// [rad/s] The angular velocity of the base, in its own frame, of
// ATurnOfTheBaseDoesNotChangeTheRate.
constexpr std::array<scalar_t, 3> kBaseAngularVelocity = {0.4, -0.3, 0.8};

class FootCollisionCbfConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const mpc_config::TaskFile& task = atlas_.config().task;
    spheres_ = checkedValue(
        footCollisionConstraintConfigFromConfig(task.collision_constraint));
    const FootCollisionCbfConstraint::Config cbfConfig =
        checkedValue(footCollisionCbfConstraintConfigFromConfig(
            task.collision_constraint, task.collision_cbf_constraint));
    gamma_ = cbfConfig.gamma;
    distances_ = std::make_unique<FootCollisionConstraint>(
        atlas_.referenceManager(), atlas_.pinocchioInterface(),
        atlas_.adWrenchModel(), spheres_, "testFootCollisionCbf_distances",
        atlas_.modelSettings());
    cbf_ = std::make_unique<FootCollisionCbfConstraint>(
        atlas_.referenceManager(), atlas_.pinocchioInterface(),
        atlas_.adWrenchModel(), cbfConfig, atlas_.modelSettings());
    state_ = atlas_.nominalState();
    input_ = atlas_.makeInput(atlas_.wrenchModel(), /*loadedFoot=*/0,
                              /*normalForce=*/400.0);
  }

  /** h of every pair at `state`, from the distances of FootCollisionConstraint.
   */
  vector_t barriers(const vector_t& state) const {
    const vector_t rows = distances_->getValue(kTime, state, PreComputation());
    vector_t h(rows.size());
    for (Eigen::Index row = 0; row < rows.size(); ++row) {
      const scalar_t radius =
          kFootCollisionPairs[static_cast<size_t>(row)].radius ==
                  FootCollisionRadius::kKnee
              ? spheres_.kneeCollisionSphereRadius
              : spheres_.footCollisionSphereRadius;
      const scalar_t distance = rows[row] + 2.0 * radius;
      h[row] = distance * distance - 4.0 * radius * radius;
    }
    return h;
  }

  /** `state` with the joints moved for `duration` at the joint velocities of
   * input_. */
  vector_t jointsMoved(const vector_t& state, scalar_t duration) const {
    const CentroidalMpcRobotModel<scalar_t>& model = atlas_.wrenchModel();
    vector_t moved = state;
    model.setJointAngles(
        moved, model.getJointAngles(state) +
                   duration * model.getJointVelocities(state, input_));
    return moved;
  }

  /** jointsMoved(), with the base turned for `duration` at kBaseAngularVelocity
   * as well. */
  vector_t turnedAndMoved(const vector_t& state, scalar_t duration) const {
    const CentroidalMpcRobotModel<scalar_t>& model = atlas_.wrenchModel();
    vector_t moved = jointsMoved(state, duration);
    const vector3_t angularVelocity(kBaseAngularVelocity.data());
    const vector4_t coefficients = model.getBaseOrientation(state);
    const Eigen::Quaterniond orientation(
        Eigen::Map<const Eigen::Quaterniond>(coefficients.data()));
    const Eigen::Quaterniond turned =
        orientation *
        Eigen::Quaterniond(Eigen::AngleAxisd(duration * angularVelocity.norm(),
                                             angularVelocity.normalized()));
    model.setBaseOrientation(moved, vector4_t(turned.coeffs()));
    return moved;
  }

  DrcAtlasContactTestModel atlas_{"testFootCollisionCbfConstraint_"};
  FootCollisionSpheres spheres_;
  scalar_t gamma_ = 0.0;
  std::unique_ptr<FootCollisionConstraint> distances_;
  std::unique_ptr<FootCollisionCbfConstraint> cbf_;
  vector_t state_;
  vector_t input_;
};

TEST_F(FootCollisionCbfConstraintTest, AtRestARowIsGammaTimesTheSquaredMargin) {
  ASSERT_GT(gamma_, 0.0)
      << "the shipped Atlas sets collision_cbf_constraint.gamma";
  vector_t atRest = input_;
  atRest.tail(atlas_.wrenchModel().getJointDim()).setZero();
  const vector_t rows = cbf_->getValue(kTime, state_, atRest, PreComputation());
  ASSERT_EQ(rows.size(), static_cast<Eigen::Index>(kNumFootCollisionPairs));
  const vector_t expected = gamma_ * barriers(state_);
  for (Eigen::Index row = 0; row < rows.size(); ++row) {
    EXPECT_NEAR(rows[row], expected[row], 1.0e-9) << "row " << row;
  }
}

TEST_F(FootCollisionCbfConstraintTest,
       TheRateIsTheDerivativeOfTheSquaredMarginAlongTheJointMotion) {
  vector_t atRest = input_;
  atRest.tail(atlas_.wrenchModel().getJointDim()).setZero();
  const vector_t rate =
      cbf_->getValue(kTime, state_, input_, PreComputation()) -
      cbf_->getValue(kTime, state_, atRest, PreComputation());
  const vector_t numerical = (barriers(jointsMoved(state_, kMotionStep)) -
                              barriers(jointsMoved(state_, -kMotionStep))) /
                             (2.0 * kMotionStep);
  ASSERT_GT(numerical.cwiseAbs().maxCoeff(), 1.0e-3)
      << "the joint motion moves no pair, so the check proves nothing";
  for (Eigen::Index row = 0; row < rate.size(); ++row) {
    EXPECT_NEAR(rate[row], numerical[row], 1.0e-6) << "row " << row;
  }
}

TEST_F(FootCollisionCbfConstraintTest, ATurnOfTheBaseDoesNotChangeTheRate) {
  // The row takes the base at rest; turning the whole robot while its joints
  // move changes no relative distance faster.
  const CentroidalMpcRobotModel<scalar_t>& model = atlas_.wrenchModel();
  vector_t atRest = input_;
  atRest.tail(model.getJointDim()).setZero();
  const vector_t rate =
      cbf_->getValue(kTime, state_, input_, PreComputation()) -
      cbf_->getValue(kTime, state_, atRest, PreComputation());
  const vector_t numerical = (barriers(turnedAndMoved(state_, kMotionStep)) -
                              barriers(turnedAndMoved(state_, -kMotionStep))) /
                             (2.0 * kMotionStep);
  for (Eigen::Index row = 0; row < rate.size(); ++row) {
    EXPECT_NEAR(rate[row], numerical[row], 1.0e-6) << "row " << row;
  }
}

TEST_F(FootCollisionCbfConstraintTest, TheJacobiansAreThoseOfTheValue) {
  expectStateInputDerivativesMatchFiniteDifferences(*cbf_, state_, input_);
}

TEST_F(FootCollisionCbfConstraintTest, GammaScalesTheSquaredMargin) {
  const vector_t before =
      cbf_->getValue(kTime, state_, input_, PreComputation());
  cbf_->setGamma(2.0 * gamma_);
  EXPECT_DOUBLE_EQ(cbf_->getGamma(), 2.0 * gamma_);
  const vector_t after =
      cbf_->getValue(kTime, state_, input_, PreComputation());
  const vector_t expected = before + gamma_ * barriers(state_);
  for (Eigen::Index row = 0; row < after.size(); ++row) {
    EXPECT_NEAR(after[row], expected[row], 1.0e-9) << "row " << row;
  }
  // A clone carries the rate it was cloned with.
  const std::unique_ptr<FootCollisionCbfConstraint> clone(cbf_->clone());
  EXPECT_DOUBLE_EQ(clone->getGamma(), 2.0 * gamma_);
}

TEST_F(FootCollisionCbfConstraintTest, IsActiveOnlyWhileAFootSwings) {
  atlas_.setStance();
  EXPECT_FALSE(cbf_->isActive(kTime));
  atlas_.setSwing(/*swingFoot=*/0);
  EXPECT_TRUE(cbf_->isActive(kTime));
  atlas_.setSwing(/*swingFoot=*/1);
  EXPECT_TRUE(cbf_->isActive(kTime));
}

}  // namespace
}  // namespace ocs2::humanoid
