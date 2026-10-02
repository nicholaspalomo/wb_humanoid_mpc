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

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

/**
 * PinocchioEndEffectorKinematics on a model whose configuration is larger than its velocity: an unbounded revolute
 * joint stores (cos, sin) of its angle, so nq = 3 and nv = 2 here, as a quaternion base makes nq = nv + 1. Pinocchio's
 * frame Jacobians have nv columns and refuse a matrix of any other width; the position and orientation-error
 * linearizations used to hand them a 6 x nq one, which was only right while nq = nv.
 */
namespace ocs2 {
namespace {

using quaternion_t = Eigen::Quaternion<scalar_t>;
using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;

constexpr char kTipName[] = "tip";

/** A wheel about z with an unbounded angle, carrying an elbow about y, carrying the tip frame. */
pinocchio::Model makeWheelArm() {
  pinocchio::Model model;
  const pinocchio::JointIndex wheel =
      model.addJoint(/*parent=*/0, pinocchio::JointModelRUBZ(), pinocchio::SE3::Identity(), /*joint_name=*/"wheel");
  const pinocchio::SE3 elbowPlacement(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.3, 0.1, 0.0));
  const pinocchio::JointIndex elbow = model.addJoint(wheel, pinocchio::JointModelRY(), elbowPlacement, /*joint_name=*/"elbow");
  const pinocchio::SE3 tipPlacement(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()).toRotationMatrix(), Eigen::Vector3d(0.2, -0.05, 0.1));
  // The frame tree a URDF parser builds: a frame per joint, the body frame under its joint's.
  model.addJointFrame(wheel);
  model.addJointFrame(elbow);
  model.addBodyFrame(kTipName, elbow, tipPlacement);
  return model;
}

/**
 * The state is the two joint angles (wheel, elbow), a chart of the configuration whose coordinates are the velocity
 * coordinates themselves: Pinocchio's v-space Jacobians are then the state Jacobians, unchanged.
 */
class WheelArmMapping final : public PinocchioStateInputMapping<scalar_t> {
 public:
  WheelArmMapping() = default;
  ~WheelArmMapping() override = default;
  WheelArmMapping* clone() const override { return new WheelArmMapping(*this); }

  vector_t getPinocchioJointPosition(const vector_t& state) const override {
    vector_t q(3);
    q << std::cos(state(0)), std::sin(state(0)), state(1);
    return q;
  }

  vector_t getPinocchioJointVelocity(const vector_t& /*state*/, const vector_t& input) const override { return input; }

  std::pair<matrix_t, matrix_t> getOcs2Jacobian(const vector_t& /*state*/, const matrix_t& Jq, const matrix_t& Jv) const override {
    return {Jq, Jv};
  }

 private:
  WheelArmMapping(const WheelArmMapping& rhs) = default;
};

class EndEffectorKinematicsTangentJacobiansTest : public ::testing::Test {
 protected:
  EndEffectorKinematicsTangentJacobiansTest()
      : pinocchioInterface_(makeWheelArm()), kinematics_(pinocchioInterface_, mapping_, {kTipName}), state_(2) {
    state_ << 0.7, -0.4;
  }

  /** Puts the kinematics at `state`, with the frame placements and the joint Jacobians the linearizations read. */
  void updateAt(const vector_t& state) {
    const pinocchio::Model& model = pinocchioInterface_.getModel();
    pinocchio::Data& data = pinocchioInterface_.getData();
    const vector_t q = mapping_.getPinocchioJointPosition(state);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    kinematics_.setPinocchioInterface(pinocchioInterface_);
  }

  /** Central differences of `function` in the state, at state_. Leaves the kinematics at state_. */
  matrix_t centralDifferences(const std::function<vector_t(const vector_t&)>& function) {
    const scalar_t step = 1e-6;
    matrix_t jacobian(3, state_.size());
    for (Eigen::Index i = 0; i < state_.size(); ++i) {
      vector_t plus = state_;
      vector_t minus = state_;
      plus(i) += step;
      minus(i) -= step;
      updateAt(plus);
      const vector_t valuePlus = function(plus);
      updateAt(minus);
      const vector_t valueMinus = function(minus);
      jacobian.col(i) = (valuePlus - valueMinus) / (2.0 * step);
    }
    updateAt(state_);
    return jacobian;
  }

  WheelArmMapping mapping_;
  PinocchioInterface pinocchioInterface_;
  PinocchioEndEffectorKinematics kinematics_;
  vector_t state_;
};

TEST_F(EndEffectorKinematicsTangentJacobiansTest, TheModelHasALargerConfigurationThanVelocity) {
  // The case under test, and the reason a 6 x nq frame Jacobian cannot be filled: Pinocchio refuses it.
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  ASSERT_EQ(model.nq, 3);
  ASSERT_EQ(model.nv, 2);
  updateAt(state_);
  pinocchio::Data data(pinocchioInterface_.getData());
  matrix_t configurationSized = matrix_t::Zero(6, model.nq);
  EXPECT_THROW(pinocchio::getFrameJacobian(model, data, model.getBodyId(kTipName), pinocchio::LOCAL_WORLD_ALIGNED, configurationSized),
               std::invalid_argument);
}

TEST_F(EndEffectorKinematicsTangentJacobiansTest, ThePositionLinearizationIsTheDerivativeOfThePosition) {
  updateAt(state_);
  const VectorFunctionLinearApproximation position = kinematics_.getPositionLinearApproximation(state_).front();
  EXPECT_TRUE(position.f.isApprox(kinematics_.getPosition(state_).front()));
  ASSERT_EQ(position.dfdx.rows(), 3);
  ASSERT_EQ(position.dfdx.cols(), state_.size());
  const matrix_t expected = centralDifferences([this](const vector_t& state) { return vector_t(kinematics_.getPosition(state).front()); });
  EXPECT_TRUE(position.dfdx.isApprox(expected, /*prec=*/1e-7)) << position.dfdx << "\nvs\n" << expected;
}

TEST_F(EndEffectorKinematicsTangentJacobiansTest, TheOrientationErrorLinearizationIsTheDerivativeOfTheError) {
  const quaternion_t reference(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, -0.5).normalized()));
  updateAt(state_);
  const VectorFunctionLinearApproximation error = kinematics_.getOrientationErrorLinearApproximation(state_, {reference}).front();
  EXPECT_TRUE(error.f.isApprox(kinematics_.getOrientationError(state_, {reference}).front()));
  ASSERT_EQ(error.dfdx.cols(), state_.size());
  const matrix_t expected = centralDifferences(
      [this, &reference](const vector_t& state) { return vector_t(kinematics_.getOrientationError(state, {reference}).front()); });
  EXPECT_TRUE(error.dfdx.isApprox(expected, /*prec=*/1e-6)) << error.dfdx << "\nvs\n" << expected;
}

TEST_F(EndEffectorKinematicsTangentJacobiansTest, TheOrientationErrorWrtAPlaneIsLinearizedAboutItsFrozenReference) {
  // The error with respect to a plane is the error from the orientation that has the plane's normal as its z-axis
  // and is otherwise the frame's own. The linearization holds that reference fixed, so it is the derivative of the
  // orientation error from it.
  const vector3_t normal = vector3_t(0.1, -0.2, 1.0).normalized();
  updateAt(state_);
  const VectorFunctionLinearApproximation planeError = kinematics_.getOrientationErrorWrtPlaneLinearApproximation(state_, {normal}).front();
  ASSERT_EQ(planeError.dfdx.cols(), state_.size());
  // Its value is NOT compared with getOrientationErrorWrtPlane(): that one is rotationMatrixDistanceToPlane, while the
  // linearization's f is the quaternion distance below, a different residual. Only the Jacobian's columns are at stake
  // here.

  const pinocchio::Data& data = pinocchioInterface_.getData();
  const Eigen::Matrix3d rotation = data.oMf[pinocchioInterface_.getModel().getBodyId(kTipName)].rotation();
  quaternion_t correction;
  correction.setFromTwoVectors(rotation * normal, normal);
  const quaternion_t frozenReference = quaternion_t(rotation) * correction;
  const VectorFunctionLinearApproximation frozen = kinematics_.getOrientationErrorLinearApproximation(state_, {frozenReference}).front();
  EXPECT_TRUE(planeError.dfdx.isApprox(frozen.dfdx, /*prec=*/1e-9)) << planeError.dfdx << "\nvs\n" << frozen.dfdx;
  EXPECT_GT(frozen.dfdx.norm(), 0.1) << "the case under test: an error that moves with the state";
}

}  // namespace
}  // namespace ocs2
