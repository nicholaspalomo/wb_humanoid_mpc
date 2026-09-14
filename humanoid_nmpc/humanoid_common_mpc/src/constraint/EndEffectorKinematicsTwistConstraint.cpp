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

#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"

#include <cmath>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

namespace {
/** The quaternion-distance orientation error is half the rotation angle, so its rate is half the angular velocity. */
constexpr scalar_t kHalfAngleScaling = 0.5;
/** 1 + v.n below this value means the end-effector normal is (almost) opposite to the plane normal. */
constexpr scalar_t kAntiParallelThreshold = 1e-6;
}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorKinematicsTwistConstraint::EndEffectorKinematicsTwistConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                                                           size_t numConstraints,
                                                                           Config config)
    : StateInputConstraint(ConstraintOrder::Linear),
      endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
      numConstraints_(numConstraints),
      ground_plane_normal_(0.0, 0.0, 1.0),
      config_(std::move(config)) {
  if (endEffectorKinematicsPtr_->getIds().size() != 1) {
    throw std::runtime_error("[EndEffectorKinematicsTwistConstraint] this class only accepts a single end-effector!");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorKinematicsTwistConstraint::EndEffectorKinematicsTwistConstraint(const EndEffectorKinematicsTwistConstraint& rhs)
    : StateInputConstraint(rhs),
      endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      ground_plane_normal_(rhs.ground_plane_normal_),
      constrainYawRateAboutNormal_(rhs.constrainYawRateAboutNormal_),
      config_(rhs.config_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void EndEffectorKinematicsTwistConstraint::configure(Config&& config) {
  // Config matrices are always 6D (full pose). The constraint slices to numConstraints_ rows at evaluation time.
  assert(config.b.rows() == 6);
  assert(config.Ax.size() > 0 || config.Av.size() > 0);
  assert((config.Ax.size() > 0 && config.Ax.rows() == 6) || config.Ax.size() == 0);
  assert((config.Ax.size() > 0 && config.Ax.cols() == 6) || config.Ax.size() == 0);
  assert((config.Av.size() > 0 && config.Av.rows() == 6) || config.Av.size() == 0);
  assert((config.Av.size() > 0 && config.Av.cols() == 6) || config.Av.size() == 0);
  config_ = std::move(config);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

matrix3_t EndEffectorKinematicsTwistConstraint::getAngularVelocityToOrientationErrorRateMap(const vector_t& state) const {
  const auto orientation = endEffectorKinematicsPtr_->getOrientation(state).front();
  const vector3_t v = orientation.toRotationMatrix() * vector3_t::UnitZ();  // end-effector normal in the world frame
  const vector3_t& n = ground_plane_normal_;

  const scalar_t cosAngle = v.dot(n);
  const scalar_t w = 1.0 + cosAngle;
  if (w < kAntiParallelThreshold) {
    // The end-effector normal points (almost) away from the plane normal: the shortest-arc correction and hence the
    // orientation error are not differentiable there. Fall back to damping the full angular velocity.
    return kHalfAngleScaling * matrix3_t::Identity();
  }

  // Analytic derivative of e = -(v x n) / sqrt(2 (1 + v.n)) with respect to the angular velocity, see the header.
  const vector3_t axis = v.cross(n);
  const scalar_t s = std::sqrt(2.0 * w);
  matrix3_t map = (cosAngle * matrix3_t::Identity() - v * n.transpose()) / s + (axis * axis.transpose()) / (s * s * s);
  // The tilt error carries no information about the rotation about the end-effector normal. Adding that rate in the
  // plane-normal direction makes the orientation rows full rank and stops a stance foot pivoting on the spot; without
  // it the last row of a 6D constraint is identically zero.
  if (constrainYawRateAboutNormal_) {
    map.noalias() += kHalfAngleScaling * n * v.transpose();
  }
  return map;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t EndEffectorKinematicsTwistConstraint::getValue(scalar_t time,
                                                        const vector_t& state,
                                                        const vector_t& input,
                                                        const PreComputation& preComp) const {
  vector_t f = config_.b.head(numConstraints_);

  if (config_.Ax.size() > 0) {
    // Foot pose: position and orientation error with respect to the ground normal. The orientation error (a kinematics
    // call) is only evaluated when the active rows use it, mirroring getLinearApproximation.
    const auto Ax = config_.Ax.topRows(numConstraints_);
    vector6_t footPose = vector6_t::Zero();
    footPose.head<3>() = endEffectorKinematicsPtr_->getPosition(state).front();
    if (!Ax.rightCols(3).isZero(0.0)) {
      footPose.tail<3>() = endEffectorKinematicsPtr_->getOrientationErrorWrtPlane(state, {ground_plane_normal_}).front();
    }
    f.noalias() += Ax * footPose;
  }

  if (config_.Av.size() > 0) {
    vector6_t twist = endEffectorKinematicsPtr_->getTwist(state, input).front();
    // The angular columns of Av act on the rate of the orientation residual, not on the raw angular velocity. The
    // mapping is skipped (and its kinematics call avoided) when those columns are inactive, e.g. for a
    // translation-only constraint.
    if (!config_.Av.topRightCorner(numConstraints_, 3).isZero(0.0)) {
      twist.tail(3) = getAngularVelocityToOrientationErrorRateMap(state) * twist.tail(3);
    }
    f.noalias() += config_.Av.topRows(numConstraints_) * twist;
  }
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

VectorFunctionLinearApproximation EndEffectorKinematicsTwistConstraint::getLinearApproximation(scalar_t time,
                                                                                               const vector_t& state,
                                                                                               const vector_t& input,
                                                                                               const PreComputation& preComp) const {
  VectorFunctionLinearApproximation linearApproximation =
      VectorFunctionLinearApproximation::Zero(numConstraints_, state.size(), input.size());

  linearApproximation.f = config_.b.head(numConstraints_);

  if (config_.Ax.size() > 0) {
    const matrix_t Ax = config_.Ax.topRows(numConstraints_);
    const auto positionApprox = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();
    linearApproximation.f.noalias() += Ax.leftCols(3) * positionApprox.f;
    linearApproximation.dfdx.noalias() += Ax.leftCols(3) * positionApprox.dfdx;

    if (!Ax.rightCols(3).isZero(0.0)) {
      const auto orientationApprox =
          endEffectorKinematicsPtr_->getOrientationErrorWrtPlaneLinearApproximation(state, {ground_plane_normal_}).front();
      linearApproximation.f.noalias() += Ax.rightCols(3) * orientationApprox.f;
      linearApproximation.dfdx.noalias() += Ax.rightCols(3) * orientationApprox.dfdx;
    }
  }

  if (config_.Av.size() > 0) {
    const auto twistApprox = endEffectorKinematicsPtr_->getTwistLinearApproximation(state, input).front();
    matrix_t Av = config_.Av.topRows(numConstraints_);
    if (!Av.rightCols(3).isZero(0.0)) {
      // The mapping is treated as constant at the linearization point: it depends on the state, but that dependence
      // enters the residual multiplied by the angular velocity and is therefore second order.
      const matrix_t mappedAngularGains = Av.rightCols(3) * getAngularVelocityToOrientationErrorRateMap(state);
      Av.rightCols(3) = mappedAngularGains;
    }
    linearApproximation.f.noalias() += Av * twistApprox.f;
    linearApproximation.dfdx.noalias() += Av * twistApprox.dfdx;
    linearApproximation.dfdu.noalias() += Av * twistApprox.dfdu;
  }

  return linearApproximation;
}

}  // namespace ocs2::humanoid
