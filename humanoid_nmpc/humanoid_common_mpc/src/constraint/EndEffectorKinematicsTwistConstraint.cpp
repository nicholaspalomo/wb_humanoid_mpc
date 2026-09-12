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
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

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

matrix3_t EndEffectorKinematicsTwistConstraint::getOrientationErrorRateMapping(const vector_t& state) const {
  // Get the current frame orientation
  const auto q_frame = endEffectorKinematicsPtr_->getOrientation(state).front();
  const Eigen::Matrix<scalar_t, 3, 3> R = q_frame.toRotationMatrix();
  const vector3_t v = R * ground_plane_normal_;  // foot z-axis in world frame (unit vector)

  // Baseline orientation error: e0 = rotationMatrixDistanceToPlane(R, planeNormal)
  const vector3_t e0 = rotationMatrixDistanceToPlane<scalar_t>(R, ground_plane_normal_);

  // Compute M via tangent-space perturbation:
  // For each axis i, a perturbation δω = eps·eᵢ causes δv = (eps·eᵢ) × v on the unit sphere.
  // We evaluate the orientation error at the perturbed v to get M_col_i = (e_pert - e0) / eps.
  const scalar_t eps = 1e-7;
  matrix3_t M;
  for (int i = 0; i < 3; i++) {
    vector3_t omega_i = vector3_t::Zero();
    omega_i(i) = eps;
    vector3_t v_pert = v + omega_i.cross(v);
    v_pert.normalize();

    // Compute orientation error at the perturbed foot z-axis
    const Eigen::Quaternion<scalar_t> q_corr_pert = getQuaternionFromUnitVectors<scalar_t>(v_pert, ground_plane_normal_);
    const vector3_t e_pert = quaternionDistance<scalar_t>(q_corr_pert, Eigen::Quaternion<scalar_t>::Identity());

    M.col(i) = (e_pert - e0) / eps;
  }
  return M;
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
    // foot pose is a 6D vector containing the foot position and orientation error wrt. the ground normal
    vector6_t footPose;
    footPose << endEffectorKinematicsPtr_->getPosition(state).front(),
        endEffectorKinematicsPtr_->getOrientationErrorWrtPlane(state, {ground_plane_normal_}).front();
    f.noalias() += config_.Ax.topRows(numConstraints_) * footPose;
  }

  if (config_.Av.size() > 0) {
    const auto twist = endEffectorKinematicsPtr_->getTwist(state, input).front();

    if (numConstraints_ <= 3) {
      // Translation-only: use linear velocity directly
      f.noalias() += config_.Av.topLeftCorner(numConstraints_, 3) * twist.head(3);
    } else {
      // Full 6D constraint: linear velocity rows (0-2) use raw linear velocity,
      // orientation rows (3-5) map angular velocity through the quaternion kinematic Jacobian
      // so that Av_ang * M * omega = Av_ang * d(orientationError)/dt (a proper PD law).
      const matrix3_t M = getOrientationErrorRateMapping(state);
      vector6_t mappedTwist;
      mappedTwist.head(3) = twist.head(3);
      mappedTwist.tail(3) = M * twist.tail(3);
      f.noalias() += config_.Av * mappedTwist;
    }
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
    const auto positionApprox = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();

    // Position rows (always present)
    const size_t posRows = std::min(numConstraints_, size_t(3));
    linearApproximation.f.head(posRows).noalias() += config_.Ax.topLeftCorner(posRows, 3) * positionApprox.f;
    linearApproximation.dfdx.topRows(posRows).noalias() += config_.Ax.topLeftCorner(posRows, 3) * positionApprox.dfdx;

    // Orientation rows (only for 6D constraint)
    if (numConstraints_ > 3) {
      const auto orientationApprox =
          endEffectorKinematicsPtr_->getOrientationErrorWrtPlaneLinearApproximation(state, {ground_plane_normal_}).front();
      linearApproximation.f.tail(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.f;
      linearApproximation.dfdx.bottomRows(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.dfdx;
    }
  }

  if (config_.Av.size() > 0) {
    const auto twistApprox = endEffectorKinematicsPtr_->getTwistLinearApproximation(state, input).front();

    if (numConstraints_ <= 3) {
      // Translation-only: use linear velocity part of twist directly
      linearApproximation.f.noalias() += config_.Av.topLeftCorner(numConstraints_, 3) * twistApprox.f.head(3);
      linearApproximation.dfdx.noalias() += config_.Av.topLeftCorner(numConstraints_, 3) * twistApprox.dfdx.topRows(3);
      linearApproximation.dfdu.noalias() += config_.Av.topLeftCorner(numConstraints_, 3) * twistApprox.dfdu.topRows(3);
    } else {
      // Full 6D constraint: map angular velocity through the quaternion kinematic Jacobian
      // for the orientation rows, so the constraint is a proper PD law.
      const matrix3_t M = getOrientationErrorRateMapping(state);

      // Build a block-diagonal mapping: [I_3x3, 0; 0, M_3x3]
      // Linear velocity rows: Av_lin * v_lin (unchanged)
      linearApproximation.f.head(3).noalias() += config_.Av.topLeftCorner(3, 3) * twistApprox.f.head(3);
      linearApproximation.dfdx.topRows(3).noalias() += config_.Av.topLeftCorner(3, 3) * twistApprox.dfdx.topRows(3);
      linearApproximation.dfdu.topRows(3).noalias() += config_.Av.topLeftCorner(3, 3) * twistApprox.dfdu.topRows(3);

      // Angular velocity rows: Av_ang * M * omega
      // The mapping M is treated as constant at the current linearization point
      // (it depends on state but its derivative is second-order and neglected in the linear approximation).
      const matrix3_t AvM = config_.Av.bottomRightCorner(3, 3) * M;
      linearApproximation.f.tail(3).noalias() += AvM * twistApprox.f.tail(3);
      linearApproximation.dfdx.bottomRows(3).noalias() += AvM * twistApprox.dfdx.bottomRows(3);
      linearApproximation.dfdu.bottomRows(3).noalias() += AvM * twistApprox.dfdu.bottomRows(3);
    }
  }

  return linearApproximation;
}

}  // namespace ocs2::humanoid
