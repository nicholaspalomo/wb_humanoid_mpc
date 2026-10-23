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

#pragma once

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid::test {

/**
 * A robot model that is nothing but the dimensions of a StateInputLayout - the state and input sizes of a centroidal or
 * whole-body MPC of a robot's ModelSettings - for tests of what is built from the configuration (the factory's
 * weights, the reference manager, the target calculator, the motion manager) without a robot model package and its
 * CppAD libraries. What depends only on the layout is real, on the quaternion layout of both MPCs (design section 2.3
 * of humanoid_nmpc/docs/quaternion_base_orientation/README.md): the dimensions (nx = n_t + 1, nq = 7 + nj, nv = 6 + nj)
 * and start indices, getTuningLayout() (tuningLayoutOf() placed with kQuaternionOrientationStateDim), stateFromTuning()
 * and tuningFromState() (stateFromTuningLayout(), tuningLayoutFromState()), interpolateStateTrajectory() (linear, the
 * orientation slerped) and getNeutralState() (the identity orientation). Every accessor of the state's values returns
 * the neutral state's values - zeros, the identity orientation and heading 0 - whatever the state, and so a zero
 * Jacobian, and every setter does nothing. `modelSettings` is retained and must outlive the model. Its const functions
 * are safe to call concurrently.
 */
template <typename SCALAR_T>
class LayoutRobotModel final : public MpcRobotModelBase<SCALAR_T> {
 public:
  LayoutRobotModel(const ModelSettings& modelSettings, StateInputLayout::Mpc mpc)
      : MpcRobotModelBase<SCALAR_T>(modelSettings,
                                    MpcRobotModelDimensions{.stateDim = tuningDimension(stateInputLayout(modelSettings, mpc)) + 1,
                                                            .inputDim = inputDimension(stateInputLayout(modelSettings, mpc)),
                                                            .configurationDim = kBaseConfigurationDim + modelSettings.mpc_joint_dim,
                                                            .velocityDim = kBaseVelocityDim + modelSettings.mpc_joint_dim},
                                    placeTuningLayout(tuningLayoutOf(stateInputLayout(modelSettings, mpc)))),
        mpc_(mpc) {}

  LayoutRobotModel* absl_nonnull clone() const override { return new LayoutRobotModel(this->modelSettings, mpc_); }

  size_t getBaseStartindex() const override { return mpc_ == StateInputLayout::Mpc::kCentroidal ? 6 : 0; }
  size_t getBaseOrientationStartIndex() const override { return mpc_ == StateInputLayout::Mpc::kCentroidal ? 9 : 3; }
  size_t getJointStartindex() const override { return mpc_ == StateInputLayout::Mpc::kCentroidal ? 13 : 7; }
  // The centroidal input's joint velocities, after the contact wrenches; the whole-body state's, after q and the base
  // velocity.
  size_t getJointVelocitiesStartindex() const override {
    return mpc_ == StateInputLayout::Mpc::kCentroidal ? 6 * this->modelSettings.contactNames.size()
                                                      : this->getConfigurationDim() + kBaseVelocityDim;
  }
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return 6 * contactIndex; }

  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& /*state*/) const override {
    VECTOR_T<SCALAR_T> q = VECTOR_T<SCALAR_T>::Zero(this->getConfigurationDim());
    q(6) = SCALAR_T(1.0);  // q = [p_W, xi = (x, y, z, w), q_j] of the identity orientation.
    return q;
  }
  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR3_T<SCALAR_T>::Zero(); }
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR3_T<SCALAR_T>::Zero(); }
  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR6_T<SCALAR_T>::Zero(); }
  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& /*state*/) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->getJointDim());
  }
  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*input*/) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->getJointDim());
  }
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*input*/) override {
    return VECTOR_T<SCALAR_T>::Zero(this->getVelocityDim());
  }

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*generalizedCorrdinates*/) const override {}
  void setBasePosition(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR3_T<SCALAR_T>& /*position*/) const override {}
  void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR3_T<SCALAR_T>& /*velocity*/) const override {}
  void setJointAngles(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*jointAngles*/) const override {}
  void setJointVelocities(VECTOR_T<SCALAR_T>& /*state*/,
                          VECTOR_T<SCALAR_T>& /*input*/,
                          const VECTOR_T<SCALAR_T>& /*jointVelocities*/) const override {}
  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& /*state*/, scalar_t /*heightChange*/) const override {}

  VECTOR4_T<SCALAR_T> getBaseOrientation(const VECTOR_T<SCALAR_T>& /*state*/) const override {
    return VECTOR4_T<SCALAR_T>(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(1.0));
  }
  void setBaseOrientation(vector_t& /*state*/, const vector4_t& /*orientation*/) const override {}
  scalar_t getBaseHeading(const vector_t& /*state*/) const override { return 0.0; }
  vector6_t getBasePoseEulerZyx(const vector_t& /*state*/) const override { return vector6_t::Zero(); }
  void setBasePoseEulerZyx(vector_t& /*state*/, const vector6_t& /*basePose*/) const override {}
  vector3_t getBaseOrientationEulerZyx(const vector_t& /*state*/) const override { return vector3_t::Zero(); }
  void setBaseOrientationEulerZyx(vector_t& /*state*/, const vector3_t& /*eulerZyx*/) const override {}
  vector3_t getBaseAngularVelocityLocal(const vector_t& /*state*/) const override { return vector3_t::Zero(); }
  void setBaseAngularVelocityLocal(vector_t& /*state*/, const vector3_t& /*angularVelocityLocal*/) const override {}
  void setBaseAngularVelocityFromWorldYawRate(vector_t& /*state*/, scalar_t /*yawRate*/) const override {}
  BaseAngularVelocityJacobian getBaseAngularVelocityLocalJacobian(const vector_t& /*state*/) const override {
    return BaseAngularVelocityJacobian{.orientationSlots = matrix_t::Zero(3, kQuaternionOrientationStateDim),
                                       .angularVelocitySlots = matrix3_t::Zero()};
  }
  vector6_t getBaseTwistInWorldFrame(const vector_t& /*state*/, const vector_t& /*jointVelocities*/) const override {
    return vector6_t::Zero();
  }
  vector3_t getTargetPlanarCommand(const vector_t& /*targetState*/, scalar_t /*compositeYawInertia*/) const override {
    return vector3_t::Zero();
  }
  absl::StatusOr<vector_t> stateFromTuning(const vector_t& tuningVector) const override {
    return stateFromTuningLayout(tuningVector, this->getTuningLayout());
  }
  absl::StatusOr<vector_t> tuningFromState(const vector_t& state) const override {
    return tuningLayoutFromState(state, this->getTuningLayout());
  }
  vector_t interpolateStateTrajectory(scalar_t time, const scalar_array_t& times, const vector_array_t& states) const override {
    return this->interpolateWithSlerpedOrientation(time, times, states);
  }
  vector_t getNeutralState() const override {
    vector_t state = vector_t::Zero(this->getStateDim());
    state.segment<4>(static_cast<Eigen::Index>(getBaseOrientationStartIndex())) = vector4_t(0.0, 0.0, 0.0, 1.0);
    return state;
  }
  matrix_t getBaseRotationTangentJacobian(const vector_t& /*state*/) const override {
    return matrix_t::Zero(3, kQuaternionOrientationStateDim);
  }
  matrix_t getPinocchioTangentJacobian(const vector_t& /*state*/) const override {
    return matrix_t::Zero(this->getVelocityDim(), this->getStateDim());
  }
  void fillGeneralizedCoordinatesFromMeasurement(const vector3_t& /*positionWorld*/,
                                                 const vector4_t& /*orientation*/,
                                                 const vector_t& /*jointPositions*/,
                                                 Eigen::Ref<vector_t> q) const override {
    q.setZero();
    q(6) = 1.0;
  }
  void fillGeneralizedVelocitiesFromMeasurement(const vector4_t& /*orientation*/,
                                                const vector3_t& /*linearVelocityLocal*/,
                                                const vector3_t& /*angularVelocityLocal*/,
                                                const vector_t& /*jointVelocities*/,
                                                Eigen::Ref<vector_t> v) const override {
    v.setZero();
  }

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR6_T<SCALAR_T>::Zero();
  }
  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR3_T<SCALAR_T>::Zero();
  }
  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR3_T<SCALAR_T>::Zero();
  }
  void setContactWrench(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR6_T<SCALAR_T>& /*wrench*/, size_t /*contactIndex*/) const override {}
  void setContactForce(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR3_T<SCALAR_T>& /*force*/, size_t /*contactIndex*/) const override {}
  void setContactMoment(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR3_T<SCALAR_T>& /*moment*/, size_t /*contactIndex*/) const override {}

 private:
  StateInputLayout::Mpc mpc_;
};

}  // namespace ocs2::humanoid::test
