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

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"  // NOLINT(euler-boundary): Euler boundary accessor (design section 4.2)

namespace ocs2::humanoid {

/******************************************************************************************************/
/* State Vector definition

  Define the state vector x = [q, v]^T = [p_W, xi, q_j, dp_W, w_B, qd_j]^T (13 +
  2 nj entries; its tangent has 12 + 2 nj): Pinocchio's configuration q (nq = 7
  + nj) and velocity v (nv = 6 + nj) of the Composite(Translation, Spherical)
  root

    Base linear position p_W = [p_base_x, p_base_y, p_base_z] in world frame
    Base orientation xi = [xi_x, xi_y, xi_z, xi_w], the unit quaternion from the
  base frame to the world Joint angles q_j Base linear velocity dp_W =
  [v_base_x, v_base_y, v_base_z] in world frame Base angular velocity w_B =
  [w_x, w_y, w_z] in the base frame Joint velocities qd_j

*/
/******************************************************************************************************/

/******************************************************************************************************/
/* Input Vector definition

  Define the input vector u = [W_l, W_r, qdd_j]^T

    Left contact Wrench W_l in inertial frame
    Right contact Wrench W_r in inertial frame
    Joint accelerations qdd_j

*/
/******************************************************************************************************/

/**
 * The state and input layout of the whole-body MPC (above): getters and setters
 * of the base pose, the joint angles and velocities, the joint accelerations
 * and the contact wrenches in its state and input vectors. The base orientation
 * is stored as a quaternion and its rate as the body angular velocity, so the
 * state lives on the product manifold [E(3), unit quaternion, E(2 nj + 6)]
 * (stateManifoldOf()) and the tuning layout maps onto it through the Euler
 * boundary (orientation/EulerBoundary.h; design decision D18 for the angular
 * velocity). Holds a reference to `modelSettings`, which must outlive it;
 * otherwise stateless, so its const methods are safe to call concurrently.
 * Copied only by clone(). Every accessor takes a state of getStateDim() and an
 * input of getInputDim() entries, unchecked: the solver and the control thread
 * call them on every evaluation.
 */
template <typename SCALAR_T>
class WBAccelMpcRobotModel : public MpcRobotModelBase<SCALAR_T> {
 public:
  explicit WBAccelMpcRobotModel(const ModelSettings& modelSettings)
      : MpcRobotModelBase<SCALAR_T>(
            modelSettings,
            MpcRobotModelDimensions{
                .stateDim = kBaseStateDim +
                            kStateEntriesPerJoint * modelSettings.mpc_joint_dim,
                .inputDim =
                    kWrenchDim * kNumContacts + modelSettings.mpc_joint_dim,
                .configurationDim =
                    kBaseConfigurationDim + modelSettings.mpc_joint_dim,
                .velocityDim = kBaseVelocityDim + modelSettings.mpc_joint_dim},
            placeTuningLayout(tuningLayoutOf(stateInputLayout(
                modelSettings, StateInputLayout::Mpc::kWholeBody)))) {}
  ~WBAccelMpcRobotModel() override = default;
  WBAccelMpcRobotModel& operator=(const WBAccelMpcRobotModel&) = delete;
  WBAccelMpcRobotModel(WBAccelMpcRobotModel&&) = delete;
  WBAccelMpcRobotModel& operator=(WBAccelMpcRobotModel&&) = delete;
  WBAccelMpcRobotModel* absl_nonnull clone() const override {
    return new WBAccelMpcRobotModel(*this);
  }

  /******************************************************************************************************/
  /*                                          Start indices */
  /******************************************************************************************************/

  size_t getBaseStartindex() const override { return 0; };
  size_t getBaseOrientationStartIndex() const override { return kPositionDim; }
  size_t getJointStartindex() const override { return kPoseCoefficients; };
  // Be careful, the joint Velocities are part of the state vector here: after q
  // (nq) and the base velocity (6).
  size_t getJointVelocitiesStartindex() const override {
    return kBaseStateDim + this->modelSettings.mpc_joint_dim;
  };
  // LINT.IfChange(input_indices)
  // The wrench of each of the biped's two feet: WBMpcInterface refuses any
  // other contact set, so unlike the centroidal input there are no 3-DoF
  // contacts here.
  size_t getJointAccelerationsStartindex() const {
    return kWrenchDim * kNumContacts;
  }

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  size_t getContactWrenchStartIndices(size_t contactIndex) const override {
    return kWrenchDim * contactIndex;
  };
  size_t getContactForceStartIndices(size_t contactIndex) const override {
    return getContactWrenchStartIndices(contactIndex);
  };
  size_t getContactMomentStartIndices(size_t contactIndex) const override {
    return getContactWrenchStartIndices(contactIndex) + kForceDim;
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputWeightsFromConfig.cpp:input_offsets)

  /******************************************************************************************************/
  /*                                     Generalized coordinates */
  /******************************************************************************************************/

  /** q = [p_W, xi_hat, q_j] of state.head(nq), with the quaternion safely
   * normalized (safelyNormalizedQuaternion()). */
  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(
      const VECTOR_T<SCALAR_T>& state) const override {
    VECTOR_T<SCALAR_T> q = state.head(this->getConfigurationDim());
    q.template segment<kQuaternionCoefficients>(
        getBaseOrientationStartIndex()) =
        safelyNormalizedQuaternion(
            VECTOR4_T<SCALAR_T>(state.template segment<kQuaternionCoefficients>(
                getBaseOrientationStartIndex())));
    return q;
  };

  VECTOR3_T<SCALAR_T> getBasePosition(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.head(kPositionDim);
  }

  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment(baseLinearVelocityStartIndex(), kPositionDim);
  }

  /**
   * The base velocity in the tuning order of the task file's
   * base_linear_velocity and base_angular_velocity blocks: [dp_W (x, y, z), P_3
   * w_B = the body angular velocity (z, y, x)] (design decision D18), not the
   * twist (getBaseTwistInWorldFrame()).
   */
  VECTOR6_T<SCALAR_T> getBaseComVelocity(
      const VECTOR_T<SCALAR_T>& state) const override {
    VECTOR6_T<SCALAR_T> baseVelocity;
    baseVelocity << state.segment(baseLinearVelocityStartIndex(), kPositionDim),
        state.segment(baseAngularVelocityStartIndex(), kRotationTangentDim)
            .reverse();
    return baseVelocity;
  };

  void setBaseComLinearVelocity(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR3_T<SCALAR_T>& velocity) const override {
    state.segment(baseLinearVelocityStartIndex(), kPositionDim) = velocity;
  };

  VECTOR_T<SCALAR_T> getJointAngles(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment(getJointStartindex(),
                         this->modelSettings.mpc_joint_dim);
  };

  VECTOR_T<SCALAR_T> getJointVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& /*input*/) const override {
    return state.tail(this->modelSettings.mpc_joint_dim);
  };

  // The base class declares this non-const because the centroidal model updates
  // its Pinocchio data in it; the whole-body state carries the velocities, so
  // here it is a pure read, and the const overload below serves the callers
  // that hold a const model (WBAccelDynamicsAD, whose clones share one).
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& input) override {
    return std::as_const(*this).getGeneralizedVelocities(state, input);
  }
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& /*input*/) const {
    return state.tail(this->getVelocityDim());
  }

  VECTOR_T<SCALAR_T> getJointAccelerations(
      const VECTOR_T<SCALAR_T>& input) const {
    return input.tail(this->modelSettings.mpc_joint_dim);
  }

  void setGeneralizedCoordinates(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const override {
    state.head(this->getConfigurationDim()) = generalizedCorrdinates;
  }

  void setBasePosition(VECTOR_T<SCALAR_T>& state,
                       const VECTOR3_T<SCALAR_T>& position) const override {
    state.head(kPositionDim) = position;
  }

  void setBaseLinearVelocity(VECTOR_T<SCALAR_T>& state,
                             const VECTOR3_T<SCALAR_T>& velocity) const {
    state.segment(baseLinearVelocityStartIndex(), kPositionDim) = velocity;
  }

  void setJointAngles(VECTOR_T<SCALAR_T>& state,
                      const VECTOR_T<SCALAR_T>& jointAngles) const override {
    state.segment(getJointStartindex(), this->modelSettings.mpc_joint_dim) =
        jointAngles;
  }

  void setJointVelocities(
      VECTOR_T<SCALAR_T>& state,
      VECTOR_T<SCALAR_T>& /*input*/,
      const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    state.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }

  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state,
                           scalar_t heightChange) const override {
    state[kZAxisIndex] += heightChange;
  }

  /******************************************************************************************************/
  /*                   Layout-agnostic accessors (MpcRobotModelBase), the
   * quaternion layout             */
  /******************************************************************************************************/

  VECTOR4_T<SCALAR_T> getBaseOrientation(
      const VECTOR_T<SCALAR_T>& state) const override {
    return safelyNormalizedQuaternion(
        VECTOR4_T<SCALAR_T>(state.template segment<kQuaternionCoefficients>(
            getBaseOrientationStartIndex())));
  }

  void setBaseOrientation(vector_t& state,
                          const vector4_t& orientation) const override {
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) =
        safelyNormalizedQuaternion(orientation);
  }

  scalar_t getBaseHeading(const vector_t& state) const override {
    return twistHeading(baseOrientationOf(state));
  }

  // NOLINTBEGIN(euler-boundary): Euler boundary accessor (design section 4.2)
  vector6_t getBasePoseEulerZyx(const vector_t& state) const override {
    vector6_t basePose;
    basePose << state.segment<kPositionDim>(getBaseStartindex()),
        getBaseOrientationEulerZyx(state);
    return basePose;
  }

  void setBasePoseEulerZyx(vector_t& state,
                           const vector6_t& basePose) const override {
    state.segment<kPositionDim>(getBaseStartindex()) =
        basePose.head<kPositionDim>();
    setBaseOrientationEulerZyx(state, basePose.tail<kEulerAnglesZyxDim>());
  }

  vector3_t getBaseOrientationEulerZyx(const vector_t& state) const override {
    return eulerZyxFromQuaternion(baseOrientationOf(state));
  }

  void setBaseOrientationEulerZyx(vector_t& state,
                                  const vector3_t& eulerZyx) const override {
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) =
        quaternionFromEulerZyx(eulerZyx);
  }
  // NOLINTEND(euler-boundary)

  vector3_t getBaseAngularVelocityLocal(const vector_t& state) const override {
    return state.segment<kRotationTangentDim>(baseAngularVelocityStartIndex());
  }

  void setBaseAngularVelocityLocal(
      vector_t& state, const vector3_t& angularVelocityLocal) const override {
    state.segment<kRotationTangentDim>(baseAngularVelocityStartIndex()) =
        angularVelocityLocal;
  }

  void setBaseAngularVelocityFromWorldYawRate(vector_t& state,
                                              scalar_t yawRate) const override {
    state.segment<kRotationTangentDim>(baseAngularVelocityStartIndex()) =
        quaternionRotationMatrix<scalar_t>(baseOrientationOf(state))
            .transpose() *
        vector3_t(0.0, 0.0, yawRate);
  }

  BaseAngularVelocityJacobian getBaseAngularVelocityLocalJacobian(
      const vector_t& /*state*/) const override {
    return BaseAngularVelocityJacobian{
        .orientationSlots =
            matrix_t::Zero(kRotationTangentDim, kQuaternionCoefficients),
        .angularVelocitySlots = matrix3_t::Identity()};
  }

  /**
   * [dp_W, R(xi) w_B] of the state's base velocity slots, exact. The state
   * holds the joint velocities too, so `jointVelocities` is not read; it is
   * held to the size of the interface all the same (ABSL_CHECK), as the
   * centroidal model, which reads it, is.
   */
  vector6_t getBaseTwistInWorldFrame(
      const vector_t& state, const vector_t& jointVelocities) const override {
    ABSL_CHECK_EQ(static_cast<size_t>(jointVelocities.size()),
                  this->getJointDim())
        << "WBAccelMpcRobotModel::getBaseTwistInWorldFrame: one joint velocity "
           "per MPC joint";
    return baseTwistOf(state);
  }

  /** (v_x, v_y, w_z) of the knot's base twist (getBaseTwistInWorldFrame()); the
   * knot stores the velocity, no inertia is read. */
  vector3_t getTargetPlanarCommand(
      const vector_t& targetState,
      scalar_t /*compositeYawInertia*/) const override {
    const vector6_t twist = baseTwistOf(targetState);
    return vector3_t(twist(0), twist(1), twist(kTwistYawRateIndex));
  }

  absl::StatusOr<vector_t> stateFromTuning(
      const vector_t& tuningVector) const override {
    return stateFromTuningLayout(tuningVector, this->getTuningLayout());
  }

  absl::StatusOr<vector_t> tuningFromState(
      const vector_t& state) const override {
    return tuningLayoutFromState(state, this->getTuningLayout());
  }

  vector_t interpolateStateTrajectory(
      scalar_t time,
      const scalar_array_t& times,
      const vector_array_t& states) const override {
    return this->interpolateWithSlerpedOrientation(time, times, states);
  }

  vector_t getNeutralState() const override {
    vector_t state = vector_t::Zero(this->getStateDim());
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) =
        vector4_t(0.0, 0.0, 0.0, 1.0);
    return state;
  }

  matrix_t getBaseRotationTangentJacobian(
      const vector_t& state) const override {
    return configurationTangentLift(
        vector4_t(state.segment<kQuaternionCoefficients>(
            getBaseOrientationStartIndex())));
  }

  matrix_t getPinocchioTangentJacobian(const vector_t& state) const override {
    // q = state.head(nq) = [p_W, xi_hat, q_j]: the position and the joints are
    // their own tangent, the quaternion's tangent is the body rotation E^+(xi)
    // dxi.
    const Eigen::Index jointDim =
        static_cast<Eigen::Index>(this->getJointDim());
    matrix_t jacobian =
        matrix_t::Zero(this->getVelocityDim(), this->getStateDim());
    jacobian.block<kPositionDim, kPositionDim>(0, getBaseStartindex())
        .setIdentity();
    jacobian.block<kRotationTangentDim, kQuaternionCoefficients>(
        kPositionDim, getBaseOrientationStartIndex()) =
        getBaseRotationTangentJacobian(state);
    jacobian.block(kPoseTangentDim, getJointStartindex(), jointDim, jointDim)
        .setIdentity();
    return jacobian;
  }

  void fillGeneralizedCoordinatesFromMeasurement(
      const vector3_t& positionWorld,
      const vector4_t& orientation,
      const vector_t& jointPositions,
      Eigen::Ref<vector_t> q) const override {
    q.head<kPositionDim>() = positionWorld;
    q.segment<kQuaternionCoefficients>(kPositionDim) =
        safelyNormalizedQuaternion(orientation);
    q.tail(this->modelSettings.mpc_joint_dim) = jointPositions;
  }

  void fillGeneralizedVelocitiesFromMeasurement(
      const vector4_t& orientation,
      const vector3_t& linearVelocityLocal,
      const vector3_t& angularVelocityLocal,
      const vector_t& jointVelocities,
      Eigen::Ref<vector_t> v) const override {
    const quaternion_t rotation(safelyNormalizedQuaternion(orientation));
    v.head<kPositionDim>() = rotation * linearVelocityLocal;
    v.segment<kRotationTangentDim>(kPositionDim) = angularVelocityLocal;
    v.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }

  /******************************************************************************************************/
  /*                                          Contacts */
  /******************************************************************************************************/

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input,
                                       size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex),
                         kWrenchDim);
  };

  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input,
                                      size_t contactIndex) const override {
    return input.segment(getContactForceStartIndices(contactIndex), kForceDim);
  };

  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input,
                                       size_t contactIndex) const override {
    return input.segment(getContactMomentStartIndices(contactIndex),
                         kTorqueDim);
  };

  void setContactWrench(VECTOR_T<SCALAR_T>& input,
                        const VECTOR6_T<SCALAR_T>& wrench,
                        size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), kWrenchDim) =
        wrench;
  };

  void setContactForce(VECTOR_T<SCALAR_T>& input,
                       const VECTOR3_T<SCALAR_T>& force,
                       size_t contactIndex) const override {
    input.segment(getContactForceStartIndices(contactIndex), kForceDim) = force;
  };

  void setContactMoment(VECTOR_T<SCALAR_T>& input,
                        const VECTOR3_T<SCALAR_T>& moment,
                        size_t contactIndex) const override {
    input.segment(getContactMomentStartIndices(contactIndex), kTorqueDim) =
        moment;
  };

 protected:
  // For clone(), and for the clone() of a derived model.
  WBAccelMpcRobotModel(const WBAccelMpcRobotModel& rhs)
      : MpcRobotModelBase<SCALAR_T>(rhs) {}

 private:
  // The base's entries of the state, its pose coefficients [p_W, xi] at the
  // head of q and its twist [dp_W, w_B] at the head of v: the state without the
  // joints.
  static constexpr int kBaseStateDim = kPoseCoefficients + kTwistDim;
  // The entries of each joint in the state: its angle in q and its velocity in
  // v.
  static constexpr int kStateEntriesPerJoint = 2;
  // The base's slots before the body angular velocity w_B: p_W, xi and dp_W
  // (the joint angles, between xi and dp_W, add mpc_joint_dim).
  static constexpr int kBaseAngularVelocityOffset =
      kPoseCoefficients + kPositionDim;

  // The first state slot of the base linear velocity dp_W, the start of the
  // base velocity rows: right after q.
  size_t baseLinearVelocityStartIndex() const {
    return kPoseCoefficients + this->modelSettings.mpc_joint_dim;
  }

  // The first state slot of the body angular velocity w_B = (w_x, w_y, w_z).
  size_t baseAngularVelocityStartIndex() const {
    return kBaseAngularVelocityOffset + this->modelSettings.mpc_joint_dim;
  }

  // getBaseOrientation() of a state in double precision, for every SCALAR_T.
  vector4_t baseOrientationOf(const vector_t& state) const {
    return safelyNormalizedQuaternion(
        vector4_t(state.segment<kQuaternionCoefficients>(
            getBaseOrientationStartIndex())));
  }

  // [dp_W, R(xi) w_B] of the state's base velocity slots
  // (getBaseTwistInWorldFrame()).
  vector6_t baseTwistOf(const vector_t& state) const {
    vector6_t twist;
    twist.head<kPositionDim>() =
        state.segment<kPositionDim>(baseLinearVelocityStartIndex());
    twist.tail<kRotationTangentDim>() =
        quaternionRotationMatrix<scalar_t>(baseOrientationOf(state)) *
        getBaseAngularVelocityLocal(state);
    return twist;
  }
};

}  // namespace ocs2::humanoid
