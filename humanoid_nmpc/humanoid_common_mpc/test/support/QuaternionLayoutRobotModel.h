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

#include <cstddef>
#include <optional>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

// The quaternion layouts of both MPCs (design section 2.3 of
// humanoid_nmpc/docs/quaternion_base_orientation/README.md) for the tests of
// humanoid_common_mpc that evaluate a term on a real state -
// TuningLayoutDeviation, the Tuning* costs, the CoM + ACoM tracking cost, the
// contact planner's heading - without the centroidal or whole-body package and
// its CppAD libraries.

namespace ocs2::humanoid::test {

/** The first state slot of `layout`'s block of kind `kind`; nullopt when it has
 * none. */
inline std::optional<size_t> stateStartOf(const TuningLayout& layout,
                                          TuningSegmentKind kind) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind == kind) return segment.stateStart;
  }
  return std::nullopt;
}

/** The tuning layout of the MPC `mpc` of `settings`, placed on its quaternion
 * state (kQuaternionOrientationStateDim). */
inline TuningLayout quaternionLayoutOf(const ModelSettings& settings,
                                       StateInputLayout::Mpc mpc) {
  return placeTuningLayout(tuningLayoutOf(stateInputLayout(settings, mpc)));
}

/**
 * A robot model of the quaternion layout of the centroidal or the whole-body
 * MPC, read and written slot for slot as CentroidalMpcRobotModel and
 * WBAccelMpcRobotModel do:
 *   - centroidal: x = [h / m (6), p_W (3), xi (4), q_j (nj)], u = [contact
 * wrenches (6 per contact), dq_j (nj)], the Pinocchio configuration q = [p_W,
 * xi_hat, q_j] of x.tail(nq);
 *   - whole body: x = [p_W (3), xi (4), q_j (nj), dp_W (3), w_B (3), dq_j
 * (nj)], u = [contact wrenches, ddq_j], q of x.head(nq) and v = x.tail(nv). xi
 * = (x, y, z, w) is the base-to-world quaternion; the accessors read it through
 * safelyNormalizedQuaternion(), so any positive scale and its negative describe
 * the same attitude, its rotation tangent Jacobian is
 * configurationTangentLift(xi) (E^+, 3 x 4), the heading its twist heading and
 * interpolation slerps it. The Euler boundary accessors convert with
 * orientation/EulerBoundary.h. The centroidal state holds no base angular
 * velocity: reading or writing one fails the test, where
 * CentroidalMpcRobotModel aborts. The centroidal base twist and a centroidal
 * knot's planar command need the centroidal momentum matrix, which this model
 * does not have: they are zero, and the base velocity of
 * getGeneralizedVelocities() is zero. `modelSettings` is retained by the base
 * class and must outlive the model. Its const functions are safe to call
 * concurrently.
 */
class QuaternionLayoutRobotModel final : public MpcRobotModelBase<scalar_t> {
 public:
  /** The model of the quaternion layout of the MPC `mpc`
   * (quaternionLayoutOf()). */
  QuaternionLayoutRobotModel(const ModelSettings& modelSettings,
                             StateInputLayout::Mpc mpc)
      : QuaternionLayoutRobotModel(
            modelSettings, mpc, quaternionLayoutOf(modelSettings, mpc)) {}

  /**
   * The model of `placedLayout`, a tuning layout placed on a state of its
   * getTuningLayoutStateDim() entries, with the input and the start indices of
   * the MPC `mpc`: for a layout other than the MPC's own (a malformed one, or
   * one of differences only) whose orientation and angular-velocity blocks, if
   * any, sit where its segments place them.
   */
  QuaternionLayoutRobotModel(const ModelSettings& modelSettings,
                             StateInputLayout::Mpc mpc,
                             const TuningLayout& placedLayout)
      : MpcRobotModelBase<scalar_t>(
            modelSettings,
            MpcRobotModelDimensions{
                .stateDim = getTuningLayoutStateDim(placedLayout),
                .inputDim =
                    inputDimension(stateInputLayout(modelSettings, mpc)),
                .configurationDim =
                    kBaseConfigurationDim + modelSettings.mpc_joint_dim,
                .velocityDim = kBaseVelocityDim + modelSettings.mpc_joint_dim},
            placedLayout),
        mpc_(mpc),
        orientationStart_(
            stateStartOf(placedLayout, TuningSegmentKind::kEulerZyxOrientation)
                .value_or(0)),
        angularVelocityStart_(stateStartOf(
            placedLayout, TuningSegmentKind::kZyxOrderedAngularVelocity)) {}
  ~QuaternionLayoutRobotModel() override = default;
  QuaternionLayoutRobotModel& operator=(const QuaternionLayoutRobotModel&) =
      delete;
  QuaternionLayoutRobotModel(QuaternionLayoutRobotModel&&) = delete;
  QuaternionLayoutRobotModel& operator=(QuaternionLayoutRobotModel&&) = delete;

  QuaternionLayoutRobotModel* absl_nonnull clone() const override {
    return new QuaternionLayoutRobotModel(*this);
  }

  size_t getBaseStartindex() const override { return isCentroidal() ? 6 : 0; }
  size_t getBaseOrientationStartIndex() const override {
    return orientationStart_;
  }
  size_t getJointStartindex() const override {
    return getBaseStartindex() + kBaseConfigurationDim;
  }
  // The centroidal input's joint velocities, after the contact wrenches; the
  // whole-body state's, after q and the base velocity.
  size_t getJointVelocitiesStartindex() const override {
    return isCentroidal() ? 6 * modelSettings.contactNames.size()
                          : getConfigurationDim() + kBaseVelocityDim;
  }
  size_t getContactWrenchStartIndices(size_t contactIndex) const override {
    return 6 * contactIndex;
  }

  vector_t getGeneralizedCoordinates(const vector_t& state) const override {
    vector_t q =
        state.segment(static_cast<Eigen::Index>(getBaseStartindex()),
                      static_cast<Eigen::Index>(getConfigurationDim()));
    q.segment<4>(kBaseTranslationDim) = getQuatBaseRtWorldXyzw(state);
    return q;
  }
  vector3_t getPosBaseRtWorldEwrtWorld(const vector_t& state) const override {
    return state.segment<3>(baseStart());
  }
  // NOLINTBEGIN(frame-naming): model-dependent, see MpcRobotModelBase
  vector3_t getBaseComLinearVelocity(const vector_t& state) const override {
    return isCentroidal()
               ? vector3_t(state.head<3>())
               : vector3_t(state.segment<3>(baseLinearVelocityStart()));
  }
  vector6_t getBaseComVelocity(const vector_t& state) const override {
    if (isCentroidal()) return state.head<6>();
    vector6_t baseVelocity;
    baseVelocity << state.segment<3>(baseLinearVelocityStart()),
        getAngvelBaseRtWorldEwrtBase(state).reverse();
    return baseVelocity;
  }
  // NOLINTEND(frame-naming)
  vector_t getJointAngles(const vector_t& state) const override {
    return state.segment(jointStart(), jointDim());
  }
  vector_t getJointVelocities(const vector_t& state,
                              const vector_t& input) const override {
    return isCentroidal()
               ? vector_t(input.segment(jointVelocitiesStart(), jointDim()))
               : vector_t(state.tail(jointDim()));
  }
  vector_t getGeneralizedVelocities(const vector_t& state,
                                    const vector_t& input) override {
    if (!isCentroidal())
      return state.tail(static_cast<Eigen::Index>(getVelocityDim()));
    vector_t generalizedVelocities =
        vector_t::Zero(static_cast<Eigen::Index>(getVelocityDim()));
    generalizedVelocities.tail(jointDim()) = getJointVelocities(state, input);
    return generalizedVelocities;
  }

  void setGeneralizedCoordinates(
      vector_t& state, const vector_t& generalizedCoordinates) const override {
    state.segment(baseStart(), generalizedCoordinates.size()) =
        generalizedCoordinates;
  }
  void setPosBaseRtWorldEwrtWorld(
      vector_t& state,
      const vector3_t& posBaseRtWorldEwrtWorld) const override {
    state.segment<3>(baseStart()) = posBaseRtWorldEwrtWorld;
  }
  // NOLINTBEGIN(frame-naming): model-dependent, see MpcRobotModelBase
  void setBaseComLinearVelocity(vector_t& state,
                                const vector3_t& velocity) const override {
    if (isCentroidal()) {
      state.head<3>() = velocity;
    } else {
      state.segment<3>(baseLinearVelocityStart()) = velocity;
    }
  }
  // NOLINTEND(frame-naming)
  void setJointAngles(vector_t& state,
                      const vector_t& jointAngles) const override {
    state.segment(jointStart(), jointDim()) = jointAngles;
  }
  void setJointVelocities(vector_t& state,
                          vector_t& input,
                          const vector_t& jointVelocities) const override {
    if (isCentroidal()) {
      input.segment(jointVelocitiesStart(), jointDim()) = jointVelocities;
    } else {
      state.tail(jointDim()) = jointVelocities;
    }
  }
  void adaptPosBaseRtWorldEwrtWorldZ(vector_t& state,
                                     scalar_t heightChange) const override {
    state(baseStart() + 2) += heightChange;
  }

  vector4_t getQuatBaseRtWorldXyzw(const vector_t& state) const override {
    return safelyNormalizedQuaternion(
        vector4_t(state.segment<4>(orientationStart())));
  }
  void setQuatBaseRtWorldXyzw(
      vector_t& state, const vector4_t& quatBaseRtWorldXyzw) const override {
    state.segment<4>(orientationStart()) =
        safelyNormalizedQuaternion(quatBaseRtWorldXyzw);
  }
  scalar_t getYawBaseRtWorld(const vector_t& state) const override {
    return yawOf(getQuatBaseRtWorldXyzw(state));
  }
  vector6_t getPoseeulerzyxBaseRtWorld(const vector_t& state) const override {
    vector6_t poseeulerzyxBaseRtWorld;
    poseeulerzyxBaseRtWorld << getPosBaseRtWorldEwrtWorld(state),
        getEulerzyxBaseRtWorld(state);
    return poseeulerzyxBaseRtWorld;
  }
  void setPoseeulerzyxBaseRtWorld(
      vector_t& state,
      const vector6_t& poseeulerzyxBaseRtWorld) const override {
    setPosBaseRtWorldEwrtWorld(state, poseeulerzyxBaseRtWorld.head<3>());
    setEulerzyxBaseRtWorld(state, poseeulerzyxBaseRtWorld.tail<3>());
  }
  vector3_t getEulerzyxBaseRtWorld(const vector_t& state) const override {
    return eulerZyxFromQuaternion(getQuatBaseRtWorldXyzw(state));
  }
  void setEulerzyxBaseRtWorld(vector_t& state,
                              const vector3_t& eulerZyx) const override {
    state.segment<4>(orientationStart()) = quaternionFromEulerZyx(eulerZyx);
  }

  vector3_t getAngvelBaseRtWorldEwrtBase(const vector_t& state) const override {
    if (!angularVelocityStart_.has_value()) {
      failWithoutBaseAngularVelocity("getAngvelBaseRtWorldEwrtBase");
      return vector3_t::Zero();
    }
    return state.segment<3>(static_cast<Eigen::Index>(*angularVelocityStart_));
  }
  void setAngvelBaseRtWorldEwrtBase(
      vector_t& state,
      const vector3_t& angvelBaseRtWorldEwrtBase) const override {
    if (!angularVelocityStart_.has_value()) {
      failWithoutBaseAngularVelocity("setAngvelBaseRtWorldEwrtBase");
      return;
    }
    state.segment<3>(static_cast<Eigen::Index>(*angularVelocityStart_)) =
        angvelBaseRtWorldEwrtBase;
  }
  void setTurnAngvelBaseRtWorldEwrtBase(vector_t& state,
                                        scalar_t yawRate) const override {
    if (!angularVelocityStart_.has_value()) {
      failWithoutBaseAngularVelocity("setTurnAngvelBaseRtWorldEwrtBase");
      return;
    }
    state.segment<3>(static_cast<Eigen::Index>(*angularVelocityStart_)) =
        quaternionRotationMatrix<scalar_t>(getQuatBaseRtWorldXyzw(state))
            .transpose() *
        vector3_t(0.0, 0.0, yawRate);
  }
  BaseAngularVelocityJacobian getJacobianAngvelBaseRtWorldEwrtBase(
      const vector_t& /*state*/) const override {
    // The state stores w_B itself.
    const bool holds = angularVelocityStart_.has_value();
    if (!holds)
      failWithoutBaseAngularVelocity("getJacobianAngvelBaseRtWorldEwrtBase");
    return BaseAngularVelocityJacobian{
        .orientationSlots = matrix_t::Zero(3, kQuaternionOrientationStateDim),
        .angularVelocitySlots = holds ? matrix3_t(matrix3_t::Identity())
                                      : matrix3_t(matrix3_t::Zero())};
  }

  vector6_t getTwistBaseRtWorldEwrtWorld(
      const vector_t& state,
      const vector_t& /*jointVelocities*/) const override {
    if (isCentroidal()) return vector6_t::Zero();
    vector6_t twistBaseRtWorldEwrtWorld;
    twistBaseRtWorldEwrtWorld << state.segment<3>(baseLinearVelocityStart()),
        quaternionRotationMatrix<scalar_t>(getQuatBaseRtWorldXyzw(state)) *
            getAngvelBaseRtWorldEwrtBase(state);
    return twistBaseRtWorldEwrtWorld;
  }
  vector3_t getTargetPlanarCommand(
      const vector_t& targetState,
      scalar_t /*compositeYawInertia*/) const override {
    if (isCentroidal()) return vector3_t::Zero();
    const vector6_t twistBaseRtWorldEwrtWorld =
        getTwistBaseRtWorldEwrtWorld(targetState, vector_t::Zero(jointDim()));
    return vector3_t(twistBaseRtWorldEwrtWorld(0), twistBaseRtWorldEwrtWorld(1),
                     twistBaseRtWorldEwrtWorld(5));
  }

  absl::StatusOr<vector_t> stateFromTuning(
      const vector_t& tuningVector) const override {
    return stateFromTuningLayout(tuningVector, getTuningLayout());
  }
  absl::StatusOr<vector_t> tuningFromState(
      const vector_t& state) const override {
    return tuningLayoutFromState(state, getTuningLayout());
  }
  vector_t interpolateStateTrajectory(
      scalar_t time,
      const scalar_array_t& times,
      const vector_array_t& states) const override {
    return interpolateWithSlerpedQuatBaseRtWorld(time, times, states);
  }
  vector_t getNeutralState() const override {
    vector_t state = vector_t::Zero(static_cast<Eigen::Index>(getStateDim()));
    state.segment<4>(orientationStart()) = vector4_t(0.0, 0.0, 0.0, 1.0);
    return state;
  }

  matrix_t getBaseRotationTangentJacobian(
      const vector_t& state) const override {
    return configurationTangentLift(
        vector4_t(state.segment<4>(orientationStart())));
  }
  matrix_t getPinocchioTangentJacobian(const vector_t& state) const override {
    // q = [p_W, xi_hat, q_j]: the position and the joints are their own
    // tangent, the quaternion's is E^+(xi) dxi.
    matrix_t jacobian =
        matrix_t::Zero(static_cast<Eigen::Index>(getVelocityDim()),
                       static_cast<Eigen::Index>(getStateDim()));
    jacobian.block<3, 3>(0, baseStart()).setIdentity();
    jacobian.block<3, 4>(3, orientationStart()) =
        getBaseRotationTangentJacobian(state);
    jacobian.block(kBaseVelocityDim, jointStart(), jointDim(), jointDim())
        .setIdentity();
    return jacobian;
  }
  void fillGeneralizedCoordinatesFromMeasurement(
      const vector3_t& posBaseRtWorldEwrtWorld,
      const vector4_t& quatBaseRtWorldXyzw,
      const vector_t& jointPositions,
      Eigen::Ref<vector_t> q) const override {
    q.head<3>() = posBaseRtWorldEwrtWorld;
    q.segment<4>(kBaseTranslationDim) =
        safelyNormalizedQuaternion(quatBaseRtWorldXyzw);
    q.tail(jointDim()) = jointPositions;
  }
  void fillGeneralizedVelocitiesFromMeasurement(
      const vector4_t& quatBaseRtWorldXyzw,
      const vector3_t& linvelBaseRtWorldEwrtBase,
      const vector3_t& angvelBaseRtWorldEwrtBase,
      const vector_t& jointVelocities,
      Eigen::Ref<vector_t> v) const override {
    v.head<3>() =
        quaternion_t(safelyNormalizedQuaternion(quatBaseRtWorldXyzw)) *
        linvelBaseRtWorldEwrtBase;
    v.segment<3>(kBaseTranslationDim) = angvelBaseRtWorldEwrtBase;
    v.tail(jointDim()) = jointVelocities;
  }

  vector6_t getWrenchContactEwrtParam(const vector_t& input,
                                      size_t contactIndex) const override {
    return input.segment<6>(wrenchStart(contactIndex));
  }
  vector3_t getForceContactEwrtParam(const vector_t& input,
                                     size_t contactIndex) const override {
    return input.segment<3>(wrenchStart(contactIndex));
  }
  vector3_t getTorqueContactEwrtParam(const vector_t& input,
                                      size_t contactIndex) const override {
    return input.segment<3>(wrenchStart(contactIndex) + 3);
  }
  void setWrenchContactEwrtParam(vector_t& input,
                                 const vector6_t& wrenchContactEwrtParam,
                                 size_t contactIndex) const override {
    input.segment<6>(wrenchStart(contactIndex)) = wrenchContactEwrtParam;
  }
  void setForceContactEwrtParam(vector_t& input,
                                const vector3_t& forceContactEwrtParam,
                                size_t contactIndex) const override {
    input.segment<3>(wrenchStart(contactIndex)) = forceContactEwrtParam;
  }
  void setTorqueContactEwrtParam(vector_t& input,
                                 const vector3_t& torqueContactEwrtParam,
                                 size_t contactIndex) const override {
    input.segment<3>(wrenchStart(contactIndex) + 3) = torqueContactEwrtParam;
  }

 private:
  // For clone(): every member copied, the reference to the ModelSettings
  // included.
  QuaternionLayoutRobotModel(const QuaternionLayoutRobotModel& other) = default;

  bool isCentroidal() const {
    return mpc_ == StateInputLayout::Mpc::kCentroidal;
  }
  Eigen::Index baseStart() const {
    return static_cast<Eigen::Index>(getBaseStartindex());
  }
  Eigen::Index orientationStart() const {
    return static_cast<Eigen::Index>(orientationStart_);
  }
  Eigen::Index jointStart() const {
    return static_cast<Eigen::Index>(getJointStartindex());
  }
  Eigen::Index jointDim() const {
    return static_cast<Eigen::Index>(getJointDim());
  }
  Eigen::Index jointVelocitiesStart() const {
    return static_cast<Eigen::Index>(getJointVelocitiesStartindex());
  }
  // The whole body's dp_W, after q.
  Eigen::Index baseLinearVelocityStart() const {
    return static_cast<Eigen::Index>(getConfigurationDim());
  }
  Eigen::Index wrenchStart(size_t contactIndex) const {
    return static_cast<Eigen::Index>(
        getContactWrenchStartIndices(contactIndex));
  }

  /** Fails the test, naming `caller`: the state holds no base angular velocity.
   */
  static void failWithoutBaseAngularVelocity(const char* absl_nonnull caller) {
    ADD_FAILURE() << caller
                  << " of a state that holds no base angular velocity";
  }

  StateInputLayout::Mpc mpc_;
  // The first state slot of the quaternion; 0 on a layout without an
  // orientation block.
  size_t orientationStart_ = 0;
  // The first state slot of the whole body's w_B; nullopt on the centroidal
  // layout.
  std::optional<size_t> angularVelocityStart_;
};

}  // namespace ocs2::humanoid::test
