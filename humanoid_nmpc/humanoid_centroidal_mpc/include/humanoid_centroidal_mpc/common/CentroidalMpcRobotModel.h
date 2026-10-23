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
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
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

  Define the state vector x = [h, q_b, q_j]^T (13 + nj entries; its tangent has 12 + nj)

    Centroidal momentum h = [vcom_x, vcom_y, vcom_z, L_x / mass, L_y / mass, L_z / mass]^T
    Base pose q_b = [p_base_x, p_base_y, p_base_z, xi_x, xi_y, xi_z, xi_w]: the position in the world and the unit
      quaternion from the base frame to the world (Pinocchio's Composite(Translation, Spherical) root)
    Joint angles q_j

*/
/******************************************************************************************************/

/******************************************************************************************************/
/* Input Vector definition

  Define the input vector u = [W_l, W_r, q_dot_j]^T

    Left contact Wrench W_l in inertial frame
    Right contact Wrench W_r in inertial frame
    Joint velocities q_dot_j

*/
/******************************************************************************************************/

/**
 * The state and input layout of the centroidal MPC defined above, for the scalar type of the solver or of CppAD: the
 * normalized centroidal momentum, the base pose and the joint angles of the state; the contact wrenches and the joint
 * velocities of the input. The base orientation is stored as a quaternion (x, y, z, w), so the state lives on the product
 * manifold [E(9), unit quaternion, E(nj)] (stateManifoldOf()) and the tuning layout maps onto it through the Euler
 * boundary (orientation/EulerBoundary.h). getGeneralizedVelocities() updates the model's own copy of the Pinocchio
 * interface, so a model is not thread-safe; clone() one for each thread. Every accessor takes a state of getStateDim(),
 * an input of getInputDim() and generalized coordinates of getConfigurationDim() = 7 + mpc_joint_dim entries, unchecked:
 * the solver and the control thread call them on every evaluation. The state holds no
 * base velocity: the base angular velocity accessors abort, and getBaseTwistInWorldFrame() needs the joint velocities
 * besides the state, whose size it checks (once per restart of the motion manager's stride history, not on every
 * evaluation). It and getTargetPlanarCommand() need the model in double precision and abort in the CppAD
 * instantiation, which no caller of theirs uses.
 */
template <typename SCALAR_T>
class CentroidalMpcRobotModel : public MpcRobotModelBase<SCALAR_T> {
 public:
  // The state is the centroidal model's, x = [h, p_W, xi, q_j] (ocs2_centroidal_model/CentroidalModelInfo.h): the
  // normalized centroidal momentum h = [v_com, L / mass] (centroidal_model::kCentroidalMomentumDim rows, those of the
  // momentum matrix), the base position, the base quaternion (x, y, z, w), then the joint angles.

  CentroidalMpcRobotModel(const ModelSettings& modelSettings,
                          const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                          const CentroidalModelInfoTpl<SCALAR_T>& centroidalModelInfo)
      : MpcRobotModelBase<SCALAR_T>(
            modelSettings,
            MpcRobotModelDimensions{.stateDim = centroidal_model::kJointAnglesStartInState + modelSettings.mpc_joint_dim,
                                    .inputDim = kWrenchDim * kNumContacts + modelSettings.mpc_joint_dim,
                                    .configurationDim = kBaseConfigurationDim + modelSettings.mpc_joint_dim,
                                    .velocityDim = kBaseVelocityDim + modelSettings.mpc_joint_dim},
            placeTuningLayout(tuningLayoutOf(stateInputLayout(modelSettings, StateInputLayout::Mpc::kCentroidal)))),
        pinocchioInterface_(pinocchioInterface),
        centroidalModelInfo_(centroidalModelInfo),
        pinocchioMappingPtr_(std::make_unique<CentroidalModelPinocchioMappingTpl<SCALAR_T>>(centroidalModelInfo)) {
    pinocchioMappingPtr_->setPinocchioInterface(pinocchioInterface_);
  }

  ~CentroidalMpcRobotModel() override = default;
  CentroidalMpcRobotModel* absl_nonnull clone() const override { return new CentroidalMpcRobotModel(*this); }
  // Copied only by clone(), whose copy constructor is private; never assigned or moved.
  CentroidalMpcRobotModel& operator=(const CentroidalMpcRobotModel&) = delete;
  CentroidalMpcRobotModel(CentroidalMpcRobotModel&&) = delete;
  CentroidalMpcRobotModel& operator=(CentroidalMpcRobotModel&&) = delete;

  /******************************************************************************************************/
  /*                                          Start indices                                             */
  /******************************************************************************************************/

  size_t getBaseStartindex() const override { return centroidal_model::kBasePositionStartInState; }
  size_t getBaseOrientationStartIndex() const override { return centroidal_model::kBaseQuaternionStartInState; }
  size_t getJointStartindex() const override { return centroidal_model::kJointAnglesStartInState; }
  // LINT.IfChange(input_indices)
  size_t getJointVelocitiesStartindex() const override { return kWrenchDim * kNumContacts; }

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return kWrenchDim * contactIndex; }
  size_t getContactForceStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex); }
  size_t getContactMomentStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex) + kForceDim; }
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputWeightsFromConfig.cpp:input_offsets)

  /******************************************************************************************************/
  /*                                     Generalized coordinates                                        */
  /******************************************************************************************************/

  /** q = [p_W, xi_hat, q_j] with the quaternion safely normalized (CentroidalModelPinocchioMapping::getPinocchioJointPosition()). */
  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& state) const override {
    return pinocchioMappingPtr_->getPinocchioJointPosition(state);
  }

  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment(centroidal_model::kBasePositionStartInState, kPositionDim);
  }

  // Return Com linear velocity
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& state) const override { return state.head(kPositionDim); }

  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& state) const override {
    return state.head(centroidal_model::kCentroidalMomentumDim);
  }

  void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& velocity) const override {
    state.head(kPositionDim) = velocity;
  }

  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& state) const override {
    return state.tail(this->modelSettings.mpc_joint_dim);
  }

  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& input) const override {
    return input.tail(this->modelSettings.mpc_joint_dim);
  }

  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) override {
    updateCentroidalDynamics<SCALAR_T>(pinocchioInterface_, centroidalModelInfo_, pinocchioMappingPtr_->getPinocchioJointPosition(state));
    return pinocchioMappingPtr_->getPinocchioJointVelocity(state, input);
  }

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const override {
    state.tail(this->getConfigurationDim()) = generalizedCorrdinates;
  }

  void setBasePosition(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& position) const override {
    state.segment(centroidal_model::kBasePositionStartInState, kPositionDim) = position;
  }

  void setJointAngles(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& jointAngles) const override {
    state.tail(this->modelSettings.mpc_joint_dim) = jointAngles;
  }

  void setJointVelocities(VECTOR_T<SCALAR_T>& /*state*/,
                          VECTOR_T<SCALAR_T>& input,
                          const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    input.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }

  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state, scalar_t heightChange) const override {
    state[centroidal_model::kBasePositionStartInState + kZAxisIndex] += heightChange;
  }

  /******************************************************************************************************/
  /*                   Layout-agnostic accessors (MpcRobotModelBase), the quaternion layout             */
  /******************************************************************************************************/

  VECTOR4_T<SCALAR_T> getBaseOrientation(const VECTOR_T<SCALAR_T>& state) const override {
    return safelyNormalizedQuaternion(VECTOR4_T<SCALAR_T>(state.template segment<kQuaternionCoefficients>(getBaseOrientationStartIndex())));
  }

  void setBaseOrientation(vector_t& state, const vector4_t& orientation) const override {
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) = safelyNormalizedQuaternion(orientation);
  }

  scalar_t getBaseHeading(const vector_t& state) const override { return twistHeading(baseOrientationOf(state)); }

  // NOLINTBEGIN(euler-boundary): Euler boundary accessor (design section 4.2)
  vector6_t getBasePoseEulerZyx(const vector_t& state) const override {
    vector6_t basePose;
    basePose << state.segment<kPositionDim>(getBaseStartindex()), getBaseOrientationEulerZyx(state);
    return basePose;
  }

  void setBasePoseEulerZyx(vector_t& state, const vector6_t& basePose) const override {
    state.segment<kPositionDim>(getBaseStartindex()) = basePose.head<kPositionDim>();
    setBaseOrientationEulerZyx(state, basePose.tail<kEulerAnglesZyxDim>());
  }

  vector3_t getBaseOrientationEulerZyx(const vector_t& state) const override { return eulerZyxFromQuaternion(baseOrientationOf(state)); }

  void setBaseOrientationEulerZyx(vector_t& state, const vector3_t& eulerZyx) const override {
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) = quaternionFromEulerZyx(eulerZyx);
  }
  // NOLINTEND(euler-boundary)

  vector3_t getBaseAngularVelocityLocal(const vector_t& /*state*/) const override {
    abortWithoutBaseAngularVelocity("getBaseAngularVelocityLocal");
  }

  void setBaseAngularVelocityLocal(vector_t& /*state*/, const vector3_t& /*angularVelocityLocal*/) const override {
    abortWithoutBaseAngularVelocity("setBaseAngularVelocityLocal");
  }

  void setBaseAngularVelocityFromWorldYawRate(vector_t& /*state*/, scalar_t /*yawRate*/) const override {
    abortWithoutBaseAngularVelocity("setBaseAngularVelocityFromWorldYawRate");
  }

  BaseAngularVelocityJacobian getBaseAngularVelocityLocalJacobian(const vector_t& /*state*/) const override {
    abortWithoutBaseAngularVelocity("getBaseAngularVelocityLocalJacobian");
  }

  /**
   * [v_W, w_W] of the base velocity Pinocchio's v starts with, v_b = [dp_W, w_B] = A_b^-1 (m h - A_j dq_j), the base
   * block of the centroidal momentum map m h = A_b v_b + A_j dq_j solved with the joint velocities `jointVelocities` =
   * dq_j (mpc_joint_dim entries): CentroidalModelPinocchioMapping::getPinocchioJointVelocity()'s formula, on a copy of the
   * Pinocchio data. A_b = [[m I, -m [c - p]x R], [0, I_G R]] is block upper triangular, so the angular part is w_B = R^T
   * I_G^-1 (L - (A_j dq_j)_ang) and w_W = R w_B = I_G^-1 (L - (A_j dq_j)_ang). Under single rigid body dynamics A_j is
   * zero and the joint velocities drop out: the twist is the rigid body's, w_W = (R I_nom R')^-1 L and dp_W = h_lin + w_W
   * x R r_nom, I_nom and r_nom the nominal inertia about the CoM and the CoM-to-base offset of CentroidalModelInfo. One
   * centroidal momentum matrix per call. Dies (ABSL_CHECK) on `jointVelocities` of another size than mpc_joint_dim, whose
   * product with A_j would otherwise read only the columns of the joints it holds.
   */
  vector6_t getBaseTwistInWorldFrame(const vector_t& state, const vector_t& jointVelocities) const override {
    ABSL_CHECK_EQ(static_cast<size_t>(jointVelocities.size()), this->getJointDim())
        << "CentroidalMpcRobotModel::getBaseTwistInWorldFrame: one joint velocity per MPC joint";
    if constexpr (std::is_same_v<SCALAR_T, scalar_t>) {
      // On a copy of the Pinocchio data, so that the function is const and safe to call concurrently.
      PinocchioInterfaceTpl<SCALAR_T> pinocchioInterface = pinocchioInterface_;
      updateCentroidalDynamics<SCALAR_T>(pinocchioInterface, centroidalModelInfo_, getGeneralizedCoordinates(state));
      const Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, Eigen::Dynamic>& momentumMatrix =
          getCentroidalMomentumMatrix<SCALAR_T>(pinocchioInterface);
      const Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, kPoseTangentDim> baseMomentumMatrix =
          momentumMatrix.template leftCols<kPoseTangentDim>();
      const vector6_t baseMomentum = centroidalModelInfo_.robotMass * getCentroidalMomentum(state) -
                                     momentumMatrix.rightCols(jointVelocities.size()) * jointVelocities;
      const vector6_t baseVelocity = computeFloatingBaseCentroidalMomentumMatrixInverse<scalar_t>(baseMomentumMatrix) * baseMomentum;
      vector6_t twist;
      twist.head<kPositionDim>() = baseVelocity.head<kPositionDim>();
      twist.tail<kRotationTangentDim>() =
          quaternionRotationMatrix<scalar_t>(baseOrientationOf(state)) * baseVelocity.tail<kRotationTangentDim>();
      return twist;
    } else {
      abortWithoutDoublePrecisionModel("getBaseTwistInWorldFrame");
    }
  }

  /**
   * (h_x, h_y, m h_z / compositeYawInertia) of the knot's normalized momentum: the CoM velocity, and the yaw rate
   * CentroidalMpcTargetTrajectoriesCalculator wrote into the angular part as I_zz psidot / m (0 while the inertia is not
   * positive).
   */
  vector3_t getTargetPlanarCommand(const vector_t& targetState, scalar_t compositeYawInertia) const override {
    if constexpr (std::is_same_v<SCALAR_T, scalar_t>) {
      const scalar_t yawRate = compositeYawInertia > 0.0
                                   ? centroidalModelInfo_.robotMass *
                                         targetState(centroidal_model::kAngularMomentumStartIndex + kZAxisIndex) / compositeYawInertia
                                   : 0.0;
      return vector3_t(targetState(0), targetState(1), yawRate);
    } else {
      abortWithoutDoublePrecisionModel("getTargetPlanarCommand");
    }
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
    state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex()) = vector4_t(0.0, 0.0, 0.0, 1.0);
    return state;
  }

  matrix_t getBaseRotationTangentJacobian(const vector_t& state) const override {
    return configurationTangentLift(vector4_t(state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex())));
  }

  matrix_t getPinocchioTangentJacobian(const vector_t& state) const override {
    // q = [p_W, xi_hat, q_j] of the slots from getBaseStartindex(): the position and the joints are their own tangent, the
    // quaternion's tangent is the body rotation E^+(xi) dxi.
    const Eigen::Index jointDim = static_cast<Eigen::Index>(this->getJointDim());
    matrix_t jacobian = matrix_t::Zero(this->getVelocityDim(), this->getStateDim());
    jacobian.block<kPositionDim, kPositionDim>(0, getBaseStartindex()).setIdentity();
    jacobian.block<kRotationTangentDim, kQuaternionCoefficients>(kPositionDim, getBaseOrientationStartIndex()) =
        getBaseRotationTangentJacobian(state);
    jacobian.block(kPoseTangentDim, getJointStartindex(), jointDim, jointDim).setIdentity();
    return jacobian;
  }

  void fillGeneralizedCoordinatesFromMeasurement(const vector3_t& positionWorld,
                                                 const vector4_t& orientation,
                                                 const vector_t& jointPositions,
                                                 Eigen::Ref<vector_t> q) const override {
    q.head<kPositionDim>() = positionWorld;
    q.segment<kQuaternionCoefficients>(kPositionDim) = safelyNormalizedQuaternion(orientation);
    q.tail(this->modelSettings.mpc_joint_dim) = jointPositions;
  }

  void fillGeneralizedVelocitiesFromMeasurement(const vector4_t& orientation,
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
  /*                                          Contacts                                                  */
  /******************************************************************************************************/

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex), kContactWrenchDim);
  }

  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex), kForceDim);
  }

  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment((getContactWrenchStartIndices(contactIndex) + kForceDim), kTorqueDim);
  }

  void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), kWrenchDim) = wrench;
  }

  void setContactForce(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& force, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), kForceDim) = force;
  }

  void setContactMoment(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& moment, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex) + kForceDim, kTorqueDim) = moment;
  }

  /******************************************************************************************************/
  /*                                    Custom Centroidal Methods                                       */
  /******************************************************************************************************/

  VECTOR_T<SCALAR_T> getCentroidalMomentum(const VECTOR_T<SCALAR_T>& state) const {
    return state.head(centroidal_model::kCentroidalMomentumDim);
  }

  const CentroidalModelInfoTpl<SCALAR_T>& getCentroidalModelInfo() const { return centroidalModelInfo_; }

 private:
  /** The base angular velocity accessors' answer: the centroidal state holds none, so calling one is a programming error. */
  [[noreturn]] static void abortWithoutBaseAngularVelocity(absl::string_view accessor) {
    ABSL_LOG(FATAL) << "CentroidalMpcRobotModel::" << accessor << ": the centroidal state holds no base angular velocity";
    ABSL_UNREACHABLE();
  }

  /**
   * The CppAD instantiation's answer to the accessors that evaluate the model in double precision, which it has only in
   * CppAD scalars: calling one is a programming error.
   */
  [[noreturn]] static void abortWithoutDoublePrecisionModel(absl::string_view accessor) {
    ABSL_LOG(FATAL) << "CentroidalMpcRobotModel::" << accessor << ": the CppAD model has no double-precision Pinocchio model";
    ABSL_UNREACHABLE();
  }

  /** getBaseOrientation() of a state in double precision, for every SCALAR_T. */
  vector4_t baseOrientationOf(const vector_t& state) const {
    return safelyNormalizedQuaternion(vector4_t(state.segment<kQuaternionCoefficients>(getBaseOrientationStartIndex())));
  }

  CentroidalMpcRobotModel(const CentroidalMpcRobotModel& rhs)
      : MpcRobotModelBase<SCALAR_T>(rhs),
        pinocchioInterface_(rhs.pinocchioInterface_),
        centroidalModelInfo_(rhs.centroidalModelInfo_),
        pinocchioMappingPtr_(rhs.pinocchioMappingPtr_->clone()) {
    pinocchioMappingPtr_->setPinocchioInterface(pinocchioInterface_);
  }

  PinocchioInterfaceTpl<SCALAR_T> pinocchioInterface_;
  const CentroidalModelInfoTpl<SCALAR_T> centroidalModelInfo_;
  const std::unique_ptr<CentroidalModelPinocchioMappingTpl<SCALAR_T>> pinocchioMappingPtr_;
};

}  // namespace ocs2::humanoid
