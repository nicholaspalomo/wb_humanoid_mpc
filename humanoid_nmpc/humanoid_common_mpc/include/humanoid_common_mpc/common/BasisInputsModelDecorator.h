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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <array>
#include <cassert>
#include <memory>

#include <Eigen/Core>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::humanoid {

// Spatial wrench/force/moment dimensions.
static constexpr size_t kWrenchDim = 6;
static constexpr size_t kForceDim = 3;
static constexpr size_t kMomentDim = 3;

/**
 * A decorator that wraps any MpcRobotModelBase and re-parameterizes the contact
 * wrench portion of the input vector from physical wrenches [F, τ] ∈ ℝ⁶ to
 * non-negative basis-vector scalings λ ∈ ℝᴺ⁺, where W_local = B · λ.
 *
 * Frames
 * ------
 * The basis matrix B of each contact is built by ContactWrenchConeBasisMatrix in the
 * *local contact frame* (its CoP rays use the foot-frame footprint bounds). Therefore
 *   - the input-only accessors getContactWrench/Force/Moment(input, i) return the wrench
 *     expressed in the local contact frame, and setContactWrench/Force/Moment(input, W, i)
 *     expect a local-frame wrench;
 *   - the state-aware accessors get/setContact...InWorldFrame(state, input, ...) rotate
 *     between the local contact frame and the world frame using the contact frame
 *     orientation obtained from the robot configuration (state) via Pinocchio.
 * Every consumer that needs a world-frame wrench (dynamics, inverse dynamics, external
 * torque costs, visualization, telemetry) must use the state-aware accessors.
 *
 * The wrapped model's state accessors are forwarded unchanged.
 * The input dimension changes: the 6 wrench variables per contact are replaced
 * by numBasisPerFoot_ scaling variables.
 *
 * Template parameter SCALAR_T must match the wrapped model.
 */
template <typename SCALAR_T>
class BasisInputsModelDecorator : public MpcRobotModelBase<SCALAR_T> {
 public:
  using Base = MpcRobotModelBase<SCALAR_T>;

  /**
   * @param wrappedModel       The model to decorate (ownership transferred).
   * @param basisMatrices      Per-contact basis matrices (in the local contact frame).
   *                           Must have exactly N_CONTACTS elements.
   * @param pinocchioInterface Pinocchio interface used to evaluate the contact frame
   *                           orientations for the world-frame accessors (copied).
   */
  BasisInputsModelDecorator(std::unique_ptr<MpcRobotModelBase<SCALAR_T>> wrappedModel,
                            const std::array<ContactWrenchConeBasisMatrix, N_CONTACTS>& basisMatrices,
                            const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface)
      : Base(wrappedModel->modelSettings, wrappedModel->getStateDim(), computeInputDim(basisMatrices, wrappedModel->modelSettings)),
        wrappedModel_(std::move(wrappedModel)),
        pinocchioInterface_(pinocchioInterface),
        numBasisPerFoot_(basisMatrices[0].numBasis()) {
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      assert(basisMatrices[i].numBasis() == numBasisPerFoot_);
      B_local_[i] = basisMatrices[i].getBasisMatrix();
      B_pinv_local_[i] = basisMatrices[i].getBasisMatrixPseudoInverse();
      contactFrameIndices_[i] = ocs2::humanoid::getContactFrameIndex<SCALAR_T>(pinocchioInterface_, *wrappedModel_, i);
    }
  }

  ~BasisInputsModelDecorator() override = default;
  BasisInputsModelDecorator* clone() const override { return new BasisInputsModelDecorator(*this); }

  /** Number of basis scalings per foot. */
  size_t getNumBasisPerFoot() const { return numBasisPerFoot_; }

  /** Input dimension of the wrapped (wrench-space) model. */
  size_t getWrenchInputDim() const { return wrappedModel_->getInputDim(); }

  /** Local-frame basis matrix (6 × numBasisPerFoot) for the given contact. */
  const matrix_t& getBasisMatrix(size_t contactIndex) const { return B_local_[contactIndex]; }

  /** All local-frame basis matrices. */
  const std::array<matrix_t, N_CONTACTS>& getBasisMatrices() const { return B_local_; }

  /** Local-frame pseudoinverse of the basis matrix. */
  const matrix_t& getBasisMatrixPseudoInverse(size_t contactIndex) const { return B_pinv_local_[contactIndex]; }

  /** Pinocchio frame index of the given contact. */
  pinocchio::FrameIndex getPinocchioContactFrameIndex(size_t contactIndex) const { return contactFrameIndices_[contactIndex]; }

  /**
   * Constant block-diagonal map from the basis-vector input to the wrapped model's wrench-space input,
   * with each contact wrench expressed in its *local* contact frame:
   *
   *   u_wrench_local = M · u_basis,   M = blkdiag(B_0, B_1, ..., I_{n_joints})   (wrenchInputDim × basisInputDim)
   *
   * This map is frame-independent only for the joint-velocity block. It is intended for quantities that are
   * naturally defined in the contact frame (e.g. the input regularization cost R). For the dynamics the
   * contact blocks must additionally be rotated into the world frame, see CentroidalDynamicsBasisInputsAD.
   */
  matrix_t getLocalBasisToWrenchMap() const {
    const size_t wrenchInputDim = wrappedModel_->getInputDim();
    const size_t jointDim = this->modelSettings.mpc_joint_dim;
    matrix_t M = matrix_t::Zero(wrenchInputDim, this->input_dim);
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      M.block(wrappedModel_->getContactWrenchStartIndices(i), getContactWrenchStartIndices(i), kWrenchDim, numBasisPerFoot_) = B_local_[i];
    }
    M.block(wrappedModel_->getJointVelocitiesStartindex(), getJointVelocitiesStartindex(), jointDim, jointDim).setIdentity();
    return M;
  }

  /******* Start indices *******/

  size_t getBaseStartindex() const override { return wrappedModel_->getBaseStartindex(); }
  size_t getJointStartindex() const override { return wrappedModel_->getJointStartindex(); }

  /** Joint velocities/accelerations start after the basis-vector block. */
  size_t getJointVelocitiesStartindex() const override { return numBasisPerFoot_ * N_CONTACTS; }

  /** Start index of the λ-block for the given contact. */
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return numBasisPerFoot_ * contactIndex; }
  size_t getContactForceStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex); }
  size_t getContactMomentStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex); }

  /******* Generalized coordinates (delegated to wrapped model) *******/

  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& state) const override {
    return wrappedModel_->getGeneralizedCoordinates(state);
  }
  VECTOR6_T<SCALAR_T> getBasePose(const VECTOR_T<SCALAR_T>& state) const override { return wrappedModel_->getBasePose(state); }
  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& state) const override { return wrappedModel_->getBasePosition(state); }
  VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(const VECTOR_T<SCALAR_T>& state) const override {
    return wrappedModel_->getBaseOrientationEulerZYX(state);
  }
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& state) const override {
    return wrappedModel_->getBaseComLinearVelocity(state);
  }
  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& state) const override {
    return wrappedModel_->getBaseComVelocity(state);
  }
  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& state) const override { return wrappedModel_->getJointAngles(state); }
  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) const override {
    assert(input.size() == this->input_dim);
    return input.tail(this->modelSettings.mpc_joint_dim);
  }
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) override {
    assert(input.size() == this->input_dim);
    // The wrapped model expects a wrench-space input, but only reads the joint-velocity block
    // to build the generalized velocities. The contact block is therefore left at zero.
    VECTOR_T<SCALAR_T> wrenchInput = VECTOR_T<SCALAR_T>::Zero(wrappedModel_->getInputDim());
    wrenchInput.tail(this->modelSettings.mpc_joint_dim) = input.tail(this->modelSettings.mpc_joint_dim);
    return wrappedModel_->getGeneralizedVelocities(state, wrenchInput);
  }

  /******* Setters (delegated to wrapped model) *******/

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& generalizedCoordinates) const override {
    wrappedModel_->setGeneralizedCoordinates(state, generalizedCoordinates);
  }
  void setBasePose(VECTOR_T<SCALAR_T>& state, const VECTOR6_T<SCALAR_T>& basePose) const override {
    wrappedModel_->setBasePose(state, basePose);
  }
  void setBasePosition(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& position) const override {
    wrappedModel_->setBasePosition(state, position);
  }
  void setBaseOrientationEulerZYX(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& eulerAnglesZYX) const override {
    wrappedModel_->setBaseOrientationEulerZYX(state, eulerAnglesZYX);
  }
  void setJointAngles(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& jointAngles) const override {
    wrappedModel_->setJointAngles(state, jointAngles);
  }
  void setJointVelocities(VECTOR_T<SCALAR_T>& state, VECTOR_T<SCALAR_T>& input, const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    assert(input.size() == this->input_dim);
    input.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }
  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state, scalar_t heightChange) const override {
    wrappedModel_->adaptBasePoseHeight(state, heightChange);
  }

  /******* Contact wrench accessors in the LOCAL contact frame — the core of the decoration *******/

  /**
   * Reconstructs the full 6D wrench from basis-vector scalings: W_local = B * λ.
   * The returned wrench is expressed in the *local contact frame*. Use
   * getContactWrenchInWorldFrame(state, input, i) to obtain the world-frame wrench.
   */
  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    assert(input.size() == this->input_dim);
    const VECTOR_T<SCALAR_T> lambda = input.segment(getContactWrenchStartIndices(contactIndex), numBasisPerFoot_);
    return B_local_[contactIndex].template cast<SCALAR_T>() * lambda;
  }

  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return getContactWrench(input, contactIndex).template head<kForceDim>();
  }

  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return getContactWrench(input, contactIndex).template tail<kMomentDim>();
  }

  /**
   * Sets a *local-frame* wrench by computing λ = max(0, B⁺ * W_local) and writing it into the input vector.
   * The pseudoinverse gives the minimum-norm λ, but it can produce negative scalings
   * (e.g., torsion rays canceling each other for a pure vertical force). We clamp to
   * zero to maintain the λ ≥ 0 structural constraint. The resulting wrench W' = B * λ'
   * is an approximation of the requested wrench, but is always inside the friction cone.
   * The MPC solver refines from this feasible starting point.
   *
   * Use setContactWrenchInWorldFrame(state, input, W_world, i) for a world-frame wrench.
   */
  void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const override {
    assert(input.size() == this->input_dim);
    VECTOR_T<SCALAR_T> lambda = B_pinv_local_[contactIndex].template cast<SCALAR_T>() * wrench;
    // Clamp negative scalings to zero — the pseudoinverse does not guarantee
    // non-negativity, but the basis-vector formulation requires λ ≥ 0.
    lambda = lambda.cwiseMax(static_cast<SCALAR_T>(0));
    input.segment(getContactWrenchStartIndices(contactIndex), numBasisPerFoot_) = lambda;
  }

  void setContactForce(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& force, size_t contactIndex) const override {
    VECTOR6_T<SCALAR_T> wrench = VECTOR6_T<SCALAR_T>::Zero();
    wrench.template head<kForceDim>() = force;
    setContactWrench(input, wrench, contactIndex);
  }

  void setContactMoment(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& moment, size_t contactIndex) const override {
    VECTOR6_T<SCALAR_T> wrench = VECTOR6_T<SCALAR_T>::Zero();
    wrench.template tail<kMomentDim>() = moment;
    setContactWrench(input, wrench, contactIndex);
  }

  /******* State-aware contact wrench accessors in the WORLD frame *******/

  /**
   * Rotation matrix from the local contact frame to the world frame, w_R_l, evaluated at the
   * configuration contained in the given state.
   */
  MATRIX3_T<SCALAR_T> getContactFrameRotationLocalToWorld(const VECTOR_T<SCALAR_T>& state, size_t contactIndex) const {
    const auto& model = pinocchioInterface_.getModel();
    // Local copy keeps this method const and safe to call from multiple threads.
    pinocchio::DataTpl<SCALAR_T> data = pinocchioInterface_.getData();
    updateFramePlacements(wrappedModel_->getGeneralizedCoordinates(state), model, data);
    return MATRIX3_T<SCALAR_T>(getRotationMatrixLocalToWorld(data, contactFrameIndices_[contactIndex]));
  }

  /** Rotates a local-frame wrench into the world frame using the contact frame orientation at the given state. */
  VECTOR6_T<SCALAR_T> rotateWrenchLocalToWorld(const VECTOR_T<SCALAR_T>& state,
                                               const VECTOR6_T<SCALAR_T>& wrenchLocal,
                                               size_t contactIndex) const {
    const MATRIX3_T<SCALAR_T> w_R_l = getContactFrameRotationLocalToWorld(state, contactIndex);
    VECTOR6_T<SCALAR_T> wrenchWorld;
    wrenchWorld.template head<kForceDim>() = w_R_l * wrenchLocal.template head<kForceDim>();
    wrenchWorld.template tail<kMomentDim>() = w_R_l * wrenchLocal.template tail<kMomentDim>();
    return wrenchWorld;
  }

  /** Rotates a world-frame wrench into the local contact frame using the contact frame orientation at the given state. */
  VECTOR6_T<SCALAR_T> rotateWrenchWorldToLocal(const VECTOR_T<SCALAR_T>& state,
                                               const VECTOR6_T<SCALAR_T>& wrenchWorld,
                                               size_t contactIndex) const {
    const MATRIX3_T<SCALAR_T> l_R_w = getContactFrameRotationLocalToWorld(state, contactIndex).transpose();
    VECTOR6_T<SCALAR_T> wrenchLocal;
    wrenchLocal.template head<kForceDim>() = l_R_w * wrenchWorld.template head<kForceDim>();
    wrenchLocal.template tail<kMomentDim>() = l_R_w * wrenchWorld.template tail<kMomentDim>();
    return wrenchLocal;
  }

  VECTOR6_T<SCALAR_T> getContactWrenchInWorldFrame(const VECTOR_T<SCALAR_T>& state,
                                                   const VECTOR_T<SCALAR_T>& input,
                                                   size_t contactIndex) const override {
    return rotateWrenchLocalToWorld(state, getContactWrench(input, contactIndex), contactIndex);
  }

  VECTOR3_T<SCALAR_T> getContactForceInWorldFrame(const VECTOR_T<SCALAR_T>& state,
                                                  const VECTOR_T<SCALAR_T>& input,
                                                  size_t contactIndex) const override {
    return getContactWrenchInWorldFrame(state, input, contactIndex).template head<kForceDim>();
  }

  VECTOR3_T<SCALAR_T> getContactMomentInWorldFrame(const VECTOR_T<SCALAR_T>& state,
                                                   const VECTOR_T<SCALAR_T>& input,
                                                   size_t contactIndex) const override {
    return getContactWrenchInWorldFrame(state, input, contactIndex).template tail<kMomentDim>();
  }

  void setContactWrenchInWorldFrame(const VECTOR_T<SCALAR_T>& state,
                                    VECTOR_T<SCALAR_T>& input,
                                    const VECTOR6_T<SCALAR_T>& wrenchInWorld,
                                    size_t contactIndex) const override {
    setContactWrench(input, rotateWrenchWorldToLocal(state, wrenchInWorld, contactIndex), contactIndex);
  }

  void setContactForceInWorldFrame(const VECTOR_T<SCALAR_T>& state,
                                   VECTOR_T<SCALAR_T>& input,
                                   const VECTOR3_T<SCALAR_T>& forceInWorld,
                                   size_t contactIndex) const override {
    VECTOR6_T<SCALAR_T> wrenchInWorld = VECTOR6_T<SCALAR_T>::Zero();
    wrenchInWorld.template head<kForceDim>() = forceInWorld;
    setContactWrenchInWorldFrame(state, input, wrenchInWorld, contactIndex);
  }

 private:
  BasisInputsModelDecorator(const BasisInputsModelDecorator& rhs)
      : Base(rhs),
        wrappedModel_(std::unique_ptr<MpcRobotModelBase<SCALAR_T>>(rhs.wrappedModel_->clone())),
        pinocchioInterface_(rhs.pinocchioInterface_),
        numBasisPerFoot_(rhs.numBasisPerFoot_),
        B_local_(rhs.B_local_),
        B_pinv_local_(rhs.B_pinv_local_),
        contactFrameIndices_(rhs.contactFrameIndices_) {}

  static size_t computeInputDim(const std::array<ContactWrenchConeBasisMatrix, N_CONTACTS>& basisMatrices,
                                const ModelSettings& modelSettings) {
    return basisMatrices[0].numBasis() * N_CONTACTS + modelSettings.mpc_joint_dim;
  }

  std::unique_ptr<MpcRobotModelBase<SCALAR_T>> wrappedModel_;
  PinocchioInterfaceTpl<SCALAR_T> pinocchioInterface_;
  const size_t numBasisPerFoot_;

  std::array<matrix_t, N_CONTACTS> B_local_;
  std::array<matrix_t, N_CONTACTS> B_pinv_local_;
  std::array<pinocchio::FrameIndex, N_CONTACTS> contactFrameIndices_;
};

}  // namespace ocs2::humanoid
