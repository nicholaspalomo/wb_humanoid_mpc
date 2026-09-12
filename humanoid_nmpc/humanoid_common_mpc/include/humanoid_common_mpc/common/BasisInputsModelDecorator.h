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

#include <array>
#include <cassert>

#include <Eigen/Core>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

namespace ocs2::humanoid {

// Spatial wrench/force/moment dimensions.
static constexpr size_t kWrenchDim = 6;
static constexpr size_t kForceDim = 3;
static constexpr size_t kMomentDim = 3;

/**
 * A decorator that wraps any MpcRobotModelBase and re-parameterizes the contact
 * wrench portion of the input vector from physical wrenches [F, τ] ∈ ℝ⁶ to
 * non-negative basis-vector scalings λ ∈ ℝᴺ⁺, where W = B · λ.
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
   * @param wrappedModel  The model to decorate (ownership transferred).
   * @param basisMatrices Per-contact basis matrices (in local frame).
   *                      Must have exactly N_CONTACTS elements.
   */
  BasisInputsModelDecorator(std::unique_ptr<MpcRobotModelBase<SCALAR_T>> wrappedModel,
                            const std::array<ContactWrenchConeBasisMatrix, N_CONTACTS>& basisMatrices)
      : Base(wrappedModel->modelSettings, wrappedModel->getStateDim(), computeInputDim(basisMatrices, wrappedModel->modelSettings)),
        wrappedModel_(std::move(wrappedModel)),
        numBasisPerFoot_(basisMatrices[0].numBasis()) {
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      B_local_[i] = basisMatrices[i].getBasisMatrix();
      B_pinv_local_[i] = basisMatrices[i].getBasisMatrixPseudoInverse();
    }
  }

  ~BasisInputsModelDecorator() override = default;
  BasisInputsModelDecorator* clone() const override { return new BasisInputsModelDecorator(*this); }

  /** Number of basis scalings per foot. */
  size_t getNumBasisPerFoot() const { return numBasisPerFoot_; }

  /** Local-frame basis matrix for the given contact. */
  const matrix_t& getBasisMatrix(size_t contactIndex) const { return B_local_[contactIndex]; }

  /** Local-frame pseudoinverse of the basis matrix. */
  const matrix_t& getBasisMatrixPseudoInverse(size_t contactIndex) const { return B_pinv_local_[contactIndex]; }

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
    return wrappedModel_->getGeneralizedVelocities(state, input);
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

  /******* Contact wrench accessors — the core of the decoration *******/

  /**
   * Reconstructs the full 6D wrench from basis-vector scalings: W = B * λ.
   * NOTE: This wrench is in the *local* contact frame. Callers must rotate to
   *       world frame if needed. However, since the wrapped model stores
   *       wrenches in the world frame, we return B * λ directly (the dynamics
   *       will handle the rotation).
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
   * Sets the wrench by computing λ = B⁺ * W and writing it into the input vector.
   * The pseudoinverse gives the minimum-norm λ that reproduces the desired wrench.
   */
  void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const override {
    assert(input.size() == this->input_dim);
    input.segment(getContactWrenchStartIndices(contactIndex), numBasisPerFoot_) =
        B_pinv_local_[contactIndex].template cast<SCALAR_T>() * wrench;
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

 private:
  BasisInputsModelDecorator(const BasisInputsModelDecorator& rhs)
      : Base(rhs),
        wrappedModel_(std::unique_ptr<MpcRobotModelBase<SCALAR_T>>(rhs.wrappedModel_->clone())),
        numBasisPerFoot_(rhs.numBasisPerFoot_),
        B_local_(rhs.B_local_),
        B_pinv_local_(rhs.B_pinv_local_) {}

  static size_t computeInputDim(const std::array<ContactWrenchConeBasisMatrix, N_CONTACTS>& basisMatrices,
                                const ModelSettings& modelSettings) {
    return basisMatrices[0].numBasis() * N_CONTACTS + modelSettings.mpc_joint_dim;
  }

  std::unique_ptr<MpcRobotModelBase<SCALAR_T>> wrappedModel_;
  const size_t numBasisPerFoot_;

  std::array<matrix_t, N_CONTACTS> B_local_;
  std::array<matrix_t, N_CONTACTS> B_pinv_local_;
};

}  // namespace ocs2::humanoid
