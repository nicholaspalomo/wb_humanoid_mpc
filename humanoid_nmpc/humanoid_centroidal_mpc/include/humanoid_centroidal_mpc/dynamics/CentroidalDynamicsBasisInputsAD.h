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
#include <string>

#include <pinocchio/multibody/fwd.hpp>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * CppAD-based centroidal dynamics where the MPC input is parameterized as
 * non-negative basis-vector scalings λ rather than direct contact wrenches:
 *
 *   u_basis = [λ_0, λ_1, ..., q̇_j],   W_i,local = B_i · λ_i
 *
 * The basis matrices B_i are defined in the *local contact frame* (see
 * ContactWrenchConeBasisMatrix), whereas the centroidal dynamics expect the
 * contact wrenches in the world frame. The CppAD tape therefore performs
 *
 *   W_i,world = blkdiag(w_R_l(q), w_R_l(q)) · B_i · λ_i
 *
 * with the contact frame rotation w_R_l(q) obtained from Pinocchio at the current
 * configuration q(x), and then evaluates the standard centroidal flow map with the
 * world-frame wrench input. Because the rotation is part of the tape, the linear
 * approximation captures both ∂f/∂λ and the additional ∂f/∂x contribution that
 * stems from the configuration-dependent rotation.
 *
 * The generated CppAD model name encodes the basis dimension and a hash of the
 * basis matrices, so cached libraries are never reused for a different basis.
 */
class CentroidalDynamicsBasisInputsAD final : public SystemDynamicsBaseAD {
 public:
  /**
   * @param pinocchioInterface The pinocchio model interface.
   * @param info               CentroidalModelInfo with the *original* wrench-based inputDim.
   * @param modelName          Base name for the CppAD model.
   * @param modelSettings      Build settings (CppAD model folder, recompile flags, contact names, etc.).
   * @param localBasisMatrices Per-contact basis matrices B_i (6 × numBasisPerFoot) in the local contact frame.
   */
  CentroidalDynamicsBasisInputsAD(const PinocchioInterface& pinocchioInterface,
                                  const CentroidalModelInfo& info,
                                  const std::string& modelName,
                                  const ModelSettings& modelSettings,
                                  const std::array<matrix_t, N_CONTACTS>& localBasisMatrices);

  ~CentroidalDynamicsBasisInputsAD() override = default;
  CentroidalDynamicsBasisInputsAD* clone() const override { return new CentroidalDynamicsBasisInputsAD(*this); }

  size_t getBasisInputDim() const { return basisInputDim_; }
  size_t getNumBasisPerFoot() const { return numBasisPerFoot_; }
  const std::array<matrix_t, N_CONTACTS>& getLocalBasisMatrices() const { return B_local_; }

  /**
   * Name of the generated CppAD model: modelName + "_basis<numBasisPerFoot>_<hash of all basis entries>".
   * Two different bases (dimension, friction coefficient, footprint, ...) therefore never share a cached library.
   */
  static std::string uniqueModelName(const std::string& modelName, const std::array<matrix_t, N_CONTACTS>& localBasisMatrices);

  /**
   * Converts a basis-vector input into the wrench-space input of the underlying centroidal model with all
   * contact wrenches rotated into the world frame. Exposed for testing.
   *
   * @param pinocchioInterface Interface whose frame placements have been updated for the configuration of interest.
   */
  template <typename SCALAR_T>
  VECTOR_T<SCALAR_T> toWorldFrameWrenchInput(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                             const CentroidalModelInfoTpl<SCALAR_T>& info,
                                             const VECTOR_T<SCALAR_T>& basisInput) const;

 protected:
  ad_vector_t systemFlowMap(ad_scalar_t time,
                            const ad_vector_t& state,
                            const ad_vector_t& input,
                            const ad_vector_t& parameters) const override;

 private:
  CentroidalDynamicsBasisInputsAD(const CentroidalDynamicsBasisInputsAD& rhs);

  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  CentroidalModelInfoCppAd infoCppAd_;

  std::array<matrix_t, N_CONTACTS> B_local_;
  std::array<pinocchio::FrameIndex, N_CONTACTS> contactFrameIndices_;
  size_t numBasisPerFoot_;
  size_t jointDim_;
  size_t basisInputDim_;
};

}  // namespace ocs2::humanoid
