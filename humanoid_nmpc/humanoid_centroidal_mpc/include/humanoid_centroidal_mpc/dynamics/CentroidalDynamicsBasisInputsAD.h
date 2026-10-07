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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <array>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/dynamics/SystemDynamicsBaseAD.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/fwd.hpp"

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
 * The generated CppAD model name carries basisInputsLibraryKey() of the basis, so the DYNAMICS library is never
 * reused for a different basis. The other taped terms built on the basis-vector input (their domain and, for some,
 * their body depend on it too) are protected the same way because CentroidalMpcInterface keys the whole CppAD model
 * folder by that key (CentroidalMpcInterface::keyCppAdModelFolder).
 */
class CentroidalDynamicsBasisInputsAD final : public SystemDynamicsBaseAD {
 public:
  /**
   * Validates the arguments, then builds (and, unless cached, compiles) the CppAD model. Configuration errors are an
   * InvalidArgumentError naming the task-file key to change.
   *
   * @param pinocchioInterface The pinocchio model interface.
   * @param info               CentroidalModelInfo with the *original* wrench-based inputDim.
   * @param modelName          Base name for the CppAD model.
   * @param modelSettings      Build settings (CppAD model folder, recompile flags, contact names, etc.).
   * @param localBasisMatrices Per-contact basis matrices B_i (6 × numBasisPerFoot) in the local contact frame.
   */
  static absl::StatusOr<std::unique_ptr<CentroidalDynamicsBasisInputsAD>> Create(
      const PinocchioInterface& pinocchioInterface,
      const CentroidalModelInfo& info,
      const std::string& modelName,
      const ModelSettings& modelSettings,
      const std::array<matrix_t, kNumContacts>& localBasisMatrices);

  /** The checks Create() runs before it builds anything. */
  static absl::Status validate(const PinocchioInterface& pinocchioInterface,
                               const CentroidalModelInfo& info,
                               const ModelSettings& modelSettings,
                               const std::array<matrix_t, kNumContacts>& localBasisMatrices);

  ~CentroidalDynamicsBasisInputsAD() override = default;
  CentroidalDynamicsBasisInputsAD* absl_nonnull clone() const override { return new CentroidalDynamicsBasisInputsAD(*this); }
  // Copied only by clone(), whose copy constructor is private; never assigned or moved.
  CentroidalDynamicsBasisInputsAD& operator=(const CentroidalDynamicsBasisInputsAD&) = delete;
  CentroidalDynamicsBasisInputsAD(CentroidalDynamicsBasisInputsAD&&) = delete;
  CentroidalDynamicsBasisInputsAD& operator=(CentroidalDynamicsBasisInputsAD&&) = delete;

  size_t getBasisInputDim() const { return basisInputDim_; }
  size_t getNumBasisPerFoot() const { return numBasisPerFoot_; }

  /**
   * Name of the generated CppAD model: modelName + "_" + basisInputsLibraryKey(localBasisMatrices), i.e.
   * modelName + "_basis<numBasisPerFoot>_<16 hex digits of the basis content hash>". Two different bases (generator
   * set, dimension, friction coefficient, footprint, ...) therefore never share a cached dynamics library.
   */
  static std::string uniqueModelName(const std::string& modelName, const std::array<matrix_t, kNumContacts>& localBasisMatrices);

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
  /** Builds the model from arguments Create() has validated; use Create(). */
  CentroidalDynamicsBasisInputsAD(const PinocchioInterface& pinocchioInterface,
                                  const CentroidalModelInfo& info,
                                  const std::string& modelName,
                                  const ModelSettings& modelSettings,
                                  const std::array<matrix_t, kNumContacts>& localBasisMatrices);

  CentroidalDynamicsBasisInputsAD(const CentroidalDynamicsBasisInputsAD& rhs);

  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  CentroidalModelInfoCppAd infoCppAd_;

  std::array<matrix_t, kNumContacts> B_local_;
  std::array<pinocchio::FrameIndex, kNumContacts> contactFrameIndices_;
  size_t numBasisPerFoot_;
  size_t jointDim_;
  size_t basisInputDim_;
};

}  // namespace ocs2::humanoid
