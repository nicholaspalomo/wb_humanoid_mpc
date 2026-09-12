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

#include <ocs2_centroidal_model/PinocchioCentroidalDynamicsAD.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * CppAD-based centroidal dynamics where the MPC input is parameterized as
 * non-negative basis-vector scalings λ rather than direct contact wrenches.
 *
 * The CppAD tape inlines the linear mapping  u_wrench = M * u_basis,
 * where M is a block-diagonal matrix that maps each contact's λ-block through
 * its basis matrix B and passes joint velocities through unchanged.
 *
 * The mapping matrix M has the structure:
 *   M = [ B_0   0     0           ]
 *       [ 0     B_1   0           ]
 *       [ 0     0     I_{n_joints}]
 *
 * This ensures the CppAD derivative tape captures the full chain rule
 * df/du_basis = df/du_wrench * M.
 */
class CentroidalDynamicsBasisInputsAD final : public SystemDynamicsBase {
 public:
  /**
   * @param pinocchioInterface The pinocchio model interface.
   * @param info               CentroidalModelInfo with the *original* wrench-based inputDim.
   * @param modelName          Unique name for the CppAD model.
   * @param modelSettings      Build settings (CppAD model folder, recompile flags, etc.).
   * @param basisInputDim      The total dimension of the basis-vector parameterized input.
   * @param basisToWrenchMap   The M matrix (basisInputDim → original inputDim).
   */
  CentroidalDynamicsBasisInputsAD(const PinocchioInterface& pinocchioInterface,
                                  const CentroidalModelInfo& info,
                                  const std::string& modelName,
                                  const ModelSettings& modelSettings,
                                  size_t basisInputDim,
                                  matrix_t basisToWrenchMap);

  ~CentroidalDynamicsBasisInputsAD() override = default;
  CentroidalDynamicsBasisInputsAD* clone() const override { return new CentroidalDynamicsBasisInputsAD(*this); }

  vector_t computeFlowMap(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) override;
  VectorFunctionLinearApproximation linearApproximation(scalar_t time,
                                                        const vector_t& state,
                                                        const vector_t& input,
                                                        const PreComputation& preComp) override;

 private:
  CentroidalDynamicsBasisInputsAD(const CentroidalDynamicsBasisInputsAD& rhs) = default;

  PinocchioCentroidalDynamicsAD pinocchioCentroidalDynamicsAd_;

  /// Maps basis-vector input to wrench-based input:  u_wrench = M_ * u_basis
  matrix_t M_;
};

}  // namespace ocs2::humanoid
