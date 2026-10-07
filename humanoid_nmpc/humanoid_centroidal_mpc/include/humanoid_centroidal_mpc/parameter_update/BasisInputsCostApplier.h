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
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies what the basis-vector contact inputs of the centroidal MPC (humanoid_nmpc/docs/contact_basis_vectors/
 * README.md) add to a reload. input_weights is indexed in wrench space (forces, moments, joint velocities), while the
 * OCP input is [lambda, joint velocities], so R is its basis-space image R_basis = M^T R_wrench M + reg * blkdiag(S, 0),
 * with exactly the transform the OCP factory used (CentroidalMpcInterface::getBasisInputsCostTransformConfig()) and the
 * regularization's weight and shape reloaded (contacts.basis_scaling_regularization, contacts.basis_regularization); a
 * reload that would make the lambda block indefinite or names an unknown shape is refused by field and the running R
 * kept. The generator set is not reloaded: it fixes M and the input dimension. And it applies
 * contacts.basis_non_negativity_barrier to every foot's lambda >= 0 barrier.
 *
 * Without basis-vector inputs (no transform) it applies nothing: R is QuadraticCostWeightsApplier's then, which is
 * created with QuadraticCostWeightsApplier::InputCost::kBasisVectorInputs exactly when this one has a transform.
 * Not thread-safe (HotFieldApplier).
 */
class BasisInputsCostApplier final : public HotFieldApplier {
 public:
  /**
   * The fields it applies: input_weights (under basis-vector inputs), contacts.basis_scaling_regularization,
   * contacts.basis_regularization and contacts.basis_non_negativity_barrier.
   */
  static absl::Span<const absl::string_view> staticFields();

  /**
   * The applier of the running transform `basisCostTransform` (nullopt: the inputs are not basis vectors, and it applies
   * nothing) for an OCP input of `inputDim`, or the InvalidArgument that says which is inconsistent: inputDim is not the
   * transform's basisInputDim(), its map does not have wrenchInputDim rows, its numBasisInputs exceeds inputDim, or it
   * fails validateBasisInputsCostTransformConfig(). The setGains() R is written with performs no size check, so a
   * wrench-versus-basis dimension mix-up is refused here, at construction, rather than corrupting the input cost online.
   */
  static absl::StatusOr<std::unique_ptr<BasisInputsCostApplier>> Create(std::optional<BasisInputsCostTransformConfig> basisCostTransform,
                                                                        size_t inputDim);

  absl::string_view name() const override { return "BasisInputsCostApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;

 private:
  explicit BasisInputsCostApplier(std::optional<BasisInputsCostTransformConfig> basisCostTransform)
      : basisCostTransform_(std::move(basisCostTransform)) {}

  /// The running transform: its regularization follows the last reload that was applied. Empty without basis inputs.
  std::optional<BasisInputsCostTransformConfig> basisCostTransform_;
};

}  // namespace ocs2::humanoid
