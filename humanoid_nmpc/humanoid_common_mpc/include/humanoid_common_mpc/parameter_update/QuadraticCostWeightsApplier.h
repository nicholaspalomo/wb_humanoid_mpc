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

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies the weights of the quadratic costs: Q of state_weights to the running quadratic state cost
 * (stateInputQuadraticCost or stateQuadraticCost), R of input_weights to the running quadratic input cost
 * (stateInputQuadraticCost or inputQuadraticCost), and terminal_cost_scaling times final_state_weights to the quadratic
 * terminal cost (terminalCost). Under the CoM + ACoM tracking cost the base-pose blocks of Q and Q_final are zeroed
 * first, as the factory does, because that cost regulates the base pose; whether the problem carries it is read off the
 * running problem, never the reloaded file. A Q whose size is not the problem's state dimension, and an R whose size is
 * not its input dimension, are refused by field.
 *
 * Shared by both formulations. Not thread-safe (HotFieldApplier).
 */
class QuadraticCostWeightsApplier final : public HotFieldApplier {
 public:
  /** What the problem's inputs are, which decides who writes R. */
  enum class InputCost {
    // The inputs are what input_weights is indexed on (contact wrenches, then joint velocities or accelerations): R is
    // input_weights, written here.
    kInputWeights,
    // The inputs are basis-vector scalings of the contact wrenches: R is the basis-space image of input_weights, which
    // the centroidal BasisInputsCostApplier writes with its regularization; this applier leaves it alone.
    kBasisVectorInputs,
  };

  /** The fields it applies: state_weights, input_weights, final_state_weights and terminal_cost_scaling. */
  static absl::Span<const absl::string_view> staticFields();

  explicit QuadraticCostWeightsApplier(InputCost inputCost) : inputCost_(inputCost) {}

  absl::string_view name() const override { return "QuadraticCostWeightsApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;

 private:
  InputCost inputCost_;
};

}  // namespace ocs2::humanoid
