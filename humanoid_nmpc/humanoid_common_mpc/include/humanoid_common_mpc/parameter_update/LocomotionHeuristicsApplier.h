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

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies the COEFFICIENTS of the locomotion heuristics (humanoid_nmpc/docs/locomotion_heuristics/README.md) to the
 * LocomotionHeuristicLayer, exactly as the cost weights beside them - all of them or, if any is rejected, none. Which
 * heuristics are listed is structural and is not reloaded; the layer says so once per distinct edit to a list. The layer
 * lives on the reference manager, which is shared rather than cloned per worker, so one reconfigure() reaches every
 * thread's view of it; it is safe from the solver thread because the reload runs before any worker of the next solve.
 *
 * Not thread-safe (HotFieldApplier).
 */
class LocomotionHeuristicsApplier final : public HotFieldApplier {
 public:
  /** The fields it applies: the ten coefficient blocks of locomotion_heuristics (not its three lists). */
  static absl::Span<const absl::string_view> staticFields();

  /**
   * The applier of `layer`, the MPC interface's (shared with its reference manager, which holds it as a shared_ptr); null:
   * the block is launch-time only and nothing is applied.
   */
  explicit LocomotionHeuristicsApplier(std::shared_ptr<LocomotionHeuristicLayer> layer) : layer_(std::move(layer)) {}

  absl::string_view name() const override { return "LocomotionHeuristicsApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;

 private:
  std::shared_ptr<LocomotionHeuristicLayer> layer_;
};

}  // namespace ocs2::humanoid
