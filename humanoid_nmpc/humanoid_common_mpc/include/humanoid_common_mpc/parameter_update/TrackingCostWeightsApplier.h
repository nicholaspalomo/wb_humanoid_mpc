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
 * Applies the weights of the tracking costs on named links and legs: task_space_costs to each listed link's
 * EndEffectorKinematicsQuadraticCost (taskSpaceKinematicsCostName(name)), and left_leg_torque_cost / right_leg_torque_cost
 * to the ExternalTorqueQuadraticCostAD of the first and the second contact (externalTorqueCostName()). The terms the
 * running problem does not carry are skipped. Which links are tracked is structural: a reloaded entry whose link cost the
 * running problem does not carry (added or renamed), and a carried link cost that no entry names any more, are reported
 * by the reload (HotUpdateTarget::reportNotApplied()) as taking effect at the next start, and change no term. Not
 * thread-safe (HotFieldApplier).
 */
class TrackingCostWeightsApplier final : public HotFieldApplier {
 public:
  /** The fields it applies: task_space_costs, left_leg_torque_cost and right_leg_torque_cost. */
  static absl::Span<const absl::string_view> staticFields();

  TrackingCostWeightsApplier() = default;

  absl::string_view name() const override { return "TrackingCostWeightsApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;
};

}  // namespace ocs2::humanoid
