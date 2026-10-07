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
 * Applies what the reference manager owns: terrain_height, which it takes (setTerrainHeight()) and builds the next
 * solve's swing trajectories and landing targets on, with the contact-implicit terms following in that same solve
 * (ContactImplicitApplier::beforeEverySolve()); and swing_trajectory_config, to its SwingTrajectoryPlanner. Both take
 * effect at the next solve, whose references are built after this reload. Without a reference manager, the swing
 * trajectories are not reloaded and a reloaded ground goes to the contact-implicit terms at once. A terrain height that
 * is not finite is refused by its field.
 *
 * Not thread-safe (HotFieldApplier); the reference manager's ground is an atomic.
 */
class ReferenceManagerApplier final : public HotFieldApplier {
 public:
  /** The fields it applies: terrain_height and swing_trajectory_config. */
  static absl::Span<const absl::string_view> staticFields();

  ReferenceManagerApplier() = default;

  absl::string_view name() const override { return "ReferenceManagerApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;
};

}  // namespace ocs2::humanoid
