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
#include <optional>
#include <string>

#include "ocs2_core/Types.h"

#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "legged_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "legged_common_mpc/contact_planning/problem/ContactLogicRule.h"

namespace ocs2::legged {

/**
 * `minimum_double_support`, a minimum support count: after a touch-down at
 * node k, until k + n_ds (the node of the touch-down itself included), a foot
 * that is in contact may lift only while at least `minSupportCount` feet stay
 * in contact. The count defaults to every foot of the planner
 * (ContactLogicState::numFeet()), so that no foot in contact lifts during the
 * hold. For a biped that is the double support: after a touch-down the other
 * foot may not lift off before k + n_ds (a lift-off at the very node the other
 * foot lands would be a weight transfer with no double support at all). The
 * registry name and the configuration key (min_double_support_duration) keep
 * the biped's name.
 *
 * Not thread-safe, like every ContactPlanningTerm: the planner that owns it
 * configures and evaluates it from one thread at a time.
 */
class MinimumDoubleSupportRule final : public ContactLogicRule {
 public:
  /** The rule the registry builds: every foot of the planner is the minimum
   * support. */
  MinimumDoubleSupportRule() = default;
  /** A rule whose hold keeps at least `minSupportCount` feet in contact. A
   * count of at least the planner's foot count keeps every foot in contact
   * down, like the default. */
  explicit MinimumDoubleSupportRule(size_t minSupportCount)
      : minSupportCount_(minSupportCount) {}

  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool propagate(const ContactLogicState& state,
                 MiqpAssignment& a,
                 bool& changed) const override;

 private:
  scalar_t minDoubleSupportDuration_ = 0.0;
  // The feet that stay in contact during the hold; empty: every foot of the
  // planner.
  std::optional<size_t> minSupportCount_;
};

}  // namespace ocs2::legged
