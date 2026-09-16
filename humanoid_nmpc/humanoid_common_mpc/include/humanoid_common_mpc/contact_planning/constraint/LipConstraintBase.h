/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <optional>

#include "humanoid_common_mpc/contact_planning/problem/LipConstraint.h"
#include "humanoid_common_mpc/contact_planning/problem/LipIndices.h"

namespace ocs2::humanoid {

/** Base of the planner's constraint terms: the bound LIP indices and the slack penalty of a soft term. */
class LipConstraintBase : public LipConstraint {
 public:
  void bind(const Layout& layout) override { idx_.bind(layout); }
  SlackPenalty slackPenalty() const override { return penalty_; }

 protected:
  /** The term's own penalty when given, the shared default otherwise; throws on a negative one. */
  void configurePenalty(const ContactPlanningConfig& config, const std::optional<SlackPenalty>& own, const char* term);
  /** "slack (Z, z)" for describe(). */
  std::string penaltyText() const;
  LipIndices idx_;
  SlackPenalty penalty_;
};

}  // namespace ocs2::humanoid
