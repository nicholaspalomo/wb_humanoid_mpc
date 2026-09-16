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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/constraint/LipConstraintBase.h"

namespace ocs2::humanoid {

/** `foot_motion_in_swing_only` (hard, running nodes): +-dp_{ij} + M c_i <= M, a foot only moves while it is not in contact. */
class FootMotionInSwingOnlyConstraint final : public LipConstraintBase {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  NodeSet nodeSet() const override { return NodeSet::RUNNING; }
  Softness softness() const override { return Softness::HARD; }
  void addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const override;

 private:
  scalar_t bigM_ = 0.0;
};

}  // namespace ocs2::humanoid
