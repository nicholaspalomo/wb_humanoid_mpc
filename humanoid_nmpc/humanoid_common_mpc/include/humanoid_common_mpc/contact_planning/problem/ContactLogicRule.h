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

#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicScan.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"

namespace ocs2::humanoid {

/**
 * A logical rule on the contact binaries, enforced outside the QP by propagation: one pass over the fixed prefix of a
 * partial assignment that fixes implied values and rejects contradictions. The problem runs the listed rules in order
 * to a fixpoint (ContactPlanningProblem::propagate), recomputing the shared scan of the prefix before every pass.
 */
class ContactLogicRule : public ContactPlanningTerm {
 public:
  /** Returns false on a contradiction. `changed` is set when a value was fixed. */
  virtual bool propagate(const ContactLogicState& state, const ContactLogicScan& scan, MiqpAssignment& assignment, bool& changed) const = 0;
};

}  // namespace ocs2::humanoid
