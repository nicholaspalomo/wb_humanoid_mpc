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

#include "humanoid_common_mpc/contact_planning/constraint/ZmpPinnedInFlightConstraint.h"

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the flight rows are written for a biped");

std::string ZmpPinnedInFlightConstraint::describe() const {
  return "|zmp - c| <= bigM (c_L + c_R) per axis: no horizontal acceleration in flight";
}

void ZmpPinnedInFlightConstraint::addRows(const ContactPlanningContext& ctx, int /*node*/, RowBuilder& rows) const {
  const scalar_t M = ctx.bigM;
  for (int axis = 0; axis < 2; ++axis) {
    for (const scalar_t sign : {1.0, -1.0}) {
      // sign (zmp - c) - M c_L - M c_R <= 0
      rows.addHard({{idx_.com[axis], -sign}}, {{idx_.zmp[axis], sign}, {idx_.contact[0], -M}, {idx_.contact[1], -M}}, -kLipLooseBound, 0.0);
    }
  }
}

}  // namespace ocs2::humanoid
