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

#include "legged_common_mpc/contact_planning/constraint/NoFlightConstraint.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/LipCoefficients.h"
#include "legged_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::legged {

namespace {
// The lower bound of the contact flags' sum: at least one foot in contact.
constexpr scalar_t kMinContactFlagSum = 1.0;
}  // namespace

std::string NoFlightConstraint::describe() const {
  return absl::StrCat("1 <= sum_i c_i <= ", idx_.contact.size(),
                      " on the running nodes");
}

void NoFlightConstraint::addRows(const ContactPlanningContext& /*ctx*/,
                                 int /*node*/,
                                 RowBuilder& rows) const {
  // One row over the contact binaries of every foot of the layout, in foot
  // order; its upper bound, every foot in contact, is the foot count.
  Coefficients uCoefficients;
  uCoefficients.reserve(idx_.contact.size());
  for (const int contactIndex : idx_.contact) {
    uCoefficients.emplace_back(contactIndex, 1.0);
  }
  rows.addHard(/*xCoefficients=*/{}, uCoefficients,
               /*lower=*/kMinContactFlagSum,
               /*upper=*/static_cast<scalar_t>(idx_.contact.size()));
}

}  // namespace ocs2::legged
