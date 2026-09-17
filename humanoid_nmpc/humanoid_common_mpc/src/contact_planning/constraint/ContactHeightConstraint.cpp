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

#include "humanoid_common_mpc/contact_planning/constraint/ContactHeightConstraint.h"

#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::humanoid {

std::string ContactHeightConstraint::describe() const {
  std::ostringstream out;
  out << "|z - " << nominalHeight_ << "| <= " << tolerance_ << " + " << kHeightBigM << " (1 - c_i) per foot, " << penaltyText();
  return out.str();
}

void ContactHeightConstraint::configure(const ContactPlanningConfig& config) {
  if (config.contactHeight.tolerance < 0.0) throw std::invalid_argument("[contact_height] tolerance must be non-negative");
  if (config.shared.comHeight <= 0.0)
    throw std::invalid_argument("[contact_height] shared.comHeight must be resolved to a positive height");
  tolerance_ = config.contactHeight.tolerance;
  nominalHeight_ = config.shared.comHeight;
  configurePenalty(config, config.contactHeight.slack, "contact_height");
}

void ContactHeightConstraint::addRows(const ContactPlanningContext& /*ctx*/, int /*node*/, RowBuilder& rows) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (const scalar_t sign : {1.0, -1.0}) {
      // sign (z - z_nom) + M c_i <= tol + M
      rows.addSoft({{idx_.height, sign}}, {{idx_.contact[foot], kHeightBigM}}, -kLipLooseBound,
                   tolerance_ + kHeightBigM + sign * nominalHeight_, penalty_);
    }
  }
}

}  // namespace ocs2::humanoid
