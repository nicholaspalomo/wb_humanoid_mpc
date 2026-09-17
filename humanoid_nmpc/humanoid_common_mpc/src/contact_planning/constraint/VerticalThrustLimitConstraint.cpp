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

#include "humanoid_common_mpc/contact_planning/constraint/VerticalThrustLimitConstraint.h"

#include <sstream>

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the thrust row is written for a biped");

std::string VerticalThrustLimitConstraint::describe() const {
  std::ostringstream out;
  out << "az + g <= a_max (c_L + c_R), a_max = " << maxContactAcceleration_ << " m/s^2: ballistic in flight, thrust-bounded in contact";
  return out.str();
}

void VerticalThrustLimitConstraint::configure(const ContactPlanningConfig& config) {
  maxContactAcceleration_ = config.verticalDoubleIntegrator.maxContactAcceleration;
  gravity_ = config.shared.gravity;
}

void VerticalThrustLimitConstraint::addRows(const ContactPlanningContext& /*ctx*/, int /*node*/, RowBuilder& rows) const {
  // az - a_max c_L - a_max c_R <= -g
  rows.addHard({}, {{idx_.heightAccel, 1.0}, {idx_.contact[0], -maxContactAcceleration_}, {idx_.contact[1], -maxContactAcceleration_}},
               -kLipLooseBound, -gravity_);
}

}  // namespace ocs2::humanoid
