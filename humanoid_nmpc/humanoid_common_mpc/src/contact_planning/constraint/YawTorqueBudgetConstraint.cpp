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

#include "humanoid_common_mpc/contact_planning/constraint/YawTorqueBudgetConstraint.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the yaw torque budget rows are written for a biped");

std::string YawTorqueBudgetConstraint::describe() const {
  std::ostringstream out;
  out << "|tau_i| <= T_t c_i + (T_c - T_t) (c_L + c_R - 1) / 2 on the running nodes, T_t = " << params_.torsionalFrictionTorque
      << " N m, T_c = " << params_.doubleSupportYawCouple << " N m (from the model)";
  return out.str();
}

void YawTorqueBudgetConstraint::configure(const ContactPlanningConfig& config) {
  if (config.yawTorqueBudget.torsionalFrictionTorque < 0.0 || config.yawTorqueBudget.doubleSupportYawCouple < 0.0) {
    throw std::invalid_argument("[yaw_torque_budget] yaw torque limits must be >= 0");
  }
  params_ = config.yawTorqueBudget;
  flightPossible_ = config.formulation.hasFlightModel();
}

void YawTorqueBudgetConstraint::addRows(const ContactPlanningContext& /*ctx*/, int /*node*/, RowBuilder& rows) const {
  const scalar_t torsion = params_.torsionalFrictionTorque;
  // A pair of feet never carries less torsion than one of them alone; a couple below the torsion is no bonus.
  const scalar_t doubleSupportShare = std::max(0.0, 0.5 * (params_.doubleSupportYawCouple - torsion));
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (const scalar_t sign : {1.0, -1.0}) {
      if (!flightPossible_) {
        rows.addHard({},
                     {{idx_.yawTorque[foot], sign},
                      {idx_.contact[foot], -torsion},
                      {idx_.contact[0], -doubleSupportShare},
                      {idx_.contact[1], -doubleSupportShare}},
                     -kLipLooseBound, -doubleSupportShare);
        continue;
      }
      // +-tau_i <= T_t c_i + share min_j c_j, as one row per j: the intersection is the minimum, which is the product
      // c_L c_R at every integer point and therefore the same budget as above wherever a foot is down.
      for (size_t other = 0; other < N_CONTACTS; ++other) {
        rows.addHard({}, {{idx_.yawTorque[foot], sign}, {idx_.contact[foot], -torsion}, {idx_.contact[other], -doubleSupportShare}},
                     -kLipLooseBound, 0.0);
      }
    }
  }
}

}  // namespace ocs2::humanoid
