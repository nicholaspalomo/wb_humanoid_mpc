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

#include "humanoid_common_mpc/contact_planning/constraint/YawTorqueBudgetConstraint.h"

#include <cmath>
#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/LipCoefficients.h"
#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::humanoid {

static_assert(kNumContacts == kBipedFootCount,
              "the yaw torque budget rows are written for a biped");

std::string YawTorqueBudgetConstraint::describe() const {
  return absl::StrCat(
      "|tau_i| <= B(c_i, c_other) on the running nodes: T_t alone, (T_t + T_c) "
      "/ 2 in double support, 0 swinging or in "
      "flight (the concave envelope over the binaries), T_t = ",
      params_.maxTorqueFootEwrtFootZ,
      " N m, T_c = ", params_.doubleSupportMaxTorqueFeetmeanEwrtWorldZ,
      " N m (from the model)");
}

void YawTorqueBudgetConstraint::configure(const ContactPlanningConfig& config) {
  params_ = config.yawTorqueBudgetParameters;
}

void YawTorqueBudgetConstraint::addRows(const ContactPlanningContext& /*ctx*/,
                                        int /*node*/,
                                        RowBuilder& rows) const {
  // The budget of foot i over its own binary c_i and the other foot's c_o is
  // B(1, 0) = T_t (alone), B(1, 1) = D = (T_t + T_c) / 2 (double support), B(0,
  // 1) = B(0, 0) = 0 (swinging, flight). B is bilinear in the binaries, B = T_t
  // c_i + (D - T_t) c_i c_o, so it is written as the minimum of the two planes
  // through three corners each that over-estimate it at the fourth - its
  // concave envelope, exact at every corner and the tightest relaxation there
  // is:
  //   D >= T_t:  |tau_i| <= T_t c_i + (D - T_t) c_o  and  |tau_i| <= D c_i;
  //   D <  T_t:  |tau_i| <= T_t c_i + (D - T_t) (c_i + c_o - 1)  and  |tau_i|
  //   <= T_t c_i.
  // A single row through (1, 0), (0, 1) and (1, 1) used to stand for all of it,
  // which put -(D - T_t) on flight: with T_c < T_t a foot in the air was
  // allowed a yaw torque, and with T_c > T_t flight was infeasible outright.
  // The single- and double-support yaw-torque bounds of a foot, and the excess
  // of the double-support one over the single-support one.
  const scalar_t maxTorqueFootEwrtFootZ = params_.maxTorqueFootEwrtFootZ;
  const scalar_t doubleSupportMaxTorqueFootEwrtFootZ =
      0.5 * (params_.maxTorqueFootEwrtFootZ +
             params_.doubleSupportMaxTorqueFeetmeanEwrtWorldZ);
  const scalar_t excessMaxTorqueFootEwrtFootZ =
      doubleSupportMaxTorqueFootEwrtFootZ - maxTorqueFootEwrtFootZ;
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const size_t other = 1 - foot;
    for (const scalar_t sign : {1.0, -1.0}) {
      if (excessMaxTorqueFootEwrtFootZ >= 0.0) {
        rows.addHard(/*xCoefficients=*/{},
                     {{idx_.torqueFeetEwrtWorldZ[foot], sign},
                      {idx_.contact[foot], -maxTorqueFootEwrtFootZ},
                      {idx_.contact[other], -excessMaxTorqueFootEwrtFootZ}},
                     -kLipLooseBound, /*upper=*/0.0);
        rows.addHard(
            /*xCoefficients=*/{},
            {{idx_.torqueFeetEwrtWorldZ[foot], sign},
             {idx_.contact[foot], -doubleSupportMaxTorqueFootEwrtFootZ}},
            -kLipLooseBound,
            /*upper=*/0.0);
      } else {
        rows.addHard(/*xCoefficients=*/{},
                     {{idx_.torqueFeetEwrtWorldZ[foot], sign},
                      {idx_.contact[foot], -(maxTorqueFootEwrtFootZ +
                                             excessMaxTorqueFootEwrtFootZ)},
                      {idx_.contact[other], -excessMaxTorqueFootEwrtFootZ}},
                     -kLipLooseBound, -excessMaxTorqueFootEwrtFootZ);
        rows.addHard(
            /*xCoefficients=*/{},
            {{idx_.torqueFeetEwrtWorldZ[foot], sign},
             {idx_.contact[foot], -maxTorqueFootEwrtFootZ}},
            -kLipLooseBound,
            /*upper=*/0.0);
      }
    }
  }
}

}  // namespace ocs2::humanoid
