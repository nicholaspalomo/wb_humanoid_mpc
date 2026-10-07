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

#include "humanoid_common_mpc/contact_planning/constraint/FootMotionInSwingOnlyConstraint.h"

#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

std::string FootMotionInSwingOnlyConstraint::describe() const {
  return absl::StrCat("+-dp_{ij} + M c_i <= M on the running nodes (a foot only moves while it is not in contact), M = ", bigM_);
}

void FootMotionInSwingOnlyConstraint::configure(const ContactPlanningConfig& config) {
  bigM_ = config.shared.bigM;
}

void FootMotionInSwingOnlyConstraint::addRows(const ContactPlanningContext& ctx, int /*node*/, RowBuilder& rows) const {
  const scalar_t M = ctx.bigM;
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    for (int axis = 0; axis < 2; ++axis) {
      for (const scalar_t sign : {1.0, -1.0}) {
        rows.addHard(/*xCoefficients=*/{}, {{idx_.footDelta[foot][axis], sign}, {idx_.contact[foot], M}}, -kLipLooseBound, M);
      }
    }
  }
}

}  // namespace ocs2::humanoid
