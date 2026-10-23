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

#include "humanoid_common_mpc/mrt/ContactEstimateIntake.h"

#include <algorithm>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

ContactEstimateIntake::ContactEstimateIntake(const char* absl_nonnull controller)
    : refusal_(makeControllerEvent(ControllerEventCode::kContactEstimateRefused,
                                   controller,
                                   /*value0=*/0.0,
                                   /*value1=*/static_cast<double>(kNumContacts))) {}

void ContactEstimateIntake::resetEstimator(absl::string_view estimatorName) {
  refusal_ = makeControllerEvent(ControllerEventCode::kContactEstimateRefused, refusal_.controller, /*value0=*/0.0,
                                 /*value1=*/static_cast<double>(kNumContacts), estimatorName);
  refusalReported_ = false;
}

bool ContactEstimateIntake::take(const std::vector<bool>& estimated, contact_flag_t& measured, ControllerEventSink& sink) {
  if (estimated.size() == kNumContacts) {
    std::copy(estimated.begin(), estimated.end(), measured.begin());
    refusalReported_ = false;
    return true;
  }
  numRefused_.fetch_add(1);
  if (!refusalReported_) {
    refusalReported_ = true;
    refusal_.values[0] = static_cast<double>(estimated.size());
    sink.post(refusal_);
  }
  return false;
}

absl::Status checkContactEstimator(robot::model::ContactEstimator& estimator,
                                   const robot::model::RobotState& robotState,
                                   absl::string_view estimatorName) {
  const std::vector<bool> flags = robot::model::estimateContactFlags(estimator, robotState);
  if (flags.size() != kNumContacts) {
    return absl::InvalidArgumentError(absl::StrCat("the contact estimator '", estimatorName, "' reports ", flags.size(),
                                                   " contact flags, but the controller has ", kNumContacts,
                                                   " contact points: it would never be taken"));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
