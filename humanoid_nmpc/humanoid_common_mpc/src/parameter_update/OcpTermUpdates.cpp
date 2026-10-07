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

#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

#include <exception>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"

namespace ocs2::humanoid {

absl::Status exceptionsToStatus(absl::FunctionRef<void()> call) {
  try {  // NOLINT(exceptions): the boundary to OCS2's throwing term API, converted to a Status at once.
    call();
  } catch (const std::exception& e) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InternalError(e.what());
  } catch (...) {  // NOLINT(exceptions): the boundary of the try above; no exception may escape preSolverRun().
    return absl::UnknownError("an exception that is not a std::exception");
  }
  return absl::OkStatus();
}

void reportFailedTermUpdate(absl::string_view name, const absl::Status& status) {
  LOG(WARNING) << "[MpcParameterUpdaterModule] Failed to update " << name << ": " << status.message();
}

void reportNotApplied(absl::string_view source, absl::string_view what, const absl::Status& status) {
  LOG(WARNING) << "[MpcParameterUpdaterModule] " << what << " of " << source
               << " was not applied, the running values are kept: " << status.message();
}

void setSoftTermWeight(std::vector<OptimalControlProblem>& problems,
                       const std::vector<std::string>& contactNames,
                       absl::string_view termSuffix,
                       scalar_t weight) {
  const vector_t scaleParam = (vector_t(1) << weight).finished();
  for (OptimalControlProblem& ocp : problems) {
    for (const std::string& footName : contactNames) {
      updateTermIfPresent<StateInputSoftConstraint>(*ocp.softConstraintPtr, absl::StrCat(footName, termSuffix),
                                                    [&](StateInputSoftConstraint& softCon) { setPenaltyParameters(softCon, scaleParam); });
    }
  }
}

}  // namespace ocs2::humanoid
