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

#include "humanoid_multi_contact/stability/JointLimits.h"

#include <cstddef>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::multi_contact {

absl::StatusOr<JointLimits> jointLimitsOf(absl::string_view name) {
  for (size_t i = 0; i < kJointLimitsNames.size(); ++i) {
    if (kJointLimitsNames[i] == name) return static_cast<JointLimits>(i);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown joint limits '", name,
                   "' (known: ", absl::StrJoin(kJointLimitsNames, ", "), ")"));
}

absl::string_view jointLimitsName(JointLimits limits) {
  switch (limits) {
    // kJointLimitsNames lists the names in the order of the values.
    case JointLimits::kPositionAndEffort:
    case JointLimits::kPositionOnly:
    case JointLimits::kRelaxedPosition:
      return kJointLimitsNames[static_cast<size_t>(limits)];
  }
  LOG(FATAL) << "unknown JointLimits " << static_cast<int>(limits);
}

bool enforcesEffortLimits(JointLimits limits) {
  return limits == JointLimits::kPositionAndEffort;
}

bool relaxesPositionLimits(JointLimits limits) {
  return limits == JointLimits::kRelaxedPosition;
}

}  // namespace ocs2::humanoid::multi_contact
