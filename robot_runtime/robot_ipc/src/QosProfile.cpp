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

#include "robot_ipc/QosProfile.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace robot::ipc {
namespace {

// LINT.IfChange(qos_profile_names)
constexpr absl::string_view kLatestName = "latest";
constexpr absl::string_view kReliableLatestName = "reliable_latest";
constexpr absl::string_view kReliableEventsName = "reliable_events";
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/qos_profile.py:qos_profile_names)

}  // namespace

absl::string_view qosProfileName(QosProfile profile) {
  switch (profile) {
    case QosProfile::kLatest:
      return kLatestName;
    case QosProfile::kReliableLatest:
      return kReliableLatestName;
    case QosProfile::kReliableEvents:
      return kReliableEventsName;
  }
  return "unknown";
}

absl::StatusOr<QosProfile> parseQosProfile(absl::string_view name) {
  std::vector<absl::string_view> names;
  for (const QosProfile profile : kAllQosProfiles) {
    if (qosProfileName(profile) == name) {
      return profile;
    }
    names.push_back(qosProfileName(profile));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown QoS profile '", name,
                   "' (valid: ", absl::StrJoin(names, ", "), ")"));
}

QosPolicy qosPolicy(QosProfile profile) {
  switch (profile) {
    case QosProfile::kLatest:
      return QosPolicy{.reliability = Reliability::kBestEffort,
                       .writerHistoryDepth = 1};
    case QosProfile::kReliableLatest:
      return QosPolicy{.reliability = Reliability::kReliable,
                       .writerHistoryDepth = 1};
    case QosProfile::kReliableEvents:
      return QosPolicy{.reliability = Reliability::kReliable,
                       .writerHistoryDepth = kReliableEventsHistoryDepth};
  }
  return QosPolicy{};
}

}  // namespace robot::ipc
