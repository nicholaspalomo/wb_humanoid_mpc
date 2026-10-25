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

#include "humanoid_common_mpc_app/robot/StoredConfigFileKinds.h"

#include <cstddef>
#include <optional>

#include "absl/strings/string_view.h"

#include "humanoid_mpc_msgs/config_file_kind.nproto.h"

namespace ocs2::humanoid {

std::optional<size_t> storedConfigFileKindIndex(msgs::ConfigFileKind kind) {
  for (size_t index = 0; index < kStoredConfigFileKinds.size(); ++index) {
    if (kStoredConfigFileKinds[index] == kind) return index;
  }
  return std::nullopt;
}

absl::string_view configFileKindName(msgs::ConfigFileKind kind) {
  switch (kind) {
    case msgs::ConfigFileKind::kTask:
      return "task file";
    case msgs::ConfigFileKind::kReference:
      return "reference file";
    case msgs::ConfigFileKind::kJointPdGains:
      return "PD gains file";
    case msgs::ConfigFileKind::kUnspecified:
      return "file of no kind";
  }
  // A number outside the enumerators, cast from the wire.
  return "file of no kind";
}

}  // namespace ocs2::humanoid
