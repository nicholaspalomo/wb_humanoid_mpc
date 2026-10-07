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

#include "humanoid_common_mpc_app/node/MpcFiles.h"

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::node {

absl::Status validateFileFlag(absl::string_view flag, const std::string& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(flag, " is not given; name the robot's file"));
  }
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return absl::NotFoundError(absl::StrCat(flag, "=", path, ": no such file"));
  }
  return absl::OkStatus();
}

absl::Status validateMpcFiles(const MpcFiles& files) {
  const std::pair<absl::string_view, const std::string* absl_nonnull> flags[] = {
      {"--task_file", &files.taskFile},
      {"--reference_file", &files.referenceFile},
      {"--urdf_file", &files.urdfFile},
      {"--gait_file", &files.gaitFile},
  };
  for (const std::pair<absl::string_view, const std::string* absl_nonnull>& flag : flags) {
    const absl::Status valid = validateFileFlag(flag.first, *flag.second);
    if (!valid.ok()) return valid;
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::node
