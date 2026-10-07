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

#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"

#include <cstdlib>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

namespace ocs2::humanoid {

absl::Status checkRetiredMpcLinkFlag(absl::string_view value) {
  if (value.empty()) return absl::OkStatus();
  return absl::FailedPreconditionError(
      absl::StrCat("--mpc_link=", value,
                   ": the flag is retired. The robot process reaches its MPC over the bus only, in simulation as on the robot: start the "
                   "MPC node of the formulation (humanoid_centroidal_mpc_node or humanoid_wb_mpc_node; make launch-<robot>-sim runs both "
                   "sides) and drop --mpc_link from the command line."));
}

absl::StatusOr<std::vector<int>> parseCoreList(absl::string_view value, const std::vector<int>& defaultCores) {
  const absl::string_view trimmed = absl::StripAsciiWhitespace(value);
  if (trimmed == "default") return defaultCores;
  if (trimmed.empty() || trimmed == "none") return std::vector<int>();
  std::vector<int> cores;
  for (absl::string_view part : absl::StrSplit(trimmed, ',')) {
    int core = -1;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(part), &core) || core < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("'", value, "' names no cores: give `default`, `none` or a comma-separated list of CPU numbers such as 4,5"));
    }
    cores.push_back(core);
  }
  return cores;
}

}  // namespace ocs2::humanoid
