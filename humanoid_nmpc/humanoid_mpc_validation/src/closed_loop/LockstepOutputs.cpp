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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_mpc_validation/closed_loop/LockstepOutputs.h"

#include <filesystem>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

std::string metricsFileName(const std::string& robot, const std::string& scenario) {
  return absl::StrCat(robot, "_", scenario, ".json");
}

std::string timeSeriesFileName(const std::string& robot, const std::string& scenario) {
  return absl::StrCat(robot, "_", scenario, "_timeseries.txt");
}

std::string recordedStatesFileName(const std::string& robot, const std::string& scenario) {
  return absl::StrCat(robot, "_", scenario, "_states.txt");
}

absl::Status writeLockstepOutputs(const LockstepResult& result,
                                  const std::string& outputDir,
                                  const std::string& robot,
                                  const std::string& scenario,
                                  const GoldenProvenance& provenance) {
  const std::filesystem::path directory(outputDir);
  RETURN_IF_ERROR(writeJsonFile((directory / metricsFileName(robot, scenario)).string(), result.metrics));
  GoldenFile timeSeries = result.timeSeries;
  GoldenProvenance seriesProvenance = provenance;
  for (const std::pair<std::string, std::string>& note : timeSeries.provenance.notes) seriesProvenance.notes.push_back(note);
  timeSeries.provenance = std::move(seriesProvenance);
  RETURN_IF_ERROR(writeGoldenFile((directory / timeSeriesFileName(robot, scenario)).string(), timeSeries));
  if (!result.recordedStates.records.empty()) {
    GoldenProvenance statesProvenance = provenance;
    statesProvenance.notes.emplace_back("robot", robot);
    statesProvenance.notes.emplace_back("scenario", scenario);
    RETURN_IF_ERROR(writeGoldenFile((directory / recordedStatesFileName(robot, scenario)).string(),
                                    toGoldenFile(result.recordedStates, std::move(statesProvenance))));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::validation
