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

#include "humanoid_wb_mpc_ros2/WBMpcPoseCommand.h"

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <ocs2_mpc/MPC_Settings.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

/** NotFound naming `path` when it does not exist; `what` says which input it is. */
absl::Status checkInputFileExists(absl::string_view what, const std::string& path) {
  if (!std::filesystem::exists(path)) {
    return absl::NotFoundError(absl::StrCat("[WBMpcPoseCommand] ", what, " not found: ", path));
  }
  return absl::OkStatus();
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> WBMpcPoseCommand::Create(const std::string& taskFile,
                                                                           const std::string& urdfFile,
                                                                           const std::string& referenceFile) {
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  RETURN_IF_ERROR(checkInputFileExists("reference file", referenceFile));
  ASSIGN_OR_RETURN(const bool verbose, ModelSettings::loadInterfaceVerbose(taskFile));
  // The loaders below report a malformed file by throwing; their messages name the key.
  try {
    // The prefix the whole-body MPC loads its model settings with (WBMpcInterface).
    std::unique_ptr<ModelSettings> modelSettings = std::make_unique<ModelSettings>(taskFile, urdfFile, "wb_mpc_", verbose);
    const scalar_t mpcHorizon = mpc::loadSettings(taskFile, "mpc", verbose).timeHorizon_;
    return std::unique_ptr<WBMpcPoseCommand>(new WBMpcPoseCommand(std::move(modelSettings), referenceFile, mpcHorizon));
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(
        absl::StrCat("[WBMpcPoseCommand] cannot load ", taskFile, ", ", urdfFile, " and ", referenceFile, ": ", error.what()));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcPoseCommand::WBMpcPoseCommand(std::unique_ptr<ModelSettings> modelSettings, const std::string& referenceFile, scalar_t mpcHorizon)
    : modelSettings_(std::move(modelSettings)),
      mpcRobotModel_(std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_)),
      calculator_(std::make_unique<WBMpcTargetTrajectoriesCalculator>(referenceFile, *mpcRobotModel_, mpcHorizon)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
TargetTrajectories WBMpcPoseCommand::toTargetTrajectories(const vector4_t& commandLineTarget, const SystemObservation& observation) {
  return calculator_->commandedPositionToTargetTrajectories(commandLineTarget, observation.time, observation.state);
}

}  // namespace ocs2::humanoid
