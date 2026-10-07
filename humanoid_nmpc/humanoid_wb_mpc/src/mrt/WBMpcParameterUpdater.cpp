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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_wb_mpc/mrt/WBMpcParameterUpdater.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"

namespace ocs2::humanoid {

absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeWholeBodyMpcParameterUpdater(
    MPC_BASE* absl_nullable mpc,
    const WBMpcInterface& interface,
    const std::string& taskFile,
    const std::string& referenceFile,
    std::vector<MpcParameterUpdaterModule::ReferenceFileReloader> referenceFileReloaders) {
  MpcParameterUpdaterModule::Options options;
  options.taskFile = taskFile;
  options.referenceFile = referenceFile;
  options.runningTask = interface.taskFile();
  options.layout = stateInputLayout(interface.modelSettings(), StateInputLayout::Mpc::kWholeBody);
  options.inputDim = interface.getMpcRobotModel().getInputDim();
  options.referenceManager = interface.getSwitchedModelReferenceManagerPtr().get();
  ASSIGN_OR_RETURN(options.appliers, wholeBodyHotFieldAppliers(interface));
  ASSIGN_OR_RETURN(std::unique_ptr<MpcParameterUpdaterModule> updater, MpcParameterUpdaterModule::Create(mpc, std::move(options)));
  for (MpcParameterUpdaterModule::ReferenceFileReloader& reloader : referenceFileReloaders) {
    updater->addReferenceFileReloader(std::move(reloader));
  }
  return std::shared_ptr<MpcParameterUpdaterModule>(std::move(updater));
}

}  // namespace ocs2::humanoid
